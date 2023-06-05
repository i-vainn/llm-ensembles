from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt_neox import GPTNeoXForCausalLM
from .models import GPTNeoXEnsembleForCausalLM, DecoderEnsemble
from .utils import TextDataset
from .utils import get_computeprob_response


class ModelProvider:
    """Base class for a model provider, currently we support API models, but this class can be subclassed with support
    for local GPU models"""
    def batched_conditional_logprobs(self, reqs, batch_size):
        """Calculates conditional logprobs for each request. Each request is of the form:
        {"context": context, "completion": completion}
        The completion should start with a space where applicable (For example middle of sentence or after an 'Answer:'
        directive)
        """
        raise NotImplementedError


class APIModelProvider(ModelProvider):
    """Provider that calculates conditional logprobs through a REST API"""

    def _logprobs(self, text):
        """Runs the given text through the API and calculates logprobs for the full text.
        Returns the (possibly different) decoded text, and a list of {"token": str, "logprob": float, "offset": int}
        for each token
        """
        raise NotImplementedError

    def _find_completion_start_token(self, context, token_data):
        # Finds the offset of the first token that contains the completion text
        completion_start = len(context)
        index = 0
        while index < len(token_data) - 1 and \
                token_data[index + 1]['offset'] <= completion_start:
            index += 1
        return index

    def _conditional_logprobs(self, context, completion):
        assert not context.endswith(" ")

        if not context and completion.startswith(" "):
            # For the unconditional case, remove leading space which is sometimes auto-added by the tokenizer
            # This is only relevant for doc_prob tasks, which usually don't start without a space
            # TODO: Maybe do this only for J1 and not openai which doesn't auto-add the space
            completion = completion[1:]

        prompt = context + completion
        prompt, token_data = self._logprobs(prompt)
        # Make sure the detokenized text equals the original text, so we can find the starting position
        assert not context or prompt == context + completion, (context + completion, prompt)

        # Find the first token that contains the completion
        completion_start_token_index = 0 if not context else self._find_completion_start_token(context, token_data)

        # Take the logprobs only for the completion tokens
        logprobs = [token['logprob'] for token in token_data[completion_start_token_index:]]

        # Tokens might not be aligned to the requested completion if completion starts in the middle of a token
        # In this case aligned_completion will contain the full completion represented in the returned tokens
        return {
            "logprobs": logprobs,
            "completion": completion,
            "aligned_completion": prompt[token_data[completion_start_token_index]['offset']:]
        }

    def batched_conditional_logprobs(self, reqs, batch_size=1):
        res = []
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            jobs = []
            for req in reqs:
                jobs.append(pool.submit(self._conditional_logprobs, req['context'], req['completion']))
            for job in tqdm(jobs):
                req_res = job.result()
                res.append(req_res)
        return res

    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        for i in range(retries):
            res = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=data)
            if res.status_code == 200:
                return res.json()
            print(f"API call failed with {res}. Waiting {retry_grace_time} seconds")
            time.sleep(retry_grace_time)
        raise TimeoutError(f"API request failed {retries} times, giving up!")


class OpenAIModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("OPENAI_ENDPOINT", "https://api.openai.com/v1/completions")
    _API_KEY = os.environ.get('OPENAI_API_KEY', None)

    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set OPENAI_API_KEY env var for running through OpenAI"

    def _logprobs(self, text, add_start_text=False):
        # add_start_text=True starts with the '<|endoftext|>' token to get logprob for the first actual token as well,
        # but not clear how the paper did the evaluations, some results are better with it, some without
        start_text = "" if not add_start_text else "<|endoftext|>"
        text = start_text + text

        endpoint = self._ENDPOINT # .format(engine=self._model)
        req = {
            "prompt": text,
            "echo": True,
            "max_tokens": 0,
            "logprobs": 0,
            "model": self._model,
        }
        data = self._api_call(endpoint, req, self._API_KEY)
        result = data['choices'][0]
        text = result['text']
        assert text.startswith(start_text)
        text = text[len(start_text):]

        # Make sure the start text, if used, is a single token
        assert not start_text or result['logprobs']['tokens'][0] == start_text

        tokens = [{
            "token": token,
            "offset": offset - len(start_text),
            "logprob": logprob
        } for token, offset, logprob in zip(
            result['logprobs']['tokens'][1:], result['logprobs']['text_offset'][1:],
            result['logprobs']['token_logprobs'][1:])
        ]
        return text, tokens


class AI21ModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("AI21_STUDIO_ENDPOINT", "https://api.ai21.com/studio/v1/{model}/complete")
    _API_KEY = os.environ.get('AI21_STUDIO_API_KEY', None)

    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set AI21_STUDIO_API_KEY env var for running through AI21 Studio"

    def _logprobs(self, text, add_start_text=False):
        endpoint = self._ENDPOINT.format(model=self._model)
        req = {
            "prompt": text,
            "numResults": 1,
            "maxTokens": 0,
            "topKReturn": 0
        }
        data = self._api_call(endpoint, req, self._API_KEY)
        prompt_data = data['prompt']
        text = prompt_data['text']
        tokens = [{
            "token": token_data['generatedToken']['token'],
            "offset": token_data['textRange']['start'],
            "logprob": token_data['generatedToken']['logprob'],
        } for token_data in prompt_data['tokens']]
        return text, tokens

class HuggingFaceModelProvider(APIModelProvider):

    def __init__(self, model_name):
        CACHE_DIR = '/data3/imoshkov/huggingface'

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name[0][0], cache_dir=CACHE_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name[0][0], cache_dir=CACHE_DIR)
        # if not isinstance(model_name, list):
        #     model_name = [model_name]
        # self.model = DecoderEnsemble([name[0] for name in model_name])
        # self.model = GPTNeoXEnsembleForCausalLM()
        # for name in model_name:
        #     self.model.add_model(AutoModelForCausalLM.from_pretrained(name[0], cache_dir=CACHE_DIR, revision=name[1]))
        self.model.cuda()

    def _logprobs(self, text, add_start_text=False):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = self.model(input_ids.cuda())
            log_probs = outputs.logits.log_softmax(dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(), skip_special_tokens=True)

        log_probs_for_tokens = log_probs[0, torch.arange(log_probs.size(1) - 1), input_ids.squeeze()[1:]]

        offsets = [0]
        for item in input_ids:
            for index, token in enumerate(item):
                # offsets.append(len(self.tokenizer._decode(token.item())) + offsets[-1])
                if index + 1 == len(item): continue
                if token in self.tokenizer(*self.tokenizer.all_special_tokens)['input_ids']:
                    offsets.append(offsets[-1])
                else:
                    offsets.append(len(self.tokenizer._decode(token.item())) + offsets[-1])

        result = [{
                "token": token.replace('Ġ', ' '),
                "offset": offset,
                "logprob": log_prob.item()
            } for token, offset, log_prob in 
            zip(tokens[1:], offsets[1:], log_probs_for_tokens)
        ]

        return text, result

_PROVIDER_MAP = {
    "openai": OpenAIModelProvider,
    "ai21": AI21ModelProvider,
    "huggingface": HuggingFaceModelProvider,
}


def make_model(provider, model):
    assert provider in _PROVIDER_MAP, f"No model provider '{provider}' implemented"
    return _PROVIDER_MAP[provider](model)
