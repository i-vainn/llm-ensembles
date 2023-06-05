from concurrent.futures import ThreadPoolExecutor
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm
import requests
import os
import time

# from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel as MegatronGPTEnsembleModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_ensemble_model import MegatronGPTEnsembleModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.utils.app_state import AppState
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.modules.common.text_generation_utils import generate, get_computeprob_response
from nemo.utils.model_utils import inject_model_parallel_rank
from nemo.core.config import hydra_runner

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from pytorch_lightning.trainer.trainer import Trainer

from apex.transformer import tensor_parallel, parallel_state

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
        print(context, token_data)
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
            "aligned_completion": prompt[token_data[completion_start_token_index]['offset']:],
            "text" : prompt
        }

    def batched_conditional_logprobs(self, reqs, batch_size):
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

class RequestDataSet(Dataset):
    def __init__(self, sentences, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.untokenized_sentences = sentences
        self.sentences = []
        for sentence in self.untokenized_sentences:
            self.sentences.append(torch.tensor(self.tokenizer.text_to_ids(sentence)))

    def __len__(self,):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

def hacky_DDP_initialize(model):
    if parallel_state.is_unitialized():

        class RequestDataSet(Dataset):
            def __init__(self, sentences):
                super().__init__()
                self.sentences = sentences

            def __len__(self):
                return len(self.sentences)

            def __getitem__(self, idx):
                return self.sentences[idx]

        # run empty predict to initialize the DDP
        ds = RequestDataSet([""])
        request_dl = DataLoader(dataset=ds, batch_size=1)
        model.trainer.predict(model, request_dl)

class NemoModelProvider(APIModelProvider):

    def __init__(self, model, cfg_trainer, cfg_evaluation, cfg_model):

        # initializing trainer and model
        self._save_restore_connector = NLPSaveRestoreConnector()
        self._cfg_model = cfg_model
        self._cfg_evaluation = cfg_evaluation
        self._trainer = Trainer(
            strategy=NLPDDPStrategy(),    
            **cfg_trainer,
        )        
        assert (
        cfg_trainer.devices * cfg_trainer.num_nodes
        == cfg_model.tensor_model_parallel_size * cfg_model.pipeline_model_parallel_size
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"
        
        if self._cfg_model.gpt_model_file:
            pretrained_cfg = MegatronGPTEnsembleModel.restore_from(
                restore_path=model,
                trainer=self._trainer,
                return_config=True,
                save_restore_connector=self._save_restore_connector,
            )
            pretrained_cfg['target'] = 'nemo.collections.nlp.models.language_modeling.megatron_gpt_ensemble_model.MegatronGPTEnsembleModel'
            OmegaConf.set_struct(pretrained_cfg, True)
            with open_dict(pretrained_cfg):
                pretrained_cfg.sequence_parallel = False
                pretrained_cfg.activations_checkpoint_granularity = None
                pretrained_cfg.activations_checkpoint_method = None
            self._model = MegatronGPTEnsembleModel.restore_from(restore_path=model, trainer=self._trainer, override_config_path=pretrained_cfg, save_restore_connector=self._save_restore_connector)
        else:
            self._app_state = AppState()
            if self._cfg_model.tensor_model_parallel_size > 1 or self._cfg_model.pipeline_model_parallel_size > 1:
                self._app_state.model_parallel_size = self._cfg_model.tensor_model_parallel_size * self._cfg_model.pipeline_model_parallel_size
                self._app_state.tensor_model_parallel_size = self._cfg_model.tensor_model_parallel_size
                self._app_state.pipeline_model_parallel_size = self._cfg_model.pipeline_model_parallel_size
                (
                    self._app_state.tensor_model_parallel_rank,
                    self._app_state.pipeline_model_parallel_rank,
                    self._app_state.model_parallel_size,
                    self._app_state.data_parallel_size,
                    self._app_state.pipeline_model_parallel_split_rank,
                    self._app_state.virtual_pipeline_model_parallel_rank,
                ) = fake_initialize_model_parallel(
                    world_size=self._app_state.model_parallel_size,
                    rank=self._trainer.global_rank,
                    tensor_model_parallel_size_=self._cfg_model.tensor_model_parallel_size,
                    pipeline_model_parallel_size_=self._cfg_model.pipeline_model_parallel_size,
                    pipeline_model_parallel_split_rank_=self._cfg_model.pipeline_model_parallel_split_rank,
                )

            checkpoint_path = inject_model_parallel_rank(os.path.join(self._cfg_model.checkpoint_dir, self._cfg_model.checkpoint_name))

            self._model = MegatronGPTEnsembleModel.load_from_checkpoint(checkpoint_path, hparams_file=self._cfg_model.hparams_file, trainer=self._trainer)
        hacky_DDP_initialize(self._model)
        # self._model.model.add_model(self._model.model.language_model)
        self._model = self._model.cuda()
        self._model.freeze()

    def batched_conditional_logprobs(self, reqs, batch_size):

        def pad_collate(batch, eos_id=50256):

            tokens_pad = pad_sequence(batch, batch_first=True, padding_value=eos_id)
            lens = torch.tensor([tokens_pad.size()[1] - 1 for i in range(tokens_pad.size()[0])])

            return (tokens_pad.cuda(), lens.cuda())

        # add all prompts from request (AI21 dataset) to single list
        data = []
        for req in reqs:
            context, completion = req['context'], req['completion']
            data.append(context + completion)

        # Initializing dataloader
        ds = RequestDataSet(data, self._model.tokenizer)
        request_dl = DataLoader(dataset=ds, collate_fn=pad_collate, batch_size=batch_size)

        # Generate method with batch_size
        response = []
        cut = 0
        for batch in tqdm(request_dl, mininterval=60):

            res = generate(
                self._model,
                inputs=batch,
                tokens_to_generate=1,
                min_tokens_to_generate=0,
                all_probs=True,
                **self._cfg_evaluation,
            )
            
            if res:
                res = get_computeprob_response(self._model.tokenizer, res, batch)            
                try:
                    result = self._conditional_logprobs(res, reqs[cut:(cut + batch_size)])
                except IndexError:
                    result = self._conditional_logprobs(res, reqs[cut:len(reqs)])

                for item in result:
                    response.append(item)
            
                cut += batch_size

        return response

    def _conditional_logprobs(self, response, requests):
        
        # Reinterpretation of results as list of dicts ([{},...,{}])
        results = []
        batch_size = len(response['sentences'])
        for index in range(batch_size):
            results.append({'token': response['tokens'][index],
                                'offset': response['offsets'][index],
                                'logprob': response['full_logprob'][index],
                                'text': response['sentences'][index],
            })
        
        ret = []

        # Extracting needed logprobs from results
        for result, request in zip(results, requests):
            context, completion = request['context'], request['completion']
            text = result['text']
            input_len = len(self._model.tokenizer.text_to_ids(text))
            logprobs = torch.tensor(result['logprob']).cpu()

            result = self._logprobs(result)
            completion_start_token_index = self._find_completion_start_token(context, result)
            completion_tokens = self._model.tokenizer.text_to_ids(text[result[completion_start_token_index]['offset']:])
            completion_tokens = torch.tensor(completion_tokens, dtype=torch.long).unsqueeze(0)
            completion_len = len(completion_tokens[0])
            try:
                logprobs = logprobs[(input_len-1) - completion_len:input_len].unsqueeze(0)
                logprobs = torch.gather(logprobs, 2, completion_tokens.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
                completion_tokens = completion_tokens[1:]
                completion_tokens = torch.tensor(completion_tokens, dtype=torch.long).unsqueeze(0)
                completion_len = len(completion_tokens[0])
                logprobs = logprobs[(input_len-1) - completion_len:input_len].unsqueeze(0)
                logprobs = torch.gather(logprobs, 2, completion_tokens.unsqueeze(-1)).squeeze(-1)
            logprobs = logprobs.tolist()

            ret.append({
                "logprobs": logprobs,
                "completion": completion,
                "aligned_completion": text[result[completion_start_token_index]['offset']:],
            })

        return ret

    def _logprobs(self, result):

        tokens = []
        for token, offset, logprob in zip(result['token'],result['offset'],result['logprob']):
            tokens.append({'token' : token,
                           'offset' : offset,
                           'logprob' : logprob})

        return tokens

    def _find_completion_start_token(self, context, token_data):
        # Finds the offset of the first token that contains the completion text
        completion_start = len(context)
        index = 0
        while index < len(token_data) - 1 and \
                token_data[index + 1]['offset'] <= completion_start:
            index += 1

        return index

class OpenAIModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("OPENAI_ENDPOINT", "https://api.openai.com/v1/engines/{engine}/completions")
    _API_KEY = os.environ.get('OPENAI_API_KEY', "8cE9rHDzzriPyvxwsFECqdYRJvqufUNL")

    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set OPENAI_API_KEY env var for running through OpenAI"

    def _logprobs(self, text, add_start_text=False):
        # add_start_text=True starts with the '<|endoftext|>' token to get logprob for the first actual token as well,
        # but not clear how the paper did the evaluations, some results are better with it, some without
        start_text = "" if not add_start_text else "<|endoftext|>"
        text = start_text + text

        endpoint = self._ENDPOINT.format(engine=self._model)
        req = {
            "prompt": text,
            "echo": True,
            "max_tokens": 0,
            "logprobs": 0,
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
    _API_KEY = os.environ.get('AI21_STUDIO_API_KEY', "8cE9rHDzzriPyvxwsFECqdYRJvqufUNL")

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


_PROVIDER_MAP = {
    "openai": OpenAIModelProvider,
    "ai21": AI21ModelProvider,
    "nemo": NemoModelProvider
}


def make_model(provider, model, cfg_trainer, cfg_evaluation, cfg_model):
    assert provider in _PROVIDER_MAP, f"No model provider '{provider}' implemented"
    return _PROVIDER_MAP[provider](model, cfg_trainer, cfg_evaluation, cfg_model)
