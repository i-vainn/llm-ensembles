import threading
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPTNeoXForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from itertools import cycle
from utils import permute_vocab

CACHE_DIR = '/data/imoshkov/huggingface'

class DecoderEnsemble(GPTNeoXForCausalLM):
    def __init__(self, model_names):
        base_model = AutoModelForCausalLM.from_pretrained(model_names[0], cache_dir=CACHE_DIR)
        base_tokenizer = AutoTokenizer.from_pretrained(model_names[0], cache_dir=CACHE_DIR)
        super(DecoderEnsemble, self).__init__(base_model.config)
        self.models = nn.ModuleList([
            AutoModelForCausalLM.from_pretrained(name, cache_dir=CACHE_DIR)
            for name in model_names
        ])
        self.caches = [None for _ in model_names]
        self.logit_permutations = []
        self.available_models = {name: 1 for name in model_names}
        for tokenizer in model_names:
            target_tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_dir=CACHE_DIR)
            perm = permute_vocab(base_tokenizer.vocab, target_tokenizer.vocab)
            self.logit_permutations.append(perm)

    def set_models(self, models):
        for model in self.available_models:
            self.available_models[model] = 1 if model in models else 0

    # def model_forward(self, model_name, input_ids, attention_mask, logit_permutation, output_list):
    #     model = self.models[model_name]
    #     cur_device = next(model.parameters()).device
    #     with torch.no_grad():
    #         outputs = model(input_ids=input_ids.to(cur_device), attention_mask=attention_mask.to(cur_device))
    #         output_list.append(outputs.logits.cpu()[:,:,logit_permutation])

    # def forward(self, input_ids, attention_mask=None, **kwargs):
    #     threads = []
    #     outputs = []
    #     for idx, model_name in enumerate(self.models):
    #         if not self.available_models[model_name]:
    #             continue
    #         thread = threading.Thread(target=self.model_forward, args=(model_name, input_ids, attention_mask, self.logit_permutations[idx], outputs))
    #         thread.start()
    #         threads.append(thread)

    #     for thread in threads:
    #         thread.join()

    #     ensemble_logits = torch.stack(outputs).mean(dim=0)
    #     return CausalLMOutputWithPast(logits=ensemble_logits)
    def forward(self, input_ids, attention_mask=None, **kwargs):
        logits = []
        for idx, model_name in enumerate(self.available_models):
            if not self.available_models[model_name]:
                continue

            model = self.models[idx]
            kwargs['past_key_values'] = self.caches[idx]
            outputs = model(input_ids=input_ids, **kwargs)
            self.caches[idx] = outputs.past_key_values
            logits.append(outputs.logits[:,:,self.logit_permutations[idx]])
        
        # cut = min(map(lambda x: len(x[:,-1]), logits))
        # print(cut)
        outputs.logits = torch.stack(logits).mean(dim=0)
        return outputs

    def _reset_cache(self):
        self.caches = [None for _ in self.models]

class FakeEnsemble(GPTNeoXForCausalLM):
    def __init__(self, model_names):
        base_model = AutoModelForCausalLM.from_pretrained(model_names[0], cache_dir=CACHE_DIR)
        super(FakeEnsemble, self).__init__(base_model.config)
        self.inner_model = AutoModelForCausalLM.from_pretrained(model_names[0], cache_dir=CACHE_DIR)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.inner_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return outputs #CausalLMOutputWithPast(logits=outputs.logits)