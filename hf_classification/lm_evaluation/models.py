from typing import Optional, Tuple, Union

import torch
from torch import nn
from transformers.models.gpt_neox import GPTNeoXForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel
from .utils import permute_vocab


CACHE_DIR = '/data3/imoshkov/huggingface'


class GPTNeoXEnsembleForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.models = nn.ModuleList([])

    def add_model(self, model):
        self.models.append(model)

    def forward(
        self,
        input_ids = None,
        **kwargs
    ):
        outputs = None
        for model in self.models:
            output = model(input_ids, **kwargs)
            if outputs is None:
                outputs = output
            else:
                outputs.logits += output.logits
        outputs.logits /= len(self.models)

        return outputs
    
    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.FloatTensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     head_mask: Optional[torch.FloatTensor] = None,
    #     past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, CausalLMOutputWithPast]:
    #     return_dict = return_dict if return_dict is not None else self.models[0].config.use_return_dict

    #     lm_logits = None
    #     for model in self.models:

    #         hiddens = model.gpt_neox(
    #             input_ids,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             head_mask=head_mask,
    #             inputs_embeds=inputs_embeds,
    #             past_key_values=past_key_values,
    #             use_cache=use_cache,
    #             output_attentions=output_attentions,
    #             output_hidden_states=output_hidden_states,
    #             return_dict=return_dict,
    #         )
    #         outputs = model.embed_out(hiddens[0])

    #         if lm_logits is None:
    #             lm_logits = outputs
    #         else:
    #             lm_logits += outputs
        
    #     lm_logits /= len(self.models)

    #     lm_loss = None
    #     if labels is not None:
    #         # move labels to correct device to enable model parallelism
    #         labels = labels.to(lm_logits.device)
    #         # we are doing next-token prediction; shift prediction scores and input ids by one
    #         shift_logits = lm_logits[:, :-1, :].contiguous()
    #         labels = labels[:, 1:].contiguous()
    #         loss_fct = nn.CrossEntropyLoss()
    #         lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

    #     if not return_dict:
    #         output = (lm_logits,) + hiddens[1:]
    #         return ((lm_loss,) + output) if lm_loss is not None else output

    #     return CausalLMOutputWithPast(
    #         loss=lm_loss,
    #         logits=lm_logits,
    #         past_key_values=hiddens.past_key_values,
    #         hidden_states=hiddens.hidden_states,
    #         attentions=hiddens.attentions,
    #     )

class DecoderEnsemble(GPT2LMHeadModel):
    def __init__(self, model_names):
        base_model = AutoModelForCausalLM.from_pretrained(model_names[0], cache_dir=CACHE_DIR)
        base_tokenizer = AutoTokenizer.from_pretrained(model_names[0], cache_dir=CACHE_DIR)
        super(DecoderEnsemble, self).__init__(base_model.config)
        self.models = nn.ModuleList([
            AutoModelForCausalLM.from_pretrained(name, cache_dir=CACHE_DIR)
            for name in model_names
        ])

        self.logit_permutations = []
        for tokenizer in model_names:
            target_tokenizer = AutoTokenizer.from_pretrained(tokenizer, cache_dir=CACHE_DIR)
            perm = permute_vocab(base_tokenizer.vocab, target_tokenizer.vocab)
            self.logit_permutations.append(perm)

    def forward(self, input_ids, **kwargs):
        logits = []
        for idx, model in enumerate(self.models):
            outputs = model(input_ids=input_ids, **kwargs)
            logits.append(outputs.logits[:,:,self.logit_permutations[idx]])
        
        outputs.logits = torch.stack(logits).mean(dim=0)
        return outputs
