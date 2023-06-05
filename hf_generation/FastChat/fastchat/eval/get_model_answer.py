import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, StoppingCriteriaList
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import ray

from fastchat.model import get_conversation_template
from ensemble_model import DecoderEnsemble, FakeEnsemble
from utils import StoppingCriteriaSub

CACHE_DIR = '/data/imoshkov/huggingface'

def run_eval(model_paths, model_id, question_file, answer_file, num_gpus):
    # split question file into num_gpus files
    answer_file = 'table/answer/{}.jsonl'.format('__'.join(x.split('/')[-1] for x in model_paths))
    ques_jsons = []
    with open(os.path.expanduser(question_file), "r") as ques_file:
        for line in ques_file:
            ques_jsons.append(line)

    chunk_size = len(ques_jsons) // num_gpus
    ans_handles = []
    for i in range(0, len(ques_jsons), chunk_size):
        ans_handles.append(
            # get_model_answers.remote(
            #     model_path, model_id, ques_jsons[i : i + chunk_size]
            # )
            get_model_answers(
                model_paths, model_id, ques_jsons[i : i + chunk_size]
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        # ans_jsons.extend(ray.get(ans_handle))
        ans_jsons.extend(ans_handle)

    with open(os.path.expanduser(answer_file), "w") as ans_file:
        for line in ans_jsons:
            ans_file.write(json.dumps(line) + "\n")


# @ray.remote(num_gpus=1)
@torch.inference_mode()
def get_model_answers(model_paths, model_id, question_jsons):
    # model_path = os.path.expanduser(model_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=CACHE_DIR,)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, cache_dir=CACHE_DIR,
    # ).cuda()
    model_names = [
        'togethercomputer/RedPajama-INCITE-Base-3B-v1',
        'databricks/dolly-v2-3b',
        'stabilityai/stablelm-base-alpha-3b',
    ]
    # device = 'cuda:1'
    model = DecoderEnsemble(model_paths).cuda()#to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
    # tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=CACHE_DIR,)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path, cache_dir=CACHE_DIR,
    # ).to(device)

    ans_jsons = []
    for i, line in enumerate(tqdm(question_jsons)):
        ques_json = json.loads(line)
        idx = ques_json["question_id"]
        qs = ques_json["text"]
        conv = get_conversation_template(model_id)
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt]).input_ids
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer('###')['input_ids'])])
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),#.to(device),
            stopping_criteria=stopping_criteria,
            do_sample=True,
            # use_cache=False,
            temperature=0.7,
            max_new_tokens=1024,
        )
        model._reset_cache()
        output_ids = output_ids[0][len(input_ids[0]) :]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        outputs = outputs.split('###')[0].strip()
        ans_id = shortuuid.uuid()
        ans_jsons.append(
            {
                "question_id": idx,
                "text": outputs,
                "answer_id": ans_id,
                "model_id": model_id,
                "metadata": {},
            }
        )
    return ans_jsons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-paths", nargs='+', type=str, required=True)
    parser.add_argument("--model-id", type=str, default=str(torch.randint(0, 9999, (1,)).item()))
    parser.add_argument("--question-file", type=str, default="table/question.jsonl")
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    # ray.init()
    run_eval(
        args.model_paths,
        args.model_id,
        args.question_file,
        args.answer_file,
        args.num_gpus,
    )
