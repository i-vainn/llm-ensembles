import torch
import time
import sys
import os
import pandas as pd
from collections import defaultdict

from tasks_config import get_all_tasks_of_type, get_task_config
from utils import load_json_lines, ensure_path_exists, dump_json, dump_dataframe, load_json, add_model
from tasks import multiple_choice_evaluate, doc_probs_evaluate
from model_providers import make_model
import json
from nemo.core.config import hydra_runner


def run_evaluation(task, model, batch_size, local_data_path, t0_data):
    conf = get_task_config(task.split('_')[0])
    
    if task.split('_')[0] in t0_data:
        dataset = load_json_lines(os.path.join(conf['test_dataset']['path'], task))
    elif local_data_path:
        dataset = load_json_lines(os.path.join(local_data_path, conf['relative_test_dataset']))
    else:
        dataset = load_json_lines(conf['test_dataset'])
    task_type = conf['type']
    if task_type == "multiple_choice":
        return multiple_choice_evaluate(dataset, model, batch_size)
    elif task_type == "doc_probs":
        return doc_probs_evaluate(dataset, model)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

@hydra_runner(config_path="config", config_name="config")
def main(cfg):
    results_dir = cfg.get("results_path", "results")
    local_data_path = cfg.get("data_path", None)
    ensure_path_exists(results_dir)

    tasks = cfg.tasks
    batch_size = cfg.batch_size
    supported_tasks = ['arc-challenge', 'arc-easy', 'race-middle', 'race-high', 'winogrande', 'rte', 'hellaswag', 'boolq', 'piqa', 'storycloze', 'copa']

    t0_data = ['rte-p', 'winogrande-p', 'cb', 'copa-p', 'wic', 'wsc', 'anli-r1', 'anli-r2', 'anli-r3', 'hellaswag-p', 'storycloze-p']
    if tasks == ["all_mc"]:
        tasks = get_all_tasks_of_type("multiple_choice")
    elif tasks == ["all_docprobs"]:
        tasks = get_all_tasks_of_type("doc_probs")
    elif tasks[0] in t0_data:
        task = tasks[0]
        conf = get_task_config(task)
        tasks = [task + '_' + str(num) + '.jsonl' for num in range(conf['test_dataset']['amount'])]

    results = defaultdict(dict)
    if cfg.model.gpt_model_file:
        models = cfg.model.gpt_model_file
    else:
        models = [cfg.model.checkpoint_name]
    model = None
    model_name = None
    for mname in models:
        if '.nemo' in mname or '.ckpt' in mname:
            provider = "nemo"
        else:
            provider, mname = mname.split("/")
        new_model = make_model(provider, mname, cfg.trainer, cfg.evaluation, cfg.model)
        if model is not None:
            add_model(model, new_model)
            model_name += '__' + mname.split('/')[-1]
        else:
            model = new_model
            model_name = mname.split('/')[-1]
    for task in tasks:
        assert task in supported_tasks or task.split('_')[0] in t0_data, f"{task} is not in the list of supported tasks: {supported_tasks}"
        results_file = os.path.join(results_dir, f"{provider}.{model_name}.{task}.json")
        result = None
        try:
            # Load existing result if found
            result = load_json(results_file)
        except:
            result = None
        if result is None:
            print(f"Running {task} on {model_name}")
            result = run_evaluation(task, model, batch_size, local_data_path, t0_data)
            dump_json(result, results_file)
        print(f"{provider}/{model_name}/{task}/{result}")
        try:
            results[task][model_name] = result[get_task_config(task.split('_')[0])["main_metric"]]
        except KeyError:
            pass

    # Summarize
    try:
        df = pd.DataFrame([{"task": k, "metric": get_task_config(k.split('_')[0])["main_metric"], **v} for k, v in results.items()])
        df.loc['-'] = ["", "average"] + [df[model].mean() for model in df.columns[2:]]
        print(df)

        dump_dataframe(df, os.path.join(results_dir, "summary.csv"))
    except:
        pass

def extract_results_path(command):
    for arg in command:
        if arg.startswith("results_path="):
            return arg.split("=")[1]
    return None

if __name__ == "__main__":
    results_path = extract_results_path(sys.argv)
    start_time = time.time()
    torch.cuda.empty_cache()

    main()

    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated()

    if results_path:
        stats_file = os.path.join(results_path, "stats.json")
        with open(stats_file, "w") as f:
            stats = dict(peak_memory=peak_memory / (1024 ** 2), elapsed_time=elapsed_time)
            json.dump(stats, f)

