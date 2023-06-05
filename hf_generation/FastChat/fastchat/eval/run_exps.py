import os
import subprocess
import queue
import time
import shlex

from itertools import combinations


def run_experiment_on_device(device, experiment):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    
    with open('/dev/null', 'w') as devnull:
        process = subprocess.Popen(shlex.split(experiment), stdout=devnull, stderr=devnull, env=env)
    
    return process


def run_experiments_on_devices(experiments, devices, interval=60):
    experiments_queue = queue.Queue()
    for exp in experiments:
        experiments_queue.put(exp)

    running_experiments = {}

    while not experiments_queue.empty() or running_experiments:
        for device in devices:
            if device not in running_experiments and not experiments_queue.empty():
                next_experiment = experiments_queue.get()
                print(f"Running experiment '{next_experiment}' on device {device}")
                running_experiments[device] = (next_experiment, run_experiment_on_device(device, next_experiment))

        time.sleep(interval)

        for device in list(running_experiments.keys()):
            experiment, process = running_experiments[device]
            if process.poll() is not None:
                print(f"Experiment '{experiment}' on device {device} completed with {process.returncode}.")
                del running_experiments[device]

def get_experiments():
    group1 = [
        'togethercomputer/RedPajama-INCITE-Instruct-3B-v1',
        'stabilityai/stablelm-tuned-alpha-3b',
        'EleutherAI/pythia-2.8b',
        'databricks/dolly-v2-3b',
    ]
    group2 = [
        # 'gpt2-xl',
        # 'cerebras/Cerebras-GPT-1.3B',
        # 'facebook/opt-1.3b',
        # 'EleutherAI/gpt-neo-1.3B'
        'databricks/dolly-v2-7b',
        'stabilityai/stablelm-tuned-alpha-7b',
        'mosaicml/mpt-7b-instruct',
    ]

    model_groups = [
        group1, group2
    ]

    expert_sets = []
    for group in model_groups:
        for n_experts in range(1, len(group)):
            for experts in combinations(group, n_experts):
                expert_sets.append(experts)
    
    experiments = []
    for expert_set in expert_sets:
        cmd = 'python get_model_answer.py --model-paths {}'.format(
            ' '.join(expert_set),
        )
        experiments.append(cmd)
    
    return experiments


if __name__ == "__main__":
    devices = [0, 1, 2]
    experiments = get_experiments()

    run_experiments_on_devices(experiments, devices)
