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
        'gpt2-xl',
        'cerebras/Cerebras-GPT-1.3B',
        'facebook/opt-1.3b',
        'EleutherAI/gpt-neo-1.3B'
    ]
    group2 = [
        'cerebras/Cerebras-GPT-2.7B',
    ]
    group3 = [
        'facebook/opt-2.7b',
    ]
    group4 = [
        'EleutherAI/gpt-neo-2.7B'
    ]

    model_groups = [
        group1, group2, group3, group4
    ]

    expert_sets = []
    for group in model_groups:
        for n_experts in range(1, len(group)+1):
            for experts in combinations(group, n_experts):
                expert_sets.append(experts)
    
    experiments = []
    for expert_set in expert_sets:
        # results_file_name = '__'.join([model[:-5] for model in expert_set] + ([f'softmax'] if softmax else []))

        cmd = 'python run_eval.py --tasks {} --models {}'.format(
            ' '.join(['arc-challenge', 'arc-easy', 'race-middle', 'race-high', 'winogrande', 'rte', 'hellaswag', 'boolq', 'piqa']),
            ' '.join([os.path.join('huggingface', name) for name in expert_set]),
        )
        experiments.append(cmd)
    
    return experiments


if __name__ == "__main__":
    devices = [0, 2, 3]
    experiments = get_experiments()
    # print(experiments)
    run_experiments_on_devices(experiments, devices)
