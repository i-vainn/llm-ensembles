import os
import subprocess
import queue
import time
import shlex

from itertools import combinations


def run_experiment_on_device(device, experiment):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    if experiment.startswith('USE_SOFTMAX='):
        softmax = experiment.split()[0][-1]
        env["USE_SOFTMAX"] = softmax
        experiment = experiment[len('USE_SOFTMAX=x '):]
    
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
    base_model_path = '/data/imoshkov/converted_checkpoints/'
    base_results_path = '/data/imoshkov/results/'
    group1 = [
        'gpt1.3b_default.nemo',
        'gpt1.3b_alibi.nemo',
        'gpt1.3b_rope.nemo',
        'gpt1.3b_sandwich.nemo',
    ]
    group2 = [
        'gpt3-843m-multi-1.1t-gtc-ibs.nemo',
        'gpt3-843m-multi-1.1t-gtc-ibs-wavg.nemo',
        'gpt3-843m-multi-1.1t-gtc-llr.nemo',
    ]
    model_groups = [
        group1, group2
    ]

    expert_sets = []
    for group in model_groups:
        for n_experts in range(1, len(group) + 1):
            for experts in combinations(group, n_experts):
                expert_sets.append(experts)
    
    experiments = []
    for expert_set in expert_sets:
        for softmax in [0, 1]:
            if len(expert_set) == 1 and softmax == 1: continue
            results_file_name = '__'.join([model[:-5] for model in expert_set] + ([f'softmax'] if softmax else []))
            cmd = 'USE_SOFTMAX={} python run_eval.py model.gpt_model_file="[{}]" results_path={}'.format(
                softmax,
                ', '.join([os.path.join(base_model_path, model_name) for model_name in expert_set]),
                os.path.join(base_results_path, results_file_name)
            )
            experiments.append(cmd)
    
    return experiments


if __name__ == "__main__":
    devices = [0, 1, 2]
    experiments = get_experiments()

    run_experiments_on_devices(experiments, devices)
