import os

# Base path for our datasets. Note that storycloze path needs to be manually supplied
MC_BASE_PATH = "https://storage.googleapis.com/ai21-public-data/lm_evaluation/datasets/multiple_choice"

# Base path for our doc_prob datasets
DOC_PROBS_BASE_PATH = "https://storage.googleapis.com/ai21-public-data/lm_evaluation/datasets/doc_probs/max_seq_len_1024-4096KB"

# By default, this metric will be used in multiple-choice tasks. For ARC and RACE the answer-context
# normalized logprobs metric will be used as per the GPT3 paper
MC_DEFAULT_METRIC = "acc_norm_tokens"

# Path to T0 prompted data
MC_T0_PATH = "set your path to T0 data"

_TASKS_CONFIG = {
    "arc-challenge": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/arc-challenge/test.jsonl",
        "relative_test_dataset": f"arc-challenge.jsonl",
        "main_metric": "acc_norm_ans_ctx"
    },
    "arc-easy": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/arc-easy/test.jsonl",
        "relative_test_dataset": f"arc-easy.jsonl",
        "main_metric": "acc_norm_ans_ctx"
    },
    "hellaswag": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/hellaswag/validation.jsonl",
        "relative_test_dataset": f"hellaswag.jsonl",
        "main_metric": MC_DEFAULT_METRIC
    },
    "piqa": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/piqa/validation.jsonl",
        "relative_test_dataset": f"piqa.jsonl",
        "main_metric": MC_DEFAULT_METRIC
    },
    "race-high": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/race-high/test.jsonl",
        "relative_test_dataset": f"race-high.jsonl",
        "main_metric": "acc_norm_ans_ctx"
    },
    "race-middle": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/race-middle/test.jsonl",
        "relative_test_dataset": f"race-middle.jsonl",
        "main_metric": "acc_norm_ans_ctx"
    },
    "storycloze": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/storycloze/storycloze.jsonl",
        "relative_test_dataset": "storycloze.jsonl",
        "main_metric": MC_DEFAULT_METRIC
    },
    "winogrande": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/winogrande/validation.jsonl",
        "relative_test_dataset": f"winogrande.jsonl",
        "main_metric": MC_DEFAULT_METRIC
    },
    "rte": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/rte/validation.jsonl",
        "relative_test_dataset": f"rte.jsonl",
        "main_metric": MC_DEFAULT_METRIC
    },
    "boolq": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/boolq/validation.jsonl",
        "relative_test_dataset": f"boolq.jsonl",
        "main_metric": MC_DEFAULT_METRIC
    },
    "copa": {
        "type": "multiple_choice",
        "test_dataset": f"{MC_BASE_PATH}/copa/copa.jsonl",
        "relative_test_dataset": f"copa.jsonl",
        "main_metric": MC_DEFAULT_METRIC
    },
    "rte-p": {
        "type": "multiple_choice",
        "test_dataset": {"path": f"{MC_T0_PATH}/rte", "amount": 10},
        "main_metric": MC_DEFAULT_METRIC
    },
    "winogrande-p": {
        "type": "multiple_choice",
        "test_dataset": {"path": f"{MC_T0_PATH}/winogrande", "amount": 6},
        "main_metric": MC_DEFAULT_METRIC
    },
    "cb": {
        "type": "multiple_choice",
        "test_dataset": {"path": f"{MC_T0_PATH}/cb", "amount": 15},
        "main_metric": MC_DEFAULT_METRIC
    },
    "copa-p": {
        "type": "multiple_choice",
        "test_dataset": {"path": f"{MC_T0_PATH}/copa", "amount": 8},
        "main_metric": MC_DEFAULT_METRIC
    },
    "wic": {
        "type": "multiple_choice",
        "test_dataset": {"path": f"{MC_T0_PATH}/wic", "amount": 10},
        "main_metric": MC_DEFAULT_METRIC
    },
    "wsc": {
        "type": "multiple_choice",
        "test_dataset": {"path": f"{MC_T0_PATH}/wsc", "amount": 10},
        "main_metric": MC_DEFAULT_METRIC
    },
    "anli-r1": {
        "type": "multiple_choice",
        "test_dataset": {"path": f"{MC_T0_PATH}/anli-r1", "amount": 15},
        "main_metric": MC_DEFAULT_METRIC
    },
    "anli-r2": {
        "type": "multiple_choice",
        "test_dataset": {"path": f"{MC_T0_PATH}/anli-r2", "amount": 15},
        "main_metric": MC_DEFAULT_METRIC
    },
    "anli-r3": {
        "type": "multiple_choice",
        "test_dataset": {"path": f"{MC_T0_PATH}/anli-r3", "amount": 15},
        "main_metric": MC_DEFAULT_METRIC
    },
    "hellaswag-p": {
        "type": "multiple_choice",
        "test_dataset": {"path": f"{MC_T0_PATH}/hellaswag", "amount": 10},
        "main_metric": MC_DEFAULT_METRIC
    },
    "storycloze-p": {
        "type": "multiple_choice",
        "test_dataset": {"path": f"{MC_T0_PATH}/storycloze", "amount": 6},
        "main_metric": MC_DEFAULT_METRIC
    }
}

# Add doc-prob tasks
_TASKS_CONFIG.update({
    name: {
        "type": "doc_probs",
        "test_dataset": f"{DOC_PROBS_BASE_PATH}/{name}.jsonl",
        "main_metric": "doc_logprob_per_byte"

    } for name in [
        "arxiv",
        "books3",
        "c4",
        "dm_math",
        "enron_emails",
        "freelaw",
        "github",
        "gutenberg",
        "hackernews",
        "nih_exporter",
        "open_subtitles",
        "phil_papers",
        "pile_cc",
        "pubmed_abstracts",
        "pubmed_central",
        "stackexchange",
        "ubuntu_irc",
        "uspto",
        "youtube_subtitles"
    ]
})


def get_task_config(task_name):
    assert task_name in _TASKS_CONFIG, f"No task '{task_name}'"
    return _TASKS_CONFIG[task_name]


def get_all_tasks_of_type(task_type):
    return [task_name for task_name, task_config in _TASKS_CONFIG.items() if task_config['type'] == task_type]
