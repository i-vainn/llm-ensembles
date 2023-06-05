# LM Ensemble Evaluation

**This is my fork of [lm-evaluation repo](https://gitlab-master.nvidia.com/dpykhtar/lm-evaluation/) modified for LM Ensemble evaluation.**

To create and evaluate ensemble you should add multiple checkpoint paths (list) to `model.gpt_model_file` in `lm_evaluation/config/config.yaml` (or via CLI).

# Megatron GPT Evaluation
This repo contains code for running the evaluations and reproducing the results from the [Jurassic-1 Technical Paper](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf) (see [blog post](https://www.ai21.com/blog/announcing-ai21-studio-and-jurassic-1)), with current support for running the tasks through both the [AI21 Studio API](https://studio.ai21.com/) and [OpenAI's GPT3 API](https://beta.openai.com/).


## Installation
```
git clone https://gitlab-master.nvidia.com/dpykhtar/lm-evaluation.git
cd lm-evaluation
pip install -e .
```

## Usage
The entry point for running the evaluations is lm_evaluation/run_eval.py, which receives a list of tasks and models to run.<br />
List of available tasks: `['arc-challenge', 'arc-easy', 'race-middle', 'race-high', 'winogrande', 'rte', 'hellaswag', 'boolq', 'piqa']`.<br />
To evaluate model on T0 data you have to specify `MC_T0_PATH` at lm_evaluation/tasks_config.py with manually prepared T0 evaluation data.

Examples:
```console
# Evaluate winogrande and rte on NeMo GPT model
python -m lm_evaluation.run_eval \
    tasks="[winogrande, rte]" \
    model.gpt_model_file=/path/to/nemo/gpt_model.nemo \
    model.tensor_model_parallel_size=1 \
    batch_size=16 \
    trainer.devices=1 \
    trainer.precision=16 \
    data_path=/path/to/stored/data \
    results_path=/path/to/save/results

# Evaluate boolq on NeMo GPT checkpoint:
python -m lm_evaluation.run_eval \
    tasks="[boolq]" \
    model.checkpoint_dir=/path/to/ckpt \
    model.checkpoint_name=CKPT_NAME \
    model.tensor_model_parallel_size=2 \
    batch_size=64 \
    trainer.devices=2 \
    trainer.precision=bf16 \
    data_path=/path/to/stored/data \
    results_path=/path/to/save/results
```
To specify more evaluation parameters you can use .yaml config at lm_evaluation/config/config.yaml

## Datasets
The repo currently support the zero-shot multiple-choice and document probability datasets reported in the [Jurassic-1 Technical Paper](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf).

### Multiple Choice
Multiple choice datasets are formatted as described in the [GPT3 paper](https://arxiv.org/abs/2005.14165), and the default reported evaluation metrics are those described there.

All our formatted datasets except for storycloze are publically available and referenced in [lm_evaluation/tasks_config.py](lm_evaluation/tasks_config.py). Storycloze needs to be [manually downloaded](https://cs.rochester.edu/nlp/rocstories/) and formatted, and the location should be configured through the environment variable 'STORYCLOZE_TEST_PATH'.

### Document Probabilities
Document probability tasks include documents from 19 data sources, including [C4](https://www.tensorflow.org/datasets/catalog/c4) and datasets from ['The Pile'](https://arxiv.org/abs/2101.00027).

Each document is pre-split at sentence boundaries to sub-documents of up to 1024 GPT tokens each, to ensure all models see the same inputs/contexts regardless of tokenization, and to support evaluation of models which are limited to sequence lengths of 1024.

Each of the 19 tasks have ~4MB of total text data.

## Additional Configuration

### Results Folder
By default all results will be saved to the folder 'results', and rerunning the same tasks will load the existing results. The results folder can be changed using the environment variable LM_EVALUATION_RESULTS_DIR.
