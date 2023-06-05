import os
import smart_open
import json
import re
import urllib.request
import torch


def load_http_text(url):
    with urllib.request.urlopen(url) as f:
        return f.read().decode("utf-8")


def load_text(path):
    if path.startswith("http://") or path.startswith("https://"):
        return load_http_text(path)
    else:
        with smart_open.open(path) as f:
            return f.read()


def load_json(path):
    return json.loads(load_text(path))


def load_json_lines(path):
    return [json.loads(line) for line in load_text(path).split("\n") if line]


def dump_json(data, path):
    with smart_open.open(path, "w") as f:
        json.dump(data, f)


def dump_dataframe(df, path):
    with smart_open.open(path, "w") as f:
        df.to_csv(f, index=False)


def ensure_path_exists(path):
    if "://" in path:
        # Buckets like GS/S3 don't need to pre-create the prefix/folder
        return

    if not os.path.exists(path):
        os.makedirs(path)


def word_count(text):
    # Count words in text, this isn't well-defined but we count regex full words and
    # single-char non-words (e.g. punctuation), similar to word tokenizers
    return len(re.findall(r"\w+|[^\w\s]", text))

class TextDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_computeprob_response(tokenizer, response, inputs):

    all_offsets = []
    for item in inputs[0]['input_ids']:
        offsets = [0]
        for index, token in enumerate(item, 1):
            if index == len(item):
                continue
            if token in tokenizer(*tokenizer.all_special_tokens)['input_ids']:
                offsets.append(offsets[-1])
            else:
                offsets.append(len(tokenizer._decode(token.item())) + offsets[-1])
        all_offsets.append(offsets[:-1])
    all_logprobs = response.logits.log_softmax(dim=-1)
    
    compute_prob_response = {}
    new_token_ids = []
    new_tokens = []
    new_texts = []
    log_probs = []
    offsets = []
    for batch_id in range(len(response.logits)):
        token_len = int(inputs[1][batch_id])
        new_token_id = inputs[0]['input_ids'][batch_id][:token_len].tolist()
        new_text = tokenizer.decode(new_token_id)

        new_token_ids.append(new_token_id)
        new_tokens.append(response.logits[batch_id].argmax(-1)[:token_len])
        new_texts.append(new_text)
        log_probs.append(all_logprobs[batch_id][:token_len])
        offsets.append(all_offsets[batch_id][:-1])
    compute_prob_response['sentences'] = new_texts
    compute_prob_response['tokens'] = new_tokens
    compute_prob_response['token_ids'] = new_token_ids
    compute_prob_response['logprob'] = log_probs
    compute_prob_response['offsets'] = offsets
    
    return compute_prob_response

def permute_vocab(reference, target):
    """
        This is a naive method for matching two vocabs
        of different sizes. It permutes tokens of target vocab
        so that their possitions match reference vocab tokens.
    """
    assert len(set(reference).difference(target)) == 0, \
            "Reference vocab should be subset of target vocab."

    permutation = [0] * len(reference)
    for key, val in target.items():
        if key in reference:
            permutation[reference[key]] = val
    
    return permutation