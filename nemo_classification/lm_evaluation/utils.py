import os
import json
import re
import urllib.request

from nemo.collections.nlp.modules.common.megatron.module import Float16Module


def add_model(base_model, new_model):
    if isinstance(new_model._model.model, Float16Module):
        language_model = new_model._model.model.module.language_model
    else:
        language_model = new_model._model.model.language_model
    
    if isinstance(base_model._model.model, Float16Module):
        base_model._model.model.module.add_model(language_model)
    else:
        base_model._model.model.add_model(language_model)


def load_http_text(url):
    with urllib.request.urlopen(url) as f:
        return f.read().decode("utf-8")


def load_text(path):
    if path.startswith("http://") or path.startswith("https://"):
        return load_http_text(path)
    else:
        with open(path) as f:
            return f.read()


def load_json(path):
    return json.loads(load_text(path))


def load_json_lines(path):
    return [json.loads(line) for line in load_text(path).split("\n") if line]


def dump_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def dump_dataframe(df, path):
    with open(path, "w") as f:
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
