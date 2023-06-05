import subprocess
import shlex


tables_to_compare = [
    'dolly-v2-7b.jsonl',
    'RedPajama-INCITE-Instruct-3B-v1__pythia-2.8b__dolly-v2-3b.jsonl',
    # 'stablelm-tuned-alpha-7b.jsonl',
    # 'RedPajama-INCITE-Instruct-3B-v1__stablelm-tuned-alpha-3b__dolly-v2-3b.jsonl',
    # 'RedPajama-INCITE-Instruct-3B-v1__stablelm-tuned-alpha-3b__pythia-2.8b.jsonl',
    'pythia-2.8b__dolly-v2-3b.jsonl',
    'RedPajama-INCITE-Instruct-3B-v1__dolly-v2-3b.jsonl',
    # 'RedPajama-INCITE-Instruct-3B-v1__stablelm-tuned-alpha-3b.jsonl',
    'RedPajama-INCITE-Instruct-3B-v1__pythia-2.8b.jsonl',
    'dolly-v2-3b.jsonl',
    # 'stablelm-tuned-alpha-3b.jsonl',
    'pythia-2.8b.jsonl',
    'RedPajama-INCITE-Instruct-3B-v1.jsonl',
]

cmd = """python eval_gpt_review.py -q table/question.jsonl -a table/answer/{} table/answer/{} \
-p table/prompt.jsonl -r table/reviewer.jsonl -o table/gpt3.5-review/{} \
"""
res = []
for i, table1 in enumerate(tables_to_compare, 1):
    for table2 in tables_to_compare[i:]:
        name1 = table1.split('/')[-1][:-6]
        name2 = table2.split('/')[-1][:-6]
        output = name1 + '-vs-' + name2
        brrr = cmd.format(table1, table2, output)
        res.append(brrr)
        if len(res) == 28:
            break
        subprocess.run(shlex.split(brrr))
        print(f'Processed {len(res)}!!')
# print(res[1])