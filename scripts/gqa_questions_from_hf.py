"""Rebuild GQA {tier}_questions.json (from questions1.2.zip) out of the lmms-lab-encoder/GQA parquet.

For clusters that cannot reach downloads.cs.stanford.edu. Writes the format eval.py reads:
{question_id: {question, answer, isBalanced, types, semantic, entailed, equivalent, ...}}.

    hf download lmms-lab-encoder/GQA testdev_balanced_instructions/testdev-00000-of-00001.parquet \
        --repo-type dataset --local-dir /tmp/gqa_hf
    python scripts/gqa_questions_from_hf.py \
        --parquet /tmp/gqa_hf/testdev_balanced_instructions/testdev-00000-of-00001.parquet \
        --out playground/data/eval/gqa/data/testdev_balanced_questions.json
"""

import argparse
import json

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--parquet', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    questions = {}
    for row in pq.read_table(args.parquet).to_pylist():
        qid = row.pop('id')
        # The original annotations map "start:end" token spans to object ids.
        row['annotations'] = {key: {a['value']: a['objectId'] for a in value or []}
                              for key, value in (row.get('annotations') or {}).items()}
        for key in ('entailed', 'equivalent'):
            value = row.get(key)
            if isinstance(value, str):
                row[key] = json.loads(value.replace("'", '"')) if value else []
        questions[qid] = row
    with open(args.out, 'w') as handle:
        json.dump(questions, handle)
    print(f'{len(questions)} questions -> {args.out}')


if __name__ == '__main__':
    main()
