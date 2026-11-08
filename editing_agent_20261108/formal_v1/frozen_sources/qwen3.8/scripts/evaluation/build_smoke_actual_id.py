#!/usr/bin/env python3
"""Choose deterministic coverage and high-budget inputs; never an accuracy sample."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
from evaluation_inputs_actual_id import validate_row
from media_inputs import atomic_json, sha_file


def budget_proxy(row):
    content = row['model_input']['messages'][1]['content']
    return sum((32 if x.get('media_role') == 'motion_reference' else 16) * 264192
               if x['type'] == 'video' else 451584 if x['type'] == 'image' else 0 for x in content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(x) for x in args.dataset.open() if x.strip()]
    selected = {}
    def add(row):
        selected[row['sample_id']] = row
    for row in random.Random(20261107).sample(rows, min(16, len(rows))):
        add(row)
    strata = {}
    for row in rows:
        audit = row['audit']
        key = (audit['history_topology'], audit['selection_kind'], audit.get('selection_axis'))
        if key not in strata or budget_proxy(row) > budget_proxy(strata[key]):
            strata[key] = row
    for row in strata.values():
        add(row)
    for field in ('exact_selection_reference_in_completed_history', 'singleton_only_candidate', 'explicit_history_textual_clue'):
        for value in (False, True):
            subset = [r for r in rows if r.get('audit_only', {}).get(field) is value]
            if subset:
                add(max(subset, key=budget_proxy))
    for row in sorted(rows, key=budget_proxy, reverse=True)[:4]:
        add(row)
    chosen = sorted(selected.values(), key=lambda r: r['sample_id'])
    for row in chosen:
        validate_row(row)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'eval.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n' for row in chosen))
    atomic_json(args.output / 'selection.json', {'source_dataset_sha256': sha_file(args.dataset),
        'sample_sha256': sha_file(path), 'rows': len(chosen), 'not_an_accuracy_sample': True,
        'selection_policy': '16 fixed-seed uniform + max-budget per topology/kind/axis + diagnostics true/false + top 4 budget proxies',
        'strata': dict(Counter('/'.join(str(row['audit'].get(k)) for k in ('history_topology', 'selection_kind', 'selection_axis')) for row in chosen)),
        'selected': [{'sample_id': row['sample_id'], 'budget_proxy': budget_proxy(row), 'audit': row['audit'],
                      'singleton_only_candidate': row['audit_only']['singleton_only_candidate'],
                      'exact_selection_reference_in_completed_history': row['audit_only']['exact_selection_reference_in_completed_history'],
                      'explicit_history_textual_clue': row['audit_only']['explicit_history_textual_clue']} for row in chosen]})
    print(json.dumps({'rows': len(chosen), 'output': str(path)}))


if __name__ == '__main__':
    main()
