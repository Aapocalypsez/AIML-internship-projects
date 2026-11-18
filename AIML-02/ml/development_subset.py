"""Create an explicitly limited development cohort without changing split membership."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def subset(source, output, limits):
    with Path(source).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    selected = []
    for split, limit in limits.items():
        candidates = [r for r in rows if r['split'] == split]
        random.Random(42).shuffle(candidates)
        selected.extend(candidates[:limit])
    with Path(output).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(selected)
    report = {'purpose': 'Limited development baseline, not a full-dataset benchmark',
              'seed': 42, 'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              'images': {s: sum(r['split'] == s for r in selected) for s in limits},
              'patients': {s: len({r['patient_id'] for r in selected if r['split'] == s}) for s in limits}}
    Path(output).with_suffix('.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='data/splits/classifier.csv')
    p.add_argument('--output', default='data/splits/classifier_development.csv')
    p.add_argument('--train', type=int, default=2048)
    p.add_argument('--val', type=int, default=512)
    p.add_argument('--test', type=int, default=512)
    a = p.parse_args()
    if min(a.train, a.val, a.test) < 1:
        p.error('All split limits must be positive')
    print(subset(a.source, a.output, {'train': a.train, 'val': a.val, 'test': a.test}))
