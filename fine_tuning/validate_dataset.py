import json, sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv)>1 else Path('data/sample_eval/engineering_train.jsonl')
rows=[]
for i,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
    if not line.strip(): continue
    obj=json.loads(line); rows.append(obj)
    if 'prompt' not in obj or 'completion' not in obj:
        raise SystemExit(f'Line {i} must contain prompt and completion')
print(f'OK: {len(rows)} training rows found in {path}')
print('Approx characters:', sum(len(r['prompt'])+len(r['completion']) for r in rows))
