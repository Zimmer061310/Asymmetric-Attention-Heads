import json, time
from pathlib import Path
import yaml
root = Path('/home/featurize/work/AAH-v3-pro6000')
status_path = root / 'paper_results/backend_4096_realized_attention_ncu/status.jsonl'
log_path = root / 'paper_results/backend_4096_realized_attention_ncu/checkpoint_janitor.log'
def log(msg):
    with log_path.open('a') as f:
        f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + msg + '\n')
def exp_name(config):
    with (root / config).open() as f:
        cfg = yaml.safe_load(f)
    return cfg['experiment']['name'], Path(cfg['experiment'].get('out_dir', 'experiments'))
seen=set()
while True:
    try:
        if status_path.exists():
            for line in status_path.read_text().splitlines():
                if not line.strip():
                    continue
                row=json.loads(line)
                if row.get('status') != 'ok':
                    continue
                key=(row.get('backend'), row.get('method'), row.get('config'))
                if key in seen:
                    continue
                name,out_dir=exp_name(row['config'])
                deleted=0; bytes_deleted=0
                for p in (root / out_dir).glob(f'{name}*.pt'):
                    try:
                        sz=p.stat().st_size
                        p.unlink()
                        deleted += 1; bytes_deleted += sz
                        log(f'deleted {sz} {p}')
                    except FileNotFoundError:
                        pass
                seen.add(key)
                log(f'row_complete backend={row.get("backend")} method={row.get("method")} deleted={deleted} bytes={bytes_deleted}')
    except Exception as exc:
        log('error ' + repr(exc))
    time.sleep(300)
