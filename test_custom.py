import json
import subprocess
import sys
import utils.post_process as post_process

def main(config = 'cfg/custom/config.json'):
    cfg = json.load(open(config))
    # data_cfg = json.load(open(data_config))
    base = f"python test.py --weight {cfg['trained_weight']} --device {cfg['device']} --name {cfg['base_name']} --data {cfg['data']} --img-size {cfg['img_size']} --batch-size 1 --save-json --task test --verbose --save-conf"
    if 'single_cls' in cfg and cfg['single_cls']:
        base += ' --single-cls'
    
    if 'exist_ok' in cfg and cfg['exist_ok']:
        base += ' --exist-ok'
    subprocess.run(base, shell=True)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()