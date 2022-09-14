import json
import subprocess

def main(config = 'cfg/custom/config.json'):
    cfg = json.load(open(config))
    # data_cfg = json.load(open(data_config))
    base = f"python test.py --weight {cfg['trained_weight']} --device {cfg['device']} --name {cfg['base_name']} --data {cfg['data']} --img-size {cfg['img_size']} --batch-size {cfg['batch_size']} --save-json --task test "
    if 'single_cls' in cfg and cfg['single_cls']:
        base += ' --single-cls'
    subprocess.run(base, shell=True)
if __name__ == '__main__':
    main()
