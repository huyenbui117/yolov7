import json
import subprocess
import sys  
def main(config = 'cfg/custom/config.json'):
    cfg = json.load(open(config))
    # data_cfg = json.load(open(data_config))
    #
    base = f"python train.py --epochs {cfg['epochs']} --device {cfg['device']} --batch-size {cfg['batch_size']} --data {cfg['data']} --img-size {cfg['img_size']}  --cfg {cfg['cfg']} --name {cfg['base_name']} --hyp {cfg['hyp']} --weights '{cfg['weights']}' --workers 8 --entity baby_team --bbox_interval 1"
    if 'single_cls' in cfg and cfg['single_cls']:
        base += ' --single-cls'
    
    if 'upload_dataset' in cfg and cfg['upload_dataset']:
        base += ' --upload_dataset'
    
    if 'exist_ok' in cfg and cfg['exist_ok']:
        base += ' --exist-ok'
    print(base)
    subprocess.run(base, shell=True)
if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
