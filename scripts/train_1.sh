#!/bin/bash
rm -rf /data/baby/Workspace/huyenbk/iai-baby/data/baby/test.cache  
rm -rf /data/baby/Workspace/huyenbk/iai-baby/data/baby/train.cache           
rm -rf /data/baby/Workspace/huyenbk/iai-baby/data/baby/val.cache  
python utils/convert_label.py --config cfg/custom/config_1.json
python train_custom.py cfg/custom/config_1.json
python test_custom.py cfg/custom/config_1.json