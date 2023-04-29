#!/bin/bash
rm -rf huyenbk/iai-baby/data/baby/test.cache           
rm -rf huyenbk/iai-baby/data/baby/train.cache           
rm -rf huyenbk/iai-baby/data/baby/val.cache  
python train_custom.py cfg/custom/config.json
python test_custom.py cfg/custom/config.json
