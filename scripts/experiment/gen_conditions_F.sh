#!/bin/sh

cmd="gen_conditions.py --seed=5 --query-trials=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20"

python $cmd F-nfb-10-cb0
python $cmd F-nfb-10-cb1
python $cmd --text-fb F-fb-10-cb0
python $cmd --text-fb F-fb-10-cb1
python $cmd --text-fb --video-fb F-vfb-10-cb0
python $cmd --text-fb --video-fb F-vfb-10-cb1
