#!/bin/bash

gpuoffset=2

for i in `seq 0 3`
do
python3 new_stage_depth.py "data/stage${i}/dataCollection.djcdc" "CaloGranOut/${1}out_stage_${i}" > $i.log 
done
wait

for i in `seq 4 7`
do
python3 new_stage_depth.py "data/stage${i}/dataCollection.djcdc" "CaloGranOut/${1}out_stage_${i}"  > $i.log 
done
wait
