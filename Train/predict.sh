#!/bin/bash

gpuoffset=2

for i in `seq 0 7`
do
predict.py  "out_stage_${i}/KERAS_model.h5"  "out_stage_${i}/trainsamples.djcdc" "test/stage${i}/dataCollection.djcdc"  "out_stage_${i}/pred"
done
wait
