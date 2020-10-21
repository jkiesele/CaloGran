#!/bin/bash

gpuoffset=2

global_out=/eos/home-j/jkiesele/CaloGranOut

function z_gran {

for i in `seq 0 7`
do
out="${global_out}/out_stage_${i}"
predict.py  "${out}/KERAS_model.h5"  "${out}/trainsamples.djcdc" "test/stage${i}/dataCollection.djcdc"  "${out}/pred"
done
wait

}


function xy_gran {

s=0

for i in A B C D E
do
out="${global_out}/out_stage_${s}${i}"
echo $out
predict.py  "${out}/KERAS_model.h5"  "${out}/trainsamples.djcdc" "test/stage${s}${i}/dataCollection.djcdc"  "${out}/pred"

done

}


xy_gran