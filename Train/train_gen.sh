#!/bin/bash

stages=( "${1}" )
if [[ $2 ]]
then
lat=$2
else
lat=( "Z" )
fi

add=$3

exec 2>&1

cd 
source calogranenv.sh

traindir=$CALOGRAN/Train

s=0
for l in $lat
do
   if [ $l == "Z" ]
   then
     l=""
   fi
   for s in $stages
   do
       stage="stage${s}${l}"
       out="${add}out_${stage}"
       outdir="${traindir}/CaloGranOut/${out}"
       mkdir -p "${outdir}"
        python3 $traindir/new_stage_depth.py "${traindir}/data/${stage}/dataCollection.djcdc" $outdir > "${outdir}/out.log"
        predict.py "${outdir}/KERAS_model.h5" "${outdir}/trainsamples.djcdc" "${traindir}/testdata/${stage}/dataCollection.djcdc" "${outdir}/pred" > "${outdir}/pred.log"
   done
done

sleep 5