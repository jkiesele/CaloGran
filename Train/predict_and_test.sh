#!/bin/zsh

stage=$1

echo "using /eos/home-c/cneubuse/miniCalo2/results/${stage}"

dataclass="TrainData_${stage}"

out_train="/data/ml/jkiesele/miniCalo_sw_comp/${dataclass}"
out_test="/data/ml/jkiesele/miniCalo_sw_comp/${dataclass}_test"
out_predict="/eos/home-c/cneubuse/miniCalo2/results/${stage}/prediction"

mkdir "/eos/home-c/cneubuse/miniCalo2/results/${stage}"

predict.py train_$stage/KERAS_model.h5 $out_test/dataCollection.dc $out_predict

makePlots.py $out_predict/tree_association.txt $out_predict/../
