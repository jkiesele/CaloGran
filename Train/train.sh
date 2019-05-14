#!/bin/zsh

stage=$1

echo "using ${stage}"

dataclass="TrainData_${stage}"
in_train="/data/ml/jkiesele/miniCalo_sw_comp/${dataclass}"
out_dir="train_${stage}"

python "${stage}.py" $in_train/dataCollection.dc $out_dir
