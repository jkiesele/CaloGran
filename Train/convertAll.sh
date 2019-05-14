
#!/bin/zsh

stage=$1

trainfiles="/eos/home-c/cneubuse/miniCalo2/prod/${stage}/files.txt"
testfiles="/eos/home-c/cneubuse/miniCalo2/test/${stage}/files.txt"
dataclass="TrainData_${stage}"

out_train="/data/ml/jkiesele/miniCalo_sw_comp/${dataclass}"
out_test="/data/ml/jkiesele/miniCalo_sw_comp/${dataclass}_test"


convertFromRoot.py -i $trainfiles -o $out_train -c $dataclass

convertFromRoot.py -r $out_train/snapshot.dc
convertFromRoot.py -r $out_train/snapshot.dc
convertFromRoot.py -r $out_train/snapshot.dc
convertFromRoot.py -r $out_train/snapshot.dc


convertFromRoot.py --testdatafor $out_train/dataCollection.dc -i $testfiles -o $out_test

convertFromRoot.py -r $out_test/snapshot.dc
convertFromRoot.py -r $out_test/snapshot.dc
convertFromRoot.py -r $out_test/snapshot.dc
convertFromRoot.py -r $out_test/snapshot.dc