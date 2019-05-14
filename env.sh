
#! /bin/bash
THISDIR=`pwd`
cd /afs/cern.ch/user/j/jkiesele/work/TESTDJ/DeepJetCore
source env.sh
cd $THISDIR
export CALOGRAN=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=$CALOGRAN

cd $CALOGRAN
export PYTHONPATH=$CALOGRAN/modules:$PYTHONPATH
export PYTHONPATH=$CALOGRAN/modules/datastructures:$PYTHONPATH
