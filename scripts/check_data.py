#!/usr/bin/env python3


from argparse import ArgumentParser
from plotting_tools import plotevent
import numpy as np

parser = ArgumentParser('Make some plots')
parser.add_argument('inputFile')
args = parser.parse_args()

infile = str(args.inputFile)

from DeepJetCore.TrainData import TrainData
import matplotlib.pyplot as plt

td=TrainData()
td.readFromFile(infile)

feat = td.transferFeatureListToNumpy()[0]
truth = td.transferTruthListToNumpy()[0]
nevents = min(len(feat),10)


for e in range(nevents):
    
    print('true energy', truth[e])
    print('reco sum   ', np.sum(feat[e,:,:,:,0]))
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x [idx]")
    ax.set_zlabel("y [idx]")
    ax.set_ylabel("z [idx]")
    ax.grid(False)
    
    print('plotting...')
    plotevent(e,feat,ax, usegrid=True)
    
    plt.show()
    plt.close(fig)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x [mm]")
    ax.set_zlabel("y [mm]")
    ax.set_ylabel("z [mm]")
    ax.grid(False)
    
    print('plotting...')
    plotevent(e,feat,ax, usegrid=False)
    
    plt.show()
    
    #exit()