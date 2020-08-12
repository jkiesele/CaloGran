import os, commands, sys
import numpy as np
from string import digits 
import ROOT as r
from ROOT import TFile, TCanvas, TPad, TChain, TH1D, TH2D, THStack, gROOT, gStyle, gPad
import glob, argparse, math
import util as ut
#from util import *

#gStyle.SetPadTopMargin(0.06)

parser = argparse.ArgumentParser() 
parser.add_argument('--energies', nargs='*', default=[10, 50, 80], help='compare resolutions for a certain energy bin?')

args, _ = parser.parse_known_args()                                       

# fixed parameters for layer depth calculations
totalDepthCM=250
x0=1.265
lamda=25.4

# prepare for loop through all stages
drawEnergies = args.energies
# dict stage->[energy,energyErr,reso,resoErr]
resos=dict([
  (0, []),
  (1, []),
  (2, []),
  (3, []),
  (4, []),
  (5, []),
  (6, []),
  (7, []),
])

#fixed energy bin
energyBin = 10
maxEnergy = 110
numBins = int(math.floor(float(maxEnergy)/float(energyBin)))

for st in ut.stages():
  print "stage: ", st

  hists = []
  energies = []
  # fill arrays with resol/lin
  meanEn = np.empty(numBins, np.dtype('float64'))
  meanEnErr = np.empty(numBins, np.dtype('float64'))
  parR = np.empty(numBins, np.dtype('float64'))
  parRerr = np.empty(numBins, np.dtype('float64'))
  parL = np.empty(numBins, np.dtype('float64'))
  parLerr = np.empty(numBins, np.dtype('float64'))
  
  for ibin in range(0,numBins):
    hists.append(r.TH1D("tmp_"+str(ibin),"tmp_"+str(ibin),400,0.,2.))
    energies.append(int(ibin*energyBin))
  energies.append(maxEnergy)

  if numBins>len(ut.colors()):
    for i in range(0,numBins-len(ut.colors())):
      ut.colors().append( ut.colors()[i] )
      ut.alpha().append(0.3)
      ut.lines().append(ut.lines()[i])
      ut.width().append(ut.width()[i])
      ut.fill().append(3001)

  leg = r.TLegend(0.25,0.55,0.3,0.85)
  leg.SetTextFont(132)
  leg.SetTextSize(0.05)
  leg.SetFillColor(0)
  leg.SetFillStyle(0)

  ifile = "/eos/user/c/cneubuse/miniCalo2/pred/stage"+str(st)+"/out.root"

  miX=0.8
  maX=1.18
  #  if args.stage>1:
  #    miX=0.8
  #    maX=1.2

  print ifile
  tfile = TFile.Open(ifile, 'read')

  for event in tfile.tree:
    enBin = int(math.floor(event.true_energy/float(energyBin)))
    hists[enBin].Fill( eval("event.pred_energy/event.true_energy") )  
  
  for inBin in range(0,numBins):

    hists[inBin].SetDirectory(0)
    hists[inBin].SetLineColor(1)
    hists[inBin].SetLineWidth(1)
    hists[inBin].SetLineStyle(1)
    hists[inBin].SetMarkerColor(20)
    hists[inBin].Scale(1./hists[inBin].GetEntries())

    # r.gROOT.SetBatch(True)
    histClone = hists[inBin]

    pars = ut.fitGauss(histClone, 1)
    parL[inBin] = pars[0]
    parLerr[inBin] = pars[1]
    parR[inBin] = pars[2]/pars[0]
    parRerr[inBin] = 1/parL[inBin] * math.sqrt(math.pow(pars[3],2)+math.pow(parLerr[inBin]*parR[inBin],2))
    
    meanEn[inBin] = float(energies[inBin]+energies[inBin+1])/2.
    meanEnErr[inBin] = energyBin/2.

  tfile.Close()

  # fill dict with energy and resoltion
  for en in drawEnergies:
    binE = energies.index(en)
    resos[st].append([parR[binE], parRerr[binE]])

 
multi = r.TMultiGraph() 
legR = r.TLegend(0.5,0.65,0.8,0.85,"#bf{#splitline{Homo PbW Calorimeter}{#it{Geant4 simulation}}}")
legR.SetTextFont(132)
legR.SetTextSize(0.05)
legR.SetFillColor(0)
legR.SetFillStyle(0)

for ie,en in enumerate(drawEnergies):
  arrSt=np.empty(len(ut.stages()), np.dtype('float64'))
  arrStErr=np.empty(len(ut.stages()), np.dtype('float64'))
  arrRes=np.empty(len(ut.stages()), np.dtype('float64'))
  arrResErr=np.empty(len(ut.stages()), np.dtype('float64'))

  for sta in ut.stages():
    arrSt[sta] = ut.layer(sta) #longGranX[sta]
    arrStErr[sta] = 0.
    #print resos[sta][0]
    #print resos[sta][1]
    arrRes[sta] = resos[sta][ie][0]
    arrResErr[sta] = resos[sta][ie][1]

  res = r.TGraphErrors(len(ut.stages()), arrSt, arrRes, arrStErr, arrResErr)
  res.SetMarkerStyle(20)
  res.SetLineColor(ut.colors()[ie])
  res.SetMarkerColor(ut.colors()[ie])
  res.GetYaxis().SetTitle("#sigma_{E_{pred}}/#LTE_{pred}#GT")
  res.GetXaxis().SetTitle("# layer")
  multi.Add(res)
  
  legTitle = str(int(en+energyBin/2.))+"GeV"
  legR.AddEntry(res, legTitle, "pl")
  
#r.gROOT.SetBatch(False)
canE = TCanvas("canE", "",600,500)
#canE.SetLogx()
canE.cd()
multi.Draw("alp")
multi.GetYaxis().SetTitle("#sigma_{E_{pred}}/#LTE_{pred}#GT")
multi.GetXaxis().SetTitle("# layer") ##X_{0}")
multi.Draw("alp")
gPad.Modified()
#gPad.Update()
legR.Draw()

canE.Print("plots/all_stages_longitudinal_layer.pdf")

#var = raw_input("Please enter something: ")
