import os, commands, sys
import numpy as np
from string import digits 
import ROOT as r
from ROOT import TFile, TCanvas, TPad, TChain, TH1D, TH2D, THStack, gROOT, gStyle, gPad
import glob, argparse, math
from util import *

parser = argparse.ArgumentParser()                                                                                                                                           
parser.add_argument('--stage', type=int, default=0, help='which stage?')
parser.add_argument('--energyBin', type=int, default=10, help='how big should the energy bins are?')
parser.add_argument('--lin', action='store_true', help='write out linearity?')

args, _ = parser.parse_known_args()                                       

stage = args.stage
energyBin = args.energyBin

maxEnergy = 110
numBins = int(math.floor(float(maxEnergy)/float(energyBin)))
hists = []
# fill arrays with resol/lin
energies = []
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

if numBins>len(colors()):
  for i in range(0,numBins-len(colors())):
    colors().append(colors()[i])
    alpha().append(0.3)
    lines().append(lines()[i])
    width().append(width()[i])
    fill().append(3001)

multi = THStack("multi","multi")
legTitle = "stage "+str(args.stage)
leg = r.TLegend(0.25,0.55,0.3,0.85)
leg.SetTextFont(132)
leg.SetTextSize(0.05)
leg.SetFillColor(0)
leg.SetFillStyle(0)

ifile = inFilePath+"/stage"+str(stage)+"/out.root"

miX=0.5
maX=1.5
if args.stage>1:
  miX=0.8
  maX=1.2

print ifile
tfile = TFile.Open(ifile, 'read')

for event in tfile.tree:
  enBin = int(math.floor(event.true_energy/float(energyBin)))
  hists[enBin].Fill( eval("event.pred_energy/event.true_energy") )  
  
outFile = TFile.Open("rootFiles/stage_"+str(stage)+".root","recreate")
outFile.cd()

for inBin in range(0,numBins):
  print inBin
  hists[inBin].SetDirectory(0)
  hists[inBin].SetLineColor(1)
  hists[inBin].SetLineWidth(1)
  hists[inBin].SetLineStyle(1)
  hists[inBin].SetMarkerColor(20)
  hists[inBin].Scale(1./hists[inBin].GetEntries())

  multi.Add(hists[inBin], "histE")
  leg.AddEntry(hists[inBin], str(energies[inBin])+"-"+str(energies[inBin+1])+"GeV", "lf")
  
  r.gROOT.SetBatch(True)
  histClone = hists[inBin]

  pars = fitGauss(histClone, 1)
  parL[inBin] = pars[0]
  parLerr[inBin] = pars[1]
  parR[inBin] = pars[2]/pars[0]
  parRerr[inBin] = 1/parL[inBin] * math.sqrt(math.pow(pars[3],2)+math.pow(parLerr[inBin]*parR[inBin],2))

  meanEn[inBin] = float(energies[inBin]+energies[inBin+1])/2.
  meanEnErr[inBin] = energyBin/2.
  
  legText = str(energies[inBin])+"-"+str(energies[inBin+1])+"GeV"

  drawEnergyDist(histClone, legTitle, legText, miX, maX)
  histClone.Write()

tfile.Close()
  
res = r.TGraphErrors(numBins, meanEn, parR, meanEnErr, parRerr)
res.SetMarkerStyle(20)
lin = r.TGraphErrors(numBins, meanEn, parL, meanEnErr, parLerr)
lin.SetMarkerStyle(20)

resLeg = "#sigma_{E_{pred}}/#LTE_{pred}#GT"
linLeg = "#LTE_{pred}/E_{true}#GT"

if not args.lin:
  drawResoLin(res, legTitle, resLeg, "resolution_stage_"+str(stage), True, 0, 0.2, "ap")
else:
  drawResoLin(lin, legTitle, linLeg, "linearity_stage_"+str(stage), False, 0.95, 1.05, "ap")


hist_reso = TH1D("hist_reso", "", numBins,0.,maxEnergy)
hist_lin = TH1D("hist_lin", "", numBins,0.,maxEnergy)
for ir,re in enumerate(parR):
  hist_reso.Fill(meanEn[ir], re)
  bin_r=hist_reso.FindBin(meanEn[ir], re)
  hist_reso.SetBinError(bin_r, parRerr[ir])
  hist_lin.Fill(meanEn[ir], parL[ir])
  bin_r=hist_lin.FindBin(meanEn[ir], parL[ir])
  hist_lin.SetBinError(bin_r, parLerr[ir])

hist_reso.SetName("reso")
hist_reso.SetDirectory(0)
hist_reso.Write()
hist_lin.SetName("lin")
hist_lin.SetDirectory(0)
hist_lin.Write()
outFile.Close()
