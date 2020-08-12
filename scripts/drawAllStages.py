import os, commands, sys
import numpy as np
from string import digits 
import ROOT as r
from ROOT import TFile, TCanvas, TPad, TChain, TH1D, TH2D, THStack, gROOT, gStyle, gPad
import glob, argparse, math
from util import * 

parser = argparse.ArgumentParser()                                                                                                                                           
parser.add_argument('--single', action='store_true', help='write out sinlge reso plots?')

args, _ = parser.parse_known_args()                                       

multi = THStack("multi","multi")
multiLin = THStack("multiLin","multiLin")
legTitle = "#bf{#splitline{Homo PbW Calorimeter}{#it{Geant4 simulation}}}"
legT = r.TLegend(0.6,0.8,0.95,0.95, legTitle)
legT.SetTextSize(0.05)
leg = r.TLegend(0.6,0.5,0.9,0.8)
leg.SetTextFont(132)
leg.SetTextSize(0.05)
leg.SetFillColor(0)
leg.SetFillStyle(0)

for st in stages():
  ifile = "/eos/user/c/cneubuse/miniCalo2/results/macros/rootFiles/stage_"+str(st)+".root"
  
  print ifile

  tfile = TFile.Open(ifile, 'read')
  hist = tfile.Get("reso")
  hist.SetDirectory(0)
  hist.SetLineColor(colors()[st])
  hist.SetMarkerColor(colors()[st])
  hist.SetMarkerStyle(style()[st])
  hist.SetName("stage"+str(st))
  multi.Add(hist, "pe1")
  leg.AddEntry(hist, "long. layer: "+str( layer(st) ), "lfp")
  
  if args.single:
    drawResoLin( hist, legTitle, "#sigma_{E_{pred}}/#LT E_{pred}#GT", "reso_stage_"+str(st), True, 0, 0.2, "pe")
  else:
    drawResoLin( hist, legTitle, "#sigma_{E_{pred}}/#LT E_{pred}#GT", "reso_stage_"+str(st), True, 0, 0.2, False)

  histLin = tfile.Get("lin")
  histLin.SetDirectory(0)
  histLin.SetLineColor(colors()[st])
  histLin.SetMarkerColor(colors()[st])
  histLin.SetMarkerStyle(style()[st])
  histLin.SetName("stage"+str(st))
  multiLin.Add(histLin, "pe1")

  if args.single:
    drawResoLin( histLin, legTitle, "#LT E_{pred}#GT/E_{true}", "lin_stage_"+str(st), False, 0.95, 1.05, "pe")
  else:
     drawResoLin( histLin, legTitle, "#LT E_{pred}#GT/E_{true}", "lin_stage_"+str(st), False, 0.95, 1.05, False)

  tfile.Close()
  
r.gROOT.SetBatch(False)
can = TCanvas("can","")
multi.Draw("nostack")
multi.GetXaxis().SetTitle("E [GeV]")
multi.GetYaxis().SetTitle("#sigma_{E_{pred}}/#LT E_{pred}#GT")
multi.GetXaxis().SetRangeUser(0,102)
multi.SetMaximum(0.1)
multi.Draw("nostack")
leg.Draw()
legT.Draw()
can.Print("plots/allStages_reso.pdf")

canLin = TCanvas("canLin","")
multiLin.Draw("nostack")
multiLin.GetXaxis().SetTitle("E [GeV]")
multiLin.GetYaxis().SetTitle("#LT E_{pred}#GT/E_{true}")
multiLin.GetXaxis().SetRangeUser(0,102)
multiLin.SetMaximum(1.15)
multiLin.SetMinimum(0.95)
multiLin.Draw("nostack")
leg.Draw()
legT.Draw()
canLin.Print("plots/allStages_lin.pdf")


input("Press a key!")
