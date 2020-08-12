import os, commands, sys
import numpy as np
from string import digits 
import ROOT as r
from ROOT import TFile, TCanvas, TPad, TChain, TH1D, TH2D, THStack, gROOT, gStyle, gPad
import glob, argparse, math

# input file path 
inFilePath = "/afs/cern.ch/user/c/cneubuse/public/CaloGran/pred"

# define stages and draw options
def stages():
    return [0,1,2,3,4,5,6,7]
def colors():
    return [632,880,602,615,403,432,618,623,801,416]
def alpha():
    return [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
def lines():
    return [1,1,1,1,1,1,1,1,1,1]
def width():
    return [2,2,2,2,2,2,2,2,2,2]
def fill():
    return [0,0,0,0,0,0,0,0,0,0]
def style(): 
    return [20,21,22,23,24,25,26,27]

def layer(stage):
    lay=dict([
        (0, 1),
        (1, 6),
        (2, 10),
        (3, 12),
        (4, 15),
        (5, 20),
        (6, 30),
        (7, 60)
    ]) 
    return lay[stage]

def longGranX(stage):
    longX=dict([
        (0, totalDepthCM/layer[0]/x0),
        (1, totalDepthCM/layer[1]/x0),
        (2, totalDepthCM/layer[2]/x0),
        (3, totalDepthCM/layer[3]/x0)
    ])
    return longX[stage]

def longGranL(stage):
    longL=dict([
        (0, totalDepthCM/layer[0]/lamda),
        (1, totalDepthCM/layer[0]/lamda),
        (2, totalDepthCM/layer[0]/lamda),
        (3, totalDepthCM/layer[0]/lamda)
    ])
    return longL[stage]

def fitGauss( hist, color ):
    gauss=r.TF1("gauss","gaus", 0,2)
    gauss.SetLineColor(color)
    hist.Fit(gauss, "RQ")
    twoSigm=gauss.GetParameter(2)
    gauss.SetLineStyle(1)
    hist.Fit(gauss,"RQ","", gauss.GetParameter(1)-twoSigm, gauss.GetParameter(1)+twoSigm)
    
    g2=r.TF1("g2","gaus", 0.5,1.5)
    g2.FixParameter(0,gauss.GetParameter(0)) 
    g2.FixParameter(1,gauss.GetParameter(1))
    g2.FixParameter(2,gauss.GetParameter(2))
    g2.SetLineColor(color)
    g2.SetLineStyle(2)
    hist.GetListOfFunctions().Add(g2)
    
    return [gauss.GetParameter(1), gauss.GetParError(1), gauss.GetParameter(2), gauss.GetParError(2)];
    
def drawEnergyDist( hist, legT, leg, minX, maxX ):
    
    histClone = hist.Clone()
    legD = r.TLegend(0.25,0.55,0.7,0.85, legT)
    legD.SetTextFont(132)
    legD.SetTextSize(0.05)
    legD.SetFillColor(0)
    legD.SetFillStyle(0)
    legD.AddEntry(hist, leg, "lp")
    
    r.gROOT.SetBatch(True)
    canD = TCanvas("canD", "",600,500)
    canD.cd()
    histClone.Draw("histe")
    histClone.GetXaxis().SetTitle("E_{pred}/E_{true}")
    histClone.GetYaxis().SetTitle("norm")
    histClone.GetXaxis().SetRangeUser(minX,maxX)
    histClone.Draw("func same")
    legD.Draw()
    gPad.Update()
    gPad.SaveAs("plots/"+str(legT).replace(" ","_")+"_energyDists_"+str(leg)+".pdf")
    
# reso/lin, title of legend, y axis label, outfile name e.g. "resolution", bool if fit with stoch. and constant term, minimum y, maximum y 
def drawResoLin( graph, legT, yTitle, name, fit, minY, maxY, draw):
    legR = r.TLegend(0.5,0.65,0.8,0.85, legT)
    legR.SetTextFont(132)
    legR.SetTextSize(0.05)
    legR.SetFillColor(0)
    legR.SetFillStyle(0)
    
    if fit:
        
        freso = r.TF1('reso', 'sqrt(pow([0]/sqrt(x),2) + pow([1],2))',1, 100)
        freso.SetLineColor(graph.GetLineColor())
        graph.Fit(freso)
        sampl = "{0:.2f}".format(freso.GetParameter(0)*100)
        const = "{0:.2f}".format(freso.GetParameter(1)*100)
        legR.AddEntry(graph, "#frac{"+str(sampl)+"%}{#sqrt{E}} #oplus "+str(const)+"%", "lp")

    else:
       
        legR.AddEntry(graph, graph.GetName(), "lp")

    if draw:
        r.gROOT.SetBatch(False)
        canE = TCanvas("canE_"+str(name), "",600,500)
        canE.cd()
        graph.Draw(draw)
        graph.GetYaxis().SetTitle(yTitle)
        graph.GetXaxis().SetTitle("E_{true} [GeV]")
        graph.GetXaxis().SetRangeUser(0,105)
        graph.GetYaxis().SetRangeUser(minY,maxY)
        graph.Draw(draw)
        legR.Draw()
        
        canE.Print("plots/"+name+".pdf")
        r.gROOT.SetBatch(True)

