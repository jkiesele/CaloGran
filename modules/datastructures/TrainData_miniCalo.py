
from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as djfto

def fileTimeOut(a,b):
    return djfto(a,b)

class TrainData_miniCalo(TrainData):
    
    def __init__(self):
        import numpy 
        TrainData.__init__(self)
        
        self.treename="events"
        
        self.undefTruth=['']
    
        self.truthclasses=[]
        
        self.remove=False
        self.weight=False
        
        self.weightbranchX='true_energy'
        self.weightbranchY='true_x'
        
        #is already flat
        self.referenceclass='flatten'
        self.weight_binX = numpy.array([0,0.1,40000],dtype=float) 
        self.weight_binY = numpy.array([-40000,40000],dtype=float) 
        
        
        
        
        self.regtruth='true_energy'

        self.regressiontargetclasses=['E']
        
        self.registerBranches([self.regtruth])
        
        self.reduceTruth(None)
        
        self.rebinx = 1
        self.rebiny = 1
        self.rebinz = 1
        
    #not needed, override
    def produceBinWeighter(self, filename):
        return self.make_empty_weighter()
    
    #not needed, override
    def produceMeansFromRootFile(self,orig_list, limit):
        import numpy
        return numpy.array([1.,])
    
    

    def formatPrediction(self, predicted_list):
        
        format_names = ['pred_E']
        out_pred = predicted_list
        
        return out_pred,  format_names
        
        
        
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
    
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        
        rfile = ROOT.TFile(filename)
        tree = rfile.Get(self.treename)
        self.nsamples=tree.GetEntries()
        
        from DeepJetCore.preprocessing import read4DArray
        
        
        x = read4DArray(filename,
                                      self.treename,
                                      "rechit",
                                      self.nsamples,
                                      xsize=50, 
                                      ysize=50, 
                                      zsize=125, 
                                      fsize=4,
                                      rebinx=self.rebinx,
                                      rebiny=self.rebiny,
                                      rebinz=self.rebinz)
        
        
        x = x / 1e6
        

        Tuple = self.readTreeFromRootToTuple(filename)  
        
        
        energytruth  =  numpy.array(Tuple[self.regtruth])
        
        print(x.shape)

        
        self.w=[]
        self.x=[x]
        self.y=[energytruth]
        
        
        
        
        
        
        
class TrainData_stage0(TrainData_miniCalo):
    
    def __init__(self):
        import numpy 
        TrainData_miniCalo.__init__(self)        
        
        #xsize=50, 
        #ysize=50, 
        #zsize=125, 
        #fsize=4,
        
        self.rebinx=50
        self.rebiny=50
        self.rebinz=125
        
        
        
        
        
        
        
        