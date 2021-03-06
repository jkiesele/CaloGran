
from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as djfto
import numpy as np
import uproot
import ROOT
import tensorflow as tf

def fileTimeOut(a,b):
    return djfto(a,b)

class TrainData_c(TrainData):
    
    def __init__(self):
        TrainData.__init__(self)
        
        
        self.rebinx=1
        self.rebiny=1
        self.rebinz=1
        
    def fileIsValid(self, filename):
        try:
            fileTimeOut(filename, 2)
            tree = uproot.open(filename)["B4"]
            f=ROOT.TFile.Open(filename)
            t=f.Get("B4")
            if t.GetEntries() < 1:
                return False
        except Exception as e:
            return False
        return True
        
    def readAndReshape(self, tree,branchname):
        entry = np.array( list(tree[branchname].array()) ,dtype='float32')
        a = np.reshape(entry, [-1, 60,30,30,1])
        return np.transpose(a, axes=[0,2,3,1,4]).copy(order='C') # B x y z F
    
    ###TBI
    def rebin(self,x):
        # say for 15:
        #
        # -1 , 15, 2, ... and sum 2
        #
        #
        if self.rebinx>1 or self.rebiny>1 or self.rebinz>1:
            
            pool = tf.keras.layers.AveragePooling3D(pool_size=(self.rebinx, self.rebiny, self.rebinz))
            
            x = pool.call(x)
            x = x.numpy()
            
            x[:,:,:,:,0] *= self.rebinx * self.rebiny * self.rebinz #keep energy, the rest doesn't really matter
           
            
        return x
        
    def convertFromSourceFileToArrays(self, filename, weighterobjects, istraining):  
    
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        
        
        tree = uproot.open(filename)["B4"]
        
        
        rechit_energy = self.readAndReshape(tree, "rechit_energy")
        rechit_x   = self.readAndReshape(tree, "rechit_x")
        rechit_y   = self.readAndReshape(tree, "rechit_y")
        rechit_z   = self.readAndReshape(tree, "rechit_z")
        rechit_vxy =self.readAndReshape(tree, "rechit_vxy")
        rechit_vz  = self.readAndReshape(tree, "rechit_vz")
        
        
        
        x = np.concatenate([rechit_energy,
                            rechit_x,
                            rechit_y,
                            rechit_z,
                            rechit_vxy,
                            rechit_vz],axis=-1)

        x = self.rebin(x)
        
        true_energy = np.array( list(tree["true_energy"].array()) ,dtype='float32')
        
        return [x], [true_energy], []
        
    def convertFromSourceFile(self, filename, weighterobjects, istraining):  
        return self.convertFromSourceFileToArrays(filename, weighterobjects, istraining)
        
        
        
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        
        pred_energy = predicted[0]#there is one last 1 dimension
        truth_energy = np.expand_dims(truth[0], axis=1)
        
        from root_numpy import array2root
        out = np.core.records.fromarrays(np.concatenate([pred_energy,truth_energy], axis=-1).transpose(), 
                                             names='pred_energy, true_energy')
        
        array2root(out, outfilename+".root", 'tree')
        
        
        
        
        
class TrainData_c_stage0(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=30
        self.rebiny=30
        self.rebinz=60 #1
        
class TrainData_c_stage1(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=30
        self.rebiny=30
        self.rebinz=10  #6
        
class TrainData_c_stage2(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=30
        self.rebiny=30
        self.rebinz=6  #10
        
class TrainData_c_stage3(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=30
        self.rebiny=30
        self.rebinz=5 #12
        
class TrainData_c_stage4(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=30
        self.rebiny=30
        self.rebinz=4 #15
        
class TrainData_c_stage5(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=30
        self.rebiny=30
        self.rebinz=3 #20
        
class TrainData_c_stage6(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=30
        self.rebiny=30
        self.rebinz=2 #30
        
class TrainData_c_stage7(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=30
        self.rebiny=30
        self.rebinz=1 #60
        
        

class TrainData_c_stage0A(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=10 #3
        self.rebiny=10 #3
        self.rebinz=60 #1
        
        
class TrainData_c_stage0B(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=6 #5
        self.rebiny=6 #5
        self.rebinz=60 #1
        
        
class TrainData_c_stage0C(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=3 #10
        self.rebiny=3 #10
        self.rebinz=60 #1
        
        
class TrainData_c_stage0D(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=2 #15
        self.rebiny=2 #15
        self.rebinz=60 #1
        
        
class TrainData_c_stage0E(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=1 #30
        self.rebiny=1 #30
        self.rebinz=60 #1
        
        
        
class TrainData_c_stage5A(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=10 #3
        self.rebiny=10 #3
        self.rebinz=3 #10
        
        
class TrainData_c_stage5B(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=6 #5
        self.rebiny=6 #5
        self.rebinz=3 #10
        
        
class TrainData_c_stage5C(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=3 #10
        self.rebiny=3 #10
        self.rebinz=3 #10
        
        
class TrainData_c_stage5D(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=2 #15
        self.rebiny=2 #15
        self.rebinz=3 #10
        
        
class TrainData_c_stage5E(TrainData_c):
    
    def __init__(self):
        TrainData_c.__init__(self)        
        
        self.rebinx=1 #30
        self.rebiny=1 #30
        self.rebinz=3 #10
        