from DeepJetCore.training.training_base import training_base

from keras.layers import Dense, Dropout, Flatten, Convolution2D, Convolution3D, Convolution1D, Conv2D, LSTM, LocallyConnected2D, BatchNormalization

from keras.models import Model

from keras.layers import Concatenate, Add, Multiply


from Layers import split_layer, simple_correction_layer
from DeepJetCore.DJCLayers import ScalarMultiply, Print, SelectFeatures

from tools import create_conv_resnet, normalise_but_energy


from argparse import ArgumentParser
parser = ArgumentParser('Run all trainings')

parser.add_argument("--xy",   help="",  default=False, action="store_true")

train = training_base(testrun=False, parser=parser,resumeSilently=True)


nodes_lin = 16
nodes_nonlin = 32

#1x1x10
def mymodel(Inputs, momentum=0.6):
    
    x = Inputs[0]
    x = normalise_but_energy(x, name='batchnorm_input',momentum=momentum)
    
    xystride = 1
    xyrnkern = 1
    if x.shape[-3] > 6:
        xystride=2
        xyrnkern=3
        
    onlysum=False
        
    i = 0
    while x.shape[-2] > 6 or x.shape[-3] > 6:
        x = create_conv_resnet(x, name='rn_'+str(i),
                       kernel_dumb=(xystride,xystride,2),
                       nodes_lin=16,
                       nodes_nonlin=32, 
                       kernel_nonlin_a=(1,xyrnkern,3), 
                       kernel_nonlin_b=(xyrnkern,1,3), 
                       lambda_reg=1e-5,
                       dropout=0.05)
        x = normalise_but_energy(x, name='batchnorm_conv_'+str(i),momentum=momentum)
        print('added conv block',i)
        i+=1
    
    x = Flatten()(x)
    if onlysum:
        x = Dense(1, name="energy_last")(x)
    else:
        x = Dense(128,activation='elu')(x)
        x = Dense(64,activation='elu',name='dense_last')(x)
        x = Dense(1, name="energy_last")(x)

    model = Model(inputs=Inputs, outputs=[x])
    return model

from Losses import huber_loss_calo, reduced_mse
from Losses import binned_global_correction_loss,binned_global_correction_loss_rel, binned_global_correction_loss_random


# also dows all the parsing

if not train.modelSet():
    train.setModel(mymodel)

    train.compileModel(learningrate=0.001,
                       loss=['mean_squared_error'],)
                       #metrics=usemetrics)

print(train.keras_model.summary())
#exit()

from tools import offset_plotter

pltr = offset_plotter(train,relative=True)
from DeepJetCore.training.DeepJet_callbacks import PredictCallback

predcb=PredictCallback(samplefile=train.val_data.getSamplePath(train.val_data.samples[0]),
                       function_to_apply=pltr.make_plot,  
                       after_n_batches=-1, #1000, 
                       batchsize=10000,
                       on_epoch_end=True,
                       use_event=-1)

from training_scheduler import scheduled_training, Learning_sched

learn=[]

learn.append(Learning_sched(lr=1e-4,     
                            nepochs=1,   
                            batchsize=256,#128
                            #loss=[binned_global_correction_loss_random]))
                            loss=[huber_loss_calo]))

learn.append(Learning_sched(lr=1e-4,     
                            nepochs=19, 
                            batchsize=512,
                            loss = [huber_loss_calo]))

learn.append(Learning_sched(lr=1e-5,     
                            nepochs=60, 
                            batchsize=1280,
                            loss = [huber_loss_calo]))

def freeze_batchnorm(train):
    from DeepJetCore.modeltools import fixLayersContaining
    train.keras_model = fixLayersContaining(train.keras_model, "batchnorm")#fix all but

learn.append(Learning_sched(lr=1e-5,     
                            nepochs=20, 
                            funccall=freeze_batchnorm,
                            batchsize=1280,
                            loss = [huber_loss_calo]))

def freeze_all_but_last(train):
    from DeepJetCore.modeltools import fixLayersContaining
    train.keras_model = fixLayersContaining(train.keras_model, "last", invert=True)#fix all but
    

#basically a chi2 fit here
learn.append(Learning_sched(lr=1e-5,     
                            nepochs=50,  
                            funccall=freeze_all_but_last,
                            batchsize=1280,
                            loss=[binned_global_correction_loss_random]))



#usemetrics
scheduled_training(learn, train, 
                   verbose=2,
                   clipnorm=None,
                   additional_callbacks=[predcb])












