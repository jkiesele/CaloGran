from DeepJetCore.training.training_base import training_base

from keras.layers import Dense, Dropout, Flatten, Convolution2D, Convolution3D, Convolution1D, Conv2D, LSTM, LocallyConnected2D, BatchNormalization

from keras.models import Model

from keras.layers import Concatenate, Add, Multiply


from Layers import split_layer, simple_correction_layer
from DeepJetCore.DJCLayers import ScalarMultiply, Print, SelectFeatures

from tools import create_conv_resnet, normalise_but_energy


def mymodel(Inputs, momentum=0.6):
    
    x = Inputs[0]
    x = normalise_but_energy(x)
    print('>>>>> x shape',x.shape)
    #x = create_conv_resnet(x, name='rn1',
    #                   kernel_dumb=(1,1,2),
    #                   nodes_lin=16,
    #                   nodes_nonlin=24, 
    #                   kernel_nonlin_a=(1,1,3), 
    #                   kernel_nonlin_b=(1,1,3), 
    #                   lambda_reg=0,
    #                   dropout=-1)
    #x = create_conv_resnet(x, name='rn2',
    #                   kernel_dumb=(1,1,3),
    #                   nodes_lin=16,
    #                   nodes_nonlin=24, 
    #                   kernel_nonlin_a=(1,1,3), 
    #                   kernel_nonlin_b=(1,1,3), 
    #                   lambda_reg=0,
    #                   dropout=-1)
    #
    x = Flatten()(x)
    x = Dense(128,activation='elu')(x)
    x = Dense(32,activation='elu')(x)
    x = ScalarMultiply(10.)(x)
    x = Dense(1, name="energy")(x)
    
    model = Model(inputs=Inputs, outputs=[x])
    return model


from Losses import huber_loss_calo, reduced_mse, loss_calo
from Losses import binned_global_correction_loss,binned_global_correction_loss_rel, binned_global_correction_loss_random


# also dows all the parsing
train = training_base(testrun=False)

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

learn.append(Learning_sched(lr=1e-3,     
                            nepochs=1,   
                            batchsize=100,
                            #loss=[binned_global_correction_loss_random]))
                            loss=[loss_calo]))

learn.append(Learning_sched(lr=1e-6,     
                            nepochs=50, 
                            batchsize=1100))

learn.append(Learning_sched(lr=1e-6,     
                            nepochs=20,  
                            batchsize=2100,
                            loss=[binned_global_correction_loss_random]))


import keras
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=train.outputDir+'logs')
#usemetrics
scheduled_training(learn, train, 
                   verbose=1,
                   clipnorm=None,
                   additional_callbacks=[predcb])












