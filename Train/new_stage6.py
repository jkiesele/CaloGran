from DeepJetCore.training.training_base import training_base

from keras.layers import Dense, Dropout, Flatten, Convolution2D, Convolution3D, Convolution1D, Conv2D, LSTM, LocallyConnected2D, BatchNormalization

from keras.models import Model

from keras.layers import Concatenate, Add, Multiply


from Layers import split_layer, simple_correction_layer
from DeepJetCore.DJCLayers import ScalarMultiply, Print, SelectFeatures

from tools import create_conv_resnet


def normalise_but_energy(x, momentum=0.6):
    e = SelectFeatures(0,1)(x)
    r = SelectFeatures(1,x.shape[-1])(x)
    r = BatchNormalization(momentum=momentum)(r)
    return Concatenate()([e,r])


def mymodel(Inputs, momentum=0.6):
    
    x = Inputs[0]
    x = normalise_but_energy(x) # ... 1 x 1 x 6 x F
    x = create_conv_resnet(x, name='rn1',
                       kernel_dumb=(1,1,2),
                       nodes_lin=24,
                       nodes_nonlin=32, 
                       kernel_nonlin_a=(1,1,3), 
                       kernel_nonlin_b=(1,1,3), 
                       lambda_reg=0,
                       dropout=-1)
    x = create_conv_resnet(x, name='rn2',
                       kernel_dumb=(1,1,3),
                       nodes_lin=24,
                       nodes_nonlin=32, 
                       kernel_nonlin_a=(1,1,4), 
                       kernel_nonlin_b=(1,1,4), 
                       lambda_reg=0,
                       dropout=-1)
    
    e = SelectFeatures(0,1)(x)
    e = Flatten()(e)
    x = Flatten()(x)
    x = Dense(32,activation='elu')(x)
    x = ScalarMultiply(10.)(x)
    x = Concatenate()([e,x])
    x = Dense(1, name="energy", kernel_initializer='ones')(x)

    model = Model(inputs=Inputs, outputs=[x])
    return model


from Losses import huber_loss_calo, reduced_mse
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
                            nepochs=50,   
                            batchsize=100,
                            #loss=[binned_global_correction_loss_random]))
                            loss=[reduced_mse]))

learn.append(Learning_sched(lr=1e-4,     
                            nepochs=50, 
                            batchsize=1000,
                            loss = [huber_loss_calo]))


learn.append(Learning_sched(lr=1e-6,     
                            nepochs=20,  
                            batchsize=2000,
                            loss=[binned_global_correction_loss_random]))




import keras
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=train.outputDir+'logs')
#usemetrics
scheduled_training(learn, train, 
                   verbose=1,
                   clipnorm=None,
                   additional_callbacks=[predcb])











