

import collections
import keras.backend as K
    
class Learning_sched(object):
    def __init__(self, 
                 lr=None, 
                 nepochs=None,
                 batchsize=None, 
                 loss_weights=None, 
                 loss=None, 
                 funccall=None):
        
        self.lr=lr
        self.nepochs=nepochs
        self.batchsize=batchsize
        if not loss_weights is None:
            self.loss_weights=loss_weights
        elif not loss is None:
            self.loss_weights=[1. for i in range(len(loss))]
        else:
            self.loss_weights=loss_weights
        self.loss=loss
        self.funccall=funccall
        

    def fill(self, rhs):
        if not rhs.lr is None:
            self.lr=rhs.lr
        if not rhs.nepochs is None:
            self.nepochs=rhs.nepochs
        if not rhs.batchsize is None:
            self.batchsize=rhs.batchsize
        if not rhs.loss_weights is None:
            self.loss_weights=rhs.loss_weights
        if not rhs.loss is None:
            self.loss=rhs.loss
        if not rhs.funccall is None:
            self.funccall=rhs.funccall
        




def scheduled_training(learn, train,clipnorm,metrics=None, **trainargs):

    totalepochs=0
    trainepochs=0
    lrs = Learning_sched()
    for ilrs in learn:
        
        lrs.fill(ilrs)
        
        if train.trainedepoches>totalepochs:
            print('skipping already trained epochs: '+str(train.trainedepoches))
            remainingepochs=lrs.nepochs
            if lrs.nepochs+totalepochs<train.trainedepoches:
                totalepochs+=lrs.nepochs
                trainepochs=0
            else:
                trainepochs=totalepochs+lrs.nepochs-train.trainedepoches
                totalepochs=train.trainedepoches
        else:
            trainepochs=lrs.nepochs
        

    
        if trainepochs > 0:
            if not lrs.funccall is None:
                lrs.funccall(train)
            if not lrs.loss_weights is None:
                train.compileModel(learningrate=0.01, #anyway overwritten
                   clipnorm=clipnorm,
                   loss=lrs.loss,
                   metrics=metrics,
                   loss_weights=lrs.loss_weights)
                print(train.keras_model.summary())
            
            for l in train.keras_model.layers:
                if l.trainable and l.weights and len(l.weights):
                    print('trainable '+l.name)
                    
            K.set_value(train.keras_model.optimizer.lr, lrs.lr)
            print('set learning rate to '+str(lrs.lr))
            print('training for epochs: '+str(trainepochs))
            
            model,history = train.trainModel(nepochs=totalepochs+trainepochs, 
                                     batchsize=int(lrs.batchsize),
                                     **trainargs)
            totalepochs+=lrs.nepochs
            remainingepochs=0
    
    