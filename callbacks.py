import pickle
from keras import callbacks as ckbs
from keras.utils import np_utils
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, matthews_corrcoef
import pandas as pd
import copy
import warnings

class FullModelCheckpoint(ckbs.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if (epoch <= 1):
            self.best_weights = copy.copy(self.model.get_weights())
        self.best = -999
        if self.save_best_only:
            current = logs.get(self.monitor)
            if current is None:
                warnings.warn('Can save best model only with %s available, '
                              'skipping.' % (self.monitor), RuntimeWarning)
            else:
                if (self.monitor_op(current, self.best)) or (self.best == -999):
                    if self.verbose > 0:
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                              ' saving model to %s'
                              % (epoch, self.monitor, self.best,
                                 current, filepath))
                    self.best = current
                    self.best_weights = copy.copy(self.model.get_weights())
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: %s did not improve' %
                              (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            self.best_weights = copy.copy(self.model.get_weights())
            
class LossHistory(ckbs.Callback):
    def __init__(self, X_train, y_train, X_test, y_test, score_func):
        self.X_test = X_test
        self.X_train = X_train
        self.y_test = y_test
        self.y_train = y_train
        self.score_func = score_func
        
    def get_history(self):
        histories = {'train_losses' : self.train_losses, 'val_losses' : self.val_losses, 
                         'add_train_losses' : self.add_train_losses,'add_test_losses' : self.add_val_losses}
        return histories
        
    def on_train_begin(self, logs={}):
        self.train_losses = []
        self.val_losses = []
        self.add_val_losses = []
        self.add_train_losses = []
        self.best_score = 0 
        
    def on_epoch_end(self, epoch, logs={}):
        try:
            self.train_losses.append(mse(self.y_train, self.model.predict(self.X_train)))
        except ValueError:
            print("Warning: ValueError was caused by MSE calculation")
            self.train_losses.append(-999)
        self.val_losses.append(logs.get('val_loss'))        

        if self.score_func == 'accuracy':
            true_train = np_utils.probas_to_classes(self.y_train)
            pred_train = np_utils.probas_to_classes(self.model.predict(self.X_train))
            true_test = np_utils.probas_to_classes(self.y_test)
            pred_test = np_utils.probas_to_classes(self.model.predict(self.X_test))
            try:
                self.add_train_losses.append(accuracy_score(true_train, pred_train))
            except ValueError:
                print("Warning: ValueError was caused by accuracy_score calculation")
                self.add_train_losses.append(-999)
            try:
                val_score = accuracy_score(true_test, pred_test)
            except:
                print("Warning: ValueError was caused by accuracy_score calculation")
                val_score = -999
            self.add_val_losses.append(val_score)
        elif self.score_func == 'r2':
            try:
                val_score = r2_score(self.y_test, self.model.predict(self.X_test), multioutput='uniform_average')
            except:
                print("Warning: ValueError was caused by r2_score calculation")
                val_score = -999
            self.add_val_losses.append(val_score)
            try:
                self.add_train_losses.append(r2_score(self.y_train, self.model.predict(self.X_train)))    
            except:
                print("Warning: ValueError was caused by r2_score calculation")
                self.add_train_losses.append(-999)
        elif self.score_func == 'mttc':
            try:
                val_score = matthews_corrcoef(self.y_test, self.model.predict(self.X_test))
            except:
                print("Warning: ValueError was caused by r2_score calculation")
                val_score = -999
            self.add_val_losses.append(val_score)
            try:
                self.add_train_losses.append(matthews_corrcoef(self.y_train, self.model.predict(self.X_train)))    
            except:
                print("Warning: ValueError was caused by r2_score calculation")
                self.add_train_losses.append(-999)
        self.best_score = max(self.best_score, val_score)