from sklearn.cross_validation import KFold
import os
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error, f1_score
from sklearn.metrics import matthews_corrcoef as mttc
from sklearn.preprocessing import MinMaxScaler
from .inswrapper import DNNWrapper, AEWrapper, SAEWrapper, DAEWrapper, SKLearnWrapper
import pickle
import copy
from .utils.error import ParamNotFoundError,InvalidParamError
from .utils.default_params import *
import json
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from keras.utils.np_utils import to_categorical
np.random.seed(5531)

class LearnManager():
    def __init__(self, main_params, data, val_data, search_space):
        self._default_params = get_default_params()
        self.search_space = search_space
        self.main_params = main_params
        self.nb_iter = 1
        self._create_log_directory(self._get_param('log_path'))
        self._parameters = []
        self._split_data(data, val_data)
        params_to_save = copy.copy(main_params)
        self._log_history = self._create_log_history(main_params)
        
    def get_best_model(self, extra):
        suff = '_1'
        path = self._get_param('log_path') + str(extra.call_log['values'].index(extra.optimum)+1)+suff+'_model.inspkl'
        with open(path, 'rb') as inp:
            model = pickle.load(inp)
        return model
    
    def _create_log_history(self, main_params):
        lh = {'default_params' : self._default_params, 'main_params' : copy.copy(main_params)}
        prev_log_history_path = self._get_param("prev_log_history_path")
        if (prev_log_history_path != None):
            prev_log_history_path = os.path.abspath(prev_log_history_path)
            self.main_params['prev_log_history_path'] = prev_log_history_path
            if (os.path.isfile(prev_log_history_path)):
                lh = json.load(open(prev_log_history_path))
        return lh
    
    def _convert_to_array(self,X):
        if (isinstance(X, pd.DataFrame) or (isinstance(X, pd.Series))):
            X = X.get_values()
        return X
    
    def _split_big(self, arr, thrs):
        size = len(np.concatenate(arr))
        narr = []
        for a in arr:
            if (len(a)/size > thrs):
                split = max(2, len(a)/(size*thrs))
                splitted = np.array_split(a, split)
                narr.append(splitted[0])
                narr.append(splitted[1])
            else:
                narr.append(a)
        return np.array(narr)
        
    def _cross_val_by(self, data, by, delete_by_column):
        if (isinstance(data[0], pd.DataFrame)):
            datacolumns = data[0].columns.get_values()
            if (isinstance(by, str)):
                by = np.where(datacolumns==by)[0][0]
            data[0] = self._convert_to_array(data[0])
            data[1] = self._convert_to_array(data[1])
        by_column = np.array([d[by] for d in data[0]])
        indexes = [np.where(by_column == s)[0] for s in np.unique(by_column)]
        
        if (delete_by_column):
            data[0] = np.delete(data[0], by, axis=1)
            
        if (self._get_param('normalize')):
            data[0] = self._normalize(data[0])
            
        tt_split = self._get_param('train_test_split')
        test_size = 1./tt_split
        indexes = self._split_big(indexes, test_size)

        self._splits = []
        kfold = KFold(len(indexes), int(tt_split), self._get_param('cross_val_shuffle'), self._get_param('random_state'))
        self._cv = True
        self._data = data
        for train, test in kfold:
            train_ind = np.concatenate(indexes[train])
            test_ind = np.concatenate(indexes[test])
            self._splits.append([train_ind, test_ind])
    
    def _normalize(self, data, val_data = None):
        mms = MinMaxScaler()
        if (val_data is not None):
            mms.fit(np.concatenate([data, val_data]))
            return mms.transform(data), mms.transform(val_data)
        else:
            return mms.fit_transform(data)
                    
    def _split_data(self, data, val_data):
        self._cv = False
        val_data_exist = val_data is not None
        #CROSSVALIDATION BY COLUMN
        if (self._get_param('cross_val_by') is not None):
            self._cross_val_by(data, self._get_param('cross_val_by'), self._get_param('delete_by_col'))
            return          
        data[0] = self._convert_to_array(data[0])
        data[1] = self._convert_to_array(data[1])
        if (self._get_param('normalize')):
            if (val_data_exist):
                data[0], val_data[0] = self._normalize(data[0], val_data[0])
            else:
                data[0] = self._normalize(data[0])
        tt_split = self._get_param('train_test_split')
        if (val_data_exist):
            val_data[0] = self._convert_to_array(val_data[0])
            val_data[1] = self._convert_to_array(val_data[1])
            self._splits = [[data[0], val_data[0], data[1], val_data[1]]]
        elif (tt_split < 1):
            self._splits = [train_test_split(data[0], data[1], test_size=tt_split, 
                                           random_state=self._get_param('random_state'))]
        else:
            #CROSS-VALIDATION
            kfold = KFold(len(data[0]),n_folds = int(tt_split), shuffle=self._get_param('cross_val_shuffle'), 
                         random_state = self._get_param('random_state'))
            self._splits = [] 
            self._cv = True
            self._data = data
            for train_fold, test_fold in kfold:
                self._splits.append([train_fold, test_fold])
                
    def _eval_metric(self, y_true, y_pred):
        metric = self._get_param('metric')
        if (self._get_param('to_categorical')):
            y_true = np.argmax(y_true,axis=1)
            y_pred = np.argmax(y_pred,axis=1)
        if (metric == 'r2'):
            from sklearn.metrics import r2_score
            return r2_score(y_true, y_pred, multioutput='uniform_average')
        if (metric == 'f1'):
            try:
                if (len(y_true[0].shape) > 0):
                    return f1_score(y_true, y_pred.round(), average = 'weighted')
                else:
                    return f1_score(y_true, y_pred.round())
            except:
                return f1_score(y_true, y_pred.round())
        if (metric == 'mttc'):
            return mttc(y_true, y_pred.round())
        if (metric == 'mae'):
            from sklearn.metrics import mean_absolute_error
            return mean_absolute_error(y_true, y_pred)
        if (metric == 'mse'):
            from sklearn.metrics import mean_squared_error
            return mean_squared_error(y_true, y_pred)
        if (metric == 'accuracy'):
            from sklearn.metrics import accuracy_score
            return accuracy_score(y_true, y_pred.round())
        if (metric == 'categorical_accuracy'):
            from sklearn.metrics import accuracy_score
            return accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
        
    def train(self, **hyper_params):
        self.crossval_iter = 1
        scores = self._get_results_from_log_history(self.nb_iter)
        if (scores == None):
            scores = []
            if (self._cv):
                for train, test in self._splits:
                    X_train = self._data[0][train]
                    X_test = self._data[0][test]
                    y_train = self._data[1][train]
                    y_test = self._data[1][test]
                    X_train, y_train = self._sampling(X_train, y_train)
                    if (self._get_param('to_categorical')):
                        y_train = to_categorical(y_train)
                        y_test = to_categorical(y_test)
                    scores.append(self._train_one(X_train, X_test, y_train, y_test, **hyper_params))
                    self.crossval_iter += 1
            else:
                for X_train, X_test, y_train, y_test in self._splits:
                    X_train, y_train = self._sampling(X_train, y_train)
                    if (self._get_param('to_categorical')):
                        y_train = to_categorical(y_train)
                        y_test = to_categorical(y_test)
                    scores.append(self._train_one(X_train, X_test, y_train, y_test, **hyper_params))
                    self.crossval_iter += 1
        self.nb_iter += 1
        return min(scores)
    
    def _sampling(self, X, Y):
        if (self._get_param('balance_dataset') == 'oversampling'):
            try:
                X, Y = SMOTE(random_state = self._get_param('random_state'), ratio = self._get_param("oversampling_ratio"), n_jobs=4, k_neighbors=self._get_param('smote_neighbors')).fit_sample(X, Y)
            except:
                print("Oversampling fail")
                return X, Y
        return X, Y
            
    
    def _get_results_from_log_history(self, nb_iter):
        if (str(nb_iter) in self._log_history):
            scores = []
            for cv in self._log_history[str(nb_iter)].keys():
                scores.append(self._log_history[str(nb_iter)][cv]['val_score'])
            if (len(scores) == self._get_param('train_test_split')):
                return scores
            else:
                return None
        else:
            return None
                
    def _on_iteration_end(self, scores, hyper_params, model):
        hyper_params['val_score'] = scores
        self._parameters.append(hyper_params)
        self._save_model(model)
        model_nb = self.nb_iter
        if (self.crossval_iter == 1):
            self._log_history[str(model_nb)] = {}
        self._log_history[str(model_nb)][str(self.crossval_iter)] = hyper_params
        with open(self._get_param('log_path') + '/' + 'log_history.json', 'w') as fp:
            json.dump(self._log_history, fp)
        if (len(self._splits) > 1):
            model_nb = str(self.nb_iter) + '_' + str(self.crossval_iter)
        print('Training was completed. Model {0}, score = {1}, params = {2}'.format(model_nb, scores, hyper_params))
        
    def _on_iteration_start(self, hyper_params):
        if (self._get_param('verbose') > 0):
            model_nb = self.nb_iter
            print('Training was started. Model {0}, params = {1}'.format(model_nb, hyper_params))
        
    def _save_model(self, model):
        if (self._get_param('dump')):
            model_nb = str(self.nb_iter) + '_' + str(self.crossval_iter)
            path = self._get_param('log_path') + '/' + str(model_nb) + "_model.inspkl"
            with open(path, 'wb') as output:
                pickle.dump(copy.deepcopy(model), output, protocol=pickle.HIGHEST_PROTOCOL)      
        
    def _create_log_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.main_params['log_path'] = os.path.abspath(self.main_params['log_path']) + '/'
    
    def _get_param(self, param_name):
        if (param_name in self.main_params):
            return self.main_params[param_name]
        elif (param_name in self._default_params):
            return self._default_params[param_name]
        else:
            raise ParamNotFoundError(param_name)
        
class SKLearnManager(LearnManager):
    def _train_one(self, X_train, X_test, y_train, y_test, **hyper_params):
        wrapper = SKLearnWrapper(self, **hyper_params)
        self._on_iteration_start(hyper_params)
        model = wrapper.fit(X_train, y_train)
        score = self._eval_metric(y_test, model.predict(X_test))
        self._on_iteration_end(score, hyper_params, wrapper)
        return score
    
class NNLearnManager(LearnManager):
    def _save_model(self, model):
        if (self._get_param('dump')):
            model_nb = self.nb_iter
            model_nb = str(self.nb_iter) + '_' + str(self.crossval_iter)
            path = self._get_param('log_path') + '/' + str(model_nb) + "_model.inspkl"
            model_arch = model.model.to_json()
            weights = model.model.get_weights()
            pickle.dump([model_arch, weights, model.loss_history], open(path, 'wb'))
        
    def _train_one(self, X_train, X_test, y_train, y_test, **hyper_params):
        model_name = self._get_param('model_name')
        if (model_name == 'dnn'):
            model = DNNWrapper(self, **hyper_params)
        elif (model_name == 'ae'):
            model = AEWrapper(self, **hyper_params)
        elif (model_name == 'sae'):
            model = SAEWrapper(self, **hyper_params)
        elif (model_name == 'dae'):
            model = DAEWrapper(self, **hyper_params)
        else:
            raise InvalidParamError('model_name', self._get_param['model_name'])
        self._on_iteration_start(hyper_params)
        nn = model.fit(X_train, X_test, y_train, y_test)
        score = self._eval_metric(y_test, nn.predict(X_test))
        self._on_iteration_end(score, hyper_params, model)
        return score
