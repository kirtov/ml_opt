import json
from .. inswrapper import DNNWrapper, AEWrapper, SAEWrapper, DAEWrapper, SKLearnWrapper
import copy
import pickle
import numpy as np

class Loader():
    def __init__(self, log_history_path):
        with (open(log_history_path)) as f:
            self._log_history = json.load(f)
        self.params = copy.copy(self._log_history['default_params'])
        self.params.update(self._log_history['main_params'])
        self.best_model = None
   
    def get_best_models(self, by='mean', get='max'):
        """
            Returns best model (models for crossval)
            
            Parameters:
                by: str
                Best fold will be fold with maximum of func. applied to scores of fold
                'mean','min','max', 'median' supported now
                
                get: str
                Get model with minimum ('min') or maximum ('max') score
        """
        if (self.best_model != None):
            return self.best_model
        if (get == 'min'):
            best_score = 999
        elif (get == 'max'):
            best_score = -999
        best_params = {}
        best_model_name = ""
        crossval = 1
        for i in range(1, len(self._log_history)-1):
            scores = []
            for m in self._log_history[str(i)]:
                scores.append(self._log_history[str(i)][m]['val_score'])
            score = min(scores)
            if (by == 'mean'):
                score = np.mean(scores)
            elif (by == 'median'):
                score = np.median(scores)
            elif (by == 'max'):
                score = max(scores)
            best = False
            if (get == 'max'):
                best = best_score < score
            elif (get == 'min'):
                best = best_score > score
            if (best):
                best_score = score
                best_params = self._log_history[str(i)][str(scores.index(max(scores))+1)]
                best_model = str(i)
                crossval = len(self._log_history[str(i)])
        self.params.update(best_params)
        self.best_params = best_params
        if (self.params['dump'] == False):
            return self.params
        else:
            wrappers = []
            for i in range(1, crossval + 1):
                best_model_name =best_model  + '_' + str(i) + "_model.inspkl"
                path = self.params['log_path'] + '/' + best_model_name
                path = path.replace('//','/')
                if (self.params['class'] == 'sklearn'):
                    wrapper = pickle.load(open(path, 'rb'))
                elif (self.params['class'] == 'nn'):
                    if (self.params['model_name'] == 'dnn'):
                        wrapper = DNNWrapper(**self.params)
                    if (self.params['model_name'] == 'ae'):
                        wrapper = AEWrapper(**self.params)
                    if (self.params['model_name'] == 'sae'):
                        wrapper = SAEWrapper(**self.params)
                    if (self.params['model_name'] == 'dae'):
                        wrapper = DAEWrapper(**self.params)
                    [arch, weights, lh] = pickle.load(open(path, 'rb'))
                    wrapper._compile()
                    wrapper.model.set_weights(weights)
                    wrapper.loss_history = lh
                wrappers.append(copy.copy(wrapper))
            self.best_model = wrappers[0]
            return wrappers
        
    def get_model_by_id(self, num):
        scores=[]
        for m in self._log_history[str(num)]:
            scores.append(self._log_history[str(num)][m]['val_score'])
        params = self._log_history[str(num)][str(scores.index(max(scores))+1)]
        model_name = str(num) + '_' + str(scores.index(max(scores))+1) + "_model.inspkl"
        self.params.update(params)
        path = self.params['log_path'] + '/' + model_name
        path = path.replace('//','/')
        if (self.params['dump'] == False):
            return self.params
        else:
            if (self.params['class'] == 'sklearn'):
                wrapper = pickle.load(open(path, 'rb'))
            elif (self.params['class'] == 'nn'):
                if (self.params['model_name'] == 'dnn'):
                    wrapper = DNNWrapper(**self.params)
                if (self.params['model_name'] == 'ae'):
                    wrapper = AEWrapper(**self.params)
                if (self.params['model_name'] == 'sae'):
                    wrapper = SAEWrapper(**self.params)
                if (self.params['model_name'] == 'dae'):
                    wrapper = DAEWrapper(**self.params)
                [arch, weights, lh] = pickle.load(open(path, 'rb'))
                wrapper._compile()
                wrapper.model.set_weights(weights)
                wrapper.loss_history = lh
            return wrapper
            
    def get_scores(self):
        scs = []
        for i in range(1, len(self._log_history)-1):
            scores = []
            for m in self._log_history[str(i)]:
                scores.append(self._log_history[str(i)][m]['val_score'])
            scs.append(min(scores))
        return scs
    
    def get_count_of_models(self):
        return len(self._log_history)-1
    
    def get_crossval_scores(self):
        crossval_scores = []
        for i in range(1, len(self._log_history) - 1):
            crossval = len(self._log_history[str(i)])
            crossval_scores.append(np.array([self._log_history[str(i)][str(j)]['val_score'] for j in range(1, crossval+1)]))
        return np.array(crossval_scores)
    
    def get_best_score(self, by = 'mean'):
        """
            Returns index and crossval scores of best model

            Parameters:
                by: str
                Best fold will be fold with maximum of func. applied to scores of fold
                'mean', 'min' or 'max'
        """
        crossval_scores = []
        scores = []
        for i in range(1, len(self._log_history) - 1):
            crossval = len(self._log_history[str(i)])
            cur_scores = np.array([self._log_history[str(i)][str(j)]['val_score'] for j in range(1, crossval+1)])
            score = np.mean(cur_scores)
            if (by == 'min'):
                score = cur_scores.min()
            elif (by == 'max'):
                score = cur_scores.max()
            scores.append(score)
            crossval_scores.append(cur_scores)
        index_of_max = scores.index(max(scores))
        return index_of_max+1, np.array(crossval_scores[index_of_max])