import os
import optunity as opt
import sys
from .learnmanager import LearnManager, NNLearnManager, SKLearnManager
from sklearn.grid_search import ParameterGrid
import copy
import numpy as np

class Optimizer():
    def __init__(self, **main_params):
        self.main_params = main_params
        
    def _create_train_manager(self,data, val_data, search_space):
        cls = self.main_params['class']
        if (cls == 'sklearn'):
            return SKLearnManager(self.main_params, data, val_data, search_space)
        elif (cls == 'nn'):
            return NNLearnManager(self.main_params, data, val_data, search_space)
        else:
            raise InvalidParamError('class')
     
    
class GridSearchOptimizer(Optimizer):
    """GridSearchOptimizer handle all grids combinations of hyperparameters
    """
    def _preprocess_params(self, param):
        to_expose = ['units']
        for ex in to_expose:
            if (ex in param):
                add_params = {}
                [add_params.update({(ex + str(i+1)) : x}) for x,i in zip(param[ex], np.arange(len(param[ex])))]
                param.update(add_params)
                param.pop(ex, None)
        return param
        
    def optimize(self, data, search_space, val_data = None):
        """
        Parameters:
        -----------
        data: [X, Y] - arrays
            This data will be used for crossval training (considering 'train_test_split' parameter)

        search_space: dict
            Dict with parameters to optimize. E.g. 'units' : [[1000,1000,500], [2500,1000]]

        val_data: [X, Y] - arrays
            Default - None. If specified than optimizer metric will be evaluated on val_data. Also if specified than 'train_test_split' parameter will be ignored.

        """
        train_manager = self._create_train_manager(data, val_data, search_space)
        param_grid = ParameterGrid(search_space)
        all_params = []
        for p in param_grid:
            all_params.append(p)
        for key in search_space.keys():
            if (isinstance(search_space[key], dict)):
                new_params=[]
                for param in all_params:
                    if (search_space[key][param[key]] is None):
                        new_params.append(param)
                    else:
                        param_grid = ParameterGrid(search_space[key][param[key]])
                        add_params = [p for p in param_grid]
                        for aparam in add_params:
                            tparam = copy.copy(param)
                            tparam.update(aparam)
                            new_params.append(tparam)
                all_params = new_params
        for param in all_params:
            param_to_pass = self._preprocess_params(param)
            train_manager.train(**param)
    
class OptunityOptimizer(Optimizer):
    """OptunityOptimizer uses optunity lib to optimize hyperparameters
    """
    
    def optimize(self, data, search_space, val_data = None, num_evals = 50, optimize='max', solver_name='particle swarm'):
        """
        Parameters:
        -----------
        data: [X, Y] - arrays
            This data will be used for crossval training (considering 'train_test_split' parameter)

        search_space: dict
            Dict with parameters to optimize. E.g. 'units' : [100,1000]

        val_data: [X, Y] - arrays
            Default - None. If specified than optimizer metric will be evaluated on val_data. Also if specified than 'train_test_split' parameter will be ignored.

        num_evals: int
            Count of iterations for optunity optimizer

        optimize: str
            'max'/'min' supported

        solver_name: str
            Default 'particle swarm'. Only default parameter supported now.     
        """
        train_manager = self._create_train_manager(data, val_data, search_space)
        sys.setrecursionlimit(100000)
        if (optimize == 'max'):
            self.retr, self.extra, self.info = opt.maximize_structured(f=train_manager.train, num_evals=num_evals,search_space=search_space)
        elif (optimize == 'min'):
            self.retr, self.extra, self.info = opt.minimize_structured(f = train_manager.train, search_space = search_space,
                                    num_evals = num_evals)
        else:
            raise(InvalidParamError('optimize', optimize))
        #load and ret best
        best_model = train_manager.get_best_model(self.extra)
        return best_model