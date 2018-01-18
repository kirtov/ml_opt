from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU, ThresholdedReLU
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation,Dropout
from keras.layers import Input
from keras.optimizers import Adagrad, Adadelta, Adam, Adamax, SGD, RMSprop, Nadam
from keras.layers.normalization import BatchNormalization
from .custom_optimizers.optimizers import Eve
from keras.regularizers import l2, l1, l1_l2
from .callbacks import *
import numpy as np
from keras import backend as K
import theano
from .utils.error import ParamNotFoundError,InvalidParamError
from .utils.metrics import f1, mttc
from .utils.losses import focal_loss
from .custom_layers import WeightInput
from .custom_init.new_keras import selu_normal
from .custom_activations.new_keras import selu
from .custom_layers.dropout import AlphaDropout

class InsModelWrapper():
    def __init__(self, manager=None, **hyper_params):
        self.manager = manager
        self.hyper_params = hyper_params

    def _check_in_params(self, param_name):
        check = param_name in self.hyper_params
        if (self.manager == None):
            return check
        else:
            return check or (param_name in self.manager.main_params)

    def _get_param(self, param_name):
        if (param_name in self.hyper_params):
            return self.hyper_params[param_name]
        else:
            if (self.manager is not None):
                return self.manager._get_param(param_name)
            else:
                raise ParamNotFoundError(param_name)

    def predict(self, X):
        return self.model.predict(X)

class DNNWrapper(InsModelWrapper):
    def _get_init(self, layer):
        def_name = 'init'
        p_name = def_name + layer
        init_f = None
        if (p_name in self.hyper_params):
            init_f = self._get_param(p_name)
        elif (def_name in self.hyper_params):
            init_f = self._get_param(def_name)
        else:
            init_f = 'glorot_uniform'
            
        if (init_f == 'selu_normal'):
            init_f = selu_normal()
        return init_f
        
    def _get_dense(self, layer, output_dim=None):
        def_name = 'units'
        p_name = def_name + layer
        kwargs = {}
        kwargs['init'] = self._get_init(layer)
        if (layer == '1'):
            kwargs['input_dim'] = self._get_param('input_dim')
        #output layer
        if (output_dim != None):
            kwargs['output_dim'] = output_dim
        elif (self._check_in_params(p_name)):
            kwargs['output_dim'] = int(self._get_param(p_name))
        else:
            kwargs['output_dim'] = int(self._get_param(def_name))
        #Regularizers
        if (output_dim == None):
            if ((('kernel_regularizer' + str(layer)) in self.hyper_params) or ('kernel_regularizer' in self.hyper_params)):
                kwargs['kernel_regularizer'] = self._get_regularizer(layer, 'kernel_regularizer')
            if ((('bias_regularizer' + str(layer)) in self.hyper_params) or ('bias_regularizer' in self.hyper_params)):
                kwargs['bias_regularizer'] = self._get_regularizer(layer, 'bias_regularizer')
            if ((('activity_regularizer' + str(layer)) in self.hyper_params) or ('activity_regularizer' in self.hyper_params)):
                kwargs['activity_regularizer'] = self._get_regularizer(layer, 'activity_regularizer')
        return Dense(**kwargs)

    def _get_act_by_name(self, act):
        #TODO: parametric activation functios
        str_act = ['relu', 'tanh', 'sigmoid', 'linear','softmax','softplus','softsign','hard_sigmoid']
        if (act == 'selu'):
            return Activation(selu)
        if (act in str_act):
            return Activation(act)
        else:
            return {'prelu': PReLU(), 'elu' : ELU(), 'lrelu' : LeakyReLU(),
                   'trelu':ThresholdedReLU()}[act]

    def _get_reg_by_name(self, reg, c):
        #TODO: parametric regularizers
        if (reg is 'None'):
            return None
        if (reg == 'l1'):
            return l1(c)
        elif (reg == 'l2'):
            return l2(c)
        elif (reg == 'l1_l2'):
            return l1_l2(c)

    def _get_activation(self, layer):
        if (layer == 'output'):
            p_name = 'output_activation'
            return self._get_act_by_name(self._get_param(p_name))
        def_name = 'activation'
        p_name = def_name + layer
        if (self._check_in_params(p_name)):
            return self._get_act_by_name(self._get_param(p_name))
        else:
            return self._get_act_by_name(self._get_param(def_name))

    def _get_dropout(self, layer):
        def_name = 'dropout'
        p_name = def_name + layer
        drop_prob = None
        if (self._check_in_params(p_name)):
            drop_prob = self._get_param(p_name)
        else:
            drop_prob = self._get_param(def_name)
        if (isinstance(drop_prob, str) and 'a' in drop_prob):
            drop_prob = float(drop_prob[1:])
            return AlphaDropout(drop_prob)
        else:
            return Dropout(float(drop_prob))
        
    def _get_regularizer(self, layer, def_name):
        p_name = def_name + str(layer)
        if (self._check_in_params(p_name)):
            norm = self._get_param(p_name)
        else:
            norm = self._get_param(def_name)
        if ('c' in self.hyper_params):
            return self._get_reg_by_name(norm, self._get_param('c'))
        else:
            return None

    def _create_callbacks(self, X_train, X_test, y_train, y_test):
        using = self._get_param('standart_ckbs')
        callbacks = []
        self._lh_ckb = None
        self._mc_ckb = None
        if ('LH' in using):
            self._lh_ckb =LossHistory(X_train, y_train, X_test, y_test, self._get_param('metric'))
            callbacks.append(self._lh_ckb)
        if ('MC' in using):
            if (self._check_in_params("mc_monitor")):
                mc_monitor = self._get_param("mc_monitor")
            else:
                mc_monitor = self._get_param("callbacks_monitor")
            
            if (self._check_in_params("mc_mode")):
                mc_mode = self._get_param('mc_mode')
            else:
                mc_mode = 'auto'
            self._mc_ckb = FullModelCheckpoint(monitor=mc_monitor,
                                         filepath="",
                                         verbose=0, save_best_only=True, mode=mc_mode)
            callbacks.append(self._mc_ckb)
        if ('ES' in using):
            if (self._check_in_params("mc_monitor")):
                es_monitor = self._get_param("es_monitor")
            else:
                es_monitor = self._get_param("callbacks_monitor")
                
            if (self._check_in_params("es_mode")):
                es_mode = self._get_param('es_mode')
            else:
                es_mode = 'auto'
            
            callbacks.append(ckbs.EarlyStopping(monitor=es_monitor, patience=self._get_param('patience'), verbose=0, mode=es_mode))
        callbacks += [copy.copy(c) for c in self._get_param('add_callbacks')]
        return callbacks

    def _on_train_end(self):
        if ('LH' in self._get_param('standart_ckbs')):
            self.loss_history = self._lh_ckb.get_history()
        else:
            self.loss_history = {}
        if ('MC' in self._get_param('standart_ckbs')):
            self.model.set_weights(self._mc_ckb.best_weights)
        #garbage
        self.hyper_params['manager'] = None
        del self.manager, self._mc_ckb, self._lh_ckb

    def _get_optimizer_by_name(self, name, params):
        if (name == 'adam'):
            return Adam(**params)
        elif (name == 'adagrad'):
            return Adagrad(**params)
        elif (name == 'adadelta'):
            return Adadelta(**params)
        elif (name == 'sgd'):
            return SGD(**params)
        elif (name == 'rmsprop'):
            return RMSprop(**params)
        elif (name == 'adamax'):
            return Adamax(**params)
        elif (name == 'nadam'):
            return Nadam(**params)
        elif (name == 'eve'):
            return Eve(**params)

    def _get_optimizer(self):
        optimizer_name = self._get_param('optimizer')
        if (self.manager is not None):
            search_space = self.manager.search_space
        else:
            search_space = self.hyper_params
        optimizer_params = {}
        if (isinstance(search_space['optimizer'], dict) and search_space['optimizer'][optimizer_name] is not None):
            for opn in search_space['optimizer'][optimizer_name]:
                optimizer_params[opn] = self.hyper_params[opn]

        return self._get_optimizer_by_name(optimizer_name, optimizer_params)
    
    def _get_metrics(self):
        metrics = self._get_param('metrics')
        all_metrics = {'binary_accuracy' : 'binary_accuracy','categorical_accuracy' : 'categorical_accuracy', 'mae' : 'mae','mse' : 'mse', 'f1' : f1, 'mttc' : mttc, 'matthews_correlation' : 'matthews_correlation'}
        return list(map(lambda x: all_metrics[x], metrics))
    
    def _get_batch_norm(self):
        return BatchNormalization()
        

    def _calc_count_of_layers(self):
        return len([x for x in self.hyper_params if 'units' in x])
    
    def _get_loss(self, loss_name):
        if (loss_name == 'focal_loss'):
            return focal_loss
        else:
            return loss_name

    def _compile(self):
        dnn = Sequential()
        if (self._check_in_params('count_of_layers')):
            count_of_layers = int(self._get_param('count_of_layers'))
        else:
            count_of_layers = self._calc_count_of_layers()
        pos = 1
        if (self._check_in_params('deep_feature_selection') and self._get_param('deep_feature_selection') == True):
            dnn.add(WeightInput.WeightInput(self._get_param('input_dim'), self._get_param('input_dim')))

        for l in [str(i) for i in range(1,count_of_layers + 1)]:
            if (self._check_in_params('units') or self._check_in_params('units'+l)):
                dnn.add(self._get_dense(l))
            if (self._check_in_params('activation') or self._check_in_params('activation'+l)):
                dnn.add(self._get_activation(l))
            if (self._check_in_params('batch_norm') and int(l) in self._get_param('batch_norm')):
                dnn.add(self._get_batch_norm())
            if (self._check_in_params('dropout') or self._check_in_params('dropout'+l)):
                dnn.add(self._get_dropout(l))
            if (l == str(count_of_layers)):
                dnn.add(self._get_dense(l, self._get_param('output_dim')))
                if (self._check_in_params('output_activation')):
                    dnn.add(self._get_activation('output'))
        dnn.compile(loss=self._get_loss(self._get_param('loss')), optimizer=self._get_optimizer(),
                    metrics=self._get_metrics())
        self.model = dnn
        self.model = dnn
        return self.model

    def fit(self, X_train, X_test, y_train, y_test):
        self.model = self._compile()
        self.model.fit(X_train, y_train, validation_data=[X_test,y_test],epochs=self._get_param('nb_epoch'),
                callbacks=self._create_callbacks(X_train, X_test, y_train, y_test), class_weight = self._get_param('loss_weights'), batch_size=self._get_param('batch_size'), verbose=self._get_param('verbose'))
        self._on_train_end()
        return self.model

class SKLearnWrapper(InsModelWrapper):
    def _get_model_by_name(self, name):
        if (name == 'RandomForestRegressor'):
            self.hyper_params['n_jobs'] = -1
            return RandomForestRegressor(**self.hyper_params)
        elif (name == 'RandomForestClassifier'):
            self.hyper_params['n_jobs'] = -1
            return RandomForestClassifier(**self.hyper_params)
        elif (name == 'GradientBoostingRegressor'):
            return GradientBoostingRegressor(**self.hyper_params)
        elif (name == 'GradientBoostingClassifier'):
            return GradientBoostingClassifier(**self.hyper_params)
        elif (name == 'ElasticNet'):
            return ElasticNet(**self.hyper_params)
        elif (name == 'LinearRegression'):
            return LinearRegression(**self.hyper_params)
        elif (name == 'LogisticRegression'):
            return LogisticRegression(**self.hyper_params)
        elif (name == 'KNeighborsRegressor'):
            return KNeighborsRegressor(**self.hyper_params)
        elif (name == 'KNeighborsClassifier'):
            return KNeighborsClassifier(**self.hyper_params)
        else:
            raise InvalidParamError('model_name', name)

    def _preprocess_params(self):
        if (self._check_in_params('n_estimators')):
            self.hyper_params['n_estimators'] = int(self._get_param('n_estimators'))

    def fit(self, X_train, y_train):
        self._preprocess_params()
        self.model = self._get_model_by_name(self._get_param('model_name'))
        self.model.fit(X_train, y_train)
        return self.model

class AEWrapper(DNNWrapper):
    def _get_model(self, input_shape):
        self.inputs = Input(shape=(input_shape,))

        self.encoder = self._get_dense('1')(self.inputs)
        self.act = self._get_activation('1')
        self.encoder = self.act(self.encoder)

        if (self._check_in_params('dropout') or self._check_in_params('dropout1')):
            drop = self._get_dropout('1')
            self.encoder = drop(self.encoder)

        self.decoder = Dense(input_shape)(self.encoder)
        model = Model(input=self.inputs, output=self.decoder)
        return model

    def _compile(self):
        input_shape = self._get_param('input_dim')
        self.model = self._get_model(input_shape)
        self.model.compile(loss=self._get_param('loss'), optimizer=self._get_optimizer(), metrics=self._get_param('metrics'))
        self.encoder_layer = Model(input=self.inputs, output=self.encoder)
        return self.model

    def fit(self, X_train, X_test, y_train, y_test):
        self.model = self._compile()
        self.model.fit(X_train, y_train, validation_data=[X_test,y_test],nb_epoch=self._get_param('nb_epoch'),
                callbacks=self._create_callbacks(X_train, X_test, y_train, y_test),
                     batch_size=self._get_param('batch_size'), verbose=self._get_param('verbose'))
        self.encoder_layer = Model(input=self.inputs, output=self.encoder)
        self._on_train_end()
        return self.model

    def transform(self, x):
        x = np.float32(x)
        return self.encoder_layer.predict(x)


class SAEWrapper(DNNWrapper):
    def _get_hidden_layers(self):
        if (self._check_in_params('count_of_layers')):
            count_of_layers = int(self._get_param('count_of_layers'))
        else:
            count_of_layers = self._calc_count_of_layers()
        hidden_layers = []
        def_name = 'units'
        for l in range(1,count_of_layers + 1):
            if (self._check_in_params(def_name + str(l))):
                hidden_layers.append(self._get_param(def_name + str(l)))
            elif (self._check_in_params(def_name)):
                hidden_layers.append(self._get_param(def_name))
            else:
                raise ParamNotFoundError(def_name)
        return hidden_layers

    def _get_model(self, input_shape):
        inputs = Input(shape=(input_shape,))
        state = inputs
        if (self._check_in_params('hidden_layers')):
            hidden_layers = self._get_param('hidden_layers')
        else:
            hidden_layers = self._get_hidden_layers()
        self.encoders = []
        self.acts = []
        self.drops = []
        self.encoder_layers = []
        count = 0
        for n_out in hidden_layers:
            count += 1
            encoder = Dense(n_out)
            self.encoders.append(encoder)
            act = self._get_activation(str(count))#self._get_act_by_name(self.activation)
            self.acts.append(act)
            encoded = act(encoder(state))
            if (self._check_in_params('dropout') or self._check_in_params('dropout' + str(count))):
                drop = self._get_dropout(str(count))
                self.drops.append(drop)
                encoded = drop(encoded)
            self.encoder_layers.append(Model(input=inputs, output=encoded))
            state = encoded

        llist = hidden_layers[:-1]
        for n_out in reversed(llist):
            count += 1
            act = self._get_activation(str(count))#self._get_act_by_name(self.activation)
            decoder = act(Dense(n_out)(state))
            if (self._check_in_params('dropout') or self._check_in_params('dropout' + str(count))):
                drop = self._get_dropout(str(count))
                decoder = drop(decoder)
            state = decoder
        decoder = Dense(input_shape)(state)
        ae = Model(input=inputs, output=decoder)
        return ae

    def _compile(self):
        input_dim = self._get_param('input_dim')
        self.model = self._get_model(input_dim)
        self.model.compile(loss=self._get_param('loss'), optimizer=self._get_param('optimizer'), metrics=self._get_param('metrics'))
        return self.model

    def fit(self, X_train, X_test, y_train, y_test):
        self.model = self._compile()
        self.model.fit(X_train, y_train, validation_data=[X_test,y_test],nb_epoch=self._get_param('nb_epoch'),
                callbacks=self._create_callbacks(X_train, X_test, y_train, y_test),
                     batch_size=self._get_param('batch_size'), verbose=self._get_param('verbose'))

        self._on_train_end()
        return self.model

    def transform(self, x, num):
        x = np.float32(x)
        return self.encoder_layers[num - 1].predict(x)


class DAEWrapper(SAEWrapper):
    def _get_corrupted_input(self, X, corruption_level):
        return self._get_param('theano_rng').binomial(size=(len(X),len(X[0])), n=1, p=1 - corruption_level,
                                        dtype=theano.config.floatX) * X

    def fit(self, X_train, X_test, y_train, y_test):
        X_train = self._get_corrupted_input(X_train, self._get_param('corruption_level'))
        X_train = X_train.eval()
        X_test = self._get_corrupted_input(X_test, self._get_param('corruption_level'))
        X_test = X_test.eval()
        return SAEWrapper.fit(self,X_train,X_test,y_train,y_test)
