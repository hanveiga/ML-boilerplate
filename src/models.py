from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
import mlflow


class BaselineModel(object):
    def __init__(self):
        self.n_dim_x = 1600
        self.hyperspace = {
            'num_filters': hp.choice('num_filters', [10, 20, 30])
            }
        self.epochs = 10

    def _functional_setup(self,hyperparameters=[]):

        if hyperparameters == []:
            num_filters = 100
            filter_size = 4
            pool_size = 4
            # Set up the default model
            self.model = Sequential([
            Conv1D(num_filters, filter_size, input_shape=(self.n_dim_x,2)),
            MaxPooling1D(pool_size=pool_size),
            Flatten(),
            Dense(8000, activation='linear'),
            ])
        else:
            filter_size = 4
            pool_size = 4
            self.model = Sequential([
            Conv1D(hyperparameters['num_filters'], filter_size, input_shape=(self.n_dim_x,2)),
            MaxPooling1D(pool_size=pool_size),
            Flatten(),
            Dense(8000, activation='linear'),
            ])

        self.model.compile(
            self.get_optimizer(),
            loss=self.get_loss(),
            metrics=self.get_metrics(),
        )

    def predict(self,inputs):
        return self.model.predict(inputs)

    def train(self,train_inputs,train_outputs,val_inputs,val_outputs,epochs=10):
        self.model.fit( train_inputs, train_outputs, self.epochs,  validation_data=(val_inputs, val_outputs))

    def get_loss(self):
        return 'mse'

    def get_optimizer(self):
        return 'adam'

    def get_metrics(self):
        return ['mse']

class Optimizer(object):
    """ Base class for regressor """
    def __init__(self, model, data,epochs=1):
        self.model_constructor = model
        self.data = data
        self.best_model = 0
        self.model = self.model_constructor()
        self.epochs = epochs

    def hyper_parameter_opt(self):
        trials = Trials()
        hpo_evals = 5

        self.best_space = fmin(fn=self._objective, space=self.model.hyperspace,
                               algo=tpe.suggest, max_evals=hpo_evals,
                               trials=trials)

        best_trial = self._get_best_trial(trials)

        return best_model

    @staticmethod
    def _performance_metrics(y_test, y_pred):
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return({'rmse':rmse, 'vss':len(y_test)})

    def _objective(self, space):

        lowest_l2_error = np.nan
        for k in range(self.data.nfolds):
            x_train, y_train, x_test, y_test = self.data.get_kth_fold(k=k)
            print(space["num_filters"])
            self.model._functional_setup(hyperparameters=space)
            self.model.train(x_train,y_train,x_test, y_test,self.epochs)
            y_pred = self.model.predict(x_test)
            l2_error = np.mean((y_pred-y_test)**2,axis=0)
            # compute performance
            print(l2_error.shape)
            total_l2_error = np.sum(l2_error)

            if (total_l2_error < lowest_l2_error):
                print(total_l2_error)
                self.best_model = self.model 

    def _fit_regressor(self,space,x_train,y_train,x_test, y_test,epochs=1):
        self.model._functional_setup(space)
