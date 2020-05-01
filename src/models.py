from hyperopt import hp
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
import mlflow


class BaselineModel(object):
    def __init__(self):
        self.n_dim_x = 1600
        self.hyperspace = {
            'num_filters': hp.choice('num_filters', [50, 100, 200])
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
            self.model = Sequential([
            Conv1D(self.hyperspace["num_filters"], filter_size, input_shape=(self.n_dim_x,2)),
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

    def train(self,train_inputs,train_outputs,val_inputs,val_outputs):
        self.model.fit( train_inputs, train_outputs, self.epochs,  validation_data=(val_inputs, val_outputs))

    def get_loss(self):
        return 'mse'

    def get_optimizer(self):
        return 'adam'

    def get_metrics(self):
        return ['mse']

class Optimizer(object):
    """ Base class for regressor """
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.best_model = 0

    def hyper_parameter_opt(self, hpo_evals):
        trials = Trials()

        self.hpo_iter = 0

        self.best_space = fmin(fn=self._objective, space=self.hyperspace,
                               algo=tpe.suggest, max_evals=hpo_evals,
                               trials=trials)

        best_trial = self._get_best_trial(trials)

        r2   = -best_trial['result']['loss']
        mbe  = best_trial['result']['mbe']
        mae  = best_trial['result']['mae']
        rmse = best_trial['result']['rmse']

        return best_model

    @staticmethod
    def _performance_metrics(y_test, y_pred):

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return({'rmse':rmse, 'vss':len(y_test)})

    def _objective(self, space):

        self.hpo_iter += 1

        for k in range(self.data.k_folds):

            x_train, y_train, x_test, y_test = self.data.get_kth_fold(k=k)

            model = self._fit_regressor(space, x_train, y_train)

            y_pred = model.predict(x_test)


    @staticmethod
    def _get_best_trial(trials):

        ok_list = [t for t in trials if STATUS_OK == t['result']['status']]

        losses = [float(t['result']['loss']) for t in ok_list]

        # Loss is the negative r2-score, so take minimum to get best trial.
        index_of_min_loss = np.argmin(losses)

        return ok_list[index_of_min_loss]