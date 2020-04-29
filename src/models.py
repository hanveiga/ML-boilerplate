from hyperopt import hp
import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
import mlflow


class BaselineModel(object):
    def __init__(self):
        self.n_dim_x = 1600
        self._model_setup()
        self.hyperspace = {
            'num_filters': hp.choice('num_filters', [50, 100, 200])
            }

    def _model_setup(self):
        num_filters = 100
        filter_size = 4
        pool_size = 4

        self.model = Sequential([
        Conv1D(num_filters, filter_size, input_shape=(self.n_dim_x,2)),
        MaxPooling1D(pool_size=pool_size),
        Flatten(),
        Dense(8000, activation='linear'),
        ])

        self.model.compile(
            'adam',
            loss='mse',
            metrics=['mse'],
        )
    

    def predict(self,inputs):
        return self.model.predict(inputs)

    def train(self,train_inputs,train_outputs,val_inputs,val_outputs):
        self.model.fit( [train_inputs], [train_outputs], epochs=10,  validation_data=([val_inputs], [val_outputs]))

    def load_model(self,path):
        self.model = mlflow.keras.load_model(path)

class Optimizer(object):
    def __init__(self):
        pass

    def optimize(self, hpo_evals):
        hpo_t0 = time()
        trials = Trials()

        self.best_space = fmin(fn=self._objective, space = self.hyperspace, \
                                algo = tpe.suggest, max_evals = hpo_evals, trials=trials)
        
        self.hpo_time = time() - hpo_t0
        self.best_trial = self._get_best_trial(trials)
        self._hpo_results()