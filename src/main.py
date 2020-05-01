from data import Dataset, Data
from preprocessing import load_dataset, preprocess_data, MLDataset
from models import BaselineModel, Optimizer
from matplotlib import pyplot as plt
import os
import numpy as np
import mlflow.keras
import argparse

import config
from config import abs_dataset_folderpath, abs_output_data_folderpath


def run(options):
    # load dataset
    dataset = load_dataset(os.path.join(abs_output_data_folderpath,"processed_dataset.pkl"))

    mldata = MLDataset(dataset,10)
    #train_inputs, train_outputs, val_inputs, val_outputs = preprocess_data(dataset)

    mlflow.keras.autolog()
    # setup model
    training = True
    if options.action == 'training':
        model = BaselineModel()
        model._functional_setup()
        train_inputs,train_outputs,val_inputs,val_outputs = mldata.get_kth_fold(0)
        model.train(train_inputs,train_outputs,val_inputs,val_outputs)
        mlflow.keras.save_model(model.model,"models")

    if options.action == 'optimize':
        model = BaselineModel()
        optimizer = Optimizer(BaselineModel, mldata)
        optimizer.hyper_parameter_opt(10)
        best_model = optimizer.best_model

    if options.action == 'predict':
        model = BaselineModel()
        model.load_model("models/")

        preds = model.predict(val_inputs[0:20])
        for i in range(20):
            plt.figure()
            plt.plot(preds[i],label="pred")
            plt.plot(val_outputs[i],label="real")
            plt.savefig(f"test{i}.png")


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("action",
            type=str, help="training, optimize, predict")

    args = parser.parse_args()
    run(args)