import numpy as np
import pickle
import os
from sklearn.model_selection import KFold
from data import Dataset, Data

from config import abs_dataset_folderpath, abs_output_data_folderpath


class MLDataset(object):
    """ Collection of Data objects that are appropriate for the ML model"""
    def __init__(self,data,nfolds):
        self.inputs, self.outputs = self.preprocess_data(data)
        self.nfolds = nfolds
        self.nData = len(self.inputs)
        self.kf = KFold(nfolds)
        self.kf.get_n_splits(range(self.nData))
        self.folds = list(self.kf.split(range(self.nData)))
    
    def preprocess_data(self, dataset):
        inputs = []
        outputs = []
        for entry in dataset.records:
            inp = np.zeros((1600,2))
            inp[:,0] = entry.wavelength
            inp[:,1] = entry.flux
            inputs.append(inp)
            outputs.append(entry.density)

        return np.array(inputs), np.array(outputs)

    def get_kth_fold(self,k=0):
        train_indices, test_indices = self.folds[k]
        train_input = self.inputs[train_indices]
        train_output = self.outputs[train_indices]
        test_input = self.inputs[test_indices]
        test_output = self.outputs[test_indices]

        return train_input, train_output, test_input, test_output


def load_dataset(dataset_path):
    dataset = pickle.load(open(dataset_path,"rb"))
    return dataset

def prepare_data_for_model(dataset):
    pass

def split_datasets_train_val_test(dataset):
    pass

def feature_engineering(dataset):
    # absolute max
    records = []
    modified_dataset = Dataset(records)

    max_wavelength = 1600
    max_comoving_dist = 8000

    # pad everything with 0 outside of range
    for data_entry in dataset.records:
        raw_wavelength = data_entry.wavelength
        new_wavelength = np.zeros(max_wavelength)
        new_flux = np.zeros(max_wavelength)
        new_flux[0:len(raw_wavelength)] = data_entry.flux[:]
        new_wavelength[0:len(raw_wavelength)] = raw_wavelength[:]
        data_entry.wavelength = new_wavelength
        data_entry.flux = new_flux

        raw_comov_dist = data_entry.comoving_dist
        new_comov_dist = np.zeros(max_comoving_dist)
        new_density = np.zeros(max_comoving_dist)
        new_comov_dist[0:len(raw_comov_dist)] = raw_comov_dist[:]
        new_density[0:len(data_entry.density)] = np.log10(data_entry.density[:])
        data_entry.comoving_dist = new_comov_dist
        data_entry.density = new_density

        modified_dataset.records.append(data_entry)

    # transform data
    for _ in range(len(modified_dataset.records)):
        data_entry = modified_dataset.records.pop(0)
        data_entry.flux = 1-data_entry.flux
        #data_entry.density = np.log10(data_entry.density)
        modified_dataset.records.append(data_entry)

    pickle.dump(modified_dataset,open(os.path.join(abs_output_data_folderpath,"processed_dataset.pkl"),"wb"))

def preprocess_data(dataset):
    n_data = len(dataset.records)
    n_train = int(0.7*n_data)
    n_val = int(0.2*n_data)
    n_test = 0.1*n_data

    inputs = []
    outputs = []

    for entry in dataset.records:
        inp = np.zeros((1600,2))
        inp[:,0] = entry.wavelength
        inp[:,1] = entry.flux
        inputs.append(inp)
        outputs.append(entry.density)

    train_inputs = inputs[0:n_train]
    train_outputs = outputs[0:n_train]
    val_inputs = inputs[n_train:n_train+n_val]
    val_outputs = outputs[n_train:n_train+n_val]

    return train_inputs, train_outputs, val_inputs, val_outputs

if __name__=='__main__':
    dataset = load_dataset(os.path.join(abs_output_data_folderpath,"dataset.pkl"))
    feature_engineering(dataset)