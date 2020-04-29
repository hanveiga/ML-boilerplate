import os
import pickle
import csv
import numpy as np

from config import abs_dataset_folderpath, abs_output_data_folderpath

class Data(object):
    """ Data entry object"""
    def __init__(self,index,comoving_dist,density,wavelength,flux):
        self.index = index
        self.comoving_dist = comoving_dist
        self.density = density
        self.wavelength = wavelength
        self.flux = flux

class Dataset(object):
    """ Collection of Data objects"""
    def __init__(self,records):
        self.records = records

def build_dataset(directory_path,output_path):
    keys = get_file_name(directory_path)    
    records = []
    for f_name in keys:
        los_fHI = load_fHI(directory_path + f_name + '.fHI')
        wavelength, flux = process_fHI(los_fHI)
        comoving_dist, density = load_density(directory_path + f_name + '.raw')
        records.append(Data(f_name,comoving_dist,density,wavelength,flux))

    dataset = Dataset(records)

    pickle.dump(dataset,open(os.path.join(output_path,"dataset.pkl"),"wb"))

def get_file_name(directory_path):
    """ Retrieves filenames in data directory """
    filenames = os.listdir(directory_path)
    names = []
    for f in filenames:
        if f[0:-4] not in names and f not in ['.DS_Store']:
            names.append(f[0:-4])
        else:
            pass

    return names

def load_fHI(filename):
    with open(filename) as csvDataFile:
        reader = csv.reader(csvDataFile, delimiter=',')
        data = list(reader)
        los_fHI_data = np.array(data[2:]).astype(float)
    return los_fHI_data

def process_fHI(fHI_vectors):
    wavelength = 1216.*(1.+fHI_vectors[:,0]/3.e5)
    flux = fHI_vectors[:,1]
    return wavelength, flux

def load_density(filename):
    los_raw_data = np.loadtxt(filename)
    return los_raw_data[:,0],los_raw_data[:,3]

def process_density(raw_vectors):
    return raw_vectors

if __name__=='__main__':
    build_dataset(abs_dataset_folderpath, abs_output_data_folderpath)