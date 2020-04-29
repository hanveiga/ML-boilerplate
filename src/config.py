import os 

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_folderpath = 'data/'
abs_dataset_folderpath = os.path.join(parent_path,dataset_folderpath)
abs_output_data_folderpath = os.path.join(parent_path,'processed_data')