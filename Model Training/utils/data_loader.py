import h5py
import scipy.io 
import numpy as np
from skimage import io

from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path
import torch
import math


def load_file(filename):
    ext = str(filename).split('.')[-1]
    if ext == 'mat':
        try:
            # Try to open with h5py for MATLAB version 7.3+
            
            mat_file = h5py.File(filename, 'r')
            keys = list(mat_file.keys())
            if not keys:
                raise ValueError("No datasets found in the .mat file.")
            dataset_name = keys[0]
            file = mat_file[dataset_name][:].T
            return file

        except OSError:
            
            # If it's not an HDF5 file, fall back to scipy for older versions
            # print(f"{filename} is not an HDF5 file, trying scipy.io.loadmat()")
            mat_file = scipy.io.loadmat(filename)
            keys = list(mat_file.keys())
            # Skip MATLAB-specific meta fields like __header__, __version__, __globals__
            data_keys = [key for key in keys if not key.startswith('__')]
            if not data_keys:
                raise ValueError("No datasets found in the .mat file.")
            dataset_name = data_keys[0]
            
            file = mat_file[dataset_name][:].T
            return file
        
    elif ext == 'npz':  
        data = np.load(filename)
        keys = list(data.keys())
        if len(keys) == 1:
            return data[keys[0]]
        else:
            raise ValueError(f"Unexpected number of datasets in .npz file {filename}. Found keys: {keys}")
    
    elif ext == 'tif':
        data = io.imread(filename)
        print(filename)
        #print(data.shape)
        return data
            


class data_loader():
    def __init__(self, dir_img:str, dir_label : str, dir_pair: str, label_suffix: str = '', pair_suffix: str = ''):
        self.dir_img = Path(dir_img)
        self.dir_label = Path(dir_label)
        self.dir_pair = Path(dir_pair)
        self.label_suffix = label_suffix
        self.pair_suffix = pair_suffix
        self.ids = [splitext(file)[0] for file in listdir(dir_img) if isfile(join(dir_img, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {dir_exp}.')

    def __len__(self):
        return len(self.ids)
    def __getitem__(self, ids):

        name = self.ids[ids]
        img_file = list(self.dir_img.glob(name + '.*'))
        

        label_file = list(self.dir_label.glob(name + self.label_suffix +  '.*'))
        #print(label_file)
        pair_file = list(self.dir_pair.glob(name + self.pair_suffix + '.*'))
        #print(pair_file)

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(label_file) == 1, f'Either no label_file or multiple D_map found for the ID {name}: {label_file}'        
        assert len(pair_file) == 1, f'Either no pair file or multiple pairs found for the ID {name}: {pair_file}'

        img = load_file(img_file[0])
        label = load_file(label_file[0])       
        pair = load_file(pair_file[0])
        
        processed_img = self.preprocess(img, type = 'img')
        processed_label = self.preprocess(label, type = 'label')
        processed_pair = self.preprocess(pair, type = 'pair')
        return {
            'image': torch.as_tensor(processed_img.copy()).float().contiguous(),
            'label': torch.as_tensor(processed_label.copy()).float().contiguous(),
            'pair' : torch.as_tensor(processed_pair.copy()).float().contiguous()
            }

        

    def preprocess(self, img,type):

        # Normalize the image 
        if type == 'img':
            if len(img.shape) == 2:

                img = img[np.newaxis,:,:]
            else:
                img = img.transpose((2, 0, 1))
                img = img
            return img
        elif type == 'label':
            if len(img.shape) == 2:
                
                img = img[np.newaxis,:,:]
            else:
                img = img.transpose((2, 1, 0))
                img = img
            return img
            # img = img.transpose((2,1, 0))
            # return img
        elif type == 'pair':
            # This is not an image but I'm returning this directly. 
            return img
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        return img






        

def expdata_preprocess(img_file, crop_save, patch_size, overlap, channel):
    # Check the size of the image 
    img = load_file(img_file) 

    if len(np.shape(img)) == 2:
        w,h = np.shape(img)
        f = 1
        img = img[np.newaxis, :,:]
    elif len(np.shape(img)) == 3:
        img = img.transpose(1,2,0)
        w,h,f = np.shape(img)

    else: 
        raise RuntimeError(f'Check the input file dimensions: {img_file}.')


    
    # Normalize based on each frame 
    for frame in range(f):
        img[:,:,frame] = (img[:,:,frame] - img[:,:,frame].min())/(img[:,:,frame].max() - img[:,:,frame].min())

    # Crop it into smaller pieces, for model input.
    # First check if the img is too small.  

    if (w < patch_size) | (h < patch_size): 
        raise ValueError(f'The image file is too small. At least {patch_size} pixels of width and height: \n {img_file}')
        

    # Start to crop 
    step_size = math.floor(patch_size * (1 - overlap))    # The steps with the overlay included.
    # initiate the x y cropping index
    x_ind = []
    y_ind = []

    # crop in x: 
    for start in range(0, w - patch_size + 1, step_size):
        x_ind.append(start)

    # Take care of the edge: 
    if (x_ind[-1] + patch_size ) < w:
        x_ind.append(w - patch_size)


    # crop in y: 
    for start in range(0, h - patch_size + 1, step_size):
        y_ind.append(start)

    if (y_ind[-1] + patch_size ) < h:
        y_ind.append(h - patch_size)

    # devide in frames, same as xy but no overlay:
    f_ind = []

    for start in range(0, f - channel + 1, channel ):
        f_ind.append(start)
    if (f_ind[-1] + channel ) < f: 
        f_ind.append(f - channel)

    # Construct the cropped images as input: 
    index = 0
    l = len(x_ind) * len(y_ind) * len(f_ind)
    # img_array = np.zeros([l,channel, patch_size, patch_size])




    for xi, i in enumerate(x_ind):
        for yj, j in enumerate(y_ind):
            for fk, k in enumerate(f_ind): 
                 # Crop the frame correctly
                frame = img[i:i + patch_size, j:j + patch_size, k:k + channel]
                # The normalization is taken on the entire image not on individual crops 
                #frame = (frame - np.min(frame))/(np.max(frame) -  np.min(frame))
                frame = np.transpose(frame , (2,0,1)) # Reorder from [batch_size, height, width, channels] to [batch_size, channels, height, width]
                #img_array[index] = frame
                np.savez(crop_save + 'frame_' + str(index) + '.npz', img = frame)
                index += 1
    print('Imgae is cropped into small pieces.\n')
    print('Patch Size = ' , patch_size ,'. ' , 'Overlap = ', overlap, '. ', 'Channel = ', channel ,'.\n')
    print('W = ', w, '. ', 'h = ', h, '. ', 'f = ', f, '. \n' )
    return 


class data_loader_exp():
    def __init__(self, dir_img:str):
        self.dir_img = Path(dir_img)
        self.ids = [splitext(file)[0] for file in listdir(dir_img) if isfile(join(dir_img, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {dir_exp}.')

    def __len__(self):
        return len(self.ids)
    def __getitem__(self, ids):

        name = self.ids[ids]
        img_file = list(self.dir_img.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'

        img = load_file(img_file[0])
        
        processed_img = img
        # img.transpose((2, 0, 1))

        # For experiment data, the normalization is taken care of in the cropping step
        #img = (img - np.min(img)) / (np.max(img) - np.min(img))

        index = int(name[6:])

        return {
            'image': torch.as_tensor(processed_img.copy()).float().contiguous(),
            'index': torch.as_tensor(index)
            }
