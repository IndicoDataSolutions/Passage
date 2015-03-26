import os
import numpy as np

def load_gender_data(ntrain=10000, ntest=10000):
    import pandas as pd
    file_loc = os.path.dirname(os.path.realpath(__file__))
    relative_path = "blogger_data_2.csv" # move dataset to examples directory
    fullpath = os.path.join(file_loc, relative_path)
    data = pd.read_csv(fullpath, nrows=ntrain+ntest)
    X = data['text'].values
    X = [str(x) for x in X] # ugly nan cleaner
    Y = data['gender'].values
    trX = X[:-ntest]
    teX = X[-ntest:]
    trY = Y[:-ntest]
    teY = Y[-ntest:]
    return trX, teX, trY, teY

def load_mnist(data_dir=None):
    if data_dir is None:
        import urllib
        import gzip
        url = 'http://yann.lecun.com/exdb/mnist/'
        fnames = [
            'train-images-idx3-ubyte.gz', 
            'train-labels-idx1-ubyte.gz', 
            't10k-images-idx3-ubyte.gz', 
            't10k-labels-idx1-ubyte.gz'
        ]
        for fname in fnames:
            if not os.path.isfile(fname):
                print 'data_dir not given and file not local - downloading mnist file:', fname
                urllib.urlretrieve(url+fname, fname)
        data_dir = ''
    fd = gzip.open(os.path.join(data_dir,'train-images-idx3-ubyte.gz'))
    loaded = np.fromstring(fd.read(), dtype=np.uint8)
    trX = loaded[16:].reshape((60000, -1))

    fd = gzip.open(os.path.join(data_dir,'train-labels-idx1-ubyte.gz'))
    loaded = np.fromstring(fd.read(), dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = gzip.open(os.path.join(data_dir,'t10k-images-idx3-ubyte.gz'))
    loaded = np.fromstring(fd.read(), dtype=np.uint8)
    teX = loaded[16:].reshape((10000, -1))

    fd = gzip.open(os.path.join(data_dir,'t10k-labels-idx1-ubyte.gz'))
    loaded = np.fromstring(fd.read(), dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trX = trX/255.
    teX = teX/255.

    trX = trX.reshape(-1, 28, 28)
    teX = teX.reshape(-1, 28, 28)
    
    return trX, teX, trY, teY