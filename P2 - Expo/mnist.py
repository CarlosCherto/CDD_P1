import struct

import numpy as np

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)












# import os, struct
# from array import array
# from cvxopt.base import matrix

# def read(digits, dataset = "training", path = "."):
#     """
#     Python function for importing the MNIST data set.
#     """

#     if dataset is "training":
#         fname_img = os.path.join(path, 'train-images-idx3-ubyte')
#         fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
#     elif dataset is "testing":
#         fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
#         fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
#     else:
#         raise ValueError, "dataset must be 'testing' or 'training'"

#     flbl = open(fname_lbl, 'rb')
#     magic_nr, size = struct.unpack(">II", flbl.read(8))
#     lbl = array("b", flbl.read())
#     flbl.close()

#     fimg = open(fname_img, 'rb')
#     magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
#     img = array("B", fimg.read())
#     fimg.close()

#     ind = [ k for k in xrange(size) if lbl[k] in digits ]
#     images =  matrix(0, (len(ind), rows*cols))
#     labels = matrix(0, (len(ind), 1))
#     for i in xrange(len(ind)):
#         images[i, :] = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
#         labels[i] = lbl[ind[i]]

#     return images, labels
