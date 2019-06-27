from __future__ import division
import re
from utils import *
from linear_discriminant import *

PATH = ""  # Data path


# --------------------------------------------------------------------------- #
# ORL data (400 * 10304)

def read_pgm(filename, byteorder='>'):
    """
    https://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm
    Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html
    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)

    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                         count=int(width)*int(height),
                         offset=len(header)).reshape((int(height), int(width)))

# Test example
# image = read_pgm(PATH + "ORL/s1/1.pgm")
# plt.imshow(image, plt.cm.gray)
# plt.show()

dat = list()
labels = list()
for i in xrange(1, 41):  # For each subject
    for j in xrange(1, 11):  # For each image
        image = read_pgm(PATH + "ORL/s%d/%d.pgm" % (i, j))
        dat.append(image.ravel())  # Flatten each image to a vector
        labels.append(i)

X = np.asarray(dat, dtype=float) / 255.
y = np.asarray(labels, dtype=int)

np.save(PATH + "ORL-data.npy", X)
np.save(PATH + "ORL-labels.npy", y)


# --------------------------------------------------------------------------- #
# PEMS data (440 * 138672)

train_dat = list()
with open(PATH + "PEMS/PEMS_train", 'r') as f:
    for line in f:
        row = re.split('\s|;', line.lstrip('[').rstrip(']\n'))
        train_dat.append(row)

test_dat = list()
with open(PATH + "PEMS/PEMS_test", 'r') as f:
    for line in f:
        row = re.split('\s|;', line.lstrip('[').rstrip(']\n'))
        test_dat.append(row)

X_train = np.asarray(train_dat).astype(float)
X_test = np.asarray(test_dat).astype(float)
X = np.vstack([X_train, X_test])
# standardize(X)

# Label vector
train_labels = list()
with open(PATH + "PEMS/PEMS_trainlabels", 'r') as f:
    for line in f:
        row = line.lstrip('[').rstrip(']\n').split(' ')
        train_labels.append(row)

test_labels = list()
with open(PATH + "PEMS/PEMS_testlabels", 'r') as f:
    for line in f:
        row = line.lstrip('[').rstrip(']\n').split(' ')
        test_labels.append(row)

y_train = np.asarray(train_labels).astype(int).ravel()
y_test = np.asarray(test_labels).astype(int).ravel()
y = np.r_[y_train, y_test]

np.save(PATH + "PEMS-data.npy", X)
np.save(PATH + "PEMS-labels.npy", y)
