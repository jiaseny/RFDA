from __future__ import division
from expr_utils import *

PATH = ""  # Data path


if __name__ == "__main__":
    # Command-line args:
    #   fname: 'ORL' or 'PEMS'
    #   rep: trial number
    fname, rep = sys.argv[1:]
    rep = int(rep)

    # ORL data (400 * 10304)
    # PEMS-SF data (440 * 138672)
    A = np.load(PATH + "%s-data.npy" % fname)
    b = np.load(PATH + "%s-labels.npy" % fname)
    n, d = A.shape

    num_iters = 50
    num_cols_list = np.linspace(1000, 5000, 6) if fname == 'ORL' else \
        np.linspace(5000, 10000, 6)

    # Relative errors w.r.t. degrees of freedom
    lmbd_list = [1., 2., 5., 10., 20., 50.]
    for lmbd in lmbd_list:
        print "lmbd = %s" % lmbd
        res = expr_err(A, b, lmbd, num_cols_list, num_iters, seed=1230+rep)
        # pckl_write(res, "%s-lmbd%d-rep%d.res" % (fname, lmbd, rep))

    # Accuracy
    num_iters_list = range(1, 11)
    res = expr_acc(A, b, lmbd, num_iters_list,
                   num_cols=5000 if fname == 'ORL' else 10000,
                   num_reps=20)
    # pckl_write(res, "%s-lmbd%d-acc.res" % (fname, lmbd))
