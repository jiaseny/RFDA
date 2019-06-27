from __future__ import division
from sklearn.model_selection import train_test_split
from linear_discriminant import *
from utils import *


def dof(S, lmbd):
    """
    Degrees of freedom for RFDA.
    """
    return np.sum(S**2 / (S**2 + lmbd))


methods = ["unif", "levr", "rdge", "count", "srht"]


def expr_err(A, b, lmbd, num_cols_list, num_iters=50, seed=123):
    """
    Compute relative error of solution/objective.
    """
    rand.seed(seed)

    model = LinearDiscriminant(A, b, lmbd=lmbd)
    G_opt = model.direct()
    obj_opt = model.obj_val(G_opt)

    num_cols_list = map(int, num_cols_list)

    # Relative error of solution
    rel_errs = np.zeros((len(methods), len(num_cols_list), num_iters))

    # # Objective sub-optimality
    # obj_errs = dict{method: np.zeros((len(num_cols_list), num_iters))
    #                 for method in methods}

    # Sampling probabilities
    probs_unif = np.ones(d) / d
    probs_levr = model.leverage_scores()
    probs_rdge = model.ridge_leverage_scores()

    for k, num_cols in enumerate(num_cols_list):
        print "\nk = %d; number of sampled columns = %d" % (k, num_cols)

        G_unif, G_unif_hist = model.iterative(
            num_cols, num_iters, method="sampling", probs=probs_unif)
        G_levr, G_levr_hist = model.iterative(
            num_cols, num_iters, method="sampling", probs=probs_levr)
        G_rdge, G_rdge_hist = model.iterative(
            num_cols, num_iters, method="sampling", probs=probs_rdge)
        G_count, G_count_hist = model.iterative(
            num_cols, num_iters, method="count-sketch")
        G_srht, G_srht_hist = model.iterative(
            num_cols, num_iters, method="SRHT")

        print "\nComputing relative error of solution ..."
        rel_errs[0, k, :] = model.rel_err(G_unif_hist, G_opt)
        rel_errs[1, k, :] = model.rel_err(G_levr_hist, G_opt)
        rel_errs[2, k, :] = model.rel_err(G_rdge_hist, G_opt)
        rel_errs[3, k, :] = model.rel_err(G_count_hist, G_opt)
        rel_errs[4, k, :] = model.rel_err(G_srht_hist, G_opt)

        # print "Computing objective sub-optimality ..."
        # obj_errs_unif[k] = [1. - model.obj_val(G)/obj_opt for G in G_unif_hist]
        # obj_errs_levr[k] = [1. - model.obj_val(G)/obj_opt for G in G_levr_hist]
        # obj_errs_rdge[k] = [1. - model.obj_val(G)/obj_opt for G in G_rdge_hist]

    return num_cols_list, rel_errs


def expr_acc(A, b, lmbd, num_iters_list, num_cols, num_reps=20):
    """
    Compute classification accuracy over multiple random train/test splits
       for varying number of iterations.
    """
    rand.seed(123)

    num_iters_list = map(int, num_iters_list)
    max_num_iters = np.max(num_iters_list)

    # Classification accuracy
    accs_opt = np.zeros(num_reps)
    accs = np.zeros((len(methods), num_reps, len(num_iters_list)))

    for rep in xrange(num_reps):
        print "\nrep = %d ..." % rep
        A_train, A_test, b_train, b_test = train_test_split(
            A, b, train_size=.6, stratify=b)

        model = LinearDiscriminant(A_train, b_train, lmbd=lmbd)
        G_opt = model.direct()

        # Sampling probabilities
        probs_unif = np.ones(d) / d
        probs_levr = model.leverage_scores()
        probs_rdge = model.ridge_leverage_scores()

        # Iterative algorithm
        _, G_unif_hist = model.iterative(
            num_cols, max_num_iters, method="sampling", probs=probs_unif)
        _, G_levr_hist = model.iterative(
            num_cols, max_num_iters, method="sampling", probs=probs_levr)
        _, G_rdge_hist = model.iterative(
            num_cols, max_num_iters, method="sampling", probs=probs_rdge)
        _, G_count_hist = model.iterative(
            num_cols, max_num_iters, method="count-sketch")
        _, G_srht_hist = model.iterative(
            num_cols, max_num_iters, method="SRHT")

        # Classification accuracy using direct solution
        W, _ = model.solve_eig(G_opt)
        accs_opt[rep] = model.classify_accuracy(W, A_test, b_test)

        # Classification accuracies for sketched solutions
        for k, num_iters in enumerate(num_iters_list):
            print "\nComputing classification accuracy ...\n"

            W_unif, _ = model.solve_eig(G_unif_hist[num_iters-1])
            W_levr, _ = model.solve_eig(G_levr_hist[num_iters-1])
            W_rdge, _ = model.solve_eig(G_rdge_hist[num_iters-1])
            W_count, _ = model.solve_eig(G_count_hist[num_iters-1])
            W_srht, _ = model.solve_eig(G_srht_hist[num_iters-1])

            accs[0, rep, k] = model.classify_accuracy(W_unif, A_test, b_test)
            accs[1, rep, k] = model.classify_accuracy(W_levr, A_test, b_test)
            accs[2, rep, k] = model.classify_accuracy(W_rdge, A_test, b_test)
            accs[3, rep, k] = model.classify_accuracy(W_count, A_test, b_test)
            accs[4, rep, k] = model.classify_accuracy(W_srht, A_test, b_test)

    return num_iters_list, num_cols, accs_opt, accs
