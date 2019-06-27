from __future__ import division
# Fast Hadamard transform: https://github.com/FALCONN-LIB/FFHT
from ffht import fht
from utils import *


class LinearDiscriminant(object):
    """
    Iterative algorithm for linear discriminant analysis.
    """
    def __init__(self, A, b, lmbd):
        """
        Args:
            A: array(n, d), design matrix.
            b: array(n, ), label vetor.
            lmbd: float, regularization parameter.
        """
        self.n, self.d = A.shape

        assert_shape(b, (self.n, ))
        self.process_input(A, b)

        assert lmbd >= 0
        self.lmbd = lmbd

        return

    def centering_matrix(self, n):
        """
        Construct centering matrix
            H = np.identity(n) - 1./n * np.ones((n, n))
        """
        H = np.identity(n) - 1./n * np.ones((n, n))  # Idempotent

        return H

    def process_input(self, A, b):
        """
        Args:
            A: array(n, d), design matrix.
            b: array(n, ), label vetor.

        Returns:
            X: array()
        """
        n = self.n

        label_cnt = Counter(b)
        self.c = len(set(b))  # Number of possible labels

        # Construct label indicator matrix X
        self.label_dict = dict(zip(label_cnt.keys(), range(self.c)))
        self.b = [self.label_dict[l] for l in b]  # Relabel

        # X[i, j] = 1 if i-th row of A is in class j
        X = coo_matrix((np.ones(n), (np.arange(n), self.b)),
                       shape=(n, self.c))
        pi_vec = 1. / np.sqrt(np.asarray(label_cnt.values()))  # Pi^{-1/2}
        self.P = X.dot(diags(pi_vec)).tocsc()  # X Pi^{-1/2}

        H = self.centering_matrix(n)
        self.B = H.dot(A)

        self.U, self.S, Vh = svd(self.B, full_matrices=False)  # Thin SVD
        self.V = Vh.T

        print "Processed inputs with %d obs, %d features, and %d classes." % \
            (self.n, self.d, self.c)

        # Dimensions
        assert_shape(self.P, (self.n, self.c))
        assert_shape(self.B, (self.n, self.d))

        return

    def direct(self):
        """
        Compute G = B^T (B B^T + lmbd I_n)^{-1} X Pi^{-1/2} directly.
        """
        R = self.B.dot(self.B.T) + self.lmbd*np.identity(self.n)
        R_inv = inv(R)

        assert isinstance(self.P, csc_matrix)
        G = self.B.T.dot(R_inv) * self.P  # np.array * sp.sparse

        assert_shape(G, (self.d, self.c))

        return G

    def solve_eig(self, G, eval_thres=1e-15):
        """
        Solve the generalized eigenvalue problem
            (G Pi^{-1/2} X^T B) W = W T
        where W are the eigenvectors and T are the eigenvalues.

        Args:
            G: computed using direct() or iterative().
        """
        assert_shape(G, (self.d, self.c))

        # Directly solve the original eigen-problem
        # assert isinstance(self.P, csc_matrix)
        # evals, W = eig(G.dot(self.P.T.dot(self.B)))

        # Much more efficient
        M = self.P.T.dot(self.B.dot(G))  # (c, c)
        evals, V = eig(M)
        W = G.dot(V)

        # Only retain real eigenvalues and eigenvectors
        real_inds = np.isreal(evals)*np.prod(np.isreal(W), axis=0, dtype=bool)
        evals = np.real(evals[real_inds])
        W = np.real(W[:, real_inds])

        # Discard eigenvalues below eval_thres
        thres_inds = (evals > eval_thres)
        evals = evals[thres_inds]
        W = W[:, thres_inds]

        # Re-order by decreasing eigenvalue
        inds = np.argsort(evals)[::-1]
        evals = evals[inds]
        W = W[:, inds]

        # assert norm(G.dot(W) - W.dot(np.diag(evals))) < 1e-10
        print "Found %d eigenvectors." % W.shape[1]
        # assert_le(W.shape[1], self.c-1)

        return W, evals

    def classify_accuracy(self, W, A_test, b_test, n_neighbors=5):
        """
        Perform classification on A_test.

        Args:
            n_neighbors: in k-NN classifier.
        """
        assert_eq(W.shape[0], self.d)
        assert_le(W.shape[1], W.shape[0]-1)  # q <= c-1
        assert_eq(A_test.shape[1], self.d)

        # Dimension reduction
        X = self.B.dot(W)
        H_test = self.centering_matrix(A_test.shape[0])
        X_test = H_test.dot(A_test.dot(W))

        # Nearest-neighbors classifier
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X, self.b)
        b_pred = clf.predict(X_test)  # Predicted labels

        b_test_new = [self.label_dict[l] for l in b_test]  # Relabel
        acc = accuracy(b_pred, b_test_new)  # Accuracy

        return acc

    def leverage_scores(self):
        """
        Computes the statistical leverage scores of the input matrix A.
        """
        scores = norm(self.V, axis=1)**2

        probs = scores / np.sum(scores)
        assert probs.shape == (self.d, ), probs

        return probs

    def ridge_leverage_scores(self, lmbd=None):
        """
        Computes the ridge leverage scores of the input matrix A.
        """
        if lmbd is None:
            lmbd = self.lmbd

        Sreg = self.S / np.sqrt(self.S**2 + lmbd)  # \Sigma_reg

        scores = norm(self.V.dot(np.diag(Sreg)), axis=1)**2

        probs = scores / np.sum(scores)
        assert probs.shape == (self.d, ), probs

        return probs

    def sampling_matrix(self, probs, num_cols):
        """
        Construct sampling-and-rescaling matrix S.

        Args:
            num_cols: int, number of columns to sample.
            probs: array(d), column-sampling probabilities.

        Returns:
            S: sp.sparse matrix (d, s).
        """
        d = self.d
        check_prob_vector(probs)

        inds = rand.choice(a=d, p=probs, size=num_cols, replace=True)
        vals = 1. / np.sqrt(num_cols * probs[inds])
        S = csc_matrix((vals, (inds, range(num_cols))), shape=(d, num_cols))
        # vals = 1. / (num_cols * probs[inds])
        # S = csr_matrix((vals, (inds, inds)), shape=(d, d))

        return S

    def count_sketch(self, num_cols):
        """
        Count-sketch matrix.
        """
        d = self.d

        # Place a random sign in each row
        inds = rand.choice(a=num_cols, size=d, replace=True)
        signs = 2*rand.binomial(n=1, p=.5, size=d) - 1   # Random signs
        S = csc_matrix((signs, (range(d), inds)), shape=(d, num_cols))

        return S

    def subsampled_hadamard(self, num_cols):
        """
        Subsampled randomized hadamard transform.

        Returns:
            d: int, augmented dimension (power of two)
            D: array(d, d), rescaled diagonal matrix of random signs
            P: array(d, num_cols), uniform sampling matrix.
        """
        d = 2**int(ceil(np.log2(self.d)))  # Dimension needs to be power of 2

        # Diagonal sign matrix
        signs = 2*rand.binomial(n=1, p=.5, size=d) - 1  # Random signs
        D = diags(signs)

        # Uniform sampling matrix
        # Each column has a single 1; each row has at most one 1
        inds = rand.choice(a=d, size=num_cols, replace=False)
        P = csc_matrix((np.ones(num_cols), (inds, range(num_cols))),
                       shape=(d, num_cols))

        # # Walsh-Hadamard matrix
        # H = hadamard(d)
        # S = 1/np.sqrt(num_cols) * D.dot((P.T.dot(H)).T)
        # assert_shape(S, (d, num_cols))  # Different d
        # return S[:self.d, :]  # Only retain self.d rows

        return d, 1./np.sqrt(num_cols) * D, P

    def iterative(self, num_cols, num_iters, method, single_sketch=True,
                  probs=None):
        """
        Args:
            num_cols: int, number of sampled columns.
            num_iters: int, max. number of iterations.
            method: str, ssketching method to construct S.
            single_sketch: boolean, whether to use a single sketch throughtout
                the iterations or generate a new sketch at every iteration.
            probs: array(d), column-sampling probabilities.
                Not required for sketching-based methods (sampling == False).
        """
        assert_ge(num_cols, self.n)

        n = self.n
        d = self.d
        c = self.c
        B = self.B

        L = np.zeros((num_iters, n, c))
        G = np.zeros((num_iters, d, c))
        Y = np.zeros((num_iters, n, c))

        L[0] = self.P.todense()

        for i in xrange(num_iters):
            if i % 50 == 0:
                print "Iterative solver: iteration %d ..." % i

            if i > 0:
                L[i] = L[i-1] - self.lmbd * Y[i-1] - B.dot(G[i-1])

            if i == 0 or not single_sketch:
                # Generate new sketching matrix
                if method == 'sampling':
                    assert probs is not None
                    S = self.sampling_matrix(probs, num_cols)
                elif method == 'count-sketch':
                    S = self.count_sketch(num_cols)
                elif method == 'SRHT':
                    # Obtain the uniform-sampling and diagonal-sign matrices;
                    # apply FHT later
                    d1, D, S = self.subsampled_hadamard(num_cols)
                else:
                    raise ValueError("Sampling method not recognized!\n")

            # Apply sketch
            if method == 'SRHT':
                B0 = np.c_[B, np.zeros((n, d1 - d))]  # Pad by zeros
                BD = (D.dot(B0.T)).T  # Multiply by (rescaled) signs

                # Apply fast Hadamard transform
                BDH = np.zeros((n, d1))
                for k in xrange(n):  # For each row
                    a = deepcopy(BD[k, :])
                    fht(a)  # Modifies in-place
                    BDH[k, :] = deepcopy(a)

                assert not np.any(np.isnan(BDH)), BDH
                # Uniform sampling matrix
                SB = S.T.dot(BDH.T)
            else:
                SB = S.T.dot(B.T)

            assert_shape(SB, (num_cols, self.n))

            # Direct inverse
            H = SB.T.dot(SB) + self.lmbd * np.identity(n)
            H_inv = inv(H)

            # # Implicit inverse via SVD
            # U, sigmas, _ = svd(SB.T, full_matrices=True)  # Full SVD
            # Ds = diags(1./(sigmas**2. + self.lmbd))
            # H_inv = U.dot(Ds.dot(U.T))

            Y[i] = H_inv.dot(L[i])
            G[i] = B.T.dot(Y[i])

        G_opt = np.sum(G, axis=0)
        G_hist = np.cumsum(G, axis=0)

        return G_opt, G_hist

    def rel_err(self, G, G_opt):
        """
        Computes the relative error for the iterative solver.

        Args:
            x: array (d,) or (num_iters, d)
        """

        return norm(G - G_opt, axis=(1, 2)) / norm(G_opt)

    def obj_val(self, G):
        """
        Computes the objective value trace(S_t^{-1} S_b).
        """
        assert_shape(G, (self.d, self.c))

        return np.trace(G.dot(self.P.T.dot(self.B)))
