import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

"""
GAUSSIAN HIDDEN MARKOV MODEL CLASS

COPYRIGHT DISCLAIMER. ALL RIGHTS RESERVED. 
UNAUTHORIZED REPRODUCTION, MODIFICATION, DISTRIBUTION, DISSEMINATION, 
RETRANSMISSION, BROADCASTING, REPUBLICATION, REPRINTING OR REPOSTING OF THIS 
FILE, VIA ANY MEDIUM WITHOUT THE PRIOR PERMISSION IS STRICTLY PROHIBITED.

WRITTEN BY SANGHYUN PARK(lkd1962@naver.com), 8 DECEMBER 2021.
"""


class GHMM:
    """
    Gaussian hidden Markov model.

    Symbols
    -------
    N : # of samples.
    T : length of a sample.
    P : # of features.
    """

    def __init__(self, k, seed=12345):
        """
        Constructs GHMM.

        Parameters
        ----------
        k : int
            # of hidden states.
        seed : int, default = 12345
            Seed for random procedure.
        """
        self._k = k
        self._seed = seed
        self._ghmm = None
        self._Y = None
        self._log_pi = None
        self._log_T = None
        self._mvns = None

    """
    MAIN LOGIC
    """

    def fit(self, Y):
        """
        Fits GHMM.

        Parameters
        ----------
        Y : np.ndarray shape : (N, P, T)
            Observed samples for fitting.

        Returns
        -------
        GHMM
            Fitted self.
        """
        self._Y = Y
        self._ghmm = GaussianHMM(self._k,
                                 random_state=self._seed).fit(*self._merge(Y))
        with np.errstate(divide='ignore'):
            self._log_pi = np.log(np.array(self._ghmm.startprob_))
            self._log_T = np.log(np.array(self._ghmm.transmat_))
        self._mvns = [multivariate_normal(mu, sigma) for mu, sigma
                      in zip(self._ghmm.means_, self._ghmm.covars_)]
        return self

    def decode(self, Y=None):
        """
        Estimates hidden states.

        Parameters
        ----------
        Y : np.ndarray shape : (N, P, T), optional
            Observed samples whose hidden states are to be estimated.
            If not given, it uses the data given for fitting.

        Returns
        -------
        np.ndarray shape : (N, T)
            Estimated hidden states.
        """
        self._check_is_fitted()
        Y = self._Y if Y is None else Y
        Y, lengths = self._merge(Y)
        X = self._ghmm.decode(Y, lengths)[1]
        return self._split(X, lengths)

    def loglike(self, Y=None):
        """
        Computes loglikelihood.
        It uses pseudo-loglikelihood approximation for efficient evaluation of
        loglikelihood.

        Parameters
        ----------
        Y : np.ndarray shape : (N, P, T), optional
            Observed samples whose loglikelihoods are to be computed.
            If not given, it uses the data given for fitting.

        Returns
        -------
        np.ndarray shape : (N)
            Computed loglikelihood.
        """
        self._check_is_fitted()
        Y = self._Y if Y is None else Y
        X = self.decode(Y)
        E = np.array([[[mvn.logpdf(Y[i, :, t]) for mvn in self._mvns]
                       for t in range(Y.shape[2])] for i in range(Y.shape[0])])
        T = np.array([[self._log_pi]
                      + [[self._log_T[X[i, t], s] for s in range(self._k)]
                         for t in range(Y.shape[2] - 1)]
                      for i in range(Y.shape[0])])
        with np.errstate(divide='ignore'):
            l = logsumexp(E + T, axis=2).sum(axis=1)
        return l

    def dim(self):
        """
        Computes dimension of parameter space.

        Returns
        -------
        int
            Computed dimension of parameter space.
        """
        self._check_is_fitted()
        return self._k * (self._k + 2 * self._Y.shape[1])

    """
    HELPERS
    """

    @classmethod
    def _merge(cls, Y):
        """
        Merges samples into one sample.

        Parameters
        ----------
        Y : np.ndarray shape : (N, P, T)
            Samples to be merged.

        Returns
        -------
        np.ndarray shape : (NT, P)
            Merged sample.
        np.ndarray shape : (N)
            Length information.
        """
        lengths = np.tile(Y.shape[2], Y.shape[0])
        Y = np.vstack([Y[i].T for i in range(Y.shape[0])])
        return Y, lengths

    @classmethod
    def _split(cls, X, lengths):
        """
        Splits merged sample.

        Parameters
        ----------
        X : np.ndarray shape : (NT)
            Merged sample to be split.
        lengths : np.ndarray shape : (N)
            Length information.

        Returns
        -------
        np.ndarray shape : (N, T)
            Split data.
        """
        X_split = [None] * lengths.size
        l = 0
        for i in range(lengths.size):
            X_split[i] = X[l:l + lengths[i]]
            l += lengths[i]
        return np.array(X_split)

    def _check_is_fitted(self):
        """
        Checks whether GHMM is fitted or not.

        Raises
        ------
        RuntimeError
            If GHMM is not fitted yet.
        """
        if self._ghmm is None:
            raise RuntimeError('GHMM is not fitted yet.')
