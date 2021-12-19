import os
from multiprocessing import Pool

from code.analysis.ConfusionMatrix import *
from code.analysis.GHMM import *
from code.utils.ProgressBar import *

"""
GAUSSIAN HIDDEN MARKOV MODEL CLASSIFIER CLASS

COPYRIGHT DISCLAIMER. ALL RIGHTS RESERVED. 
UNAUTHORIZED REPRODUCTION, MODIFICATION, DISTRIBUTION, DISSEMINATION, 
RETRANSMISSION, BROADCASTING, REPUBLICATION, REPRINTING OR REPOSTING OF THIS 
FILE, VIA ANY MEDIUM WITHOUT THE PRIOR PERMISSION IS STRICTLY PROHIBITED.

WRITTEN BY SANGHYUN PARK(lkd1962@naver.com), 8 DECEMBER 2021.
"""


class Classifier:
    """
    GHMM classifier.

    Symbols
    -------
    G : # of groups.
    N : # of samples.
    T : length of a sample.
    P : # of features.
    K : # of candidates.
    """

    def __init__(self, ks, seed=12345):
        """
        Constructs GHMM classifier.

        Parameters
        ----------
        ks : np.ndarray shape : (G)
            # of hidden states for each group.
        seed : int, default = 12345
            Seed for random procedure.
        """
        self._ks = ks
        self._seed = seed
        self._g = ks.size
        self._ghmms = None
        self._X = None

    """
    MAIN LOGIC
    """

    @classmethod
    def tune(cls, X, candidates, n_cores=-1, seed=12345, verbose=True):
        """
        Tunes hyperparameter(# of hidden states) of GHMM classifier.

        Parameters
        ----------
        X : list of np.ndarray shape : (G, N, P, T)
            Data for tuning.
        candidates : np.ndarray shape : (K)
            Candidates for tuning.
        n_cores : int, default = -1
            # of threads for multithreading.
            0 means no multithreading and any negative value means default
            setting which is # of cores supported by CPU.
        seed : int, default = 12345
            Seed for random procedure.
        verbose : bool, default = True
            If true, it shows progress bar.

        Returns
        -------
        dict
            Dictionary holding tuning results.
        """
        K = len(candidates)
        g = len(X)
        log_n = np.log([X[i].shape[0] for i in range(len(X))])
        np.random.seed(seed)
        list_of_args = [(k, np.random.randint(0, 100000), X[i])
                        for i in range(g) for k in candidates]
        ghmms = np.array(
            cls._do_parallel(cls._fit, list_of_args, n_cores, 'model',
                             'fitting', verbose),
            dtype=object).reshape(g, K)
        list_of_args = [(ghmms[i, j], X[k]) for i in range(g)
                        for j in range(K) for k in range(g)]
        l = cls._do_parallel(cls._loglike, list_of_args, n_cores, 'group',
                             'computing log-likelihood', verbose, (g, K, g))
        list_of_args = [(i, ghmms[i, j].dim(), l[i, j], log_n, g)
                        for i in range(g) for j in range(K)]
        criterion = cls._do_parallel(cls._information_criterion, list_of_args,
                                     n_cores, 'model',
                                     'computing information criterion',
                                     verbose, (g, K, 4))
        min_idx = np.nanargmin(criterion, axis=1)
        min_ = np.nanmin(criterion, axis=1)
        decision = np.array([[candidates[min_idx[i, j]] for j in range(4)]
                             for i in range(g)])
        result = dict(decision=decision,
                      min=dict(aic=min_[:, 0], bic=min_[:, 1],
                               dfaics=min_[:, 2], dfbics=min_[:, 3]),
                      criterion=dict(aic=criterion[:, :, 0],
                                     bic=criterion[:, :, 1],
                                     dfaics=-criterion[:, :, 2],
                                     dfbics=-criterion[:, :, 3]))
        return result

    def fit(self, X, n_cores=-1, verbose=True):
        """
        Fits GHMM classifier.

        Parameters
        ----------
        X : list of np.ndarray shape : (G, N, P, T)
            Observed samples for fitting.
        n_cores : int, default = -1
            # of threads for multithreading.
            0 means no multithreading and any negative value means default
            setting which is # of cores supported by CPU.
        verbose : bool, default = True
            If true, it shows progress bar.

        Returns
        -------
        Classifier
            Fitted self.
        """
        assert self._g == len(X)
        self._X = X
        list_of_args = [(self._ks[i], self._get_seed(), X[i])
                        for i in range(self._g)]
        self._ghmms = self._do_parallel(self._fit, list_of_args, n_cores,
                                        'model', 'fitting', verbose)
        return self

    def predict(self, X=None, n_cores=-1, verbose=True):
        """
        Classifies samples.

        Parameters
        ----------
        X : list of np.ndarray shape : (G, N, P, T)
            Observed samples to be classified.
            If not given, it uses the data given for fitting.
        n_cores : int, default = -1
            # of threads for multithreading.
            0 means no multithreading and any negative value means default
            setting which is # of cores supported by CPU.
        verbose : bool, default = True
            If true, it shows progress bar.

        Returns
        -------
        dict
            Dictionary holding classification results.
        """
        self._check_is_fitted()
        X = self._X if X is None else X
        list_of_args = [(self._ghmms, X[i]) for i in range(self._g)]
        y = self._do_parallel(self._predict, list_of_args, n_cores, 'group',
                              'predicting', verbose)
        result = dict(y=y, C=ConfusionMatrix(y))
        return result

    """
    HELPERS
    """

    @staticmethod
    def _fit(args):
        """
        Fits GHMM.
        This is a subroutine for `cls.tune` method and `cls.fit` method.

        Parameters
        ----------
        k : int
            # of hidden states.
        seed : int
            Seed for random procedure.
        X : np.ndarray shape : (N, P, T)
            Observed samples for fitting.

        Returns
        -------
        GHMM
            Fitted GHMM.
        """
        k, seed, X = args
        return GHMM(k, seed).fit(X)

    @staticmethod
    def _loglike(args):
        """
        Computes mean of loglikelihood.
        This is a subroutine for `cls.tune` method.

        Parameters
        ----------
        ghmm : GHMM
            Fitted GHMM.
        X : np.ndarray shape : (N, P, T)
            Observed samples whose loglikelihoods are to be computed.

        Returns
        -------
        float
            Computed mean loglikelihood.
        """
        ghmm, X = args
        return np.nanmean(ghmm.loglike(X))

    @staticmethod
    def _information_criterion(args):
        """
        Computes information criterion.
        This is a subroutine for `cls.tune` method.

        Parameters
        ----------
        idx : int
            Index indicating target group.
        dim : int
            Dimension of parameter space.
        l : np.ndarray shape : (G)
            Mean loglikelihoods of each group.
        log_n : np.ndarray shape : (G)
            Log of # of samples in each group.
        g : int
            # of groups.

        Returns
        -------
        np.ndarray shape : (4)
            Computed information criterion.
        """
        idx, dim, l, log_n, g = args
        l_mean = np.nanmean(l)
        aic = -2 * l[idx] + 2 * dim
        bic = -2 * l[idx] + log_n[idx] * dim
        with np.errstate(invalid='ignore'):
            dfaics = -(l[idx] - l_mean) * g / (g - 1)
            dfbics = -(l[idx] - l_mean - (log_n[idx] - log_n.mean()) * dim) \
                     * g / (g - 1)
        return np.array([aic, bic, dfaics, dfbics])

    @staticmethod
    def _predict(args):
        """
        Classifies samples.
        This is a subroutine for `cls.predict` method.

        Parameters
        ----------
        ghmms : list of GHMM
            Fitted GHMMs for each group.
        X : np.ndarray shape : (N, P, T)
            Observed samples to be classified.

        Returns
        -------
        np.ndarray shape : (N)
            Predicted class labels.
        """
        ghmms, X = args
        if X.size == 0:
            return np.array([])
        y = np.nanargmax(np.array([ghmms[i].loglike(X)
                                   for i in range(len(ghmms))]), axis=0)
        return y

    @classmethod
    def _do_parallel(cls, job, list_of_args, n_cores, unit, description,
                     verbose, reshape=None):
        """
        Does given job in parallel via multithreading.

        Parameters
        ----------
        job : callable
            Job to be done in parallel.
        list_of_args : list
            List of arguments to be passed to `job`.
        n_cores : int
            # of threads for multithreading.
            0 means no multithreading and any negative value means default
            setting which is # of cores supported by CPU.
        unit : str
            Unit of each job for progress bar.
        description : str
            Short description of the job for progress bar.
        verbose : bool
            If true, it shows progress bar.
        reshape : tuple of int, optional
            If given, it reshapes the result.

        Returns
        -------
        list
            List holding return values from `job`.
        """
        with ProgressBar(len(list_of_args), unit, description, verbose) as bar:
            if n_cores == 0:
                results = [None] * len(list_of_args)
                for i in range(len(list_of_args)):
                    results[i] = job(list_of_args[i])
                    bar.update()
            else:
                results = []
                if n_cores < 0:
                    n_cores = os.cpu_count()
                else:
                    n_cores = min(n_cores, os.cpu_count())
                with Pool(n_cores) as pool:
                    for result in pool.imap(job, list_of_args):
                        results.append(result)
                        bar.update()
        if reshape is None:
            return results
        else:
            return np.array(results).reshape(reshape)

    def _get_seed(self):
        """
        Generates new seed.

        Returns
        -------
        int
            Generated seed.
        """
        seed = self._seed
        np.random.seed(seed)
        self._seed = np.random.randint(0, 100000)
        return seed

    def _check_is_fitted(self):
        """
        Checks whether GHMM classifier is fitted or not.

        Raises
        ------
        RuntimeError
            If GHMM classifier is not fitted yet.
        """
        if self._ghmms is None:
            raise RuntimeError('Classifier is not fitted yet.')
