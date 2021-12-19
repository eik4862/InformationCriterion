import glob
import os
from multiprocessing import Pool

import numpy as np
from scipy.integrate import simpson
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from code.utils.ProgressBar import *

"""
PREPROCESS CLASS

COPYRIGHT DISCLAIMER. ALL RIGHTS RESERVED. 
UNAUTHORIZED REPRODUCTION, MODIFICATION, DISTRIBUTION, DISSEMINATION, 
RETRANSMISSION, BROADCASTING, REPUBLICATION, REPRINTING OR REPOSTING OF THIS 
FILE, VIA ANY MEDIUM WITHOUT THE PRIOR PERMISSION IS STRICTLY PROHIBITED.

WRITTEN BY SANGHYUN PARK(lkd1962@naver.com), 8 DECEMBER 2021.
"""


class Preprocess:
    """
    Toolbox for data preprocessing.

    Symbols
    -------
    G : # of groups.
    N : # of samples.
    T : length of a sample.
    P : # of features.
    """
    def __init__(self):
        raise NotImplementedError('This class cannot be instantiated.')

    """
    MAIN LOGIC
    """

    @classmethod
    def process(cls, dname, n_cores=-1, seed=12345, verbose=True):
        """
        Preprocesses data.
        It loads data, preprocesses data and split data into train data and
        test data.
        The splitting ratio is fixed as 6:4.

        Parameters
        ----------
        dname : {'digits', 'eeg'}
            Name of the dataset to be preprocessed.
        n_cores : int, default = -1
            # of threads for multithreading.
            0 means no multithreading and any negative value means default
            setting which is # of cores supported by CPU.
        seed : int, default = 12345
            Seed for random procedure.
        verbose : bool, default = True
            If true, it shows progress bar.

        Raises
        ------
        ValueError
            If dname is invalid.

        Returns
        -------
        list of np.ndarray shape: (G, N, P, T)
            Train data.
        list of np.ndarray shape: (G, N, P, T)
            Test data.
        """
        if dname == 'digits':
            return cls._process_digit(seed)
        elif dname == 'eeg':
            return cls._process_eeg(n_cores, seed, verbose)
        else:
            raise ValueError('Invalid data name.')

    """
    HELPERS
    """

    @classmethod
    def _process_digit(cls, seed):
        """
        Preprocesses data for handwriting recognition.

        Parameters
        ----------
        seed : int
            Seed for random procedure.

        Returns
        -------
        list of np.ndarray shape: (G, N, P, T)
            Train data.
        list of np.ndarray shape: (G, N, P, T)
            Test data.
        """
        raw = loadmat('../data/digits.mat', squeeze_me=True)
        X, y = raw['mixout'], raw['consts']['charlabels'].all() - 1
        cls._pad(X)

        v_rect = np.array(X.tolist())
        a_rect = cls._differentiate(v_rect)
        d_rect = cls._integrate(v_rect)
        d_sp = cls._rectangular_to_spherical(d_rect)
        v_sp = cls._differentiate(d_sp)
        a_sp = cls._differentiate(v_sp)

        X = cls._collect_features(d_rect, v_rect, a_rect, d_sp, v_sp, a_sp)
        X = cls._split_by_group(X, y)
        X_train, X_test = cls._train_test_split(X, seed)

        return X_train, X_test

    @classmethod
    def _process_eeg(cls, n_cores, seed, verbose):
        """
        Preprocesses data for EEG classification.

        Parameters
        ----------
        n_cores : int
            # of threads for multithreading.
            0 means no multithreading and any negative value means default
            setting which is # of cores supported by CPU.
        seed : int
            Seed for random procedure.
        verbose : bool, default = True
            If true, it shows progress bar.

        Returns
        -------
        list of np.ndarray shape: (G, N, P, T)
            Train data.
        list of np.ndarray shape: (G, N, P, T)
            Test data.
        """
        eeg, y = cls._parse_eeg(n_cores, verbose)
        X = cls._bandpower(eeg, n_cores, True, verbose)
        X = cls._normalize(X)
        bp = cls._bandpower(eeg, 0, False, verbose)
        X = cls._pca(X, bp)

        X = cls._split_by_group(X, y)
        X_train, X_test = cls._train_test_split(X, seed)

        return X_train, X_test

    @staticmethod
    def _parse_job(target):
        """
        Parses text file.
        This is a subroutine for `cls._parse_eeg` method.

        Parameters
        ----------
        target : str
            Path of the file to be parsed.

        Returns
        -------
        None
            If the file to be parsed is invalid.
        np.ndarray shape : (P, T)
            Observed sample.
        int
            Observed class label.
        """
        with open(target, 'r') as file:
            lines = list(file)
            if any(['err' in line for line in lines]) or len(lines) < 10 \
                    or 'S1' not in lines[3]:
                return None
            y = 1 if lines[0][5] == 'a' else 0
            X = np.zeros((64, 256))
            lines = lines[4:]
            i, j = -1, -1
            for line in lines:
                if '#' in line:
                    i += 1
                    j = 0
                else:
                    X[i, j] = float(line.split()[-1])
                    j += 1
        return X, y

    @staticmethod
    def _bandpower_job(eeg):
        """
        Computes bandpower.
        This is a subroutine for `cls._bandpower` method.

        Parameters
        ----------
        eeg : np.ndarray shape : (GN, P, T)
            EEG data for bandpower computation.

        Returns
        -------
        np.ndarray shape : (GN, T, P)
            Computed bandpowers.
        """
        f, Pxx = welch(eeg, fs=256, nperseg=128, axis=-1)
        df = f[0] - f[1]
        delta = simpson(Pxx[:, :, f <= 4], dx=df, axis=-1)
        theta = simpson(Pxx[:, :, np.logical_and(4 < f, f <= 12)], dx=df,
                        axis=-1)
        beta = simpson(Pxx[:, :, np.logical_and(12 < f, f <= 30)], dx=df,
                       axis=-1)
        gamma = simpson(Pxx[:, :, np.logical_and(30 < f, f <= 50)], dx=df,
                        axis=-1)
        high_gamma = simpson(Pxx[:, :, np.logical_and(50 < f, f <= 100)],
                             dx=df, axis=-1)
        return np.hstack([delta, theta, beta, gamma, high_gamma])

    @classmethod
    def _do_parallel(cls, job, list_of_args, n_cores, unit, description,
                     verbose):
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
        return results

    @classmethod
    def _pad(cls, X):
        """
        Pads data.

        Parameters
        ----------
        X : list of np.ndarray shape : (GN, P, T)
            Data to be padded.

        Returns
        -------
        list of np.ndarray shape : (GN, P, T)
            Padded data.
        """
        T = np.nanmax([sample.shape[1] for sample in X])
        for i in range(X.size):
            if X[i].shape[1] != T:
                padding = np.repeat(0, (T - X[i].shape[1]) * 3).reshape(
                    3, T - X[i].shape[1])
                X[i] = np.hstack([X[i], padding])

    @classmethod
    def _differentiate(cls, X):
        """
        Differentiates data.

        Parameters
        ----------
        X : np.ndarray shape : (GN, 3, T)
            Data to be differentiated.

        Returns
        -------
        np.ndarray shape : (GN, 3, T)
            Differentiated data.
        """
        return gaussian_filter1d(np.gradient(X, axis=-1), sigma=2, axis=-1)

    @classmethod
    def _integrate(cls, X):
        """
        Integrates data.

        Parameters
        ----------
        X : np.ndarray shape : (GN, 3, T)
            Data to be integrated.

        Returns
        -------
        np.ndarray shape : (GN, 3, T)
            Integrated data.
        """
        return gaussian_filter1d(np.cumsum(X, axis=-1), sigma=2, axis=-1)

    @classmethod
    def _rectangular_to_spherical(cls, d_rect):
        """
        Converts rectangular coordinate data to spherical coordinate data.

        Parameters
        ----------
        d_rect : np.ndarray shape : (GN, 3, T)
            Data to be converted.

        Returns
        -------
        np.ndarray shape : (GN, 3, T)
            Converted data.
        """
        r = np.linalg.norm(d_rect, axis=1)
        theta = np.arctan2(np.linalg.norm(d_rect[:, :2], axis=1), d_rect[:, 2])
        phi = np.arctan2(d_rect[:, 1], d_rect[:, 0])
        return np.moveaxis(np.array([r, theta, phi]), 0, 1)

    @classmethod
    def _parse_eeg(cls, n_cores, verbose):
        """
        Parses data for EEG classification.

        Parameters
        ----------
        n_cores : int
            # of threads for multithreading.
            0 means no multithreading and any negative value means default
            setting which is # of cores supported by CPU.
        verbose : bool
            If true, it shows progress bar.

        Returns
        -------
        np.ndarray shape : (GN, P, T)
            Observed samples.
        np.ndarray shape : (GN)
            Observed class labels.
        """
        X, y = [], []
        targets = glob.glob('../data/eeg/*')
        Xys = cls._do_parallel(cls._parse_job, targets, n_cores, 'file',
                               'parsing', verbose)
        for Xy in Xys:
            if Xy is not None:
                X.append(Xy[0])
                y.append(Xy[1])
        return np.array(X), np.array(y)

    @classmethod
    def _bandpower(cls, eeg, n_cores, moving, verbose):
        """
        Computes bandpowers.

        Parameters
        ----------
        eeg : np.ndarray shape : (GN, P, T)
            EEG data for bandpower computation.
        n_cores : int
            # of threads for multithreading.
            0 means no multithreading and any negative value means default
            setting which is # of cores supported by CPU.
        moving : bool
            If true, it computes bandpower by moving window of size 128.
        verbose : bool
            If true, it shows progress bar.

        Returns
        -------
        np.ndarray shape :
            Computed bandpowers.
        """
        if moving:
            i = 0
            eegs = []
            while 2 * i + 128 <= eeg.shape[2]:
                eegs.append(eeg[:, :, (2 * i):(2 * i) + 128])
                i += 1
            bp = np.array(cls._do_parallel(cls._bandpower_job, eegs, n_cores,
                                           'window', 'computing bandpowers',
                                           verbose))
            return gaussian_filter1d(np.moveaxis(bp, 0, 2), sigma=2, axis=-1)
        else:
            return cls._bandpower_job(eeg)

    @classmethod
    def _normalize(cls, X):
        """
        Normalizes data.

        Parameters
        ----------
        X : np.ndarray shape : (GN, P, T)
            Data to be normalized.

        Returns
        -------
        np.ndarray shape : (GN, P, T)
            Normalized data.
        """
        X = np.moveaxis(X, 2, 0)
        shape = X.shape
        X = X.reshape((shape[0] * shape[1], shape[2]))
        X = StandardScaler().fit_transform(X).reshape(shape)
        return np.moveaxis(X, 0, 2)

    @classmethod
    def _pca(cls, X, bp):
        """
        Performs PCA and projects data.
        It projects data into the subspace spanned by PCs explaining 99% of
        the variance.

        Parameters
        ----------
        X : np.ndarray shape : (GN, P, T)
            Data to be projected.
        bp : np.ndarray shape : (GN, p)
            Data for PCA.

        Returns
        -------
        np.ndarray shape : (GN, P, T)
            Projected data.
        """
        pca = PCA(svd_solver='full').fit(bp)
        k = int(np.sum(np.cumsum(pca.explained_variance_ratio_) < .99)) + 1
        pca = PCA(n_components=k, svd_solver='full').fit(bp)
        Y = []
        for i in range(X.shape[2]):
            Y.append(pca.transform(X[:, :, i]))
        return np.moveaxis(np.array(Y), 0, 2)

    @classmethod
    def _collect_features(cls, *features):
        """
        Collects features.

        Parameters
        ----------
        features : np.ndarray shape : (GN, 3, T)
            Features to be collected.

        Returns
        -------
        np.ndarray shape : (GN, P, T)
            Collected features.
        """
        X = []
        for i in range(features[0].shape[0]):
            X.append(np.vstack([feature[i] for feature in features]))
        return np.array(X)

    @classmethod
    def _split_by_group(cls, X, y):
        """
        Splits data into the groups according to the class labels.

        Parameters
        ----------
        X : np.ndarray shape : (GN, P, T)
            Data to be spilt.
        y : np.ndarray shape : (GN)
            Class labels.

        Returns
        -------
        list of np.ndarray shape : (G, N, P, T)
            Split data.
        """
        labels = np.unique(y)
        Y = [[] for _ in range(labels.size)]
        for i in range(y.size):
            Y[y[i]].append(X[i])
        for i in range(len(Y)):
            Y[i] = np.array(Y[i])
        return Y

    @classmethod
    def _train_test_split(cls, X, seed):
        """
        Splits data into train and test data.

        Parameters
        ----------
        X : list of np.ndarray shape : (G, N, P, T)
            Data to be split.
        seed : int
            Seed for random procedure.

        Returns
        -------
        list of np.ndarray shape: (G, N, P, T)
            Train data.
        list of np.ndarray shape: (G, N, P, T)
            Test data.
        """
        X_train, X_test = [None] * len(X), [None] * len(X)
        for i in range(len(X)):
            X_train[i], X_test[i] = train_test_split(X[i], test_size=.4,
                                                     random_state=seed)
        return X_train, X_test
