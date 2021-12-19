import glob
import pickle

"""
CACHE CLASS

COPYRIGHT DISCLAIMER. ALL RIGHTS RESERVED. 
UNAUTHORIZED REPRODUCTION, MODIFICATION, DISTRIBUTION, DISSEMINATION, 
RETRANSMISSION, BROADCASTING, REPUBLICATION, REPRINTING OR REPOSTING OF THIS 
FILE, VIA ANY MEDIUM WITHOUT THE PRIOR PERMISSION IS STRICTLY PROHIBITED.

WRITTEN BY SANGHYUN PARK(lkd1962@naver.com), 5 DECEMBER 2021.
"""


class Cache:
    """
    Cache.
    """

    def __init__(self):
        raise NotImplementedError('This class cannot be instantiated.')

    @classmethod
    def lookup(cls, fname):
        """
        Finds cache file.

        Parameters
        ----------
        fname : str
            Name of cache file to be found.

        Returns
        -------
        bool
            True if cache file is found. False otherwise.
        """
        return '../cache/' + fname in glob.glob('../cache/*.pkl')

    @classmethod
    def load(cls, fname):
        """
        Loads cache.

        Parameters
        ----------
        fname : str
            Name of cache file to be loaded.

        Returns
        -------
        object
            Loaded object.
        """
        with open('../cache/' + fname, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def save(cls, object_, fname):
        """
        Caches an object.

        Parameters
        ----------
        object_ : object
            Object to be cached.
        fname : str
            Name of cache file.
        """
        with open('../cache/' + fname, 'wb') as f:
            pickle.dump(object_, f)
