import numpy as np
from sklearn.metrics import confusion_matrix

"""
CONFUSION MATRIX CLASS

COPYRIGHT DISCLAIMER. ALL RIGHTS RESERVED. 
UNAUTHORIZED REPRODUCTION, MODIFICATION, DISTRIBUTION, DISSEMINATION, 
RETRANSMISSION, BROADCASTING, REPUBLICATION, REPRINTING OR REPOSTING OF THIS 
FILE, VIA ANY MEDIUM WITHOUT THE PRIOR PERMISSION IS STRICTLY PROHIBITED.

WRITTEN BY SANGHYUN PARK(lkd1962@naver.com), 8 DECEMBER 2021.
"""


class ConfusionMatrix:
    """
    Confusion matrix.

    Symbols
    -------
    G : # of groups.
    N : # of samples.
    """

    def __init__(self, y):
        """
        Constructs confusion matrix.

        Parameters
        ----------
        y : np.ndarray shape : (G, N)
            Predicted class labels.
        """
        labels = np.arange(len(y))
        y_true = np.concatenate([np.tile(i, y[i].size) for i in range(len(y))])
        self._C = confusion_matrix(y_true, np.concatenate(y), labels=labels)

    def acc(self):
        """
        Computes accuracy.

        Returns
        -------
        float
            Computed accuracy.
        """
        with np.errstate(divide='ignore'):
            return self._C.diagonal().sum() / self._C.sum()
