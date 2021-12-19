from analysis.Classifier import *
from analysis.Preprocess import *
from utils.Cache import *
from utils.Plotter import *

"""
VALIDATION OF DFAICS AND DFBICS

COPYRIGHT DISCLAIMER. ALL RIGHTS RESERVED. 
UNAUTHORIZED REPRODUCTION, MODIFICATION, DISTRIBUTION, DISSEMINATION, 
RETRANSMISSION, BROADCASTING, REPUBLICATION, REPRINTING OR REPOSTING OF THIS 
FILE, VIA ANY MEDIUM WITHOUT THE PRIOR PERMISSION IS STRICTLY PROHIBITED.

WRITTEN BY SANGHYUN PARK(lkd1962@naver.com), 9 DECEMBER 2021.
"""


def test(dname, use_cache=True):
    """
    Runs test.

    Parameters
    ----------
    dname : {'digits', 'eeg'}
        Name of the dataset to be used.
    use_cache : bool, default = True
        If true, it uses cached results to save time.

    Returns
    -------
    dict
        Dictionary holding tuning results using train data.
    dict
        Dictionary holding classification results using test data.
    """
    _check_dname(dname)

    if use_cache and Cache.lookup(f'X_{dname}.pkl'):
        X_train, X_test = Cache.load(f'X_{dname}.pkl')
    else:
        X_train, X_test = Preprocess.process(dname)
        Cache.save((X_train, X_test), f'X_{dname}.pkl')

    if use_cache and Cache.lookup(f'tuning_result_{dname}.pkl'):
        tuning_result = Cache.load(f'tuning_result_{dname}.pkl')
    else:
        candidates = np.arange(20 if dname == 'digits' else 40) + 1
        tuning_result = Classifier.tune(X_train, candidates)
        Cache.save(tuning_result, f'tuning_result_{dname}.pkl')

    if use_cache and Cache.lookup(f'classification_result_{dname}.pkl'):
        classification_result = Cache.load(
            f'classification_result_{dname}.pkl')
    else:
        classification_result = [None] * 4
        for i in range(4):
            classifier = Classifier(tuning_result['decision'][:, i])
            classifier.fit(X_train)
            classification_result[i] = classifier.predict(X_test)
        Cache.save(classification_result, f'classification_result_{dname}.pkl')

    return tuning_result, classification_result


def plot(dname, save=True):
    """
    Plots test results.

    Parameters
    ----------
    dname : {'digits', 'eeg'}
        Name of the target dataset.
    save : bool, default = True
        If true, it saves the plot as PDF.
    """
    _check_dname(dname)
    if Cache.load(f'tuning_result_{dname}.pkl'):
        tuning_result = Cache.load(f'tuning_result_{dname}.pkl')
        criterion = tuning_result['criterion']
        for i in range(criterion['aic'].shape[0]):
            Plotter.plot_criterion(dname, criterion, i, save=save)
    if Cache.load(f'classification_result_{dname}.pkl'):
        classification_result = Cache.load(
            f'classification_result_{dname}.pkl')
        acc = np.array([classification_result[i]['C'].acc() for i in range(4)])
        Plotter.plot_acc(dname, acc, save=save)


def _check_dname(dname):
    """
    Checks whether the given dname is valid or not.

    Parameters
    ----------
    dnames : {'digits', 'eeg'}
        Name of the dataset to be checked.

    Raises
    ------
    ValueError
        If dname is invalid.
    """
    if dname not in ['digits', 'eeg']:
        raise ValueError('Invalid data name.')


if __name__ == '__main__':
    # test('digits')
    # plot('digits')
    # test('eeg')
    # plot('eeg')
    pass
