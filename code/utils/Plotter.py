import matplotlib.pyplot as plt
import numpy as np

"""
PLOTTING CLASS

COPYRIGHT DISCLAIMER. ALL RIGHTS RESERVED. 
UNAUTHORIZED REPRODUCTION, MODIFICATION, DISTRIBUTION, DISSEMINATION, 
RETRANSMISSION, BROADCASTING, REPUBLICATION, REPRINTING OR REPOSTING OF THIS 
FILE, VIA ANY MEDIUM WITHOUT THE PRIOR PERMISSION IS STRICTLY PROHIBITED.

WRITTEN BY SANGHYUN PARK(lkd1962@naver.com), 9 DECEMBER 2021.
"""


class Plotter:
    """
    Plotter.

    Symbols
    -------
    G : # of groups.
    """
    _COLOR = ['#00AFBB', '#E7B800', '#FC4E07', '#BB3099', '#EE0099', '#0000AC']
    plt.rc('font', family='serif')
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{mathptmx}')

    def __init__(self):
        raise NotImplementedError('This class cannot be instantiated.')

    """
    MAIN LOGIC
    """

    @classmethod
    def plot_criterion(cls, dname, criterion, group_index, show=True,
                       save=False):
        """
        Plots information criterion.

        Parameters
        ----------
        dname : {'digits', 'eeg'}
            Name of the target dataset.
        criterion : list of np.ndarray shape : (4, G)
            Information criterion to be plotted.
        group_index : int
            Index indicating the group whose information criterion are to be
            plotted.
        show : bool, default = True
            If true, it shows the plot.
        save : bool, default = False
            If true, it saves the plot as PDF.

        Returns
        -------
        plt.figure
            Figure object holding the plot.
        """
        cls._check_dname(dname)

        criterion_name = ['aic', 'bic', 'dfaics', 'dfbics']
        criterion = np.array([criterion[k][group_index]
                              for k in criterion_name])
        k = np.arange(criterion.shape[1]) + 1
        decision = np.concatenate([np.nanargmin(criterion[:2], axis=1),
                                   np.nanargmax(criterion[2:], axis=1)]) + 1
        criterion_at_decision = [criterion[i, decision[i] - 1]
                                 for i in range(4)]

        figure, axes = cls._new_figure()
        axes_twin = axes.twinx()
        for i in range(2):
            axes.plot(k, criterion[i] / 1e4, marker='s', ms=4, lw=1.5,
                      c=cls._COLOR[i], zorder=0)
            axes.scatter(decision[i], criterion_at_decision[i] / 1e4, s=60,
                         lw=0, marker='o', c=cls._COLOR[i], zorder=1)
        for i in range(2):
            axes_twin.plot(k, criterion[i + 2] / 1e3, marker='s', ms=4, lw=1.5,
                           c=cls._COLOR[i + 2], zorder=0)
            axes_twin.scatter(decision[i + 2],
                              criterion_at_decision[i + 2] / 1e3, s=60, lw=0,
                              marker='o', c=cls._COLOR[i + 2], zorder=1)

        xticks = k if dname == 'digits' else np.arange(k.size / 2) * 2 + 2
        axes.set_xticks(xticks)
        axes.set_xlabel(r'\# of hidden states ($k$)',
                        fontdict=dict(size='large', family='serif'))
        axes.set_ylabel(r'AIC / BIC ($\times10^4$)',
                        fontdict=dict(size='large', family='serif'))
        axes_twin.set_ylabel(r'DFAICS / DFBICS ($\times10^3$)',
                             fontdict=dict(size='large', family='serif'))

        box = axes.get_position()
        axes.set_position([box.x0, box.y0 + box.height * .1, box.width,
                           box.height * .9])
        axes.legend(['AIC', 'BIC'], ncol=4, frameon=False,
                    loc='upper center', bbox_to_anchor=(.28, -.15))
        axes_twin.legend(['DFAICS', 'DFBICS'], ncol=4, frameon=False,
                         loc='upper center', bbox_to_anchor=(.72, -.15))

        cls._show_and_save(figure, show, save,
                           f'../figure/criterion_{dname}_{group_index + 1}.pdf')
        return figure

    @classmethod
    def plot_acc(cls, dname, acc, show=True, save=False):
        """
        Plots accuracy.

        Parameters
        ----------
        dname : {'digits', 'eeg'}
            Name of the target dataset.
        acc : np.ndarray shape : (4)
            Accuracy to be plotted.
        show : bool, default = True
            If true, it shows the plot.
        save : bool, default = False
            If true, it saves the plot as PDF.

        Returns
        -------
        plt.figure
            Figure object holding the plot.
        """
        cls._check_dname(dname)

        figure, axes = cls._new_figure()
        for i in range(4):
            axes.bar(i, acc[i] * 100, width=.8, lw=0, fc=cls._COLOR[i])
        if dname == 'digits':
            axes.set_ylim(90, 100)
        else:
            axes.set_ylim(60, 75)
        ylim = axes.get_ylim()
        pad = (ylim[1] - ylim[0]) * .07
        for i in range(4):
            axes.annotate(f'{np.round(acc[i] * 100, 2)}\%',
                          (i, acc[i] * 100 - pad), va='center', ha='center',
                          fontsize='large')

        axes.set_xticks(np.arange(4))
        axes.set_xticklabels(['AIC', 'BIC', 'DFAICS', 'DFBICS'])
        axes.set_xlabel('information criterion')
        axes.set_ylabel('ACC (\%)')

        cls._show_and_save(figure, show, save, f'../figure/acc_{dname}.pdf')
        return figure

    """
    HELPERS
    """

    @classmethod
    def _check_dname(cls, dname):
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

    @classmethod
    def _new_figure(cls):
        """
        Generates new figure object.

        Returns
        -------
        plt.figure
            Generated figure obejct.
        plt.axes
            Axes object in the generated figure object.
        """
        figure = plt.figure(figsize=(6, 4))
        axes = figure.subplots()
        figure.tight_layout(pad=3, h_pad=5, w_pad=3)
        axes.tick_params(labelsize='large')
        return figure, axes

    @classmethod
    def _show_and_save(cls, figure, show, save, fname):
        """
        Shows and saves the plot if needed.

        Parameters
        ----------
        figure : plt.figure
            Figure object to be displayed or saved.
        show : bool
            If true, it shows the plot.
        save : bool
            If true, it saves the plot as PDF.
        fname : str
            Name of the file to be saved.
        """
        if show:
            figure.show()
        if save and fname is not None:
            figure.savefig(fname, bbox_inches='tight')
