from tqdm import tqdm

"""
PROGRESS BAR CLASS

COPYRIGHT DISCLAIMER. ALL RIGHTS RESERVED. 
UNAUTHORIZED REPRODUCTION, MODIFICATION, DISTRIBUTION, DISSEMINATION, 
RETRANSMISSION, BROADCASTING, REPUBLICATION, REPRINTING OR REPOSTING OF THIS 
FILE, VIA ANY MEDIUM WITHOUT THE PRIOR PERMISSION IS STRICTLY PROHIBITED.

WRITTEN BY SANGHYUN PARK(lkd1962@naver.com), 5 DECEMBER 2021.
"""


class ProgressBar:
    """
    Progress bar.
    """
    def __init__(self, total, unit, description, verbose):
        """
        Constructs progress bar.

        Parameters
        ----------
        total : int
            Total count for progress bar.
        unit : str
            Unit for progress bar.
        description : str
            Short description for progress bar.
        verbose : bool
            If false, it does nothing.
        """
        if verbose:
            self._bar = tqdm(total=total, unit=unit, desc=description,
                             bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
        else:
            self._bar = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def update(self):
        """
        Updates progress bar.
        """
        if self._bar is not None:
            self._bar.update()

    def close(self):
        """
        Closes progress bar.
        """
        if self._bar is not None:
            self._bar.close()
