import snippets.datapackage as dpkg
from mass import DATA_PATH
from .util import LazyProperty
import numpy as np


class Feedback(object):
    def __init__(self, name):
        self.path = DATA_PATH.joinpath("model/%s.dpkg" % name)
        self.dp = dpkg.DataPackage.load(self.path)
        self.data = self.dp.load_resource("model.csv")

    @LazyProperty
    def fall(self):
        nfell = self\
            .data\
            .groupby(['sigma', 'phi', 'sample'])\
            .get_group((0.0, 0.0, 0))\
            .pivot('stimulus', 'kappa', 'nfell')
        fall = (nfell > 1).astype('float')
        return fall


class NoFeedback(Feedback):

    @LazyProperty
    def fall(self):
        fall = self\
            .data\
            .groupby(['sigma', 'phi', 'sample'])\
            .get_group((0.0, 0.0, 0))\
            .pivot('stimulus', 'kappa', 'nfell')
        fall[:] = np.nan
        return fall
