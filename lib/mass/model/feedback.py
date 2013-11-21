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
    def unstable(self):
        data = self.data.copy().drop(['nfell'], axis=1)
        data['unstable'] = (self.data['nfell'] > 0).astype(float)
        return data.set_index(['sigma', 'phi', 'sample'])\
                   .ix[(0.0, 0.0, 0)]

    @LazyProperty
    def fall(self):
        # make sure there's only one sample
        fall = self.unstable.pivot(
            index='stimulus',
            columns='kappa',
            values='unstable')
        return fall


class NoFeedback(Feedback):

    @LazyProperty
    def unstable(self):
        data = self.data.copy().drop(['nfell'], axis=1)
        data['unstable'] = np.nan
        return data.set_index(['sigma', 'phi', 'sample'])\
                   .ix[(0.0, 0.0, 0)]
