import dbtools
from path import path
from analysis_tools import load_datapackage as load

DATAPATH = path("../../mass-inference-data/model")
DBPATH = path("../resources/sso/metadata.db")
TABLE = "stability"
PARAMS = ['sigma', 'phi', 'kappa']

if not dbtools.Table.exists(DBPATH, "stability"):
    tbl = dbtools.Table.create(
        DBPATH, "stability",
        [('stimulus', str),
         ('kappa', int),
         ('nfell', int),
         ('stable', int),
         ('dataset', str)])

else:
    tbl = dbtools.Table(DBPATH, "stability")


for filename in DATAPATH.listdir():
    print filename

    model_dp = load(filename)
    model = model_dp['model.csv']

    nfell = model[PARAMS + ['stimulus', 'sample', 'nfell']]
    groups = nfell.groupby(['sigma', 'phi', 'sample'])
    key = (0.0, 0.0, 0)
    if key not in groups.groups:
        print "warning: key %s not in dataset" % (key,)
        continue

    fb = groups.get_group(key).drop(['sigma', 'phi', 'sample'], axis=1)
    fb['stable'] = fb['nfell'] == 0
    fb['dataset'] = str(filename.name)

    tbl.insert(fb.T.to_dict().values())
