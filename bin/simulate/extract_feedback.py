#!/usr/bin/env python

import argparse
import dbtools
import logging
from snippets import datapackage as dpkg
from mass import DATA_PATH, CPO_PATH

logger = logging.getLogger("mass.sims")


def get_table():
    dbpath = CPO_PATH.joinpath("metadata.db")

    if not dbtools.Table.exists(dbpath, "stability"):
        logger.info("Creating new table 'stability'")
        tbl = dbtools.Table.create(
            dbpath, "stability",
            [('stimulus', str),
             ('kappa', int),
             ('nfell', int),
             ('stable', int),
             ('dataset', str)])

    else:
        logger.info("Loading existing table 'stability'")
        tbl = dbtools.Table(dbpath, "stability")

    return tbl


def get_stability(dp_path):

    model_dp = dpkg.DataPackage.load(dp_path)
    model = model_dp.load_resource('model.csv')

    nfell = model[['sigma', 'phi', 'kappa', 'stimulus', 'sample', 'nfell']]
    groups = nfell.groupby(['sigma', 'phi'])
    key = (0.0, 0.0)
    if key not in groups.groups:
        logger.warning("key %s not in dataset %s", key, dp_path.name)
        return None

    fb = (groups.get_group(key)
          .drop(['sigma', 'phi'], axis=1)
          .set_index(['kappa', 'stimulus', 'sample'])
          .unstack('sample')
          .mean(axis=1)
          .reset_index()
          .rename(columns={0: 'nfell'}))

    fb['stable'] = fb['nfell'] == 0

    return fb


def save_stability(exp, tag, force=False):
    dp_path = DATA_PATH.joinpath("model", "%s_%s_fall.dpkg" % (exp, tag))

    # load the stability data
    logger.info("Loading '%s'", dp_path.relpath())
    fb = get_stability(dp_path)
    if fb is None:
        return
    fb['dataset'] = str(dp_path.name)

    # load the table we're saving it to
    tbl = get_table()

    KEY = ['stimulus', 'kappa']
    tbl_dupes = tbl.select(KEY, where=("dataset=?", dp_path.name))
    tbl_dupes['stimulus'] = map(str, tbl_dupes['stimulus'])
    tbl_dupes = set(tbl_dupes.to_records(index=False).tolist())
    fb_dupes = set(fb[KEY].to_records(index=False).tolist())

    # get the unique values and the duplicated values, because we will
    # treat them differently
    unique = fb_dupes.difference(tbl_dupes)
    dupes = fb_dupes.intersection(tbl_dupes)
    fb_idx = fb.set_index(KEY)

    if len(unique) > 0:
        logger.info("Adding %d new items", len(unique))
        tbl.insert(fb_idx.ix[unique].reset_index().T.to_dict().values())

    if len(dupes) > 0 and force:
        logger.info("Updating %d old items", len(dupes))
        for dupe in dupes:
            newvals = fb_idx.ix[[dupe]].reset_index().irow(0).to_dict()
            newvals['kappa'] = float(newvals['kappa'])
            newvals['nfell'] = int(newvals['nfell'])
            newvals['stable'] = bool(newvals['stable'])
            tbl.update(newvals, where=("stimulus=? AND kappa=? AND dataset=?",
                                       dupe + (str(dp_path.name),)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "-e", "--exp",
        required=True,
        help="Experiment version.")
    parser.add_argument(
        "-t", "--tag",
        required=True,
        help="Simulation tag. A short label for this simulation config.")
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        default=False,
        help="Force datapackages to be generated.")

    args = parser.parse_args()
    save_stability(args.exp, args.tag, force=args.force)
