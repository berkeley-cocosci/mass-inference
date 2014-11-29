from ConfigParser import SafeConfigParser
from argparse import ArgumentParser, RawTextHelpFormatter
from path import path

import os


def newcommand(name, val):
    cmd = r"\newcommand{\%s}[0]{%s}" % (name, val)
    return cmd + "\n"


def default_argparser(doc):
    root = path("..")
    config = SafeConfigParser()
    config.read(root.joinpath("config.ini"))
    results_path = os.path.abspath(root.joinpath(
        config.get("analysis", "results_path")))

    parser = ArgumentParser(description=doc, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        'latex_path', help='where to save out the latex file')
    parser.add_argument(
        '-r', '--results-path',
        help='directory where the csv results are located\ndefault: %(default)s',
        default=results_path)

    return parser


latex_spearman = r"\rho={median:.2f}\textrm{{, 95\% CI }}[{lower:.2f}, {upper:.2f}]"
latex_pearson = r"r={median:.2f}\textrm{{, 95\% CI }}[{lower:.2f}, {upper:.2f}]"
latex_percent = r"M={median:.1f}\%\textrm{{, 95\% CI }}[{lower:.1f}\%, {upper:.1f}\%]"
latex_mean = r"M={median:.1f}\textrm{{, 95\% CI }}[{lower:.1f}, {upper:.1f}]"
latex_gamma = r"\gamma={median:.2f}\textrm{{, 95\% CI }}[{lower:.2f}, {upper:.2f}]"
