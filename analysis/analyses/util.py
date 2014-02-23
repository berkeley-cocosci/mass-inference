from path import path
import pandas as pd

from mass.analysis import load_human, load_all#, load_model
# from mental_rotation.analysis import beta, bootcorr, modtheta
# from mental_rotation.analysis import bootstrap_median, bootstrap_mean


# def load_config(pth):
#     config = SafeConfigParser()
#     config.read(pth)
#     return config


def newcommand(name, val):
    cmd = r"\newcommand{\%s}[0]{%s}" % (name, val)
    return cmd + "\n"


report_spearman = "\rho={median:.2f}, 95% CI [{lower:.2f}, {upper:.2f}]"
latex_spearman = r"$\rho={median:.2f}$, 95\% CI $[{lower:.2f}, {upper:.2f}]$"

report_pearson = "r={median:.2f}, 95% CI [{lower:.2f}, {upper:.2f}]"
latex_pearson = r"$r={median:.2f}$, 95\% CI $[{lower:.2f}, {upper:.2f}]$"

report_percent = "M={median:.1f}%, 95% CI [{lower:.1f}%, {upper:.1f}%]"
latex_percent = r"$M={median:.1f}\%$, 95\% CI $[{lower:.1f}\%, {upper:.1f}\%]$"

report_mean = "M={median:.1f}, 95% CI [{lower:.1f}, {upper:.1f}]"
latex_mean = r"$M={median:.1f}$, 95\% CI $[{lower:.1f}, {upper:.1f}]$"


def run_analysis(func):
    version = 'G'
    data_path = path('../../data')
    results_path = path('../../results')
    seed = 923012
    data = load_all(version, data_path)
    pth = func(data, results_path, seed)
    print pth
    if pth.ext == ".csv":
        df = pd.read_csv(pth)
        print df
