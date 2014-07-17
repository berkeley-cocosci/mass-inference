from ConfigParser import SafeConfigParser
from path import path


def newcommand(name, val):
    cmd = r"\newcommand{\%s}[0]{%s}" % (name, val)
    return cmd + "\n"


def run_analysis(func, filename):
    root = path("..")
    config = SafeConfigParser()
    config.read(root.joinpath("config.ini"))
    results_path = root.joinpath(
        config.get("analysis", "results_path"))
    func(path(filename), results_path)


latex_spearman = r"$\rho={median:.2f}$, 95\% CI $[{lower:.2f}, {upper:.2f}]$"
latex_pearson = r"$r={median:.2f}$, 95\% CI $[{lower:.2f}, {upper:.2f}]$"
latex_percent = r"$M={median:.1f}\%$, 95\% CI $[{lower:.1f}\%, {upper:.1f}\%]$"
latex_mean = r"$M={median:.1f}$, 95\% CI $[{lower:.1f}, {upper:.1f}]$"
