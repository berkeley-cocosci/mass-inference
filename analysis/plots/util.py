import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
import numpy as np

from argparse import ArgumentParser, RawTextHelpFormatter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sns.set_style('white')

from matplotlib import rcParams
rcParams['font.family'] = 'Helvetica Neue'
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 10
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['font.size'] = 9


def load_config():
    with open(os.path.join(ROOT, "config.json"), "r") as fh:
        config = json.load(fh)
    return config


def get_query():
    return load_config()["analysis"]["query"]


def grayify(colors):
    """Based on grayify function from:

    https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/

    """
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(np.array(colors)[:, :3] ** 2, RGB_weight))
    return [str(x) for x in luminance]


def default_argparser(module):
    # switch matplotlib backends because we don't need to be interactive here
    plt.switch_backend('AGG')

    config = load_config()

    name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    dest = os.path.join(ROOT, config["paths"]["figures"], name)
    depends = ["RESULTS_PATH/{}".format(x) for x in module['__depends__']]

    if len(depends) > 0:
        description = "{}\n\nDependencies:\n\n    {}".format(
            module['__doc__'], "\n    ".join(depends))
    else:
        description = module['__doc__']

    parser = ArgumentParser(
        description=description,
        formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--to',
        nargs="*",
        default=["{}.pdf".format(dest), "{}.png".format(dest)],
        help='where to save out the results (accepts multiple values)\ndefault: %(default)s')

    parser.add_argument(
        '--results-path',
        default=os.path.join(ROOT, config["paths"]["results"]),
        help='where other results are saved\ndefault: %(default)s')

    return parser


def save(path, fignum=None, close=True, width=None, height=None,
         ext=None, dpi=None, verbose=False):
    """Save a figure from pyplot.

    Parameters:

    path [string] : The path (and filename, without the extension) to
    save the figure to.

    fignum [integer] : The id of the figure to save. If None, saves the
    current figure.

    close [boolean] : (default=True) Whether to close the figure after
    saving.  If you want to save the figure multiple times (e.g., to
    multiple formats), you should NOT close it in between saves or you
    will have to re-plot it.

    width [number] : The width that the figure should be saved with. If
    None, the current width is used.

    height [number] : The height that the figure should be saved with. If
    None, the current height is used.

    ext [string or list of strings] : (default='png') The file
    extension. This must be supported by the active matplotlib backend
    (see matplotlib.backends module).  Most backends support 'png',
    'pdf', 'ps', 'eps', and 'svg'.

    verbose [boolean] : (default=True) Whether to print information
    about when and where the image has been saved.

    """

    # get the figure
    if fignum is not None:
        fig = plt.figure(fignum)
    else:
        fig = plt.gcf()

    # set its dimenions
    if width:
        fig.set_figwidth(width)
    if height:
        fig.set_figheight(height)

    # get the dpi
    if dpi is None:
        dpi = load_config()["plots"]["dpi"]

    # make sure we have a list of extensions
    if ext is not None and not hasattr(ext, '__iter__'):
        ext = [ext]

    # Extract the directory and filename from the given path
    directory, basename = os.path.split(path)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # infer the extension if ext is None
    if ext is None:
        basename, ex = os.path.splitext(basename)
        ext = [ex[1:]]

    for ex in ext:
        # The final path to save to
        filename = "%s.%s" % (basename, ex)
        savepath = os.path.join(directory, filename)

        if verbose:
            sys.stdout.write("Saving figure to '%s'..." % savepath)

        # Actually save the figure
        plt.savefig(savepath, dpi=dpi)

    # Close it
    if close:
        plt.close()

    if verbose:
        sys.stdout.write("Done\n")


def notfilled_errorbar(ax, x, y, xerr, yerr, color, **kwargs):
    ax.errorbar(
        x, y,
        xerr=xerr, yerr=yerr,
        linestyle='',
        color=color,
        markerfacecolor='white',
        markeredgecolor=color,
        markeredgewidth=0,
        **kwargs)

    if 'zorder' in kwargs:
        kwargs['zorder'] += 1

    ax.plot(
        x, y,
        linestyle='',
        color=color,
        markerfacecolor='white',
        markeredgecolor=color,
        markeredgewidth=1,
        **kwargs)

