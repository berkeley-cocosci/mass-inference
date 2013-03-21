# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pickle
import os
import yaml
import sys
import shutil

import model_observer as mo
import analysis_tools as at
from stats_tools import normalize
from cp_helper import copy_stims
import stats_tools as st

# <markdowncell>

# ## Configuration

# <codecell>

listpath = "../stimuli/lists"
confdir = "../stimuli/meta"
exp_ver = "F"
stim_ratio = 10
nsamps = 300
nexp = 20
ntrain = 6

cmap = at.make_cmap("lh", (0, 0, 0), (.5, .5, .5), (1, 0, 0))
rso = np.random.RandomState(23)

# <markdowncell>

# ### Colors

# <codecell>

colors = {
    'red': '#FF0033',
    'orange': '#FF7000',
    'yellow': '#FFFF00',
    'green': '#00FF00',
    'blue': '#0033FF',
    'magenta': '#FF00FF'
    }

blacklist = [
    ('red', 'green'),
    ('red', 'magenta'),
    ('red', 'orange'),
    ('orange', 'yellow'),
    ('yellow', 'green'),
    ('orange', 'magenta'),
    ('orange', 'green'),
    ]
blacklist = [tuple(sorted(b)) for b in blacklist]

# <markdowncell>

# ## Load Data

# <codecell>

reload(at)
out = at.load_model('mass-all', nthresh0=(0.0, 0.2), nthresh=0.4)
rawipe, ipe_samps, rawtruth, feedback, kappas, stimuli = out

n_kappas = len(kappas)
ratios = 10 ** kappas
ratios[kappas < 0] = np.round(ratios[kappas < 0], decimals=2)
ratios[kappas >= 0] = np.round(ratios[kappas >= 0], decimals=1)

# find indices of the ratios that we want
ridx = int(np.nonzero(ratios==stim_ratio)[0][0])
r1 = list(ratios).index(1.0)

# numbers for stimuli
nums = np.array([x.split("_")[1] for x in stimuli])

# <codecell>

model_lh = mo.IPE(feedback, ipe_samps, kappas, smooth=False)
model_prior = np.log(np.ones(n_kappas) / n_kappas)[None]
model_prior[:, r1] = -5
model_prior = normalize(model_prior, axis=1)[1]
model_joint = model_prior + np.log(model_lh)

# <codecell>

fb = feedback[ridx]
lh = model_lh[ridx].copy()
pfell1 = lh * fb[..., None]
pfell0 = (1-lh) * (1-fb[..., None])
pfell = pfell0 + pfell1
prior = np.exp(model_prior)
post = np.exp(normalize(model_joint, axis=-1)[1])[ridx]

# <codecell>

hyps = [
    slice(None, r1),
    slice(r1 + 1, None)
    ]

hyp_lh = np.empty((lh.shape[0], len(hyps),))
for hi, h in enumerate(hyps):
    hyp_lh[:, hi] = lh[:, h].sum(axis=-1)

lhr = (np.log(hyp_lh[:, 1] / hyp_lh[:, 0]))

# <codecell>

# compute information gain
I0 = np.sum(prior * np.log(1. / prior), axis=-1)
I1 = np.sum(post * np.log(1. / post), axis=-1)
gain = I0 - I1
gain[gain < 0] = np.nan

# <codecell>

ok = ~np.isnan(lhr) & ~np.isnan(gain)
print np.sum(ok)

# <codecell>

lhrsort = np.sort(lhr[ok])
plt.plot(lhrsort)
plt.xlim(0, lhrsort.size)
plt.xlabel("Stimulus")
plt.ylabel("Log likelihood ratio $r_0>1/r_0<1$")
plt.title("Stimuli likelihood ratios")

# <codecell>

gainsort = np.sort(gain[ok])
plt.plot(gainsort)
plt.xlim(0, gainsort.size)
plt.xlabel("Stimulus")
plt.ylabel("Information gain from feedback w/ $r_0=10$")
plt.title("Stimuli information content")

# <codecell>

plt.plot(lhr[ok], gain[ok], '.', alpha=0.5)
x = np.linspace(-3, 3, 100)
f = lambda x: (x**2) / 10.
plt.plot(x, f(x), 'r-')
plt.xlabel("Log likelihood ratio $r_0>1/r_0<1$")
plt.ylabel("Information gain")
plt.title("Likelihood and information when $r_0=10$")
plt.xlim(-3, 3)
plt.ylim(0, 0.6)

# <codecell>

target_gain = f(lhr)
#gaindiff = (gain - target_gain)**2
#good = gaindiff < 0.005
good = np.ones(gain.shape, dtype='bool')
print np.sum(good & ok)
plt.plot(lhr[ok & good], gain[ok & good], '.')
plt.plot(x, f(x), 'r-')
plt.xlabel("Log likelihood ratio")
plt.ylabel("Information gain")

# <codecell>

guess_err = np.abs(np.sum(prior*pfell, axis=1) - fb)

# <codecell>

lowest = lhr[ok & good].min()
highest = lhr[ok & good].max()
lowedge = np.sort(lhr[ok & good])[50]
highedge = np.sort(lhr[ok & good])[-50]
edges = np.linspace(lowedge, highedge, (nexp/2)-1)
# binsize = (np.ptp([lowedge, highedge]) / (nexp-2)) / 2.
# if (np.abs(edges) < binsize).any():
#     val = edges[np.nonzero(np.abs(edges) < binsize)]
#     edges -= (binsize - np.abs(val))
bins = np.hstack([
    [lowest],
    edges,
    [highest+0.001]])
inbins = (lhr[:, None] >= bins[:-1]) & (lhr[:, None] < bins[1:])
print np.sum(inbins, axis=0)

# <codecell>

exp = np.empty((2, nexp/2), dtype='i8')
nummask = np.ones(lhr.shape, dtype='bool')
for bi in xrange(nexp/2):
    for fbtype in rso.permutation(2):
	inbin = (lhr >= bins[bi]) & (lhr < bins[bi+1])
	goodfb = fb == fbtype
	mask = (ok & good & inbin & nummask & goodfb).astype('f8')
	assert np.sum(mask) > 0
	mask[mask==0] = np.nan
	score = guess_err * mask
	argbest = np.nanargmin(score)
	best = score[argbest]
	assert not np.isnan(best)
	print nums[argbest], fb[argbest], best
	exp[fbtype, bi] = argbest
	num = nums[argbest]
	nummask[nums == num] = False
exp = exp.T.ravel()
expF0 = exp[np.nonzero(fb[exp] == 0)[0]]
expF1 = exp[np.nonzero(fb[exp] == 1)[0]]

# <codecell>

plt.plot(lhr[expF0], gain[expF0], 'bo', label='F=0')
plt.plot(lhr[expF1], gain[expF1], 'ro', label='F=1')
plt.xlabel("Likelihood ratio")
plt.ylabel("Information gain")
plt.title("Experimental stimuli")
plt.legend(loc=0)

# <codecell>

cnames = sorted(colors.keys())
pairs = [tuple(sorted((ci, cj))) for ci in cnames for cj in cnames if ci != cj]
pairs = sorted(set(pairs))
pairs = np.array([p for p in pairs if p not in blacklist])
color_pairs = []
for i in xrange(nexp+1):
    pair = pairs[i % len(pairs)]
    color_pairs.append(pair[rso.permutation(2)])
color_pairs_hr = np.array(color_pairs)
rso.shuffle(color_pairs_hr)
print color_pairs_hr
color_pairs = np.array([[colors[p] for p in pair] for pair in color_pairs_hr])
example_color_pair_hr = color_pairs_hr[0]
example_color_pair = color_pairs[0]
color_pairs_hr = color_pairs_hr[1:]
color_pairs = color_pairs[1:]

# <codecell>

angles = rso.randint(0, 360, nexp + ntrain + 3)
angles

# <codecell>

for sidx, best in enumerate(exp):
    print stimuli[best], fb[best]
    print "\t lhr   = %s" % lhr[best]
    print "\t gain  = %s" % gain[best]
    print "\t err   = %s" % guess_err[best]
    # print "\t light = %s" % color_pairs_hr[sidx, 0]
    # print "\t heavy = %s" % color_pairs_hr[sidx, 1]
    print

# <codecell>

highF0 = []
highF1 = []

for sidx in xrange(nexp):
    model_joint, model_theta = mo.ModelObserver(
	fb[[exp[sidx]]],
	ipe_samps[[exp[sidx]]],
	kappas,
	prior=None, 
	p_ignore=0, 
	smooth=False)
    r1theta = np.exp(model_theta[1, r1]) / 2.
    lowtheta = np.exp(model_theta[1, hyps[0]]).sum() + r1theta
    hightheta = np.exp(model_theta[1, hyps[1]]).sum() + r1theta
    Z = lowtheta + hightheta
    high = hightheta / Z
    if fb[exp[sidx]] == 0:
	highF0.append(high)
    else:
	highF1.append(high)

plt.clf()
plt.subplot(1, 2, 1)
plt.plot(lhr[expF0], highF0, 'b-', label="stable")
plt.plot(lhr[expF0], highF0, 'bo')
plt.plot(lhr[expF1], highF1, 'r-', label="unstable")
plt.plot(lhr[expF1], highF1, 'ro')
plt.xlabel("Log likelihood ratio")
plt.ylabel("Probability of correct ratio")
plt.ylim(0.0, 1.0)
plt.legend(loc=4)

plt.subplot(1, 2, 2)
plt.plot(gain[expF0], highF0, 'b-', label="stable")
plt.plot(gain[expF0], highF0, 'bo')
plt.plot(gain[expF1], highF1, 'r-', label="unstable")
plt.plot(gain[expF1], highF1, 'ro')
plt.xlabel("Information gain")
plt.ylabel("Probability of correct ratio")
plt.ylim(0.0, 1.0)
plt.legend(loc=4)

plt.suptitle("Effect of log likelihood and information on model mass judgments",
	     fontsize=16)

fig = plt.gcf()
fig.set_figwidth(10)
fig.set_figheight(4)

at.savefig("images/stimuli-metrics.png", close=False)


# <codecell>

i = 0
at.plot_smoothing(ipe_samps, stimuli, [exp[i]], kappas)
print stimuli[exp[i]], fb[exp[i]], lhr[exp[i]], guess_err[exp[i]]
fig = plt.gcf()
fig.set_figwidth(6)
fig.set_figheight(4)

# <markdowncell>

# ### Mass example stimulus

# <codecell>

# choose a (stable) mass example
goodex = np.nonzero((lh[:, [0]] == lh).all(axis=1) & (fb == 0))[0]
goodex = [x for x in goodex if nums[x] not in list(nums[exp])]
mass_example = goodex[np.argmin(lh[goodex, 0])]
print stimuli[mass_example], lh[mass_example]

# <markdowncell>

# ### Used stimuli numbers

# <codecell>

used_nums = list(nums[exp]) + [nums[mass_example]]

# <markdowncell>

# ### Training stimuli
# 
# We want to pick training towers that are half and half, and are really
# obvious to people (based on previous sameheight experiment results).

# <codecell>

# load sameheight model and human data
reload(at)
model_sh = at.load_model('stability-sameheight')
rawipe_sh, ipe_sh, rawtruth_sh, fb_sh, kappas_sh, sstimuli = model_sh
human, rawhuman_sh, stim_sh = at.load_old_human('stability-sameheight')

sh_nums = [x[len('stability'):] for x in stim_sh]
oknums = np.array([x not in used_nums for x in sh_nums])

tstable = (fb_sh[0] == 0) & oknums
tunstable = (fb_sh[0] == 1) & oknums

hstable = 1 - human.copy()
hsort = np.argsort(hstable)
unstable = hsort[np.nonzero(tunstable[hsort])[0]]
stable = hsort[np.nonzero(tstable[hsort])[0]][::-1]

# <codecell>

# examples
stable_example = stable[0]
unstable_example = unstable[0]
print stim_sh[stable_example]
print stim_sh[unstable_example]

# <codecell>

# training stimuli
train = np.hstack([
    unstable[1:(ntrain/2)+1],
    stable[1:(ntrain/2)+1]])
train_stims = np.sort(stim_sh[train])
print train_stims

# <markdowncell>

# ## Save stimuli lists to file
# ### Unstable example

# <codecell>

name = "stability-example-unstable-%s" % exp_ver
l = os.path.join(listpath, name)
print l
with open(l, "w") as fh:
    lines = "\n".join([stim_sh[unstable_example]])
    fh.write(lines)
copy_stims(name, "tower_originalSH")

# <markdowncell>

# ### Stable example

# <codecell>

name = "stability-example-stable-%s" % exp_ver
l = os.path.join(listpath, name)
print l
with open(l, "w") as fh:
    lines = "\n".join([stim_sh[stable_example]])
    fh.write(lines)
copy_stims(name, "tower_originalSH")

# <markdowncell>

# ### Training stimuli

# <codecell>

name = "mass-oneshot-training-%s" % exp_ver
l = os.path.join(listpath, name)
print l
with open(l, "w") as fh:
    lines = "\n".join(train_stims)
    fh.write(lines)
copy_stims(name, "tower_originalSH")

# <markdowncell>

# ### Mass example

# <codecell>

name = "mass-oneshot-example-%s" % exp_ver
l = os.path.join(listpath, "%s~kappa-%s" % (name, kappas[ridx]))
print l
with open(l, "w") as fh:
    lines = "\n".join(["%s~kappa-%s" % (stimuli[mass_example], kappas[ridx])])
    fh.write(lines)
copy_stims(name, "tower_mass_all")

# <markdowncell>

# ### Experiment stimuli

# <codecell>

name = "mass-oneshot-%s" % exp_ver
l = os.path.join(listpath, "%s~kappa-%s" % (name, kappas[ridx]))
print l
exp_stims = ["%s~kappa-%s" % (x, kappas[ridx]) for x in stimuli[exp]]
with open(l, "w") as fh:
    lines = "\n".join(exp_stims)
    fh.write(lines)
copy_stims(name, "tower_mass_all")

# <markdowncell>

# ## Save stimuli metadata
# 
# For example, stability.

# <codecell>

infodict = {}

# unstable example 
infodict[stim_sh[unstable_example]] = {
    'angle': angles[0],
    'stable': not(bool(fb_sh[0, unstable_example])),
    'full': True,
    'color0': None,
    'color1': None
    }

# stable example
infodict[stim_sh[stable_example]] = {
    'angle': angles[1],
    'stable': not(bool(fb_sh[0, stable_example])),
    'full': True,
    'color0': None,
    'color1': None
    }

# training
for k, i in enumerate(train):
    infodict[stim_sh[i]] = {
	'angle': angles[2+k],
	'stable': not(bool(fb_sh[0, i])),
	'full': False,
	'color0': None,
	'color1': None
	}

# mass example
infodict["%s~kappa-%s_cb-0" % (stimuli[mass_example], kappas[ridx])] = {
	 'angle': angles[2+ntrain],
	 'stable': not(bool(feedback[ridx, mass_example])),
	 'full': True,
	 'color0': example_color_pair[0],
	 'color1': example_color_pair[1]
	 }
infodict["%s~kappa-%s_cb-1" % (stimuli[mass_example], kappas[ridx])] = {
	 'angle': angles[2+ntrain],
	 'stable': not(bool(feedback[ridx, mass_example])),
	 'full': True,
	 'color0': example_color_pair[0],
	 'color1': example_color_pair[1]
	 }

# experiment
for k, i in enumerate(exp):
    infodict["%s~kappa-%s_cb-0" % (stimuli[i], kappas[ridx])] = {
	'angle': angles[3+ntrain+k],
	'stable': not(bool(feedback[ridx, i])),
	'full': False,
	'color0': color_pairs[k, 0],
	'color1': color_pairs[k, 1]
	}
    infodict["%s~kappa-%s_cb-1" % (stimuli[i], kappas[ridx])] = {
	'angle': angles[3+ntrain+k],
	'stable': not(bool(feedback[ridx, i])),
	'full': False,
	'color0': color_pairs[k, 0],
	'color1': color_pairs[k, 1]
	}

# write to file
infofile = os.path.join(confdir, "%s-rendering-info.pkl" % exp_ver)
with open(infofile, 'w') as fh:
    pickle.dump(infodict, fh)
    

# <codecell>

infodict = {}
    
# experiment
for k, i in enumerate(exp):
    infodict["%s~kappa-%s" % (stimuli[i], kappas[ridx])] = {
	'angle': angles[3+ntrain+k],
	'stable': not(bool(feedback[ridx, i])),
	'full': False,
	'color0': colors['red'],
	'color1': colors['blue']
	}

infofile = os.path.join(confdir, "%s-demo-rendering-info.pkl" % exp_ver)
with open(infofile, 'w') as fh:
    pickle.dump(infodict, fh)

