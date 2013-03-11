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
stim_ratios = [0.1, 10]
nsamps = 300
nexp = 40
ntrain = 6

cmap = at.make_cmap("lh", (0, 0, 0), (.5, .5, .5), (1, 0, 0))

# <markdowncell>

# ## Load Data

# <codecell>

out = at.load_model('mass-all', nthresh0=0.0, nthresh=0.4)
rawipe, ipe_samps, rawtruth, feedback, kappas, stimuli = out

n_kappas = len(kappas)
ratios = 10 ** kappas
ratios[kappas < 0] = np.round(ratios[kappas < 0], decimals=2)
ratios[kappas >= 0] = np.round(ratios[kappas >= 0], decimals=1)

# find indices of the ratios that we want
ridx = [int(np.nonzero(ratios==r)[0][0]) for r in stim_ratios]
r1 = list(ratios).index(1.0)

# numbers for stimuli
nums = np.array([x.split("_")[1] for x in stimuli])

# <codecell>

model_lh = np.log(mo.IPE(feedback, ipe_samps, kappas, smooth=False))

# <codecell>

fb = feedback[ridx]
lh = np.exp(model_lh[ridx].copy())
p = normalize(lh, axis=-1)[1]

# <codecell>

lowsl = slice(None, r1)
highsl = slice(r1 + 1, None)
# lowsl = slice(ridx[0], ridx[0] + 1)
# highsl = slice(ridx[1], ridx[1] + 1)

r1_lh = lh[0, :, r1] / 2.

# likelihood for r < 1 when r0 = 0.1
low01_lh = lh[0, :, lowsl].sum(axis=-1) + r1_lh
# likelihood for r > 1 when r0 = 0.1
high01_lh = lh[0, :, highsl].sum(axis=-1) + r1_lh
# likelihood ratio for r0 = 0.1
lhr01 = np.log(low01_lh / high01_lh)

# likelihood for r < 1 when r0 = 10
low10_lh = lh[1, :, lowsl].sum(axis=-1) + r1_lh
# likelihood for r > 1 when r0 = 10
high10_lh = lh[1, :, highsl].sum(axis=-1) + r1_lh
# likelihood ratio for r0 = 10
lhr10 = np.log(high10_lh / low10_lh)

# <codecell>

fb_00 = (fb == np.array([0, 0])[:, None]).all(axis=0)
fb_01 = (fb == np.array([0, 1])[:, None]).all(axis=0)
fb_10 = (fb == np.array([1, 0])[:, None]).all(axis=0)
fb_11 = (fb == np.array([1, 1])[:, None]).all(axis=0)

# <codecell>

same = fb_00 | fb_11
diff = fb_01 | fb_10
plt.plot(lhr01[same], lhr10[same], 'b.', alpha=0.5, label='same feedback')
plt.plot(lhr01[diff], lhr10[diff], 'r.', alpha=0.5, label='different feedback')
plt.xlabel("Log diagnosticity when $r_0=0.1$")
plt.ylabel("Log diagnosticity when $r_0=10$")
plt.title("Stimuli log diagnosticities")
plt.legend()

# <codecell>

same = fb_00 | fb_11# & informative
diff = fb_01 | fb_10# & informative

skew = np.abs(lhr01 - lhr10)
skewthresh = 0.1

# score = np.sqrt(lhr01**2 + lhr10**2)
# score[np.sign(lhr01) != np.sign(lhr10)] = np.nan
# score *= np.sign(lhr01)
# score = np.sqrt(lhr01**2 + lhr10**2)
# score[lhr10 < -lhr01] *= -1
score = (lhr01 + lhr10) / 2.

notnan = ~np.isnan(score)
ok = (skew <= skewthresh) & diff & notnan

plt.plot(lhr01[ok], lhr10[ok], 'r.', alpha=0.5)

lim = 3
plt.plot(np.linspace(-lim, lim, 10), np.linspace(-lim, lim, 10), 'k-')
plt.xlabel("Log diagnosticity when $r_0=0.1$")
plt.ylabel("Log diagnosticity when $r_0=10$")
plt.title("Approximately equivalently diagnostic stimuli")

# <codecell>

kwargs = {
    'range': [-2.5, 2.5], 
    'bins': 51, 
    'normed': False,
    'alpha': 0.5,
    }

plt.hist(score[ok & fb_01], label="no fall/fall", **kwargs)
plt.hist(score[ok & fb_10], label="fall/no fall", **kwargs)
plt.xlabel("Log diagnosticity")
plt.ylabel("Number of stimuli")
plt.title("Histogram of log diagnosticities")
plt.legend()

# <codecell>

nstim = 10
target_scores = np.linspace(score[ok].min(), score[ok].max(), nstim)
print "target scores:", target_scores
scorediff = np.abs(score[:, None] - target_scores)
scorediff[~ok] = np.inf
exp = []
for sidx in xrange(nstim):
    best = np.argmin(scorediff[:, sidx], axis=0)
    exp.append(best)
    print stimuli[best], fb[:, best], score[best], scorediff[best, sidx]
    bad = nums == nums[best]
    scorediff[bad] = np.inf
	

# <codecell>

num_each_fb = np.sum(fb[:, exp], axis=1)
more = np.max(num_each_fb) - np.min(num_each_fb)

while more > 0:
    # the feedback we have too much of and too little of
    i = np.argmax(num_each_fb)
    bad_fb = np.array([1-i, i])
    good_fb = np.array([i, 1-i])
    # multiplication mask for stimuli that have the feedback we want
    good = (fb == good_fb[:, None]).all(axis=0).astype('f8')
    good[good==0] = np.inf
    # indices of target the scores corresponding to stimuli that have
    # the feedback we don't want
    badidx = np.nonzero((fb[:, exp] == bad_fb[:, None]).all(axis=0))[0]
    # find stimuli with desired feedback that are closest to each of
    # the target scores with bad feedback
    goodstim = np.argmin(scorediff[:, badidx]*good[:, None], axis=0)
    # choose the best stimulus out of these
    argbest = badidx[np.argmin(scorediff[goodstim, badidx])]
    best = goodstim[argbest]
    exp[argbest] = best
    
    print stimuli[best], fb[:, best], score[best], scorediff[best, argbest]
    bad = nums == nums[best]
    scorediff[bad] = np.inf

    num_each_fb = np.sum(fb[:, exp], axis=1)
    more = np.max(num_each_fb) - np.min(num_each_fb)

print "num each:", num_each_fb
assert num_each_fb[0] == num_each_fb[1]

# <codecell>

low = []
high = []

for sidx in xrange(nstim):
    model_joint, model_theta = mo.ModelObserver(
	fb[:, [exp[sidx]]],
	ipe_samps[[exp[sidx]]],
	kappas,
	prior=None, 
	p_ignore=0, 
	smooth=False)

    r1theta = np.exp(model_theta[:, 1, r1]) / 2.
    lowtheta = np.exp(model_theta[:, 1, lowsl]).sum(axis=1) + r1theta
    hightheta = np.exp(model_theta[:, 1, highsl]).sum(axis=1) + r1theta
    Z = lowtheta + hightheta
    low.append((lowtheta / Z)[0])
    high.append((hightheta / Z)[1])

plt.clf()
plt.plot(score[exp], low, label="r=0.1, p(r<1)")
plt.plot(score[exp], high, label="r=10, p(r>1)")
plt.title("Model probability of correct ratio")
plt.xlabel("Score")
plt.ylim(0.3, 0.8)
plt.legend(loc=0)


# <codecell>

at.plot_smoothing(ipe_samps, stimuli, exp, kappas)
fig = plt.gcf()
fig.set_figwidth(8)
fig.set_figheight(6)

# <markdowncell>

# ### Mass example stimulus

# <codecell>

# choose a (stable) mass example
goodex = np.nonzero((lh[:, :, [0]] == lh).all(axis=-1).all(axis=0) & fb_00)[0]
goodex = [x for x in goodex if x not in exp]
mass_example = goodex[0]
print stimuli[mass_example]

# <markdowncell>

# ### Used stimuli numbers

# <codecell>

used = list(stimuli[exp]) + [stimuli[mass_example]]
used_nums = [x.split("_")[1] for x in used]

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
ok = np.array([x not in used_nums for x in sh_nums])

tstable = ~fb_sh[0] & ok
tunstable = fb_sh[0] & ok

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
for i, ix in enumerate(ridx):
    l = os.path.join(listpath, "%s~kappa-%s" % (name, kappas[ix]))
    print l
    with open(l, "w") as fh:
        lines = "\n".join(["%s~kappa-%s" % (stimuli[mass_example], kappas[ix])])
        fh.write(lines)
copy_stims(name, "tower_mass_all")

# <markdowncell>

# ### Experiment stimuli

# <codecell>

name = "mass-oneshot-%s" % exp_ver
for i, ix in enumerate(ridx):
    l = os.path.join(listpath, "%s~kappa-%s" % (name, kappas[ix]))
    print l
    exp_stims = ["%s~kappa-%s" % (x, kappas[ix]) for x in np.sort(stimuli[exp])]
    with open(l, "w") as fh:
        lines = "\n".join(exp_stims)
        fh.write(lines)
copy_stims(name, "tower_mass_all")

# <markdowncell>

# ## Save stimuli metadata
# 
# For example, stability.

# <codecell>

infofile = "mass-oneshot-%s-stiminfo.csv" % exp_ver
fh = open(os.path.join(confdir, infofile), "w")
fh.write("stimulus,stable,catch\n")
for i in exp:
    for ix in ridx:
        fh.write(",".join(
            ["%s~kappa-%s_cb-0" % (stimuli[i], kappas[ix]),
             str(not(bool(feedback[ix,i]))),
             str(False)]
             ) + "\n")
        fh.write(",".join(
            ["%s~kappa-%s_cb-1" % (stimuli[i], kappas[ix]),
             str(not(bool(feedback[ix,i]))),
             str(False)]
             ) + "\n")

for i in train:
    fh.write(",".join(
        [stim_sh[i],
         str(not(bool(fb_sh[0, i]))),
         str(False)]
         ) + "\n")

fh.close()

