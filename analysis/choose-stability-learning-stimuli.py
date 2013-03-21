# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pickle
import os
import yaml

import model_observer as mo
import analysis_tools as at
from stats_tools import normalize

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

# <markdowncell>

# ## Load Data

# <codecell>

out = at.load_model('mass-prediction-stability', nthresh0=0.0, nthresh=0.4)
rawipe, ipe_samps, rawtruth, feedback, kappas, stimuli = out

n_kappas = len(kappas)
ratios = 10 ** kappas
ratios[kappas < 0] = np.round(ratios[kappas < 0], decimals=2)
ratios[kappas >= 0] = np.round(ratios[kappas >= 0], decimals=1)

# find indices of the ratios that we want
ridx = [int(np.nonzero(ratios==r)[0][0]) for r in stim_ratios]

# <markdowncell>

# ## Compute informativeness
# ### Calculate stimuli fall probabilities

# <codecell>

model_lh = np.log(mo.IPE(feedback, ipe_samps, kappas, True))

# <codecell>

def KL(qi, idx):
    """Compute the KL divergence between qi and pi, where pi is a
    multinomial centered around idx.

    """
    pi = (np.eye(n_kappas) + (1./n_kappas))[idx][:, None]
    for i, ix in enumerate(idx):
	pi[i, :, ix] += nsamps
    pi /= nsamps + 1
    kl = np.sum(np.log(pi / qi)*pi, axis=-1)
    return kl

# <markdowncell>

# ### Order by KL divergence

# <codecell>

# figure out the starting stimulus (minimum entropy=most information)
fb = feedback[ridx]
lh = model_lh[ridx].copy()
p = normalize(lh, axis=-1)[1]
H = KL(np.exp(p), ridx)

order = [np.argmin(np.sum(H**2, axis=0))]
nums = [stimuli[order[0]].split("_")[1]]
    
joint = lh[:, order[0]].copy()
allJoint = [joint.copy()]
allH = [H[:, order[0]]]

T = 48
for t in xrange(T-1):
    # calculate possible posterior values for each stimulus
    p = normalize(joint[:, None] + lh, axis=-1)[1]
    # compute entropies
    H = KL(np.exp(p), ridx)
    # choose stimulus that would result in the lowest entropy, without
    # repeating stimuli
    for s in np.argsort(np.prod(H, axis=0)):
        num = stimuli[s].split("_")[1]
        if (s not in order) and (num not in nums):
            order.append(s)
            nums.append(num)
            allH.append(H[:, s])
            joint += lh[:, order[-1]]
            allJoint.append(joint.copy())
            break

order = np.array(order)
nums = np.array(nums)
allJoint = np.array(allJoint)
allH = np.array(allH)    

# <markdowncell>

# ### Plot model belief, ordered by informativeness

# <codecell>

plt.close('all')
cmap = at.make_cmap("lh", (0, 0, 0), (.55, .55, .55), (1, 1, 1))
for i, ix in enumerate(ridx):
    plt.figure()
    plt.clf()
    plt.suptitle(ratios[ix])
    plt.subplot(1, 2, 1)
    plt.plot(allH[:, i])
    plt.ylim(0, 3)

    at.plot_theta(
	1, 2, 2,
	np.exp(normalize(allJoint[:, i], axis=-1)[1]),
	"",
	exp=np.e,
	cmap=cmap,
	fontsize=14)

# <markdowncell>

# ## Choose stimuli
# ### Experiment stimuli
# 
# Make sure there are (approximately) equal number of stable and
# unstable stimuli with high informativeness.

# <codecell>

# all stimuli that have opposite feedback
exp = list(order[np.nonzero(fb[0, order] != fb[1, order])[0]][:nexp])
# all other stimuli
newexp = order[np.nonzero(fb[0, order] == fb[1, order])[0]]
# number of each stimulus type (stable/unstable)
fbexp = np.array([np.sum(fb[:, exp], axis=1), np.sum(1-fb[:, exp], axis=1)]).T

# add to the list of stimuli from stimuli that have the same feedback,
# until we get a list of nexp stimuli
for i in xrange(len(newexp)):
    newfb = np.array([fb[:, newexp[i]], 1-fb[:, newexp[i]]]).T
    if (fbexp==(nexp/2)).all():
        break
    if len(exp) == nexp:
        break
    if ((fbexp + newfb) > (nexp/2)).any():
        continue
    fbexp += newfb
    exp.append(newexp[i])
for i in xrange(len(newexp)):
    if newexp[i] in exp:
        continue
    if len(exp) == nexp:
        break
    exp.append(newexp[i])

# make sure we actually have nexp stimuli
assert len(exp) == nexp

exp = np.array(exp)
print stimuli[exp]
print zip(["stable", "unstable"], np.sum(fb[:, exp], axis=1))

# <markdowncell>

# ### Mass example stimulus

# <codecell>

# choose a (stable) mass example
mfall = np.nonzero((fb[:, order] == 0).all(axis=0))[0]
assert order[mfall[-1]] not in exp
mass_example = order[mfall[-1]]
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

l = os.path.join(listpath, "stability-example-unstable")
print l
with open(l, "w") as fh:
    lines = "\n".join([stim_sh[unstable_example]])
    fh.write(lines)

# <markdowncell>

# ### Stable example

# <codecell>

l = os.path.join(listpath, "stability-example-stable")
print l
with open(l, "w") as fh:
    lines = "\n".join([stim_sh[stable_example]])
    fh.write(lines)

# <markdowncell>

# ### Training stimuli

# <codecell>

l = os.path.join(listpath, "mass-learning-training")
print l
with open(l, "w") as fh:
    lines = "\n".join(train_stims)
    fh.write(lines)

# <markdowncell>

# ### Mass example

# <codecell>

for i, ix in enumerate(ridx):
    l = os.path.join(listpath, "mass-example~kappa-%s" % kappas[ix])
    print l
    with open(l, "w") as fh:
        lines = "\n".join(["%s~kappa-%s" % (stimuli[mass_example], kappas[ix])])
        fh.write(lines)

# <markdowncell>

# ### Experiment stimuli

# <codecell>

for i, ix in enumerate(ridx):
    exp_stims = ["%s~kappa-%s" % (x, kappas[ix]) for x in np.sort(stimuli[exp])]
    l = os.path.join(listpath, "mass-learning-%s~kappa-%s" % (exp_ver, kappas[ix]))
    print l
    with open(l, "w") as fh:
        lines = "\n".join(exp_stims)
        fh.write(lines)    

# <markdowncell>

# ## Save stimuli metadata
# 
# For example, stability.

# <codecell>

infofile = "mass-learning-%s-stiminfo.csv" % exp_ver
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

