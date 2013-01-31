# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# imports
import collections
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pdb
import pickle
import scipy.stats
import os
import time

import cogphysics
import cogphysics.lib.circ as circ
import cogphysics.lib.nplib as npl
import cogphysics.lib.rvs as rvs

import cogphysics.tower.analysis_tools as tat
import cogphysics.tower.mass.model_observer as mo
import cogphysics.tower.mass.learning_analysis_tools as lat

from cogphysics.lib.corr import xcorr, partialcorr

normalize = rvs.util.normalize
weightedSample = rvs.util.weightedSample

pd.set_option('line_width', 195)
LINE = "-"*195

# <codecell>

######################################################################
## Load data
######################################################################

reload(lat)

# human
training, posttest, experiment, queries = lat.load_turk(thresh=1)
hconds = sorted(experiment.keys())
for cond in hconds:
    print "%s: n=%d" % (cond, experiment[cond].shape[0])

# stims
Stims = np.array([
    x.split("~")[0] for x in zip(*experiment[experiment.keys()[0]].columns)[1]])

# model
nthresh0 = 0
nthresh = 0.4
rawipe, ipe_samps, rawtruth, feedback, kappas = lat.process_model_turk(
    Stims, nthresh0, nthresh)
nofeedback = np.empty(feedback.shape[1])*np.nan

# <codecell>

######################################################################
## Global parameters
######################################################################

n_kappas = len(kappas)
kappas = np.array(kappas)
ratios = 10 ** kappas
ratios[kappas < 0] = np.round(ratios[kappas < 0], decimals=2)
ratios[kappas >= 0] = np.round(ratios[kappas >= 0], decimals=1)
ratios = list(ratios)
kappas = list(kappas)

n_trial      = Stims.size
n_fake_data  = 2000

f_smooth = True
p_ignore = 0.0

cmap = lat.make_cmap("lh", (0, 0, 0), (.5, .5, .5), (1, 0, 0))
alpha = 0.2
#colors = ['r', '#AAAA00', 'g', 'c', 'b', 'm']
colors = ["r", "c", "k"]
#colors = cm.hsv(np.round(np.linspace(0, 220, n_cond)).astype('i8'))

plot_ratios = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
plot_ticks = [ratios.index(r) for r in plot_ratios]

# <codecell>

######################################################################
## Generate fake human data
######################################################################

reload(mo)
model_belief = {}
for cond in hconds:

    obstype, group, fbtype, ratio, cb = lat.parse_condition(cond)
    if obstype == "M" or fbtype == "vfb":
	continue

    cols = experiment[cond].columns
    order = np.argsort(zip(*cols)[0])
    undo_order = np.argsort(order)
    #nfake = experiment[cond].shape[0]
    nfake = n_fake_data

    # determine what feedback to give
    if fbtype == 'nfb':
	fb = np.empty((3, n_trial))*np.nan
	prior = np.zeros((3, n_kappas,))
	prior[0, :] = 1 # uniform
	prior[1, kappas.index(-1.0)] = 1 # r=0.1
	prior[2, kappas.index(1.0)] = 1 # r=10
	prior = normalize(np.log(prior), axis=1)[1]
	# prior = None
    else:
	ridx = ratios.index(float(ratio))
	fb = feedback[:, order][ridx]
	prior = None

    responses, model_theta = mo.simulateResponses(
	nfake, fb, ipe_samps[order], kappas, 
	prior=prior, p_ignore=p_ignore, smooth=f_smooth)

    if fbtype == "nfb":
	newcond = "M-%s-%s-0" % (group, fbtype)
	experiment[newcond] = pd.DataFrame(
	    responses[:, 0][:, undo_order], 
	    columns=cols)
	model_belief[newcond] = model_theta[0]
	
	newcond = "M-%s-%s-0.1" % (group, fbtype)
	experiment[newcond] = pd.DataFrame(
	    responses[:, 1][:, undo_order], 
	    columns=cols)
	model_belief[newcond] = model_theta[1]

	newcond = "M-%s-%s-10" % (group, fbtype)
	experiment[newcond] = pd.DataFrame(
	    responses[:, 2][:, undo_order], 
	    columns=cols)
	model_belief[newcond] = model_theta[2]


    else:
	newcond = "M-%s-%s-%s" % (group, fbtype, ratio)
	experiment[newcond] = pd.DataFrame(
	    responses[:, undo_order], 
	    columns=cols)
	model_belief[newcond] = model_theta

    # explicit judgments
    if fbtype == "fb":
	cols = queries[cond].columns
	idx = np.array(cols)-6-np.arange(len(cols))-1
	theta = np.exp(model_theta[idx])
	if float(ratio) > 1:
	    pcorrect = np.sum(theta[:, ratios.index(1.0)+1:], axis=1)
	    other = 0.1
	else:
	    pcorrect = np.sum(theta[:, :ratios.index(1.0)], axis=1)
	    other = 10
	r = np.random.rand(nfake, 1) < pcorrect[None]
	responses = np.empty(r.shape)
	responses[r] = float(ratio)
	responses[~r] = other
	queries[newcond] = pd.DataFrame(
	    responses, columns=cols)
	# print newcond, pcorrect
	

# <codecell>

######################################################################
## Conditions
######################################################################

groups = ["C", "E", "all"]

cond_labels = dict([
    (key % group, value % {'group': group}) 
    for group in groups
    for key, value in {
	    'H-%s-nfb-10': 'No feedback condition',
	    'H-%s-vfb-0.1': 'Video $r_0=0.1$ feedback condition',
	    'H-%s-vfb-10': 'Video $r_0=10$ feedback condition',
	    'H-%s-fb-0.1': 'Text $r_0=0.1$ feedback condition',
	    'H-%s-fb-10': 'Text $r_0=10$ feedback condition',
	    
	    'M-%s-nfb-0': 'Ideal observer w/ fixed uniform belief',
	    'M-%s-nfb-0.1': 'Ideal observer w/ fixed belief that $r_0=0.1$',
	    #'M-%s-nfb-1': 'Ideal observer w/ fixed belief that $r_0=1.0$',
	    'M-%s-nfb-10': 'Ideal observer w/ fixed belief that $r_0=10$',
	    'M-%s-fb-0.1': 'Ideal observer w/ text feedback',
	    'M-%s-fb-10': 'Ideal observer w/ text feedback',
	    }.items()])

conds = [cond % group 
	 for group in groups 
	 for cond in [
		 'H-%s-fb-0.1',
		 'H-%s-vfb-0.1',
		 'M-%s-fb-0.1',
		 
		 'H-%s-fb-10',
		 'H-%s-vfb-10',
		 'M-%s-fb-10',

		 'H-%s-nfb-10',
		 'M-%s-nfb-0',
		 'M-%s-nfb-0.1',
		 #'M-%s-nfb-1',
		 'M-%s-nfb-10',
		 ]]
n_cond = len(conds)

    

# <codecell>

######################################################################
## Compute likelihoods under various models
######################################################################

reload(lat)

ir1 = list(kappas).index(0.0)
ir10 = list(kappas).index(1.0)
ir01 = list(kappas).index(-1.0)

# random model
model_random = lat.CI(lat.random_model_lh(experiment, n_trial), conds)

# fixed models
theta = np.log(np.eye(n_kappas))
fb = np.empty((n_kappas, n_trial))*np.nan
model_fixed = lat.CI(lat.block_lh(
    experiment, fb, ipe_samps, theta, kappas, 
    f_smooth=f_smooth, p_ignore=p_ignore), conds)
# model_true01 = dict([(cond, model_fixed[cond][ir01]) for cond in model_fixed])
model_true1 = dict([(cond, model_fixed[cond][ir1]) for cond in model_fixed])
# model_true10 = dict([(cond, model_fixed[cond][ir10]) for cond in model_fixed])
model_true = {}
for cond in model_fixed:
    obstype, group, fbtype, ratio, cb = lat.parse_condition(cond)
    if ratio != "0":
	model_true[cond] = model_fixed[cond][ratios.index(float(ratio))]
model_uniform = lat.CI(lat.block_lh(
    experiment, nofeedback, ipe_samps, None, kappas,
    f_smooth=f_smooth, p_ignore=p_ignore), conds)
	
# learning models
model_not_fixed = lat.CI(lat.block_lh(
    experiment, feedback, ipe_samps, None, kappas, 
    f_smooth=f_smooth, p_ignore=p_ignore), conds)
model_learn = {}
for cond in model_not_fixed:
    obstype, group, fbtype, ratio, cb = lat.parse_condition(cond)
    if ratio != "0":
	model_learn[cond] = model_not_fixed[cond][ratios.index(float(ratio))]

# model_learn01 = dict([(cond, model_learn[cond][0]) for cond in model_fixed])
# model_learn10 = dict([(cond, model_learn[cond][1]) for cond in model_fixed])

# all the models
mnames = np.array([
	"Random",
	# "fixed 0.1",
	# "learning 0.1",
	# "fixed 10",
	# "learning 10",
	"Uniform",
	"Equal",
	"Correct",
	"Learning",
	# "fixed uniform",
	# "fixed r=1",
	])
mparams = np.array([0, 1, 2, 1, 2, 1, 1])
models = [
    model_random,
    # model_true01,
    # model_learn01,
    # model_true10,
    # model_learn10,
    model_uniform,
    model_true1,
    model_true,
    model_learn,
    ]

# <codecell>

######################################################################
## Plot explicit judgments
######################################################################

reload(lat)

emjs = {}
for cond in conds:
    if cond not in queries:
	continue
    obstype, group, fbtype, ratio, cb = lat.parse_condition(cond)
    emjs[cond] = np.asarray(queries[cond]) == float(ratio)
    index = np.array(queries[cond].columns, dtype='i8')
    X = np.array(index)-6-np.arange(len(index))-1
emj_stats = lat.CI(emjs, conds)

for i, group in enumerate(groups):

    fig = plt.figure(10+i)
    plt.clf()
    idx = 0

    for cidx, cond in enumerate(conds):
	if cond not in emj_stats:
	    continue
	obstype, grp, fbtype, ratio, cb = lat.parse_condition(cond)
	if (grp != group):# or obstype == "M":
	    continue
	mean, lower, upper, sums, n = emj_stats[cond].T
	sem = np.array([np.abs(mean - lower), np.abs(mean-upper)]).T

	# color = colors[(idx/3) % len(colors)]
	# if obstype == "M":
	#     linestyle = '-'
	# elif fbtype == "fb":
	#     linestyle = ':'
	# elif fbtype in ("vfb", "nfb"):
	#     linestyle = '--'

	# if fbtype == "nfb":
	#     plt.subplot(1, 3, 3)
	if ratio == "10":
	    r10ax = plt.subplot(1, 2, 2)
	    r10title = "$r_0=10$"
	elif ratio == "0.1":
	    r01ax = plt.subplot(1, 2, 1)
	    r01title = "$r_0=0.1$"

	if obstype == "M":
	    linestyle = "-"
	elif fbtype == "fb":
	    linestyle = ":"
	elif fbtype == "vfb":
	    linestyle = "--"

	color = 'k'

	# label = "%s %s r=%s (n=%d)" % (
	#     obstype, fbtype, ratio, n[0])
	# plt.errorbar(X, mean, yerr=sem, label=label, 
	# 	     linewidth=2, color=color, linestyle=linestyle)
	plt.fill_between(X, lower, upper, color=color, alpha=0.2)
	plt.plot(X, mean, label=cond_labels[cond], 
		 linewidth=4, color=color, linestyle=linestyle)

	idx += 1

    # arr = np.concatenate(allarr, axis=0)
    # lat.plot_explicit_judgments(idx, arr)

    for ax, title in (r10ax, r10title), (r01ax, r01title):
	ax.set_xlim(X.min(), X.max())
	ax.set_ylim(0.3, 1.0)
	ax.set_xticks(X)
	ax.set_xticklabels(X)
	ax.set_xlabel("Trial")
	if ax == r01ax:
	    ax.set_ylabel("Proportion correct")
	else:
	    ax.set_yticklabels([])
	ax.legend(loc=4, fontsize=11)
	ax.set_title(title)

    title = "Proportion of correct \"Which color is heavier?\" judgments"
    if group != "all":
	title += " (%s)" % group
    plt.suptitle(title, fontsize=16)
    fig = plt.gcf()
    fig.set_figwidth(11)
    fig.set_figheight(4)
    plt.subplots_adjust(top=0.84, wspace=0.1, left=0.07, right=0.95)

    if p_ignore > 0:
	lat.save("images/explicit_mass_judgments_%s_ignore" % group, close=True, ext=['png', 'pdf', 'svg'])
    else:
	lat.save("images/explicit_mass_judgments_%s" % group, close=True, ext=['png', 'pdf', 'svg'])
	    

# <codecell>

######################################################################
## Binomial analysis of explicit judgments
######################################################################

reload(lat)
for i, group in enumerate(groups):
    allarr = []

    for cidx, cond in enumerate(conds):
	if cond not in emj_stats:
	    continue
	obstype, grp, fbtype, ratio, cb = lat.parse_condition(cond)
	if (grp != group):
	    continue
	mean, lower, upper, sums, n = emj_stats[cond].T

	# arr = np.asarray(queries[cond]) == float(ratio)
	binom = [scipy.stats.binom_test(x, n[0], 0.5) for x in sums]

        print cond
        print "  ", np.round(binom, decimals=3)

    # arr = np.concatenate(allarr, axis=0)
    # binom = [scipy.stats.binom_test(x, arr.shape[0], 0.5) 
    # 	     for x in np.sum(arr, axis=0)]
    # print "All %s" % group
    # print "  ", np.round(binom, decimals=3)
    

# <codecell>

######################################################################
## Chi-square analysis of explicit judgments
######################################################################

for i, group in enumerate(groups):
    allarr = []

    for cond in conds:
	if cond not in queries:
	    continue
	obstype, grp, fbtype, ratio, cb = lat.parse_condition(cond)
	if grp != group or obstype == "M":
	    continue
	print cond

	arr = np.asarray(queries[cond]) == float(ratio)
	allarr.append(arr)

    if allarr == []:
	continue

    arr = np.concatenate(allarr, axis=0)
    df = pd.DataFrame(
	np.array([np.sum(1-arr, axis=0), np.sum(arr, axis=0)]).T,
	index=X,
	columns=["incorrect", "correct"])
    print group
    print df

    chi2, p, dof, ex = scipy.stats.chi2_contingency(df)
    print (chi2, p)
    print

# <codecell>

######################################################################
## Plot smoothed likelihoods
######################################################################

reload(lat)
fig = plt.figure(1)
plt.clf()
istim = [
    #list(Stims).index("mass-tower_00035_0110101001"),
    #list(Stims).index("mass-tower_00352_0010011011"),
    list(Stims).index("mass-tower_00357_0111011000"),
    list(Stims).index("mass-tower_00388_0011101001")
    ]
    
print Stims[istim]
lat.plot_smoothing(ipe_samps, Stims, istim, kappas)
plt.xticks([kappas[x] for x in plot_ticks], plot_ratios)
fig.set_figwidth(8)
fig.set_figheight(6)

lat.save("images/likelihood_smoothing", close=True, ext=['png', 'pdf', 'svg'])

# <codecell>

######################################################################
## Plot ideal learning observer beliefs
######################################################################

reload(lat)
beliefs = {}
for cond in conds:
    if cond not in model_belief:
	continue
    obstype, group, fbtype, ratio, cb = lat.parse_condition(cond)
    if obstype == "M" and fbtype not in ("vfb", "nfb"):
	beliefs[cond] = model_belief[cond]

lat.plot_belief(2, 2, 2, beliefs, kappas, cmap, cond_labels)
lat.save("images/ideal_observer_beliefs", close=True, ext=['png', 'pdf', 'svg'])

# <codecell>

######################################################################
## Plot likelihoods under fixed models
######################################################################

reload(lat)

for i, group in enumerate(groups):
    idx = 0
    fig = plt.figure(30+i)
    plt.clf()

    for cidx, cond in enumerate(conds):
	obstype, grp, fbtype, ratio, cb = lat.parse_condition(cond)
	
	if grp != group:
	    continue
	if obstype == "M" and fbtype == "fb":
	    continue

	if obstype == "H" and fbtype == "nfb":
	    nfbax = plt.subplot(1, 3, 3)
	    nfbtitle = "No feedback"
	elif ratio == "10":
	    r10ax = plt.subplot(1, 3, 2)
	    r10title = "Feedback generated w/ $r_0=10$"
	elif ratio == "0.1":
	    r01ax = plt.subplot(1, 3, 1)
	    r01title = "Feedback generated w/ $r_0=0.1$"
	else:
	    nfbax = plt.subplot(1, 3, 3)
	    nfbtitle = "No feedback"

	if obstype == "M":
	    linestyle = "-"
	elif fbtype == "fb":
	    linestyle = ":"
	elif fbtype == "vfb":
	    linestyle = "--"
	elif fbtype == "nfb":
	    linestyle = ":"

	# color = 'k'
	# color = colors[(idx/3) % len(colors)]
	# if obstype == "M":
	#     linestyle = '-'
	# elif fbtype == "fb":
	#     linestyle = ':'
	# else:
	#     linestyle = '--'
	# if obstype == "M" and fbtype == "nfb" and ratio == "0":
	#     color = "#996600"
	# else:
	color = 'k'
	    
	mean, lower, upper, sums, n = model_fixed[cond].T
	# mean = np.exp(mean)
	# lower = np.exp(lower)
	# upper = np.exp(upper)
	plt.fill_between(kappas, lower, upper, color=color, alpha=0.2)
	plt.plot(kappas, mean, label=cond_labels[cond], color=color, linewidth=4,
		 linestyle=linestyle)
	# plt.plot(x, sums[cidx], label=cond_labels[cond], color=color, linewidth=2,
	# 	     linestyle=linestyle)

	idx += 1

    for ax, title in [(nfbax, nfbtitle), (r10ax, r10title), (r01ax, r01title)]:
	for k in (-1.0, 0.0, 1.0):
	    ax.plot([k]*2, [-40, -22], 'k', linestyle=':', alpha=1)

	ax.set_xticks([kappas[x] for x in plot_ticks])
	ax.set_xticklabels(plot_ratios, rotation=30)
	ax.set_xlabel("Assumed mass ratio ($r$)")
	ax.set_xlim(min(kappas), max(kappas))
	ax.set_ylim(-40, -22)
	if ax == r01ax:
	    ax.set_ylabel("Log likelihood of judgments")
	else:
	    ax.set_yticklabels([])
	ax.legend(loc=4, ncol=1, fontsize=12)
	ax.set_title(title)

    title = "Likelihood of \"Will it fall?\" judgments under fixed belief models"
    if group != "all":
	title += " (%s)" % group
    plt.suptitle(title , fontsize=20)
    fig.set_figwidth(16)
    fig.set_figheight(6)

    plt.subplots_adjust(bottom=0.2, top=0.85, wspace=0.1, left=0.07, right=0.95)
	
    if p_ignore > 0:
	lat.save("images/fixed_model_performance_%s_ignore" % group, close=True, ext=['png', 'pdf', 'svg'])
    else:
	lat.save("images/fixed_model_performance_%s" % group, close=True, ext=['png', 'pdf', 'svg'])
	

# <codecell>

######################################################################
## Plot likelihoods under other models
######################################################################

n = 5#n_cond / len(groups)
width = 0.7 / n
x0 = np.arange(len(models))
colors = ["m", "r", "c", "b", "y"]
performance_table = []
table_conds = []

for i, group in enumerate(groups):
    idx = 0
    fig = plt.figure(40+i)
    plt.clf()
    performance_table.append([])
    table_conds.append([])

    #for midx, model in enumerate(mnames):
    for cidx, cond in enumerate(conds):
	# height = []
	# err = []
	    
	# for cidx, cond in enumerate(conds):
	#     obstype, grp, fbtype, ratio, cb = lat.parse_condition(cond)
	#     if (obstype == "M") or (grp != group) or (cond not in models[midx]):
	# 	continue
	#     height.append(models[midx][cond][0])
	#     err.append(models[midx][cond][0] - models[midx][cond][[1,2]])

	obstype, grp, fbtype, ratio, cb = lat.parse_condition(cond)
	if (obstype == "M") or (grp != group):
	    continue

	data = np.array([models[x][cond] for x in xrange(len(models))])
	# height = np.array([models[x][cond][0] for x in xrange(len(models))])
	# lower = np.array([models[x][cond][1] for x in xrange(len(models))])
	# upper = np.array([models[x][cond][2] for x in xrange(len(models))])
	# sums = np.array([models[x][cond][3] for x in xrange(len(models))])
	# ssize = np.array([models[x][cond][4] for x in xrange(len(models))])
	height = data[:, 0]
	lower = data[:, 1]
	upper = data[:, 2]
	# err = np.array([np.abs(models[x][cond][0] - models[x][cond][[1,2]])
	# 		for x in xrange(len(models))])# if cond in models[x]])
	err = np.array([np.abs(height - lower), np.abs(height-upper)]).T
	#performance_table[-1].append([height, lower, upper])
	table_conds[-1].append(cond)

	# performance_table[-1].append(["%.2f [%.2f, %.2f]" % (
	# 	height[x], lower[x], upper[x])
	# 	for x in xrange(height.size)])
	performance_table[-1].append(data.T)

	# height = np.array(height)
	# err = np.array(err)
	#n = height.size
	
	#color = colors[(idx/2) % len(colors)]
	color = colors[idx]
	# if obstype == "M":
	#     alpha = 1.0
	# elif fbtype == "vfb":
	#     alpha = 0.4
	# elif fbtype in ("fb", "nfb"):
	#     alpha = 0.2
	# if obstype == "M" and fbtype == "nfb" and ratio == "1":
	#     color = "#996600"

	x = x0 + width*(idx-(n/2.)) + (width/2.)
	plt.bar(x, height, yerr=err.T, 
		color=color,
		ecolor='k', align='center', width=width, 
		label=cond_labels[cond], alpha=0.7)

	idx += 1

    plt.xticks(x0, mnames)#, rotation=15)
    plt.ylim(-34, -22)
    plt.xlim(x0.min()-0.5, x0.max()+0.5)
    plt.legend(loc=4, ncol=2, fontsize=10)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Log likelihood", fontsize=14)

    title = "Likelihood of \"Will it fall?\" judgments under different models"
    if group != "all":
	title += " (%s)" % group
    plt.suptitle(title, fontsize=16)
    fig.set_figwidth(8)
    fig.set_figheight(5)
    plt.subplots_adjust(bottom=0.18, top = 0.9)

    if p_ignore > 0:
	lat.save("images/model_performance_%s_ignore" % group, close=True, ext=['png', 'pdf', 'svg'])
    else:
	lat.save("images/model_performance_%s" % group, close=True, ext=['png', 'pdf', 'svg'])

performance_table = np.array(performance_table)

# <codecell>

gidx = groups.index('all')
# group, condition, model, stat
table = performance_table[gidx, :, 3]
samplesize = performance_table[gidx, :, 4]
print "& " + " & ".join(mnames) + r"\\\hline"
for lidx, line in enumerate(table):
    midx = np.argmax(np.array(line, dtype='f8'))
    cond = table_conds[gidx][lidx]
    entries = [cond_labels[cond] + " ($n=%d$)" % samplesize[lidx][0]]
    entries.extend([
	(r"\textbf{%d}" if idx == midx else "%d") % float(line[idx])
	for idx in xrange(len(line))])
    print " & ".join(entries) + r"\\"
print "\hline"	    

# table = pd.DataFrame(
#     ptable[gidx],
#     index=[cond_labels[x] + " &" for x in table_conds[gidx]],
#     columns=["& " + x for x in mnames]) + " &"

# <codecell>

print performance_table[gidx, 0, 0]
print performance_table[gidx, 0, 3] / 56.

# <codecell>

reload(lat)
pr = lat.infer_CI(lat.infer_beliefs(
    experiment, ipe_samps, feedback, kappas, f_smooth), conds)

# <codecell>

tidx = np.array([0] + list(X))
T0 = tidx.copy()[:-1]
T1 = tidx.copy()[1:]

for i, group in enumerate(groups):
    idx = 0
    plt.figure(500+i)
    plt.clf()
    plt.figure(600+i)
    plt.clf()
	
    for cidx, cond in enumerate(conds):
	obstype, grp, fbtype, ratio, cb = lat.parse_condition(cond)
	if (grp != group) or (obstype == "M" and fbtype == "nfb" and ratio != "0"):
	    continue
	avgpr = np.mean(pr[cond], axis=0)

	if fbtype != "nfb":
	    if float(ratio) > 1:
		arr = pr[cond][:, :, ratios.index(1.0)+1:]
	    else:
		arr = pr[cond][:, :, :ratios.index(1.0)]
	    pcorrect = np.mean(np.sum(arr, axis=-1), axis=0)
	    
	    plt.figure(600+i)
	    plt.subplot(2, 3, idx+1)
	    plt.plot(T1, emj_stats[cond][:, 0], label="explicit mass judgments")
	    plt.plot(pcorrect, label="P(correct | responses)")
	    plt.title(cond_labels[cond], fontsize=12)
	    plt.ylim(0.3, 1)
	    plt.xticks(T1, T1)
	    plt.xlim(T1.min(), T1.max())

	plt.figure(500+i)
	img = lat.plot_theta(
	    3, 3, idx+1,
	    avgpr,
	    cond_labels[cond], 
	    exp=np.e,
	    vmin=0, vmax=0.1,
	    cmap=cmap)
	
	idx += 1

    plt.colorbar(img)
    plt.suptitle("P(r | J) -- %s" % group)
    fig = plt.gcf()
    fig.set_figwidth(8)
    fig.set_figheight(6)
    lat.save("images/inferred_belief_%s" % group, close=False, ext=['png', 'pdf', 'svg'])

    plt.figure(600+i)
    plt.legend(loc=0, fontsize=9)
    fig = plt.gcf()
    fig.set_figwidth(8)
    fig.set_figheight(6)
    lat.save("images/estimated_mass_judgments_%s" % group, close=False, ext=['png', 'pdf', 'svg'])

# <codecell>

hi

