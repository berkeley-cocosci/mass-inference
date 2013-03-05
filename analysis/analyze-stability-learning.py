# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Mass Learning Experiment Analysis

# <codecell>

import numpy as np
import pandas as pd
import scipy.stats

from stats_tools import normalize
import analysis_tools as at
import model_observer as mo

# <markdowncell>

# ## Load Data
# ### Human Data

# <codecell>

# conditions and suffixes we want to load
conds = ['C-vfb-10', 'C-vfb-0.1', 'C-fb-10', 'C-fb-0.1', 'C-nfb-10',
         'E-vfb-10', 'E-vfb-0.1', 'E-fb-10', 'E-fb-0.1', 'E-nfb-10']
suffixes = ['-cb0', '-cb1']

# load the data
training, posttest, experiment, queries = at.load_turk(conds, suffixes, thresh=1)

# <markdowncell>

# #### Number of participants in each condition

# <codecell>

# print out information about the number of participants
hconds = sorted(experiment.keys())
total = 0
for cond in hconds:
    print "%s: n=%d" % (cond, experiment[cond].shape[0])
    total += experiment[cond].shape[0]
print "All: n=%d" % total

# <markdowncell>

# #### Stimuli

# <codecell>

# stimuli seen by participants
Stims = np.array([
    x.split("~")[0] for x in zip(*experiment[experiment.keys()[0]].columns)[1]])
n_trial = Stims.size
print Stims

# <markdowncell>

# ### Model Simulation Data

# <codecell>

# thresholds for what counts as 'fall'
nthresh0 = 0
nthresh = 0.4

# load the data
rawipe, ipe_samps, rawtruth, feedback, kappas = at.load_model(
    "stability", nthresh0=nthresh0, nthresh=nthresh, fstim=Stims)
nofeedback = np.empty(feedback.shape[1])*np.nan
n_kappas = len(kappas)

# <markdowncell>

# #### Ratios and Kappas

# <codecell>

# round to two or one decimal places, depending on value
kappas = np.array(kappas)
ratios = 10 ** kappas
ratios[kappas < 0] = np.round(ratios[kappas < 0], decimals=2)
ratios[kappas >= 0] = np.round(ratios[kappas >= 0], decimals=1)

# convert ratios and kappas to lists, so we can index
ratios = list(ratios)
kappas = list(kappas)

# ratios that we want to use in plotting
plot_ratios = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
plot_ticks = [ratios.index(r) for r in plot_ratios]

# <markdowncell>

# ### Generate Model Responses

# <codecell>

out = at.generate_model_responses(
    hconds, experiment, queries, feedback, ipe_samps,
    kappas, ratios, n_trial, n_fake=2000)
model_experiment, model_queries, model_belief = out
experiment.update(model_experiment)
queries.update(model_queries)

# <markdowncell>

# ### Condition Labels (Model and Human)

# <codecell>

groups = ["C", "E", "all"]

cond_labels = dict([
    (key % group, value % {'group': group}) 
    for group in groups
    for key, value in {
	    'H-%s-nfb-10': 'No feedback',
	    'H-%s-vfb-0.1': 'Video $r_0=0.1$',
	    'H-%s-vfb-10': 'Video $r_0=10$',
	    'H-%s-fb-0.1': 'Text $r_0=0.1$',
	    'H-%s-fb-10': 'Text $r_0=10$',
	    
	    'M-%s-nfb-10': 'Ideal observer w/ fixed uniform belief',
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
		 'M-%s-nfb-10',
		 ]]
n_cond = len(conds)    

# <markdowncell>

# ## Model Analysis
# ### Plot of smoothed model outcome probabilities

# <codecell>

fig = plt.figure(1)
plt.clf()
istim = [
    list(Stims).index("mass-tower_00357_0111011000"),
    list(Stims).index("mass-tower_00388_0011101001")
    ]
    
print Stims[istim]
at.plot_smoothing(ipe_samps, Stims, istim, kappas)
plt.xticks([kappas[x] for x in plot_ticks], plot_ratios)
fig.set_figwidth(8)
fig.set_figheight(6)

at.savefig("images/likelihood_smoothing", close=False, ext=['png', 'pdf', 'svg'])

# <markdowncell>

# ### Plot of learning model beliefs over time

# <codecell>

beliefs = {}
for cond in conds:
    if cond not in model_belief:
	continue
    obstype, group, fbtype, ratio, cb = at.parse_condition(cond)
    if obstype == "M" and fbtype not in ("vfb", "nfb"):
	beliefs[cond] = model_belief[cond]
	
cmap = at.make_cmap("lh", (0, 0, 0), (.55, .55, .55), (1, 1, 1))
at.plot_belief(2, 2, 2, beliefs, kappas, cmap, cond_labels)
at.savefig("images/ideal_observer_beliefs", close=False, ext=['png', 'pdf', 'svg'])

# <markdowncell>

# ## Explicit mass judgments
# ### Compute summary statistics of mass judgments

# <codecell>

emjs = {}
for cond in conds:
    if cond not in queries:
	continue
    obstype, group, fbtype, ratio, cb = at.parse_condition(cond)
    emjs[cond] = np.asarray(queries[cond]) == float(ratio)
    index = np.array(queries[cond].columns, dtype='i8')
    X = np.array(index)-6-np.arange(len(index))-1
emj_stats = at.CI(emjs, conds)

# <markdowncell>

# ### Binomial analysis of explicit mass judgments

# <codecell>

for i, group in enumerate(groups):
    allarr = []

    for cidx, cond in enumerate(conds):
	if cond not in emj_stats:
	    continue
	obstype, grp, fbtype, ratio, cb = at.parse_condition(cond)
	if (grp != group) or obstype == "M":
	    continue
	mean, lower, upper, sums, n = emj_stats[cond].T
	mcond = "M-%s-fb-%s" % (grp, ratio)
	mmean = emj_stats[mcond][:, 0]
	
	binom0 = [scipy.stats.binom.sf(x-1, n[0], 0.5) for x in sums]
	binom1 = [scipy.stats.binom_test(x, n[0], 0.5) for x in sums]
	binom2 = [scipy.stats.binom.sf(
		sums[idx]-1, n[0], mmean[idx])
		for idx in xrange(len(sums))]

    print cond
    print "  mean:  ", np.round(mean, decimals=2)
    print "  1-tail:", np.round(binom0, decimals=4)
    print "  2-tail:", np.round(binom1, decimals=4)
    print "  model: ", np.round(binom2, decimals=4)
    print

# <markdowncell>

# ### Chi-square analysis of explicit judgments over time

# <codecell>

grouparr = []
for i, group in enumerate(groups):
    allarr = []

    for cond in conds:
	if cond not in queries:
	    continue
	obstype, grp, fbtype, ratio, cb = at.parse_condition(cond)
	if grp != group or obstype == "M":
	    continue
	print cond

	arr = np.asarray(queries[cond]) == float(ratio)
	allarr.append(arr)
	grouparr.append(arr)

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
    print "X2(%s, n=%s)=%s, p=%s" % (dof, df.sum(axis=1)[1], chi2, p)
    print

# <markdowncell>

# ### Plots of explicit mass judgments

# <codecell>

fig = plt.figure(10)
plt.clf()

for i, group in enumerate(groups):
    if group == "all":
	    continue

    for cidx, cond in enumerate(conds):
        if cond not in emj_stats:
            continue
        obstype, grp, fbtype, ratio, cb = at.parse_condition(cond)
        if (grp != group):
            continue
        mean, lower, upper, sums, n = emj_stats[cond].T
        sem = np.array([np.abs(mean - lower), np.abs(mean-upper)]).T

        if group == "E":
            if ratio == "10":
                plt.subplot(1, 4, 2)
                plt.title("$r_0=10$  (diagnostic order)")
            elif ratio == "0.1":
                plt.subplot(1, 4, 1)
                plt.title("$r_0=0.1$ (diagnostic order)")
        elif group == "C":
            if ratio == "10":
                plt.subplot(1, 4, 4)
                plt.title("$r_0=10$ (random order)")
            elif ratio == "0.1":
                plt.subplot(1, 4, 3)
                plt.title("$r_0=0.1$ (random order)")

        if obstype == "M":
            linestyle = "-"
        elif fbtype == "fb":
            linestyle = ":"
        elif fbtype == "vfb":
            linestyle = "--"

        color = 'k'
        plt.fill_between(X, lower, upper, color=color, alpha=0.2)
        plt.plot(X, mean, label=cond_labels[cond], 
                 linewidth=4, color=color, linestyle=linestyle)

        idx += 1

for idx in xrange(4):
    ax = plt.subplot(1, 4, idx+1)
    ax.plot([1, 40], [0.5, 0.5], 'k:')
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(0.25, 1.0)
    ax.set_xticks(X)
    if idx == 0:
	    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
	    ax.set_ylabel("Proportion correct")
    else:
    	ax.set_yticklabels([])
    ax.set_xticklabels(X)
    ax.set_xlabel("Trial")
	
    ax.legend(loc=4, fontsize=11)

title = "Proportion of correct \"Which color is heavier?\" judgments"
plt.suptitle(title, fontsize=16)
fig = plt.gcf()
fig.set_figwidth(16)
fig.set_figheight(4)
plt.subplots_adjust(top=0.85, wspace=0.1, hspace=0.2,
                    left=0.07, right=0.95)

at.savefig("images/explicit_mass_judgments", close=False, ext=['png', 'pdf', 'svg'])

# <markdowncell>

# ## Model Comparison
# ### Table of model likelihoods

# <codecell>

# compute likelihood under various models
models, mnames = at.model_lhs(
    experiment, feedback, nofeedback, ipe_samps, 
    kappas, ratios, conds, f_smooth=True, p_ignore=0.0)

# <codecell>

# make a table of the liklihoods
reload(at)
table, table_conds, table_models = at.make_performance_table(groups, conds, models, mnames)
print

# print a human-readable version for condition E
tabledf = pd.DataFrame(table[0, :, :, 3], columns=mnames[2:], index=table_conds[0])
print tabledf
print

# print a human-readable version for condition C
tabledf = pd.DataFrame(table[1, :, :, 3], columns=mnames[2:], index=table_conds[1])
print tabledf
print

# print a LaTeX version of both conditions
at.print_performance_table(table, table_conds, groups, table_models, cond_labels)

# <markdowncell>

# ### Plots of likelihoods

# <codecell>

nc = 5
nm = len(table_models)
width = 0.8 / (2*nm)
x0 = np.arange(nc)/2.
colors = ["b", "g", "m"]

for gidx, group in enumerate(groups):
    if group == "all":
        continue
        
    idx = 0
    fig = plt.figure(40+gidx)
    plt.clf()

    for midx, model in enumerate(table_models):
        height = table[gidx, :, midx, 3].copy()
        ticks = []
        for i in xrange(len(table_conds[gidx])):
            cond = table_conds[gidx][i]
            obstype, group, fbtype, ratio, cb = at.parse_condition(cond)
            if fbtype == "fb" and ratio == "0.1":
                clabel = r"B(0.1)"
            elif fbtype == "fb" and ratio == "10":
                clabel = r"B(10)"
            elif fbtype == "vfb" and ratio == "0.1":
                clabel = r"V(0.1)"
            elif fbtype == "vfb" and ratio == "10":
                clabel = r"V(10)"
            elif fbtype == "nfb":
                clabel = r"NFB"
            ticks.append(clabel)
		
        color = colors[idx]
        x = x0 + width*(idx-(nm/2.)) + width/2.
        plt.bar(x, height,
            color=color,
            ecolor='k', align='center', width=width, 
            label=model)
        plt.xticks(x0, ticks)

        idx += 1

    best = np.nanargmax(table[gidx, :, :, 3], axis=1), list(np.arange(nc))
    heights = table[gidx, :, :, 3].T[best]
    b = np.array(best[1])/2. + width*(np.array(best[0])-(nm/2.)) + (width/2.)
    for x, h in zip(b, heights):
        plt.text(x, h, '*', fontweight='bold', fontsize=16, ha='center', color='w')

    plt.ylim(-30.5, -25)
    plt.xlim(x0.min()-0.3, x0.max()+0.3)
    plt.plot([x0.min()-0.3, x0.max()+0.3], [np.log(0.5)*n_trial]*2, 
	     'k--', label="Chance", linewidth=2)

    plt.legend(loc=8, ncol=4, fontsize=12, frameon=False)
    plt.xlabel("%s-order condition" % ("Diagnostic" if group == "E" else "Random"), fontsize=14)
    plt.ylabel("Average log likelihood", fontsize=14)

    title = "Model likelihoods for \"Will it fall?\" judgments"
    plt.suptitle(title, fontsize=16)
    fig.set_figwidth(6)
    fig.set_figheight(4)
    plt.subplots_adjust(bottom=0.18, top = 0.9, left=0.11)
    
    at.savefig("images/model_performance_%s" % group, close=False, ext=['png', 'pdf', 'svg'])

# <markdowncell>

# ### Plots of likelihoods under fixed models

# <codecell>

theta = np.log(np.eye(n_kappas))
fb = np.empty((n_kappas, n_trial))*np.nan
model_fixed = at.CI(at.block_lh(
    experiment, fb, ipe_samps, theta, kappas, 
    f_smooth=True, p_ignore=0.0), conds)

for i, group in enumerate(groups):
    if group == "all":
        continue
    
    idx = 0
    fig = plt.figure(30+i)
    plt.clf()

    for cidx, cond in enumerate(conds):
        obstype, grp, fbtype, ratio, cb = at.parse_condition(cond)
        
        if grp != group:
            continue
    
        if fbtype == "nfb":
            nfbax = plt.subplot(1, 3, 3)
            nfbtitle = "No feedback"
        elif ratio == "10":
            r10ax = plt.subplot(1, 3, 2)
            r10title = "Feedback generated w/ $r_0=10$"
        elif ratio == "0.1":
            r01ax = plt.subplot(1, 3, 1)
            r01title = "Feedback generated w/ $r_0=0.1$"
                
        if obstype == "M":
            linestyle = "-"
        elif fbtype == "fb":
            linestyle = ":"
        elif fbtype == "vfb":
            linestyle = "--"
        elif fbtype == "nfb":
            linestyle = ":"
                
        color = 'k'
        mean, lower, upper, sums, n = model_fixed[cond].T
        plt.plot(kappas, sums/n, label=cond_labels[cond], 
                 color=color, linewidth=4,
                 linestyle=linestyle)

        idx += 1

    for ax, title in [(nfbax, nfbtitle), (r10ax, r10title), (r01ax, r01title)]:
        ax.set_xticks([kappas[x] for x in plot_ticks])
        ax.set_xticklabels(plot_ratios, rotation=30)
        ax.set_xlabel("Assumed mass ratio ($r$)")
        ax.set_xlim(min(kappas), max(kappas))
            
        if ax == r01ax:
            ax.set_ylabel("Log likelihood of judgments")
        else:
            ax.set_yticklabels([])
        ax.legend(loc=3, ncol=1, fontsize=12, frameon=False)
        ax.set_title(title)

    title = "Likelihood of \"Will it fall?\" judgments under fixed belief models"
    if group != "all":
        title += " (%s)" % group
    plt.suptitle(title, fontsize=20)
    fig.set_figwidth(15)
    fig.set_figheight(4)

    plt.subplots_adjust(bottom=0.2, top=0.85, wspace=0.1,
                        left=0.07, right=0.95)
    at.savefig("images/fixed_model_performance_%s" % group,
               close=False, ext=['png', 'pdf', 'svg'])
    

