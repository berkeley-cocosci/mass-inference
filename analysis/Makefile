ANALYSES := condition_counts\
	    fall_response_corrs\
	    fall_responses\
	    mass_accuracy\
	    mass_accuracy_by_stimulus\
	    mass_accuracy_by_stimulus_corrs\
	    mass_accuracy_by_trial\
	    mass_responses_by_stimulus\
	    mass_responses_by_stimulus_corrs\
	    model_belief\
	    model_log_lh\
	    num_chance\
	    num_learned_by_trial\
	    participant_fits\
	    payrate\
	    num_participants\
	    mass_accuracy_by_trial_corrs\
	    fall_responses_best_parameters

.PHONY: all list clean $(ANALYSES)

all: $(ANALYSES)
list:
	@echo $(ANALYSES) | tr " " "\n"
clean:
	rm -f results/*
	rm -f figures/*.pdf
	rm -f figures/*.png

fall_responses_best_parameters_targets := figures/fall_responses_best_parameters.pdf figures/fall_responses_best_parameters.png
fall_responses_best_parameters: $(fall_responses_best_parameters_targets)
$(fall_responses_best_parameters_targets): plots/fall_responses_best_parameters.py results/fall_responses_best_parameters.csv
	python $<

num_learned_by_trial_targets := figures/num_learned_by_trial.pdf figures/num_learned_by_trial.png
num_learned_by_trial: $(num_learned_by_trial_targets)
$(num_learned_by_trial_targets): plots/num_learned_by_trial.py results/num_learned_by_trial.csv
	python $<

model_belief_versions := static learning chance best
model_belief_pdf := $(addsuffix .pdf,$(addprefix figures/model_belief_,$(model_belief_versions)))
model_belief_png := $(model_belief_pdf:%.pdf=%.png)
model_belief_targets = $(model_belief_pdf) $(model_belief_png)
model_belief: $(model_belief_targets)
$(model_belief_targets): plots/model_belief.py results/model_belief_agg.csv results/participant_fits.csv
	python $<

mass_responses_by_stimulus_targets := figures/mass_responses_by_stimulus.pdf figures/mass_responses_by_stimulus.png
mass_responses_by_stimulus: $(mass_responses_by_stimulus_targets)
$(mass_responses_by_stimulus_targets): plots/mass_responses_by_stimulus.py results/mass_responses_by_stimulus.csv
	python $<

mass_accuracy_by_trial_targets := figures/mass_accuracy_by_trial.pdf figures/mass_accuracy_by_trial.png
mass_accuracy_by_trial: $(mass_accuracy_by_trial_targets)
$(mass_accuracy_by_trial_targets): plots/mass_accuracy_by_trial.py results/mass_accuracy_by_trial.csv
	python $<

mass_accuracy_by_stimulus_targets := figures/mass_accuracy_by_stimulus.pdf figures/mass_accuracy_by_stimulus.png
mass_accuracy_by_stimulus: $(mass_accuracy_by_stimulus_targets)
$(mass_accuracy_by_stimulus_targets): plots/mass_accuracy_by_stimulus.py results/mass_accuracy_by_stimulus.csv
	python $<

fall_responses_versions := G_A G_B H_A H_B I_A GH_A GH_B
fall_responses_pdf := $(addsuffix .pdf,$(addprefix figures/fall_responses_,$(fall_responses_versions)))
fall_responses_png := $(fall_responses_pdf:%.pdf=%.png)
fall_responses: $(fall_responses_pdf) $(fall_responses_png)
$(fall_responses_pdf) $(fall_responses_png): plots/fall_responses.py results/fall_responses.csv
	python $<

results/trial_order.csv: analyses/trial_order.py
	python $<

results/switchpoint.csv: analyses/switchpoint.py
	python $<

payrate: results/payrate.csv
results/payrate.csv: analyses/payrate.py
	python $<

participant_fits: results/participant_fits.csv
results/participant_fits.csv: analyses/participant_fits.py results/model_log_lh.csv
	python $<

results/num_learned_by_trial.csv: analyses/num_learned_by_trial.py results/switchpoint.csv results/mass_accuracy.csv
	python $<

num_chance: results/num_chance.tex
results/num_chance.csv results/num_chance.tex: analyses/num_chance.py
	python $<

model_log_lh: results/model_log_lh.csv
results/model_log_lh.csv: analyses/model_log_lh.py results/model_belief_agg.csv
	python $<

results/model_belief_agg.csv: analyses/model_belief_agg.py results/model_belief.csv
	python $<

results/model_belief.csv: analyses/model_belief.py results/trial_order.csv
	python $<

mass_responses_by_stimulus_corrs: results/mass_responses_by_stimulus_corrs.csv
results/mass_responses_by_stimulus_corrs.csv: analyses/mass_responses_by_stimulus_corrs.py results/mass_responses_by_stimulus.csv
	python $<

results/mass_responses_by_stimulus.csv: analyses/mass_responses_by_stimulus.py results/model_belief_agg.csv
	python $<

results/mass_accuracy_by_trial.csv: analyses/mass_accuracy_by_trial.py results/model_belief_agg.csv
	python $<

mass_accuracy_by_stimulus_corrs: results/mass_accuracy_by_stimulus_corrs.csv
results/mass_accuracy_by_stimulus_corrs.csv: analyses/mass_accuracy_by_stimulus_corrs.py results/mass_accuracy_by_stimulus.csv
	python $<

results/mass_accuracy_by_stimulus.csv: analyses/mass_accuracy_by_stimulus.py results/model_belief_agg.csv
	python $<

mass_accuracy: results/mass_accuracy.csv
results/mass_accuracy.csv: analyses/mass_accuracy.py results/model_belief_agg.csv
	python $<

results/fall_responses.csv: analyses/fall_responses.py
	python $<

fall_response_corrs: results/fall_response_corrs.tex
results/fall_response_corrs.csv results/fall_response_corrs.tex: analyses/fall_response_corrs.py results/fall_responses.csv
	python $<

condition_counts: results/condition_counts.csv
results/condition_counts.csv: analyses/condition_counts.py
	python $<

num_participants: results/num_participants.csv
results/num_participants.csv: analyses/num_participants.py
	python $<

mass_accuracy_by_trial_corrs: results/mass_accuracy_by_trial_corrs.csv
results/mass_accuracy_by_trial_corrs.csv: analyses/mass_accuracy_by_trial_corrs.py results/mass_accuracy_by_trial.csv
	python $<

results/fall_responses_best_parameters.csv: analyses/fall_responses_best_parameters.py
	python $<
