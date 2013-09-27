#!/bin/sh -e

function color_red() {
    echo "\x1b[31m$@\x1b[0m"
}

function color_green() {
    echo "\x1b[32m$@\x1b[0m"
}

function color_blue() {
    echo "\x1b[34m$@\x1b[0m"
}

function makedir() {
    if [ -d "$1" ]; then
	echo "exists: $1"
    else
	cmd="mkdir -p $1"
	color_green "$cmd"
	`$cmd`
    fi
}

function hardlink() {
    if [ -e "$2" ]; then
	rm "$2"
    fi
    cmd="ln $1 $2"
    color_green "$cmd"
    `$cmd`
}

function gen_render_config() {
    cmd="gen_render_config.py $@"
    color_blue "python $cmd"
    python $cmd
}

function render_movies() {
    cmd="render_movies.py $@"
    color_blue "python $cmd"
    python $cmd
}

########################################################################

OUTDIR="../resources/render/G"
SSODIR="../resources/sso"

# remove old files, if they exist
if [[ -d $OUTDIR ]]; then
    read -p "'$OUTDIR' already exists, do you want to remove it? (y/n) " -n 1
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
	cmd="rm -r $OUTDIR"
	color_red "$cmd"
	`$cmd`
    fi
fi

# create output directories for each condition
makedir "$OUTDIR/vfb-10-cb0"
makedir "$OUTDIR/vfb-10-cb1"
makedir "$OUTDIR/vfb-0.1-cb0"
makedir "$OUTDIR/vfb-0.1-cb1"
makedir "$OUTDIR/shared"

######################## 
## Stability examples ##
######################## 

gen_render_config -o "$OUTDIR/shared/stability-examples.csv" --full "$SSODIR/stability-example-stable-F/*" "$SSODIR/stability-example-unstable-F/*"
render_movies -c "$OUTDIR/shared/stability-examples.csv" -d "$OUTDIR/shared"


######################
## Pretest/posttest ##
######################

# gen_render_config -o "$OUTDIR/shared/training.csv" "$SSODIR/mass-oneshot-training-F/*"
# render_movies -c "$OUTDIR/shared/training.csv" -d "$OUTDIR/shared"


# hard link all the shared videos, so we don't have to render them
# multiple times
for i in `ls $OUTDIR/shared`; do
    hardlink "$OUTDIR/shared/$i" "$OUTDIR/vfb-10-cb0/$i"
    hardlink "$OUTDIR/shared/$i" "$OUTDIR/vfb-10-cb1/$i"
    hardlink "$OUTDIR/shared/$i" "$OUTDIR/vfb-0.1-cb0/$i"
    hardlink "$OUTDIR/shared/$i" "$OUTDIR/vfb-0.1-cb1/$i"
done


# # ###################
# # ## Mass examples ##
# # ###################

# gen_render_config -o "$OUTDIR/vfb-10-cb0/mass-example.csv" --kappa 1.0 "$SSODIR/mass-oneshot-example-F/*"
# gen_render_config -o "$OUTDIR/vfb-10-cb1/mass-example.csv" --kappa 1.0 --flip-colors "$SSODIR/mass-oneshot-example-F/*"
# gen_render_config -o "$OUTDIR/vfb-0.1-cb0/mass-example.csv" --kappa -1.0 "$SSODIR/mass-oneshot-example-F/*"
# gen_render_config -o "$OUTDIR/vfb-0.1-cb1/mass-example.csv" --kappa -1.0 --flip-colors "$SSODIR/mass-oneshot-example-F/*"

# render_movies -c "$OUTDIR/vfb-10-cb0/mass-example.csv" -d "$OUTDIR/vfb-10-cb0"
# render_movies -c "$OUTDIR/vfb-10-cb1/mass-example.csv" -d "$OUTDIR/vfb-10-cb1"
# render_movies -c "$OUTDIR/vfb-0.1-cb0/mass-example.csv" -d "$OUTDIR/vfb-0.1-cb0"
# render_movies -c "$OUTDIR/vfb-0.1-cb1/mass-example.csv" -d "$OUTDIR/vfb-0.1-cb1"


# # ################
# # ## Experiment ##
# # ################

# gen_render_config -o "$OUTDIR/vfb-10-cb0/experiment.csv" --kappa 1.0 "$SSODIR/mass-inference/*"
# gen_render_config -o "$OUTDIR/vfb-10-cb1/experiment.csv" --kappa 1.0 --flip-colors "$SSODIR/mass-inference/*"
# gen_render_config -o "$OUTDIR/vfb-0.1-cb0/experiment.csv" --kappa -1.0 "$SSODIR/mass-inference/*"
# gen_render_config -o "$OUTDIR/vfb-0.1-cb1/experiment.csv" --kappa -1.0 --flip-colors "$SSODIR/mass-inference/*"

# render_movies -c "$OUTDIR/vfb-10-cb0/experiment.csv" -d "$OUTDIR/vfb-10-cb0"
# render_movies -c "$OUTDIR/vfb-10-cb1/experiment.csv" -d "$OUTDIR/vfb-10-cb1"
# render_movies -c "$OUTDIR/vfb-0.1-cb0/experiment.csv" -d "$OUTDIR/vfb-0.1-cb0"
# render_movies -c "$OUTDIR/vfb-0.1-cb1/experiment.csv" -d "$OUTDIR/vfb-0.1-cb1"
