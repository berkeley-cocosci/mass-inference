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

function remove() {
    if [ -e "$1" ]; then
	cmd="rm -r $1"
	color_red "$cmd"
	`$cmd`
    fi
}

function hardlink() {
    if [ $(uname) == "Darwin" ]; then
	cmd="gcp -lr $1 $2"
    else
	cmd="cp -lr $1 $2"
    fi
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

function convert_videos() {
    fmts=("mp4" "ogg" "flv" "wmv")
    for i in $1/*.avi; do
	if [ -e $i ]; then
	    for fmt in ${fmts[*]}; do
		newfile="${i%.avi}.$fmt"
		if [ ! -e $i ]; then
		    cmd="-loglevel error -i $i -r 30 -b 2048k -s 640x480 $newfile"
		    color_blue "ffmpeg $cmd"
		    ffmpeg $cmd
		fi
	    done
	fi
    done
}

########################################################################

OUTDIR="../resources/render/G"
SSODIR="../resources/sso"
EXPDIR="../experiment/static/stimuli"

# remove old files, if they exist
if [[ -d "$OUTDIR" ]]; then
    read -p "'$OUTDIR' already exists, do you want to remove it? (y/n) " -n 1
    echo
    if [[ "$REPLY" =~ ^[Yy]$ ]]; then
	remove "$OUTDIR"
    fi
fi

makedir "$OUTDIR/shared"
makedir "$OUTDIR/vfb-10-cb0"
makedir "$OUTDIR/vfb-10-cb1"
makedir "$OUTDIR/vfb-0.1-cb0"
makedir "$OUTDIR/vfb-0.1-cb1"


#################### 
## Shared stimuli ##
#################### 

gen_render_config -o "$OUTDIR/shared/stability-examples.csv" --full "$SSODIR/stability-example-stable-F/*" "$SSODIR/stability-example-unstable-F/*"
render_movies -c "$OUTDIR/shared/stability-examples.csv" -d "$OUTDIR/shared"

gen_render_config -o "$OUTDIR/shared/training.csv" "$SSODIR/mass-oneshot-training-F/*"
render_movies -c "$OUTDIR/shared/training.csv" -d "$OUTDIR/shared"


###################
## Mass examples ##
###################

gen_render_config -o "$OUTDIR/vfb-10-cb0/mass-example.csv" --kappa 1.0 "$SSODIR/mass-oneshot-example-F/*"
gen_render_config -o "$OUTDIR/vfb-10-cb1/mass-example.csv" --kappa 1.0 --flip-colors "$SSODIR/mass-oneshot-example-F/*"
gen_render_config -o "$OUTDIR/vfb-0.1-cb0/mass-example.csv" --kappa -1.0 "$SSODIR/mass-oneshot-example-F/*"
gen_render_config -o "$OUTDIR/vfb-0.1-cb1/mass-example.csv" --kappa -1.0 --flip-colors "$SSODIR/mass-oneshot-example-F/*"

render_movies -c "$OUTDIR/vfb-10-cb0/mass-example.csv" -d "$OUTDIR/vfb-10-cb0"
render_movies -c "$OUTDIR/vfb-10-cb1/mass-example.csv" -d "$OUTDIR/vfb-10-cb1"
render_movies -c "$OUTDIR/vfb-0.1-cb0/mass-example.csv" -d "$OUTDIR/vfb-0.1-cb0"
render_movies -c "$OUTDIR/vfb-0.1-cb1/mass-example.csv" -d "$OUTDIR/vfb-0.1-cb1"


################
## Experiment ##
################

gen_render_config -o "$OUTDIR/vfb-10-cb0/experiment.csv" --kappa 1.0 "$SSODIR/mass-inference/*"
gen_render_config -o "$OUTDIR/vfb-10-cb1/experiment.csv" --kappa 1.0 --flip-colors "$SSODIR/mass-inference/*"
gen_render_config -o "$OUTDIR/vfb-0.1-cb0/experiment.csv" --kappa -1.0 "$SSODIR/mass-inference/*"
gen_render_config -o "$OUTDIR/vfb-0.1-cb1/experiment.csv" --kappa -1.0 --flip-colors "$SSODIR/mass-inference/*"

render_movies -c "$OUTDIR/vfb-10-cb0/experiment.csv" -d "$OUTDIR/vfb-10-cb0"
render_movies -c "$OUTDIR/vfb-10-cb1/experiment.csv" -d "$OUTDIR/vfb-10-cb1"
render_movies -c "$OUTDIR/vfb-0.1-cb0/experiment.csv" -d "$OUTDIR/vfb-0.1-cb0"
render_movies -c "$OUTDIR/vfb-0.1-cb1/experiment.csv" -d "$OUTDIR/vfb-0.1-cb1"


#####################################################
## Convert videos and copy to experiment directory ##
#####################################################

# convert videos to different formats
convert_videos "$OUTDIR/shared"
convert_videos "$OUTDIR/vfb-10-cb0"
convert_videos "$OUTDIR/vfb-10-cb1"
convert_videos "$OUTDIR/vfb-0.1-cb0"
convert_videos "$OUTDIR/vfb-0.1-cb1"

# hard link all the shared files, so we don't have to render/convert
# them multiple times
hardlink "$OUTDIR/shared/*" "$OUTDIR/vfb-10-cb0/"
hardlink "$OUTDIR/shared/*" "$OUTDIR/vfb-10-cb1/"
hardlink "$OUTDIR/shared/*" "$OUTDIR/vfb-0.1-cb0/"
hardlink "$OUTDIR/shared/*" "$OUTDIR/vfb-0.1-cb1/"

read -p "Copy to experiment stimuli directory? (y/n) " -n 1
echo
if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    remove "$EXPDIR"
    makedir "$EXPDIR"
    hardlink "$OUTDIR/vfb-10-cb0" "$EXPDIR/vfb-10-cb0"
    hardlink "$OUTDIR/vfb-10-cb1" "$EXPDIR/vfb-10-cb1"
    hardlink "$OUTDIR/vfb-0.1-cb0" "$EXPDIR/vfb-0.1-cb0"
    hardlink "$OUTDIR/vfb-0.1-cb1" "$EXPDIR/vfb-0.1-cb1"
fi
