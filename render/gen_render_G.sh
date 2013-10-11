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
    `$cmd` || true
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
    fmts=("mp4" "ogg" "webm")
    for i in $1/*.avi; do
	if [ -e $i ]; then
	    for fmt in ${fmts[*]}; do
		newfile="${i%.avi}.$fmt"
		if [ ! -e $newfile ]; then
		    cmd="-loglevel error -i $i -r 30 -b 2048k -s 640x480 $newfile"
		    color_blue "ffmpeg $cmd"
		    ffmpeg $cmd
		else
		    echo "exists: $newfile"
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

gen_render_config -o "$OUTDIR/shared/stable_example.csv" --full "$SSODIR/mass-inference-stable-example-G/*"
render_movies -c "$OUTDIR/shared/stable_example.csv" -d "$OUTDIR/shared"

gen_render_config -o "$OUTDIR/shared/unstable_example.csv" --full "$SSODIR/mass-inference-unstable-example-G/*"
render_movies -c "$OUTDIR/shared/unstable_example.csv" -d "$OUTDIR/shared"

gen_render_config -o "$OUTDIR/shared/pretest.csv" "$SSODIR/mass-inference-training-G/*"
cp "$OUTDIR/shared/pretest.csv" "$OUTDIR/shared/posttest.csv"
render_movies -c "$OUTDIR/shared/pretest.csv" -d "$OUTDIR/shared"


###################
## Mass examples ##
###################

gen_render_config -o "$OUTDIR/nfb-10-cb0/mass_example.csv" --label0 "red" --label1 "blue" --color0 "#CA0020" --color1 "#0571B0" --kappa 1.0 "$SSODIR/mass-inference-example-G/*"
gen_render_config -o "$OUTDIR/nfb-10-cb1/mass_example.csv" --label0 "red" --label1 "blue" --color0 "#CA0020" --color1 "#0571B0" --kappa 1.0 --flip-colors "$SSODIR/mass-inference-example-G/*"
gen_render_config -o "$OUTDIR/nfb-0.1-cb0/mass_example.csv" --label0 "red" --label1 "blue" --color0 "#CA0020" --color1 "#0571B0" --kappa -1.0 "$SSODIR/mass-inference-example-G/*"
gen_render_config -o "$OUTDIR/nfb-0.1-cb1/mass_example.csv" --label0 "red" --label1 "blue" --color0 "#CA0020" --color1 "#0571B0" --kappa -1.0 --flip-colors "$SSODIR/mass-inference-example-G/*"

render_movies -c "$OUTDIR/nfb-10-cb0/mass_example.csv" -d "$OUTDIR/nfb-10-cb0"
render_movies -c "$OUTDIR/nfb-10-cb1/mass_example.csv" -d "$OUTDIR/nfb-10-cb1"
render_movies -c "$OUTDIR/nfb-0.1-cb0/mass_example.csv" -d "$OUTDIR/nfb-0.1-cb0"
render_movies -c "$OUTDIR/nfb-0.1-cb1/mass_example.csv" -d "$OUTDIR/nfb-0.1-cb1"


#######################
## Experiment Part A ##
#######################

gen_render_config -o "$OUTDIR/nfb-10-cb0/experiment.csv" --label0 "red" --label1 "blue" --color0 "#CA0020" --color1 "#0571B0" --kappa 1.0 "$SSODIR/mass-inference-G/*"
gen_render_config -o "$OUTDIR/nfb-10-cb1/experiment.csv" --label0 "red" --label1 "blue" --color0 "#CA0020" --color1 "#0571B0" --kappa 1.0 --flip-colors "$SSODIR/mass-inference-G/*"
gen_render_config -o "$OUTDIR/nfb-0.1-cb0/experiment.csv" --label0 "red" --label1 "blue" --color0 "#CA0020" --color1 "#0571B0" --kappa -1.0 "$SSODIR/mass-inference-G/*"
gen_render_config -o "$OUTDIR/nfb-0.1-cb1/experiment.csv" --label0 "red" --label1 "blue" --color0 "#CA0020" --color1 "#0571B0" --kappa -1.0 --flip-colors "$SSODIR/mass-inference-G/*"

render_movies -c "$OUTDIR/nfb-10-cb0/experiment.csv" -d "$OUTDIR/nfb-10-cb0"
render_movies -c "$OUTDIR/nfb-10-cb1/experiment.csv" -d "$OUTDIR/nfb-10-cb1"
render_movies -c "$OUTDIR/nfb-0.1-cb0/experiment.csv" -d "$OUTDIR/nfb-0.1-cb0"
render_movies -c "$OUTDIR/nfb-0.1-cb1/experiment.csv" -d "$OUTDIR/nfb-0.1-cb1"


#######################
## Experiment Part B ##
#######################

gen_render_config -o "$OUTDIR/vfb-10-cb0/experiment.csv" --label0 "purple" --label1 "green" --color0 "#7B3294" --color1 "#008837" --kappa 1.0 "$SSODIR/mass-inference-G/*"
gen_render_config -o "$OUTDIR/vfb-10-cb1/experiment.csv" --label0 "purple" --label1 "green" --color0 "#7B3294" --color1 "#008837" --kappa 1.0 --flip-colors "$SSODIR/mass-inference-G/*"
gen_render_config -o "$OUTDIR/vfb-0.1-cb0/experiment.csv" --label0 "purple" --label1 "green" --color0 "#7B3294" --color1 "#008837" --kappa -1.0 "$SSODIR/mass-inference-G/*"
gen_render_config -o "$OUTDIR/vfb-0.1-cb1/experiment.csv" --label0 "purple" --label1 "green" --color0 "#7B3294" --color1 "#008837" --kappa -1.0 --flip-colors "$SSODIR/mass-inference-G/*"

render_movies -c "$OUTDIR/vfb-10-cb0/experiment.csv" -d "$OUTDIR/vfb-10-cb0"
render_movies -c "$OUTDIR/vfb-10-cb1/experiment.csv" -d "$OUTDIR/vfb-10-cb1"
render_movies -c "$OUTDIR/vfb-0.1-cb0/experiment.csv" -d "$OUTDIR/vfb-0.1-cb0"
render_movies -c "$OUTDIR/vfb-0.1-cb1/experiment.csv" -d "$OUTDIR/vfb-0.1-cb1"


#####################################################
## Convert videos and copy to experiment directory ##
#####################################################

# convert videos to different formats
convert_videos "$OUTDIR/shared"
convert_videos "$OUTDIR/nfb-10-cb0"
convert_videos "$OUTDIR/nfb-10-cb1"
convert_videos "$OUTDIR/nfb-0.1-cb0"
convert_videos "$OUTDIR/nfb-0.1-cb1"
convert_videos "$OUTDIR/vfb-10-cb0"
convert_videos "$OUTDIR/vfb-10-cb1"
convert_videos "$OUTDIR/vfb-0.1-cb0"
convert_videos "$OUTDIR/vfb-0.1-cb1"

read -p "Copy to experiment stimuli directory? (y/n) " -n 1
echo
if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    remove "$EXPDIR"
    makedir "$EXPDIR"
    hardlink "$OUTDIR/shared" "$EXPDIR/shared"
    hardlink "$OUTDIR/nfb-10-cb0" "$EXPDIR/nfb-10-cb0"
    hardlink "$OUTDIR/nfb-10-cb1" "$EXPDIR/nfb-10-cb1"
    hardlink "$OUTDIR/nfb-0.1-cb0" "$EXPDIR/nfb-0.1-cb0"
    hardlink "$OUTDIR/nfb-0.1-cb1" "$EXPDIR/nfb-0.1-cb1"
    hardlink "$OUTDIR/vfb-10-cb0" "$EXPDIR/vfb-10-cb0"
    hardlink "$OUTDIR/vfb-10-cb1" "$EXPDIR/vfb-10-cb1"
    hardlink "$OUTDIR/vfb-0.1-cb0" "$EXPDIR/vfb-0.1-cb0"
    hardlink "$OUTDIR/vfb-0.1-cb1" "$EXPDIR/vfb-0.1-cb1"
fi
