
	#!/bin/sh -e

OUTDIR="../resources/render"
SSODIR="../resources/sso"

CMD="python gen_render_config.py"
# $CMD -o $OUTDIR/F/stability-examples-F.csv --full $SSODIR/stability-example-stable-F/* $SSODIR/stability-example-unstable-F/*
# $CMD -o $OUTDIR/F/mass-oneshot-training-F.csv $SSODIR/mass-oneshot-training-F/*
# $CMD -o $OUTDIR/F-cb0/mass-example-F.csv --kappa 1.0 --full $SSODIR/mass-oneshot-example-F/*
# $CMD -o $OUTDIR/F-cb1/mass-example-F.csv --kappa 1.0 --flip-colors --full $SSODIR/mass-oneshot-example-F/*

$CMD -o $OUTDIR/G-vfb-10-cb0/experiment.csv --kappa 1.0 $SSODIR/mass-inference/*
$CMD -o $OUTDIR/G-vfb-10-cb1/experiment.csv --kappa 1.0 --flip-colors $SSODIR/mass-inference/*
$CMD -o $OUTDIR/G-vfb-0.1-cb0/experiment.csv --kappa -1.0 $SSODIR/mass-inference/*
$CMD -o $OUTDIR/G-vfb-0.1-cb1/experiment.csv --kappa -1.0 --flip-colors $SSODIR/mass-inference/*

CMD="python render_movies.py"
# $CMD -c $OUTDIR/F/stability-examples-F.csv -d $OUTDIR/F
# $CMD -c $OUTDIR/F/mass-oneshot-training-F.csv -d $OUTDIR/F
# $CMD -c $OUTDIR/F-cb0/mass-example-F.csv -d $OUTDIR/F-cb0
# $CMD -c $OUTDIR/F-cb1/mass-example-F.csv -d $OUTDIR/F-cb1

$CMD -c $OUTDIR/G-vfb-10-cb0/experiment.csv -d $OUTDIR/G-vfb-10-cb0
$CMD -c $OUTDIR/G-vfb-10-cb1/experiment.csv -d $OUTDIR/G-vfb-10-cb1
$CMD -c $OUTDIR/G-vfb-0.1-cb0/experiment.csv -d $OUTDIR/G-vfb-0.1-cb0
$CMD -c $OUTDIR/G-vfb-0.1-cb1/experiment.csv -d $OUTDIR/G-vfb-0.1-cb1
