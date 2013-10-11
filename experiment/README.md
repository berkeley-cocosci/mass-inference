# Experimental Design

## Version G (Mass prediction+inference)

The structure of this experiment goes as follows:

1. Pretest

2. Mass prediction on a scale from 1-10, to get estimates of the
   likelihood of falling given mass for each tower. People do not see
   feedback, so they can't learn whether the towers actually fall for
   a given mass ratio.
   
   Here, we should choose one color pairing and stick with it
   (e.g. red/blue). We will need to run people on both mass ratios, so
   we'll need:

	Red: #CA0020
	Blue: #0571B0

3. Mass inferences on the *same* towers, to see how well the estimated
   likelihoods from the first part of the experiment predict the
   actual inferences that people make.

   Here, we'll choose a *different* color pairing, though
   (e.g. green/purple), so people don't get confused and think
   whichever color was heavier before is still heavier. But, for half
   the people, we'll still want to flip which color is heavier, just
   to be sure.

   Purple: #7B3294
   Green: #008837

4. Posttest


So, we need to render the following sets of towers:

 * G-vfb-0.1-cb0
 * G-vfb-0.1-cb1
 * G-vfb-10-cb0
 * G-vfb-10-cb1
 * G-nfb-0.1-cb0
 * G-nfb-0.1-cb1
 * G-nfb-10-cb0
 * G-nfb-10-cb1

And there will be 4 possible conditions times 2 for counterbalancing:

 * G nfb-0.1 vfb-0.1
 * G nfb-0.1 vfb-10
 * G nfb-10 vfb-0.1
 * G nfb-10 vfb-10
