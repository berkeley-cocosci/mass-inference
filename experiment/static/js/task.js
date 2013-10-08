/* task.js
 * 
 * This file holds the main experiment code.
 * 
 * Requires:
 *   config.js
 *   psiturk.js
 *   utils.js
 */

// TODO: try to fix weird flickery behavior at the beginning of a trial
// TODO: try to fix flickery behavior at the beginning of videos
// TODO: trial order should be the same in all conditions

// Initialize flowplayer
var $f = flowplayer;
if ($f.support.firstframe) {
    $f(function (api, root) {
        // show poster when video ends
        api.bind("resume finish", function (e) {
            root.toggleClass("is-poster", /finish/.test(e.type));
            api.disable(!/finish/.test(e.type));
        });
    });
}

// Create and initialize the experiment configuration object
var $c = new Config(condition, counterbalance);

// Initalize psiturk object
var psiTurk = new PsiTurk();

// Preload the HTML template pages that we need for the experiment
psiTurk.preloadPages($c.pages);

// Objects to keep track of the current phase and state
var CURRENTVIEW;
var STATE;


/*************************
 * INSTRUCTIONS         
 *************************/

var Instructions = function() {

    // The list of pages for this set of instructions
    this.pages = $c.instructions[STATE.experiment_phase].pages;
    // The list of examples on each page of instructions
    this.examples = $c.instructions[STATE.experiment_phase].examples;
    // Time when a page of instructions is presented
    this.timestamp;
    // The flowplayer instance
    this.player;

    // Display a page of instructions, based on the current
    // STATE.index
    this.show = function() {

        // Load the next page of instructions
        psiTurk.showPage(this.pages[STATE.index]);
        // Update the URL hash
        STATE.set_hash();

        // Bind a handler to the "next" button. We have to wrap it in
        // an anonymous function to preserve the scope.
        var that = this;
        $('.next').click(function () {
            that.record_response();
        });
        
        // Load the video player
        if (this.examples[STATE.index]) {
            var video = this.examples[STATE.index] + "~stimulus";

            // Start the video immediately
            var on_ready = function (e, api) {
                api.play();
            };
            // Loop the player by restarting the current video
            var on_finish = function (e, api) {
                api.prev();
            };

            // Initialize the player and start it
            this.player = make_player(
                "#player",     // element 
                [video],       // stimuli
                video + "~A"   // background image

            ).bind("ready", on_ready)
                .bind("finish", on_finish)
                .play(0);
        }

        // Record the time that an instructions page is presented
        this.timestamp = new Date().getTime();
    };

    // Handler for when the "next" button is pressed
    this.record_response = function() {

        // Calculate the response time
        var rt = (new Date().getTime()) - this.timestamp;
        debug("'Next' button pressed");

        // Record the data. The format is: 
        // experiment phase, instructions, index, trial_phase, response time
        var data = new DataRecord();
        data.update({response: "", response_time: rt});
        psiTurk.recordTrialData(data.to_array());
        debug(data.to_array());

         // Destroy the video player
        if (this.player) this.player.unload();

        // Go to the next page of instructions, or complete these
        // instructions if there are no more pages
        if ((STATE.index + 1) >= this.pages.length) {
            this.finish();
        } else {
            STATE.set_index(STATE.index + 1);
            this.show();
        }
    };

    // Clean up the instructions phase and move on to the test phase
    this.finish = function() {
        debug("Done with instructions")

        // Record that the user has finished the instructions and 
        // moved on to the experiment. This changes their status code
        // in the database.
        // if (STATE.experiment_phase == EXPERIMENT.pretest) {
        //     psiTurk.finishInstructions();
        // }

        // Reset the state object for the test phase
        STATE.set_instructions(0);
        STATE.set_index();
        STATE.set_trial_phase();
        CURRENTVIEW = new TestPhase();
    };

    // Display the first page of instructions
    this.show();
};



/*****************
 *  TRIALS       *
 *****************/

var TestPhase = function() {

    /* Instance variables */

    // When the time response period begins
    this.timestamp; 
    // Whether the object is listening for responses
    this.listening = false;

    // Video player for the stimulus
    this.stim_player;
    // Video player for feedback
    this.fb_player;

    // List of trials in this block of the experiment
    this.trials = $c.trials[STATE.experiment_phase].slice(0, 1);
    // Information about the current trial
    this.trialinfo;
    // The current stimulus name
    this.stimulus;
    
    // Handlers to setup each phase of a trial
    this.phases = new Object();

    // Initialize a flowplayer in elem. This is different from
    // make_player (in utils.js) in that it also binds a "finish"
    // handler to set a new poster for the player, and to go to the
    // next trial phase.
    this.init_player = function (elem, name) {
        var that = this;
        var video = this.stimulus + "~" + name;
        var on_finish = function (e, api) {
            debug(name + " finished");
            api.unbind("finish");

            set_poster(elem + ".flowplayer", video + "~B");
            $(elem).addClass("is-poster");

            STATE.set_trial_phase(STATE.trial_phase + 1);
            that.show();
        };
        return make_player(elem, [video], video + "~A")
            .bind("finish", on_finish);
    };

    // Initialize a new trial. This is called either at the beginning
    // of a new trial, or if the page is reloaded between trials.
    this.init_trial = function () {
        debug("Initializing trial " + STATE.index);

        // If there are no more trials left, then we are at the end of
        // this phase
        if (STATE.index >= this.trials.length) {
            return this.finish();
        }
        
        // Load the new trialinfo and stimulus values
        this.trialinfo = this.trials[STATE.index];
        this.stimulus = this.trialinfo.stimulus;

        // Load the trial.html snippet into the body of the page
        psiTurk.showPage('trial.html');

        // Register the response handler to record responses
        var that = this;
        $('button').click(function () {
            that.record_response(this.name, this.value);
        });

        // Initialize the video players
        this.stim_player = this.init_player("#stim", "stimulus");
        this.fb_player = this.init_player("#video_feedback", "feedback");

        // Set appropriate backgrounds for phase elements
        // TODO: prestim image should be the floor
        set_poster("#prestim", this.stimulus + "~stimulus~A");
        set_poster("#fall_response", this.stimulus + "~stimulus~B");
        set_poster("#mass_response", this.stimulus + "~feedback~B");

        // Set the stimulus colors
        $(".left-color").css("background-color", this.trialinfo.color0);
        $(".right-color").css("background-color", this.trialinfo.color1);
        $("button.left-color").html(this.trialinfo.color0);
        $("button.right-color").html(this.trialinfo.color1);

        // Possibly show image (if the trials are not mass trials,
        // then we don't want to show the image).
        if (STATE.experiment_phase == EXPERIMENT.experiment) {
            $("#question-image").find("img").show();
        } else {
            $("#question-image").find("img").hide();
        }

        // Determine which feedback to show (stable or unstable)
        if (this.trialinfo.stable) {
            $("#stable-feedback").show();
        } else {
            $("#unstable-feedback").show();
        }

        // Display the question prompt
        $("#fall-question").show();

        // Update progress bar
        update_progress(STATE.index, this.trials.length);
    };

    // Phase 1: show the floor and "start" button
    this.phases[TRIAL.prestim] = function(that) {
        // Initialize the trial
        that.init_trial();

        // Actually show the prestim element
        debug("Show PRESTIM");
        $("#prestim").show();

        // Listen for a response to show the stimulus
        that.listening = true;
    };

    // Phase 2: show the stimulus
    this.phases[TRIAL.stim] = function (that) {
        debug("Show STIMULUS");

        // If the player isn't ready, then bind a ready handler to it.
        if (!that.stim_player.ready) {
            that.stim_player.bind("ready", function (e, api) {
                api.play();
            });
        }

        // Hide prestim and show stim
        replace("prestim", "stim");

        // Start playing the stimulus video
        that.stim_player.play(0);
    };

    // Phase 3: show the response options for "fall?" question
    this.phases[TRIAL.fall_response] = function (that) {
        debug("Show FALL_RESPONSE");

        // Hide stim and show fall_response
        replace("stim", "fall_response");

        // Destroy the player
        if (that.stim_player) that.stim_player.unload();

        // Listen for a response
        that.listening = true;
    };

    // Phase 4: show feedback
    this.phases[TRIAL.feedback] = function (that) {
        debug("Show FEEDBACK");

        var fb = that.trialinfo.feedback;
        var advance = function () {
            STATE.set_trial_phase(STATE.trial_phase + 1);
            that.show();
        };

        if (fb == "vfb") {
            // If we're showing video feedback, we need to show the
            // player and also display the text feedback.

            var play = function () {
                // If the player isn't ready, bind a ready handler to
                // it so it will start playing when it is ready
                if (!that.fb_player.ready) {
                    that.fb_player.bind("ready", function (e, api) {
                        api.play();
                    });
                }

                // Show the player and hide the fall responses
                $("#video_feedback").show();
                $("#fall_response").hide();

                // Play the video
                that.fb_player.play(0);
            };

            // Temporarily hide the text and video feedback
            $("#text_feedback").hide();
            $("#video_feedback").hide();

            // Show the div containing the text and video feedback
            $("#feedback").show();

            // Fade in the text feedback, then show the video
            $("#text_feedback").fadeIn($c.fade, play);

        } else if (fb == "fb") {
            // If we're only showing text feedback, we don't need to
            // bother with the video player.
            replace("fall_response", "feedback");
            setTimeout(advance, 2500);

        } else { 
            // If we're showing no feedback, just move on to the next
            // trial phase.
            setTimeout(advance, 200); 
        }
    };

    // Phase 5: show response options for "mass?" question
    this.phases[TRIAL.mass_response] = function (that) {
        // We won't query for the mass on every trial, so check to see
        // if this is a trial where we do.
        if (that.trialinfo["mass? query"]) {
            debug("Show MASS_RESPONSE");

            // Swap the fall? prompt for the mass? prompt
            $("#fall-question").hide();
            $("#mass-question").show();

            // Fade out text_feedback and fade in mass_response
            $("#text_feedback").fadeOut(
                $c.fade,
                function () {
                    replace("feedback", "mass_response");
                });

            // Destroy the player
            if (that.fb_player) that.fb_player.unload();

            // Listen for a response
            that.listening = true;

        } else {
            // Destroy the player
            if (that.fb_player) that.fb_player.unload();

            // Move on to the next trial
            STATE.set_trial_phase();
            STATE.set_index(STATE.index + 1);
            that.show();
        }
    };

    // Show the current trial at the currect phase
    this.show = function () {
        // Update the URL hash
        STATE.set_hash();
        // Call the phase setup handler
        this.phases[STATE.trial_phase](this);
        // Record when this phase started
        this.timestamp = new Date().getTime();
    };

    // Record a response (this could be either just clicking "start",
    // or actually a choice to the prompt(s))
    this.record_response = function(name, value) {
        // If we're not listening for a response, do nothing
        if (!this.listening) return;
        this.listening = false;

        var rt = (new Date().getTime()) - this.timestamp;
        var data = new DataRecord();
        data.update(this.trialinfo);
        data.update({response_time: rt});

        debug("Record response");

        // Parse the actual value of the data to record
        if (name == "play") {
            // Not a real response, so just save an empty string
            data.update({response: ""});

        } else if (name == "fall") {
            // We want a true/false boolean value
            data.update({response: Boolean(value)});

        } else if (name == "mass") {
            // Left/right refers to different colors
            // TODO: handle counterbalancing appropriately
            if (value == "left") {
                data.update({response: this.trialinfo.color0});
            } else if (value == "right") {
                data.update({response: this.trialinfo.color1});
            }
        }

        // Create the record we want to save
        psiTurk.recordTrialData(data.to_array());
        debug(data.to_array());

        // Tell the state to go to the next trial phase or trial
        if (name == "mass") {
            STATE.set_trial_phase();
            STATE.set_index(STATE.index + 1);
        } else {
            STATE.set_trial_phase(STATE.trial_phase + 1);
        }            

        // Update the page with the current phase/trial
        this.show();
    };

    // Complete the set of trials in the test phase
    this.finish = function() {
        debug("Finish test phase");

        // Reset the state object for the next experiment phase
        STATE.set_experiment_phase(STATE.experiment_phase + 1);
        STATE.set_instructions();
        STATE.set_index();
        STATE.set_trial_phase();

        // If we're at the end of the experiment, submit the data to
        // mechanical turk, otherwise go on to the next experiment
        // phase and show the relevant instructions
        if (STATE.experiment_phase >= EXPERIMENT.length) {

            // Send them to the debriefing form, but delay a bit, so
            // they know what's happening
            var debrief = function() {
                setTimeout(function () {
                    window.location = "/debrief?uniqueId=" + psiTurk.taskdata.id;
                }, 2000);
            };

            // Prompt them to resubmit the HIT, because it failed the first time
            var prompt_resubmit = function() {
                $("#resubmit_slide").click(resubmit);
                $(".slide").hide();
                $("#submit_error_slide").show();
            };

            // Show a page saying that the HIT is resubmitting, and
            // show the error page again if it times out or error
            var resubmit = function() {
                $(".slide").hide();
                $("#resubmit_slide").show();

                var reprompt = setTimeout(prompt_resubmit, 10000);
                psiTurk.saveData({
                    success: function() {
                        clearInterval(reprompt); 
                        finish();
                    }, 
                    error: prompt_resubmit
                });
            };

            // Render a page saying it's submitting
            psiTurk.showPage("submit.html")
            psiTurk.teardownTask();
            psiTurk.saveData({
                success: debrief, 
                error: prompt_resubmit
            });

        } else {
            CURRENTVIEW = new Instructions();
        }
    };

    // Initialize the current trial -- we need to do this here in
    // addition to in prestim in case someone refreshes the page in
    // the middle of a trial
    this.init_trial();

    // Start the test
    this.show();
};


// --------------------------------------------------------------------
// --------------------------------------------------------------------

/*******************
 * Run Task
 ******************/

$(document).ready(function() { 
    // Start the experiment
    STATE = new State();
    if (STATE.instructions) {
        CURRENTVIEW = new Instructions();
    } else {
        CURRENTVIEW = new TestPhase();
    }
});
