/* task.js
 * 
 * This file holds the main experiment code.
 * 
 * Requires:
 *   config.js
 *   psiturk.js
 *   utils.js
 */

// Initialize flowplayer
var $f = flowplayer;

// Create and initialize the experiment configuration object
var $c = new Config(condition, counterbalance);

// Initalize psiturk object
var psiTurk = new PsiTurk();

// Preload the HTML template pages that we need for the experiment
psiTurk.preloadPages($c.pages);

// Objects to keep track of the current phase and state
var CURRENTVIEW;
var STATE;
var PLAYER;


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

        debug("show slide " + this.pages[STATE.index]);

        // Load the next page of instructions
        $(".slide").hide();
        var slide = $("#" + this.pages[STATE.index]);
        slide.fadeIn($c.fade);

        // Update the URL hash
        STATE.set_hash();

        // Bind a handler to the "next" button. We have to wrap it in
        // an anonymous function to preserve the scope.
        var that = this;
        slide.find('.next').click(function () {
            that.record_response();
        });
        
        var example = this.examples[STATE.index];
        var player_elem = slide.find(".example");

        if (example) {
            // Set mass colors
            set_colors(example);

            if (player_elem.length > 0) {
                // Load the video player
                var videos = [get_video_formats(
                    example.stimulus + "~stimulus",
                    STATE.experiment_phase)];
                var player_id = "#" + player_elem.attr("id");

                // Start the video immediately
                var on_ready = function (e, api) {
                    api.play();
                };
                // Loop the player by restarting the current video
                var on_finish = function (e, api) {
                    api.prev();
                };

                // Initialize the player and start it
                this.player = make_player(player_id, videos)
                    .bind("ready", on_ready)
                    .bind("finish", on_finish)
                    .play(0);
            }
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
        // TODO: enable finish instructions
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

    // List of trials in this block of the experiment
    // TODO: use the full list of trials
    this.trials = $c.trials[STATE.experiment_phase].slice(0, 2);
    // Information about the current trial
    this.trialinfo;
    // The current stimulus name
    this.stimulus;
    
    // Handlers to setup each phase of a trial
    this.phases = new Object();

    // Initialize a new trial. This is called either at the beginning
    // of a new trial, or if the page is reloaded between trials.
    this.init_trial = function () {
        debug("Initializing trial " + STATE.index);
        $(".phase").hide();

        // If there are no more trials left, then we are at the end of
        // this phase
        if (STATE.index >= this.trials.length) {
            this.finish();
            return false;
        }
        
        // Load the new trialinfo and stimulus values
        this.trialinfo = this.trials[STATE.index];
        this.stimulus = this.trialinfo.stimulus;

        // Set appropriate backgrounds for phase elements
        set_poster("#prestim", this.stimulus + "~floor", STATE.experiment_phase);
        set_poster("#prefeedback", this.stimulus + "~stimulus~B", STATE.experiment_phase);
        set_poster("#fall_response", this.stimulus + "~stimulus~B", STATE.experiment_phase);
        if (this.trialinfo.stable) {
            set_poster("#mass_response", this.stimulus + "~feedback~A", STATE.experiment_phase);
        } else {
            set_poster("#mass_response", this.stimulus + "~feedback~B", STATE.experiment_phase);
        }

        // Set the stimulus colors
        set_colors(this.trialinfo);

        // Display the question prompt
        $(".question").hide();
        if (this.trialinfo["fall? query"]) {
            $("#fall-question").show();
        } else if (this.trialinfo["mass? query"]) {
            $("#mass-question").show();
        }

        // Possibly show image (if the trials are not mass trials,
        // then we don't want to show the image).
        $(".question-image").hide();
        if (STATE.experiment_phase == EXPERIMENT.experimentA) {
            $("#question-image-A").show();
        } else if (STATE.experiment_phase == EXPERIMENT.experimentB) {
            $("#question-image-B").show();
        }

        // Determine which feedback to show (stable or unstable)
        if (this.trialinfo.stable) {
            $("#stable-feedback").show();
            $("#unstable-feedback").hide();
        } else {
            $("#stable-feedback").hide();
            $("#unstable-feedback").show();
        }

        // Update progress bar
        update_progress(STATE.index, this.trials.length);

        // Register the response handler to record responses
        var that = this;
        $("body").focus().keypress(function (e) {
            that.record_response(e.keyCode);
        });

        return true;
    };

    // Phase 1: show the floor and "start" button
    this.phases[TRIAL.prestim] = function(that) {
        $("#phase-container").hide();

        // Initialize the trial
        if (that.init_trial()) {
            // Actually show the prestim element
            debug("Show PRESTIM");

            setTimeout(function () {
                $("#prestim").show();
                $("#phase-container").fadeIn($c.fade, function () {
                    // Listen for a response to show the stimulus
                    that.listening = true;
                });
            }, 100);
        }
    };

    // Phase 2: show the stimulus
    this.phases[TRIAL.stim] = function (that) {
        debug("Show STIMULUS");
            
        // Hide prestim and show stim
        show_phase("stim");

        // Start playing the stimulus video
        PLAYER.play("stimulus");
    };

    // Phase 3: show the response options for "fall?" question
    this.phases[TRIAL.fall_response] = function (that) {
        // We won't ask for fall predictions on every trial, so check
        // to see if this is a trial where we do.
        if (that.trialinfo["fall? query"]) {
            debug("Show FALL_RESPONSE");

            // Hide stim and show fall_response
            show_phase("fall_response");

            // Listen for a response
            that.listening = true;

        } else {
            // Move on to the next trial
            STATE.set_trial_phase(STATE.trial_phase + 1);
            that.show();
        }
    };

    this.phases[TRIAL.prefeedback] = function (that) {
        if (!that.trialinfo["fall? query"] && that.trialinfo["feedback"] != "nfb") {
            debug("Show PREFEEDBACK");

            // Show the prefeedback element
            $("#feedback-instructions").show();
            show_phase("prefeedback");

            // Listen for a response to show the feedback
            that.listening = true;

        } else {
            $("#feedback-instructions").hide();
            show_phase("prefeedback");

            STATE.set_trial_phase(STATE.trial_phase + 1);
            that.show();
        }
    };

    // Phase 4: show feedback
    this.phases[TRIAL.feedback] = function (that) {
        debug("Show FEEDBACK");

        var fb = that.trialinfo.feedback;
        var advance = function () {
            STATE.set_trial_phase(STATE.trial_phase + 1);
            that.show();
        };

        if (fb == "vfb" && !that.trialinfo.stable) {
            // If we're showing video feedback, we need to show the
            // player and also display the text feedback.
            
            // Show the player and hide the fall responses
            show_phase("stim", function () { 
		$("#feedback").fadeIn($c.fade); 
	    });

            // Play the video
            PLAYER.play("feedback");

        } else if (fb == "fb" || (fb == "vfb" && that.trialinfo.stable)) {
            // If we're only showing text feedback, we don't want to
            // actually play a video

	    // Show the player and hide the fall responses
            show_phase("stim", function () { 
		$("#feedback").fadeIn($c.fade); 
	    });

	    // Load the video (but don't actually play it)
            PLAYER.load("feedback");
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
            $("#feedback").hide();
            show_phase("mass_response");

            // Listen for a response
            that.listening = true;

        } else {
            $("#feedback").hide();
            
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
    this.record_response = function(key) {
        // If we're not listening for a response, do nothing
        if (!this.listening) return;

        // Record response time
        var rt = (new Date().getTime()) - this.timestamp;

        // Parse the actual value of the data to record
        var response = KEYS[STATE.trial_phase][key];
        if (response == undefined) return;
        this.listening = false;

        debug("Record response: " + response);

        var data = new DataRecord();
        data.update(this.trialinfo);
        data.update({response_time: rt, response: response});

        // Create the record we want to save
        psiTurk.recordTrialData(data.to_array());
        debug(data.to_array());

        // Tell the state to go to the next trial phase or trial
        if (STATE.trial_phase == (TRIAL.length - 1)) {
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
                $("#submit_error_slide").fadeIn($c.fade);
            };

            // Show a page saying that the HIT is resubmitting, and
            // show the error page again if it times out or error
            var resubmit = function() {
                $(".slide").hide();
                $("#resubmit_slide").fadeIn($c.fade);

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

    // Load the trial html page
    $(".slide").hide();
    $("#trial").fadeIn($c.fade);

    // Initialize the current trial -- we need to do this here in
    // addition to in prestim in case someone refreshes the page in
    // the middle of a trial
    if (this.init_trial()) {
        // Start the test
        this.show();
    };
};


// --------------------------------------------------------------------
// --------------------------------------------------------------------

/*******************
 * Run Task
 ******************/

$(document).ready(function() { 
    psiTurk.showPage("trial.html");
    
    // Start the experiment
    STATE = new State();
    PLAYER = new Player();

    if (STATE.instructions) {
        CURRENTVIEW = new Instructions();
    } else {
        CURRENTVIEW = new TestPhase();
    }
});
