/* task.js
 * 
 * This file holds the main experiment code.
 * 
 * Requires:
 *   config.js
 *   psiturk.js
 *   utils.js
 */

// TODO: document TestPhase class
// TODO: refactor Questionnaire (or remove it?)

// Initialize flowplayer
var $f = flowplayer;
if ($f.support.firstframe) {
  $f(function (api, root) {
    // show poster when video ends
    api.bind("resume finish", function (e) {
      root.toggleClass("is-poster", /finish/.test(e.type));
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

        // Record the response time
        var rt = (new Date().getTime()) - this.timestamp;
        debug("'Next' button pressed");

        // TODO: review recording of Instructions data
        psiTurk.recordTrialData(["$c.instructions", STATE.index, rt]);

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

    // When the time response period begins
    this.timestamp; 
    this.listening = false;

    this.stim_player;
    this.fb_player;

    this.trials = $c.trials[STATE.experiment_phase];
    this.stimulus;
    this.trialinfo;
    
    this.phases = new Object();

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

    this.init_trial = function () {
        debug("initializing trial");

        if (STATE.index >= this.trials.length) {
            return this.finish();
        }
        
        this.trialinfo = this.trials[STATE.index];
        this.stimulus = this.trialinfo.stimulus;

        // Load the test.html snippet into the body of the page
        psiTurk.showPage('test.html');

        // register the response handler that is defined above to handle
        // responses
        var that = this;
        $('button').click(function () {
            that.record_response(this.value);
        });

        // Initialize the video players
        this.stim_player = this.init_player("#stim", "stimulus");
        this.fb_player = this.init_player("#video_feedback", "feedback");

        // Set appropriate backgrounds
        // TODO: prestim image should be the floor
        set_poster("#prestim", this.stimulus + "~stimulus~A");
        set_poster("#fall_response", this.stimulus + "~stimulus~B");
        set_poster("#mass_response", this.stimulus + "~feedback~B");

        // Set the stimulus colors
        $(".left-color").css("background-color", this.trialinfo.color0);
        $(".right-color").css("background-color", this.trialinfo.color1);
        $("button.left-color").html(this.trialinfo.color0);
        $("button.right-color").html(this.trialinfo.color1);

        // Possibly show image
        if (STATE.experiment_phase == EXPERIMENT.experiment) {
            $("#question-image").find("img").show();
        } else {
            $("#question-image").find("img").hide();
        }

        // Determine which feedback to show
        if (this.trialinfo.stable) {
            $("#stable-feedback").show();
        } else {
            $("#unstable-feedback").show();
        }

        // Display the question
        $("#fall-question").show();

        // Update progress bar
        update_progress(STATE.index, this.trials.length);
    };

    // Phase 1: show the floor and "start" button
    this.phases[TRIAL.prestim] = function(that) {
        // Initialize the trial
        that.init_trial();

        debug("show prestim");
        $("#prestim").show();

        that.listening = true;
    };

    // Phase 2: show the stimulus
    this.phases[TRIAL.stim] = function (that) {
        debug("show stimulus");
        if (!that.stim_player.ready) {
            that.stim_player.bind("ready", function (e, api) {
                api.play();
            });
        }
        that.stim_player.play(0);
        replace("prestim", "stim");
    };

    // Phase 3: show the response options for "fall?" question
    this.phases[TRIAL.fall_response] = function (that) {
        debug("show fall responses");
        replace("stim", "fall_response");
        if (that.stim_player) that.stim_player.unload();
        that.listening = true;
    };

    // Phase 4: show feedback
    this.phases[TRIAL.feedback] = function (that) {
        debug("show feedback");

        var fb = that.trialinfo.feedback;
        var advance = function () {
            STATE.set_trial_phase(STATE.trial_phase + 1);
            that.show();
        };

        if (fb == "vfb") {
            var play = function () {
                if (!that.fb_player.ready) {
                    that.fb_player.bind("ready", function (e, api) {
                        api.play();
                    });
                }
                $("#video_feedback").show();
                $("#fall_response").hide();
                that.fb_player.play(0);
            };

            $("#text_feedback").hide();
            $("#video_feedback").hide();
            $("#feedback").show();
            $("#text_feedback").fadeIn($c.fade, play);

        } else if (fb == "fb") {
            replace("fall_response", "feedback");
            setTimeout(advance, 2500);

        } else { setTimeout(advance, 200); }
    };

    // Phase 5: show response options for "mass?" question
    this.phases[TRIAL.mass_response] = function (that) {
        if (that.trialinfo["mass? query"]) {
            debug("show mass responses");

            $("#fall-question").hide();
            $("#mass-question").show();

            replace("text_feedback", "mass_response");
            $("#text_feedback").fadeOut(
                $c.fade,
                function () {
                    replace("feedback", "mass_response");
                });

            if (that.fb_player) that.fb_player.unload();
            that.listening = true;

        } else {
            if (that.fb_player) that.fb_player.unload();
            STATE.set_trial_phase();
            STATE.set_index(STATE.index + 1);
            that.show();
        }
    };

    this.show = function () {
        STATE.set_hash();
        debug("next (trial_phase=" + STATE.trial_phase + ")");

        this.phases[STATE.trial_phase](this);
        this.timestamp = new Date().getTime();
    };

    this.record_response = function(response) {
        debug("in response handler");
        if (!this.listening) return;
        
        this.listening = false;

        // TODO: record response data for TestPhase

        if (response == "left" || response == "right") {
            STATE.set_trial_phase();
            STATE.set_index(STATE.index + 1);
        } else {
            STATE.set_trial_phase(STATE.trial_phase + 1);
        }

        this.show();
    };

    this.finish = function() {
        debug("finish test phase");

        $("button").click(function() {}); // Unbind buttons

        STATE.set_experiment_phase(STATE.experiment_phase + 1);
        STATE.set_instructions();
        STATE.set_index();
        STATE.set_trial_phase();

        if (STATE.experiment_phase >= EXPERIMENT.length) {
            CURRENTVIEW = new Questionnaire();
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


/****************
 * Questionnaire *
 ****************/

var Questionnaire = function() {

    var error_message = "<h1>Oops!</h1><p>Something went wrong submitting your HIT. This might happen if you lose your internet connection. Press the button to resubmit.</p><button id='resubmit'>Resubmit</button>";

    record_responses = function() {

        // TODO: fix recording of Questionniare data
        psiTurk.recordTrialData(['postquestionnaire', 'submit']);

        $('textarea').each( function(i, val) {
            psiTurk.recordUnstructuredData(this.id, this.value);
        });
        $('select').each( function(i, val) {
            psiTurk.recordUnstructuredData(this.id, this.value);                
        });

    };
    
    finish = function() {
        debriefing();
    };
    
    prompt_resubmit = function() {
        replaceBody(error_message);
        $("#resubmit").click(resubmit);
    };

    resubmit = function() {
        replaceBody("<h1>Trying to resubmit...</h1>");
        reprompt = setTimeout(prompt_resubmit, 10000);
        
        psiTurk.saveData({
            success: function() {
                clearInterval(reprompt); 
                finish();
            }, 
            error: prompt_resubmit}
                        );
    };

    // Load the questionnaire snippet 
    psiTurk.showPage('postquestionnaire.html');
    psiTurk.recordTrialData(['postquestionnaire', 'begin']);
    
    $("#continue").click(function () {
        record_responses();
        psiTurk.teardownTask();
            psiTurk.saveData({success: finish, error: prompt_resubmit});
    });
    
};


var debriefing = function() { 
    window.location="/debrief?uniqueId=" + psiTurk.taskdata.id; 
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
