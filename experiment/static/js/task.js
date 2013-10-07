/*
 * Requires:
 *     psiturk.js
 *     utils.js
 */

// Initialize flowplayer
var $f = flowplayer;
$f();

if ($f.support.firstframe) {
  $f(function (api, root) {
    // show poster when video ends
    api.bind("resume finish", function (e) {
      root.toggleClass("is-poster", /finish/.test(e.type));
    });
  });
}

// Initalize psiturk object
var psiTurk = PsiTurk();

// Task object to keep track of the current phase
var CURRENTVIEW;

/********************
 * HTML manipulation
 *
 * All HTML files in the templates directory are requested 
 * from the server when the PsiTurk object is created above. We
 * need code to get those pages from the PsiTurk object and 
 * insert them into the document.
 *
 ********************/


/*************************
 * INSTRUCTIONS         
 *************************/

var Instructions = function(state) {
    self = this;
    
    debug("initialize instructions");
 
    self.state = state;
    self.pages = $c.instructions[state.experiment_phase].pages;
    self.examples = $c.instructions[state.experiment_phase].examples;

    self.timestamp;
    self.player;
    
    self.show = function() {
        debug("next (index=" + self.state.index + ")");
        set_state(self.self.state);
        
        // show the next page of instructions
        psiTurk.showPage(self.pages[self.state.index]);

        // bind a handler to the "next" button
        $('.next').click(button_press);
        
        // load the player
        if (examples[state.index]) {
            var video = self.examples[self.state.index] + "~stimulus";
            self.player = Player("player", [video], true);
            set_poster("#player.flowplayer", video + "~A");
            self.player.bind("ready", function (e, api) {
                api.play();
            });
            self.player.play(0);
        }

        // Record the time that an instructions page is presented
        self.timestamp = new Date().getTime();
    };

    self.button_press = function() {
        debug("button pressed");

        // Record the response time
        var rt = (new Date().getTime()) - self.timestamp;
        psiTurk.recordTrialData(["$c.instructions", self.state.index, rt]);

         // destroy the video player
        if (self.player) {
            self.player.unload();
        }

        self.state.index = self.state.index + 1;
        if (self.state.index == self.pages.length) {
            self.finish();
        } else {
            self.show();
        }
    };

    self.finish = function() {
        debug("done with instructions")

        // Record that the user has finished the instructions and 
        // moved on to the experiment. This changes their status code
        // in the database.
        // if (state.experiment_phase == EXPERIMENT.pretest) {
        //     psiTurk.finishInstructions();
        // }

        self.state.instructions = 0;
        self.state.index = 0;
        self.state.trial_phase = TRIAL.prestim;
        CURRENTVIEW = new TestPhase(state);
    };

    self.show();
    return self;
};



/********************
 * STROOP TEST       *
 ********************/

var TestPhase = function(state) {
    self = this;

    debug("initialize test phase");

    self.starttime; // time response period begins
    self.listening = false;

    self.state = state;
    self.stim_player;
    self.fb_player;

    self.trials = $c.trials[state.experiment_phase];
    self.stimulus;
    self.trialinfo;
    
    self.phases = new Object();

    self.init_trial = function () {
        debug("initializing trial");
        
        self.trialinfo = self.trials[self.state.index];
        self.stimulus = self.trialinfo.stimulus;

        // Load the test.html snippet into the body of the page
        psiTurk.showPage('test.html');

        // register the response handler that is defined above to handle
        // responses
        $('button').click(self.response_handler);

        // Initialize the video players
        self.stim_player = Player("stim", [self.stimulus + "~stimulus"], false);
        self.fb_player = Player("video_feedback", [self.stimulus + "~feedback"], false);

        // // Hide the overlapping trial components
        // $(".phase").hide();

        // Set appropriate backgrounds
        set_poster("#prestim", self.stimulus + "~stimulus~A");
        set_poster("#stim.flowplayer", self.stimulus + "~stimulus~A");
        set_poster("#fall_response", self.stimulus + "~stimulus~B");
        set_poster("#video_feedback.flowplayer", self.stimulus + "~feedback~A");
        set_poster("#mass_response", self.stimulus + "~feedback~B");

        $("#stim").addClass("is-poster");
        $("#video_feedback").addClass("is-poster");

        // Set the stimulus colors
        $(".left-color").css("background-color", self.trialinfo.color0);
        $(".right-color").css("background-color", self.trialinfo.color1);
        $("button.left-color").html(self.trialinfo.color0);
        $("button.right-color").html(self.trialinfo.color1);

        // Possibly show image
        if (self.state.experiment_phase == EXPERIMENT.experiment) {
            $("#question-image").find("img").show();
        } else {
            $("#question-image").find("img").hide();
        }

        // Determine which feedback to show
        if (self.trialinfo.stable) {
            $("#stable-feedback").show();
        } else {
            $("#unstable-feedback").show();
        }

        // Display the question
        $("#fall-question").show();

        // Update progress bar
        update_progress(self.state.index, self.trials.length);
    };

    // Phase 1: show the floor and "start" button
    self.phases[TRIAL.prestim] = function() {
        // Initialize the trial
        self.init_trial();

        // XXX this should be the floor
        debug("show prestim");
        $("#prestim").show();

        self.listening = true;
    };

    // Phase 2: show the stimulus
    self.phases[TRIAL.stim] = function () {
        debug("show stimulus");

        var onfinish_stim = function (e, api) {
            debug("stimulus finished");
            api.unbind("finish");

            set_poster("#stim.flowplayer", self.stimulus + "~stimulus~B");
            $("#stim").addClass("is-poster");

            self.state.trial_phase = self.state.trial_phase + 1;
            self.show();
        };

        self.stim_player
            .bind("finish", onfinish_stim)
            .play(0);

        show_and_hide("stim", "prestim");
    };

    // Phase 3: show the response options for "fall?" question
    self.phases[TRIAL.fall_response] = function () {
        debug("show fall responses");
        show_and_hide("fall_response", "stim");
        if (self.stim_player) {
            self.stim_player.unload();
        }
        self.listening = true;
    };

    // Phase 4: show feedback
    self.phases[TRIAL.feedback] = function () {
        debug("show feedback");

        var fb = self.trialinfo.feedback;

        var advance = function () {
            self.state.trial_phase = self.state.trial_phase + 1;
            self.show();
        };
            
        var onfinish_feedback = function (e, api) {
            debug("done showing feedback");
            api.unbind("finish");

            set_poster("#video_feedback.flowplayer", self.stimulus + "~feedback~B");
            $("#video_feedback").addClass("is-poster");

            advance();
        };

        if (fb == "vfb") {
            $("#text_feedback").hide();
            $("#video_feedback").show();
            $("#feedback").show();
            $("#text_feedback").fadeIn($c.fade);
            $("#fall_response").hide();
         
            self.fb_player
                .bind("finish", onfinish_feedback)
                .play(0);

        } else if (fb == "fb") {
            show_and_hide("feedback", "fall_response");
            setTimeout(advance, 2500);

        } else {
            setTimeout(advance, 200);
        }
    };

    // Phase 5: show response options for "mass?" question
    self.phases[TRIAL.mass_response] = function () {
        if (self.trialinfo["mass? query"]) {
            debug("show mass responses");

            $("#fall-question").hide();
            $("#mass-question").show();

            show_and_hide("mass_response", "feedback");
            if (self.fb_player) {
                self.fb_player.unload();
            }

            self.listening = true;

        } else {
            if (self.fb_player) {
                self.fb_player.unload();
            }

            self.state.trial_phase = TRIAL.prestim;
            self.state.index = self.state.index + 1;
            self.show();
        }
    };

    self.show = function () {
        if (self.state.index >= self.trials.length) {
            self.finish();
        } else {
            set_state(state);
            debug("next (trial_phase=" + self.state.trial_phase + ")");

            self.phases[self.state.trial_phase]();
            self.starttime = new Date().getTime();
        }
    };

    self.response_handler = function(e) {
        debug("in response handler");
        if (!self.listening) return;
        
        var response = this.value;
        self.listening = false;

        if (response == "left" || response == "right") {
            self.state.trial_phase = TRIAL.prestim;
            self.state.index = self.state.index + 1;
        } else {
            self.state.trial_phase = self.state.trial_phase + 1;
        }

        self.show();
    };

    self.finish = function() {
        debug("finish test phase");

        $("button").click(function() {}); // Unbind buttons

        self.state.experiment_phase = self.state.experiment_phase + 1;
        self.state.trial_phase = undefined;
        self.state.index = 0;
        self.state.instructions = 1;

        if (self.state.experiment_phase >= EXPERIMENT.length) {
            CURRENTVIEW = new Questionnaire(state);
        } else {
            CURRENTVIEW = new Instructions(state);
        }
    };

    // Initialize the current trial -- we need to do this here in
    // addition to in prestim in case someone refreshes the page in
    // the middle of a trial
    self.init_trial();

    // Start the test
    self.show();

    return self;
};


/****************
 * Questionnaire *
 ****************/

var Questionnaire = function() {

    var error_message = "<h1>Oops!</h1><p>Something went wrong submitting your HIT. This might happen if you lose your internet connection. Press the button to resubmit.</p><button id='resubmit'>Resubmit</button>";

    record_responses = function() {

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


var debriefing = function() { window.location="/debrief?uniqueId=" + psiTurk.taskdata.id; };


// --------------------------------------------------------------------
// --------------------------------------------------------------------

/*******************
 * Run Task
 ******************/

$(document).ready(function() { 
    load_config(
        function () {
            // All pages to be loaded
            var pages = $.map($c.instructions, 
                  function(item) { 
                      return item.pages 
                  });
            pages.push("test.html");
            pages.push("postquestionnaire.html");
            psiTurk.preloadPages(pages);

            // Start the experiment
            var state = get_state();
            if (state.instructions) {
                CURRENTVIEW = new Instructions(state);
            } else {
                CURRENTVIEW = new TestPhase(state);
            }
        }
    );
});
