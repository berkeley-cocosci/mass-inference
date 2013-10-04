/*
 * Requires:
 *     psiturk.js
 *     utils.js
 */

// Initialize flowplayer
var $f = flowplayer;

// Initalize psiturk object
var psiTurk = PsiTurk();

// Task object to keep track of the current phase
var currentview;

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
    debug("initialize instructions");

    var state = state;
    var pages = INSTRUCTIONS[state.experiment_phase].pages;
    var examples = INSTRUCTIONS[state.experiment_phase].examples;

    var timestamp;
    var player;
    
    var show = function() {
        debug("next (index=" + state.index + ")");
        set_state(state);
        
        // show the next page of instructions
        psiTurk.showPage(pages[state.index]);

        // bind a handler to the "next" button
        $('.next').click(button_press);
        
        // load the player
        if (examples[state.index]) {
            player = Player([examples[state.index] + "~stimulus"], true);
            player.play(0);
        }

        // Record the time that an instructions page is presented
        timestamp = new Date().getTime();
    };

    var button_press = function() {
        debug("button pressed");

        // Record the response time
        var rt = (new Date().getTime()) - timestamp;
        psiTurk.recordTrialData(["INSTRUCTIONS", state.index, rt]);

         // destroy the video player
        if (player) {
            player.unload();
        }

        state.index = state.index + 1;
        if (state.index == pages.length) {
            finish();
        } else {
            show();
        }
    };

    var finish = function() {
        debug("done with instructions")

        // Record that the user has finished the instructions and 
        // moved on to the experiment. This changes their status code
        // in the database.
        // if (state.experiment_phase == EXPERIMENT.pretest) {
        //     psiTurk.finishInstructions();
        // }

        state.instructions = 0;
        state.index = 0;
        state.trial_phase = TRIAL.prestim;
        currentview = new TestPhase(state);
    };

    show();
};



/********************
 * STROOP TEST       *
 ********************/

var TestPhase = function(state) {
    debug("initialize test phase");

    var starttime; // time response period begins
    var listening = false;

    var state = state;
    var player;

    var fall_question = "<b>Question:</b> Will the tower fall down?";
    var mass_question = "<b>Question:</b> Which is the <b>heavy</b> color?";
    
    var phases = new Object();

    // Phase 1: show the floor and "start" button
    phases[TRIAL.prestim] = function() {
        debug("show prestim");

        set_bg_image("prestim", trialinfo(state).stimulus + "~floor");
        $("#prestim").fadeIn($c.fade);

        listening = true;
    };

    // Phase 2: show the stimulus
    phases[TRIAL.stim] = function () {
        debug("show stimulus");
        
        var onload_stim = function (e, api) {
            debug("stimulus loaded");
            // set_bg_image("player", trialinfo(state).stimulus + "~stimulus~B");
        };

        var onfinish_stim = function (e, api) {
            debug("stimulus finished");
            player.unbind("finish");
            state.trial_phase = state.trial_phase + 1;
            show();
        };

        $("#stim").show();

        player.bind("load", onload_stim);
        player.bind("finish", onfinish_stim);
        player.play(2 * state.index);
    };

    // Phase 3: show the response options for "fall?" question
    phases[TRIAL.fall_response] = function () {
        debug("show fall responses");

        // set_bg_image("responses", trialinfo(state).stimulus + "~stimulus~B");
        $("#fall_response").fadeIn($c.fade);
        listening = true;
    };

    // Phase 4: show feedback
    phases[TRIAL.feedback] = function () {
        debug("show feedback");

        var stable = trialinfo(state).stable;
        var videofb = trialinfo(state).feedback == "vfb";
        var textfb = (trialinfo(state).feedback == "vfb" || trialinfo(state).feedback == "fb");
        
        var time;
        $("#feedback").show();
        if (textfb) {
            if (videofb && stable) {
                time = 3000;
            } else {
                time = 2000;
            }
            if (stable) {
                $("#stable-feedback").fadeIn($c.fade);
            } else {
                $("#unstable-feedback").fadeIn($c.fade);
            }
        } else {
            time = 200;
        }

        var onfinish_feedback = function (e, api) {
            debug("done showing feedback");
            player.unbind("finish");
            state.trial_phase = state.trial_phase + 1;
            show();
        };

        // if videofb (video feedback) is true, then show a video
        // and text
        if (videofb && !stable) {
            player.bind("finish", onfinish_feedback);
            player.play(2 * state.index + 1);
        }
        
        // otherwise just show text
        else {
            setTimeout(onfinish_feedback, time);
        }
    };

    // Phase 5: show response options for "mass?" question
    phases[TRIAL.mass_response] = function () {
        if (trialinfo(state)["mass? query"]) {
            debug("show mass responses");
            $("#question").html(mass_question);
            $("#mass_response").fadeIn($c.fade);
            listening = true;
        } else {
            state.trial_phase = TRIAL.prestim;
            state.index = state.index + 1;
            show();
        }
    };

    var show = function () {
        if (state.index >= trials(state).length) {
            finish();
        } else {
            set_state(state);
            debug("next (trial_phase=" + state.trial_phase + ")");

            $(".left-color").css("background-color", trialinfo(state).color0);
            $(".right-color").css("background-color", trialinfo(state).color1);
            $("button.left-color").html(trialinfo(state).color0);
            $("button.right-color").html(trialinfo(state).color1);

            $("#question").html(fall_question);

            // Possibly show image
            if (state.experiment_phase == EXPERIMENT.experiment) {
                $("#question-image").find("img").show();
            } else {
                $("#question-image").find("img").hide();
            }

            // Update progress bar
            update_progress(state.index, trials(state).length);

            $(".phase").hide();        
            phases[state.trial_phase]();
            starttime = new Date().getTime();
        }
    };

    var response_handler = function(e) {
        // if (!listening) return;

        // var keyCode = e.keyCode,
        // response;

        // switch (keyCode) {
        // case 82:
        //     // "R"
        //     response="red";
        //     break;
        // case 71:
        //     // "G"
        //     response="green";
        //     break;
        // case 66:
        //     // "B"
        //     response="blue";
        //     break;
        // default:
        //     response = "";
        //     break;
        // }
        // if (response.length>0) {
        //     listening = false;
        //     var hit = response == stim[1];
        //     var rt = new Date().getTime() - wordon;

        //     psiTurk.recordTrialData(["TEST", stim[0], stim[1], stim[2], response, hit, rt]);
            
        //     remove_word();
        //     next();
        // }

        debug("in response handler");
        if (!listening) return;
        
        var response = this.value;
        listening = false;

        if (response == "left" || response == "right") {
            state.trial_phase = TRIAL.prestim;
            state.index = state.index + 1;
        } else {
            state.trial_phase = state.trial_phase + 1;
        }

        show();

        // if (response == "play") {
        //     debug("--> stim");
        //     next(TRIAL.stim);
        // } else if (response == "stable" || response == "unstable") {
        //     debug("--> feedback");
        //     next(feedback);
        // } else if (response == "left" || response == "right") {
        //     debug("--> prestim");
        //     next("prestim");
        // }
    };

    var finish = function() {
        debug("finish test phase");

        $("button").click(function() {}); // Unbind buttons

        state.experiment_phase = state.experiment_phase + 1;
        state.trial_phase = undefined;
        state.index = 0;
        state.instructions = 1;

        if (state.experiment_phase >= EXPERIMENT.length) {
            currentview = new Questionnaire(state);
        } else {
            currentview = new Instructions(state);
        }
    };
    
    // Load the test.html snippet into the body of the page
    psiTurk.showPage('test.html');

    // Initialize the video player
    var videos = $.map(trials(state), function(item) {
        return [item.stimulus + "~stimulus", item.stimulus + "~feedback"];
    });
    player = Player(videos, false);

    // var playlist = $(".fp-playlist");
    // playlist.attr("id", "stim");
    // var classes = playlist.attr("class").split(" ");
    // classes[classes.length] = "phase";
    // playlist.attr("class", classes.join(" "));

    // register the response handler that is defined above to handle any
    // key down events.
    // $("body").focus().keydown(response_handler); 
    $('button').click(response_handler);

    // Start the test
    show();
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
            psiTurk.preloadPages($.map(INSTRUCTIONS, function(item) { return item.pages }));
            psiTurk.preloadPages(["test.html", "postquestionnaire.html"]);

            // Start the experiment
            var state = get_state();
            if (state.instructions) {
                currentview = new Instructions(state);
            } else {
                currentview = new TestPhase(state);
            }
        }
    );
});
