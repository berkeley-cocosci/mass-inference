/* utils.js
 * 
 * This file contains helper utility functions/objects for use by the
 * main experiment code.
 */


// Object to hold the state of the experiment. It is initialized to
// reflect the hash in the URL (see set_hash and load_hash for
// details).
var State = function () {

    // One of the phases defined in EXPERIMENT
    this.experiment_phase;
    // 0 (false) or 1 (true)
    this.instructions;
    // Trial index
    this.index;
    // One of the phases defined in TRIAL
    this.trial_phase;

    // Update the experiment phase. Defaults to EXPERIMENT.pretest.
    this.set_experiment_phase = function (experiment_phase) {
        if (experiment_phase != undefined) {
            this.experiment_phase = experiment_phase;
        } else {
            this.experiment_phase = EXPERIMENT.pretest;
        }
    };

    // Update the instructions flag. Defaults to 1.
    this.set_instructions = function (instructions) {
        if (instructions != undefined) {
            this.instructions = instructions;
        } else {
            this.instructions = 1;
        }
    };

    // Update the trial index. Defaults to 0.
    this.set_index = function (index) {
        if (index != undefined) {
            this.index = index;
        } else {
            this.index = 0;
        }
    };

    // Update the trial phase. Defaults to TRIAL.prestim.
    this.set_trial_phase = function (trial_phase) {
        if (!this.instructions) {
            if (trial_phase != undefined) {
                this.trial_phase = trial_phase;
            } else {
                this.trial_phase = TRIAL.prestim;
            }
        } else {
            this.trial_phase = undefined;
        }
    };

    // Set the URL hash based on the current state. If
    // this.instructions is 1, then it will look like:
    //
    //     <experiment_phase>-<instructions>-<index>
    // 
    // Otherwise, if this.instructions is 0, it will be:
    //
    //     <experiment_phase>-<instructions>-<index>-<trial_phase>
    //
    // Returns the URL hash string.
    this.set_hash = function () {
        var parts = [
            this.experiment_phase,
            this.instructions,
            this.index
        ];

        if (!this.instructions) {
            parts[parts.length] = this.trial_phase;
        }

        var hash = parts.join("-");
        window.location.hash = hash;
        return hash;
    };

    // Update the State object based on the URL hash
    this.load_hash = function () {
        // get the URL hash, and remove the # from the front
        var hash = window.location.hash.slice(1);

        if (window.location.hash == "") {
            // no hash is present, so use the defaults
            this.set_experiment_phase();
            this.set_instructions();
            this.set_index();
            this.set_trial_phase();

        } else {
            // split the hash into its components and set them
            var parts = hash.split("-").map(
                function (item) {
                    return parseInt(item);
                });
            this.set_experiment_phase(parts[0]);
            this.set_instructions(parts[1]);
            this.set_index(parts[2]);
            this.set_trial_phase(parts[3]);
        }
    };

    // Return a list of the state's properties in human-readable form,
    // to be recorded as data in the database
    this.as_data = function () {
        var experiment_phase;
        var instructions = Boolean(this.instructions);
        var index = this.index;
        var trial_phase;
        
        // Find the name of the experiment phase
        for (item in EXPERIMENT) {
            if (EXPERIMENT[item] == this.experiment_phase) {
                experiment_phase = item;
                break;
            }
        }

        // Find the name of the trial phase (or just use
        // an empty string if instructions is true)
        if (!instructions) {
            for (item in TRIAL) {
                if (TRIAL[item] == this.trial_phase) {
                    trial_phase = item;
                    break
                }
            }
        } else {
            trial_phase = "";
        }

        return [experiment_phase, instructions, index, trial_phase];
    };

    // Initialize the State object components
    this.load_hash();
};

// Object to properly format rows of data
var DataRecord = function () {
    this.update = function (other) {
        _.extend(this, other);
    };

    this.to_array = function () {
        return STATE.as_data().concat([
            this.feedback,
            this.ratio,
            this.counterbalance,
            this.trial,
            this.stimulus,
            this.response,
            this.response_time,
            this.angle,
            this.color0,
            this.color1,
            this.feedback_time,
            this.presentation_time,
            this.stable
        ]);
    };
};

// Log a message to the console, if debug mode is turned on.
function debug(msg) {
    if ($c.debug) {
        console.log(msg);
    }
}

// Throw an assertion error if a statement is not true.
function AssertException(message) { this.message = message; }
AssertException.prototype.toString = function () {
    return 'AssertException: ' + this.message;
};
function assert(exp, message) {
    if (!exp) {
        throw new AssertException(message);
    }
}

// Open a new window and display the consent form
function open_window(hitid, assignmentid, workerid) {
    popup = window.open(
        '/consent?' + 
            'hitId=' + hitid + 
            '&assignmentId=' + assignmentid + 
            '&workerId=' + workerid,
        'Popup',
        'toolbar=no,' +
            'location=no,' +
            'status=no,' +
            'menubar=no,' + 
            'scrollbars=yes,' + 
            'resizable=no,' + 
            'width=' + screen.availWidth + ',' +
            'height=' + screen.availHeight + '');
    popup.onunload = function() { 
        location.reload(true) 
    };
}

// Set the background image on an element.
function set_poster(elem, image) {
    var path = "/static/stimuli/" + $c.condition + "/" + image + ".png";
    $(elem).css("background", "#FFF url(" + path + ") no-repeat");
    $(elem).css("background-size", "cover");
}

// Update the progress bar based on the current trial and total number
// of trials.
function update_progress(num, num_trials) {
    debug("update progress");
    var width = 2 + 98 * (num / (num_trials - 1.0));
    $("#indicator-stage").css({"width": width + "%"});
    $("#progress-text").html("Progress " + (num + 1) + "/" + num_trials);
}

// Fade in new_elem, and then hide old_elem
function replace(old_elem, new_elem) {
    $("#" + new_elem).fadeIn($c.fade, function () {
        $("#" + old_elem).hide();
    });
}

// Create a flowplayer object in `elem`, load a playlist of `stims`,
// and set the poster (background image) to `poster`.
function make_player(elem, stims, poster) {

    // Helper function to get the appropriate formats for each video
    var get_video_formats = function (stim) {
        var prefix = "/static/stimuli/" +  $c.condition + "/" + stim;
        var formats = [
            { webm: prefix + ".webm" },
            { mp4: prefix + ".mp4" },
            { ogg: prefix + ".ogg" }
        ];
        return formats;
    };

    // Create the new flowplayer instance
    var p = $f($(elem).flowplayer({
        debug: $c.debug,
        fullscreen: false,
        keyboard: false,
        muted: true,
        ratio: 0.75,
        splash: false,
        tooltip: false,
        playlist: stims.map(get_video_formats),
        advance: false,
        loop: false,
        embed: false
    }));

    // Set the poster
    if (poster) {
        set_poster(elem + ".flowplayer", poster);
    }

    return p;
}

// Set background and button colors to reflect the different block
// types
function set_colors(color0, color1) {
    if (color0) {
        $(".color0").css("background-color", color0);
        $("button.color0").html(color0);
    }

    if (color1) {
        $(".color1").css("background-color", color1);
        $("button.color1").html(color1);
    }
}
