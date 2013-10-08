var Player = function(elem, stims, loop) {
    var get_video_formats = function (stim) {
        var prefix = "/static/stimuli/" +  $c.condition + "/" + stim;
        var formats = [
            { webm: prefix + ".webm" },
            { mp4: prefix + ".mp4" },
            { ogg: prefix + ".ogg" }
        ];
        return formats;
    };

    var p = $f($("#" + elem).flowplayer({
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

    if (loop) {
        p.bind("finish", function (e, api) {
            api.prev();
        });
    }

    return p;
}

var State = function (experiment_phase, instructions, index, trial_phase) {

    this.experiment_phase;
    this.instructions;
    this.index;
    this.trial_phase;

    this.set_experiment_phase = function (experiment_phase) {
        if (experiment_phase != undefined) {
            this.experiment_phase = experiment_phase;
        } else {
            this.experiment_phase = EXPERIMENT.pretest;
        }
    };

    this.set_instructions = function (instructions) {
        if (instructions != undefined) {
            this.instructions = instructions;
        } else {
            this.instructions = 1;
        }
    };

    this.set_index = function (index) {
        if (index != undefined) {
            this.index = index;
        } else {
            this.index = 0;
        }
    };

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

    this.load_hash = function () {
        if (window.location.hash == "") {
            this.set_experiment_phase();
            this.set_instructions();
            this.set_index();
            this.set_trial_phase();

        } else {
            var parts = window.location.hash.slice(1).split("-").map(
                function (item) {
                    return parseInt(item);
                });
            this.set_experiment_phase(parts[0]);
            this.set_instructions(parts[1]);
            this.set_index(parts[2]);
            this.set_trial_phase(parts[3]);
        }
    };

    this.set_experiment_phase(experiment_phase);
    this.set_instructions(instructions);
    this.set_index(index);
    this.set_trial_phase(trial_phase);
};

function debug(msg) {
    if ($c.debug) {
        console.log(msg);
    }
}

function AssertException(message) { this.message = message; }
AssertException.prototype.toString = function () {
    return 'AssertException: ' + this.message;
};

function assert(exp, message) {
    if (!exp) {
        throw new AssertException(message);
    }
}

// Mean of booleans (true==1; false==0)
function boolpercent(arr) {
    var count = 0;
    for (var i=0; i<arr.length; i++) {
        if (arr[i]) { count++; } 
    }
    return 100* count / arr.length;
}

function openwindow(hitid, assignmentid, workerid) {
    popup = window.open(
        '/consent?hitId=' + hitid + '&assignmentId=' + assignmentid + '&workerId=' + workerid,
        'Popup',
        'toolbar=no,location=no,status=no,menubar=no,scrollbars=yes,resizable=no,width=' + screen.availWidth + ',height=' + screen.availHeight + '');
    popup.onunload = function() { location.reload(true) }
}

function onexit() {
    this.close()
}

function set_poster(elem, image) {
    var path = "/static/stimuli/" + $c.condition + "/" + image + ".png";
    $(elem).css("background", "#FFF url(" + path + ") no-repeat");
    $(elem).css("background-size", "cover");
}

function update_progress(num, num_trials) {
    debug("update progress");

    var width = 2 + 98*(num / (num_trials-1.0));
    $("#indicator-stage").css({"width": width + "%"});
    $("#progress-text").html(
        "Progress " + (num+1) + "/" + num_trials);
}


function show_and_hide(elem1, elem2) {
    $("#" + elem1).fadeIn($c.fade, function () {
        $("#" + elem2).hide();
    });
}
