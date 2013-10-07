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
    var state = new Object();

    if (experiment_phase != undefined) {
        state.experiment_phase = experiment_phase;
    } else {
        state.experiment_phase = EXPERIMENT.pretest;
    }

    if (instructions != undefined) {
        state.instructions = instructions;
    } else {
        state.instructions = 1;
    }

    if (index != undefined) {
        state.index = index;
    } else {
        state.index = 0;
    }

    if (!state.instructions) {
        if (trial_phase != undefined) {
            state.trial_phase = trial_phase;
        } else {
            state.trial_phase = TRIAL.prestim;
        }
    }

    return state
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
    self.close()
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

function set_state(state) {
    var parts = [
        state.experiment_phase,
        state.instructions,
        state.index
    ];

    if (!state.instructions) {
        parts[parts.length] = state.trial_phase;
    }

    window.location.hash = parts.join("-");
}

function get_state() {
    var state;
    if (window.location.hash == "") {
        state = new State();
    } else {
        var parts = window.location.hash.slice(1).split("-").map(
            function (item) {
                return parseInt(item);
            });
        state = new State(parts[0], parts[1], parts[2], parts[3]);
    }
    return state;
}

function show_and_hide(elem1, elem2) {
    $("#" + elem1).fadeIn($c.fade, function () {
        $("#" + elem2).hide();
    });
}
