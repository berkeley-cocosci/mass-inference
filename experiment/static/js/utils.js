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

var Player = function(stims, loop) {
    var get_video_formats = function (stim) {
        var prefix = "/static/stimuli/" +  $c.condition + "/" + stim;
        var formats = [
            { webm: prefix + ".webm" },
            { mp4: prefix + ".mp4" },
            { ogg: prefix + ".ogg" }
        ];
        return formats;
    };

    var p = $f($("#player").flowplayer({
        debug: $c.debug,
        // swf: '/static/lib/flowplayer/flowplayer.swf',
        fullscreen: false,
        keyboard: false,
        muted: true,
        ratio: 0.75,
        splash: false,
        tooltip: false,
        playlist: stims.map(get_video_formats),
        advance: false,
        loop: false
    }));

    if (loop) {
        p.bind("finish", function (e, api) {
            api.prev();
        });
    }

    return p;
}

function set_bg_image(elem, image) {
    var path = "/static/stimuli/" + $c.condition + "/" + image + ".png";
    $("#" + elem).css(
        "background-image", "url(" + path + ")");
}

var update_progress = function (num, num_trials) {
    debug("update progress");

    var width = 2 + 98*(num / (num_trials-1.0));
    $("#indicator-stage").animate(
        {"width": width + "%"}, $c.fade);
    $("#progress-text").html(
        "Progress " + (num+1) + "/" + num_trials);
}
