// --------------------------------------------------------------------
// slides.js -- loaded before experiment.js
//
// Written by Jessica Hamrick (jhamrick@berkeley.edu)
//
// --------------------------------------------------------------------


// --------------------------------------------------------------------
// Configuration

var videoUrl = "resources/video/";
var imageUrl = "resources/images/";

var videoWidth = 640;
var videoHeight = 480;
var videoExt = "mp4";
var imageExt = "png";

var fade = 200;

var DEBUG = true;

// --------------------------------------------------------------------
// Generic helper functions

function debug(msg) {
    if (DEBUG) {
	console.log(msg);
    }
}

// --------------------------------------------------------------------
// Configure flowplayer

var $f = flowplayer;
$f.conf.embed = false;

$f(function(api, root) {
    // when a new video is about to be loaded
    api.bind("load", function() {
	debug("flowplayer '" + this.id + "' loaded");
	api.conf.loop = false;
	api.conf.embed = false;
	
    // when a video is loaded and ready to play
    }).bind("ready", function() {
	debug("flowplayer '" + this.id + "' ready");
    });
});

// --------------------------------------------------------------------
// Media

function formatImage(image) {
    return imageUrl + image + "." + imageExt;
}

function getVideoFormats(video) {
    var prefix = videoUrl + video + ".";
    var formats = [
	{ mp4: prefix + "mp4" },
	{ ogg: prefix + "ogg" },
	{ flv: prefix + "flv" }];
    return formats;
}

function setBgImage(elem, image) {
    $("#" + elem).css(
	"background-image",
	"url(../../" + formatImage(image) + ")");
}

function preloadImages(arrayOfImages, callback) {
    var numToLoad = arrayOfImages.length;
    $(arrayOfImages).each(function () {
	var img = imageUrl + this;
	debug("preloading image '" + img + "'");
	$("<img />").load(function () {
	    numToLoad--;
	    if (numToLoad == 0 && callback) { callback(); }
	}).attr("src", img);
    });
}

// --------------------------------------------------------------------
// Slides object: each attribute corresponds to a different slide
// (which should be a div with the corresponding id). Each slide
// object should have setup and teardown functions, which are called
// before and after the slide is shown, respectively.

var slides = {

    current : undefined,

    show : function (next) {
	// clean up and hide current slide
	if (slides.current) {
	    debug("tear down slide '" + slides.current + "'");
	    slides[slides.current].teardown();
	    if (slides.current != next) {
		debug("hide slide '" + slides.current + "'");
		$("#" + slides.current).hide();
	    }
	}

	// set up and show new slide
	debug("setup slide '" + next + "'");
	slides[next].setup();

	if (slides.current != next) {
	    debug("show slide '" + next + "'");
	    $("#" + next).fadeIn(fade);
	}

	$("#" + next).focus();
	slides.current = next;
    },

    // ----------------------------------------------------------------
    index : {
	setup : function () {},
	teardown : function () {}
    },

    // ----------------------------------------------------------------
    instructions1a : {
	setup : function () {
	    $f($("#unstable-example")).bind(
		"finish", function (e, api) {
		    if (!api.playing) {
			debug("looping");
			api.play();
		    }
		}).load(getVideoFormats("unstable"));
	},

	teardown : function () {
	    $f($("#unstable-example")).unload();
	}
    },

    // ----------------------------------------------------------------
    instructions1b : {
	setup : function () {
	    $f($("#stable-example")).bind(
		"finish", function (e, api) {
		    if (!api.playing) {
			debug("looping");
			api.play();
		    }
		}).load(getVideoFormats("stable"));
	},

	teardown : function () {
	    $f($("#stable-example")).unload();
	}
    },
    
    // ----------------------------------------------------------------
    instructions1c : {
	setup : function () {},
	teardown : function () {}
    },

    // ----------------------------------------------------------------
    instructions2 : {
	setup : function () {
	    $f($("#mass-example")).bind(
		"finish", function (e, api) {
		    if (!api.playing) {
			debug("looping");
			api.play();
		    }
		}).load(getVideoFormats("mass"));
	},

	teardown : function () {
	    $f($("#mass-example")).unload();
	}
    },

    // ----------------------------------------------------------------
    instructions3 : {
	setup : function () {},
	teardown : function () {}
    },

    // ----------------------------------------------------------------
    trial : {
	setup : function () {
	    // Update progress bar
	    slides.trial.updateProgress();

	    // Possibly show image
	    if (experiment.showQuestionImage) {
		$("#question-image").find("img").show();
	    } else {
		$("#question-image").find("img").hide();
	    }
	    
	    debug("showing trial");
	    setBgImage("prestim", experiment.stimulus + "-floor");
	    $("#prestim").fadeIn(fade);
	    $(".feedback").hide();
	},

	teardown : function () {},

	// Update the progress bar
	updateProgress : function () {
	    var index = experiment.index;
	    var numTrials = experiment.numTrials;
	    var width = 2 + 98*(index / (numTrials-1.0));
	    $("#indicator-stage").animate(
		{"width": width + "%"}, fade);
	    $("#progress-text").html(
		"Progress " + (index+1) + "/" + numTrials);
	},

	// User hits the 'play' button
	play : function () {
	    if ($f($("#player")).disabled)
		$f($("#player")).disable()
	    $f($("#player")).unbind("finish").bind(
		"finish", function (e, api) {
		    if (!api.disabled) {
			debug("done showing trial");
			api.disable();
			experiment.queryResponse();
		    }
		}).load(
		    getVideoFormats(experiment.stimulus),
		    function (e, api) {
			setBgImage("player", experiment.stimulus + "B");
			$("#prestim").hide();
		    });
	},

	// Show the responses to the user
	showQuery : function () {
	    debug("showing responses");
	    setBgImage("responses", experiment.stimulus + "B");
	    $("#responses").fadeIn(fade);
	},

	showFeedback : function () {

	    var stable = experiment.textFeedback == "stable";
	    var videofb = experiment.showVideoFeedback;
	    var showTextFeedback = function () {
		$("#responses").hide();
		if (stable) {
		    $("#stable-feedback").fadeIn(fade);
		} else {
		    $("#unstable-feedback").fadeIn(fade);
		}
	    };

	    // if videofb (video feedback) is true, then show a video
	    // and text
	    if (videofb) {
		showTextFeedback();
		// re-enable player
		if ($f($("#player")).disabled)
		    $f($("#player")).disable()
		$f($("#player")).unbind("finish").bind(
		    "finish", function (e, api) {
			if (!api.disabled) {
			    debug("done showing feedback");
			    api.disable();
			    experiment.nextTrial();
			}
		    }).load(getVideoFormats(experiment.stimulus + "-fb"));
	    }
            
	    // otherwise just show text
	    else {
		showTextFeedback();
		setTimeout(function () {
		    experiment.nextTrial();
		}, 2000);
	    }
	}

    },

    // ----------------------------------------------------------------
    finished : {
	setup : function () {
	    $("#code").html(experiment.validationCode);
	},

	teardown : function () {}
    },

    // ----------------------------------------------------------------
    error : {
	setup : function () {
	    $("#error-message").html("<p>" + experiment.error + "</p>");
	},

	teardown : function () {}
    },

    // ----------------------------------------------------------------
    declined : {
	setup : function () {},
	teardown : function () {},
    },
};

// --------------------------------------------------------------------
// --------------------------------------------------------------------

$(document).ready(function () {
    var players = [
	"unstable-example", 
	"stable-example", 
	"mass-example", 
	"player"];
    $(players).each(
	function () {
	    $("#" + this).flowplayer({
		debug: DEBUG,
		fullscreen: false,
		keyboard: false,
		muted: true,
		ratio: 0.75,
		splash: true,
		tooltip: false });
	});
    preloadImages(
	["UCSeal122x122.gif", "Bayes-500h.jpg", "scales.png"],
	function () {
	    slides.show("index");
	});
});

