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
var PLAYERSETUP = false;

var VIDEO = {
    onComplete: function (event) {},
    onPlaylistItem : function (event) {},
    onPause : function (event) {
	if (jwplayer().getPosition() == jwplayer().getDuration()) {
	    VIDEO.onComplete();
	}
	else if (event.oldstate == "PLAYING") {
	    debug("not allowed to pause");
	    jwplayer().play();
	}
    },
};

// --------------------------------------------------------------------
// Generic helper functions

function debug(msg) {
    if (DEBUG) {
	console.log(msg);
    }
}

// --------------------------------------------------------------------
// Image functions

function formatImage(image) {
    return imageUrl + image + "." + imageExt;
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
// Video functions

function formatVideo(video) {
    return videoUrl + video + "." + videoExt;
}

function embedVideo(div, video, image) {
    var setup = { file: formatVideo(video) };
    if (image)
	setup.image = formatImage(image);

    if (!PLAYERSETUP) {

	// default configuration
	setup.width = videoWidth;
	setup.height = videoHeight;
	setup.controls = false;
	setup.fallback = false;
	setup.bufferlength = 5;
	setup.autostart = false;
	setup.repeat = "none";

	debug("creating video player");
	jwplayer("player").setup(setup);

	jwplayer().onPause(function (event) {
	    VIDEO.onPause(event);
	});
	jwplayer().onPlaylistItem(function (event) {
	    VIDEO.onPlaylistItem(event);
	});
	jwplayer().onComplete(function (event) {
	    VIDEO.onComplete(event);
	});

	PLAYERSETUP = true;
    }

    else {
	debug("embedding '" + setup.file + "'");
	jwplayer().load(setup);
    }
	
    $("#" + div).append($("#player"));
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

	// scroll back up to the top of the page
	$('html, body').animate({ scrollTop: 0 }, 0);

	// set up and show new slide
	if (slides.current != next) {
	    debug("show slide '" + next + "'");
	    $("#" + next).show();
	}

	debug("setup slide '" + next + "'");
	slides[next].setup();
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
	    // automatically play
	    VIDEO.onPlaylistItem = function (event) {
		debug("playing unstable example");
		$("#unstable-example").show();
		jwplayer().play();
	    };

	    // repeat the video
	    VIDEO.onComplete = function (event) {
	    	debug("looping");
	    	jwplayer().play();
	    };

	    $("#unstable-example").hide();
	    embedVideo(
		"unstable-example",
		"unstable",
		"unstableA");
	},

	teardown : function () {
	    jwplayer().stop();
	}
    },

    // ----------------------------------------------------------------
    instructions1b : {
	setup : function () {
	    // automatically play
	    VIDEO.onPlaylistItem = function (event) {
		debug("playing stable example");
		$("#stable-example").show();
		jwplayer().play();
	    };

	    $("#stable-example").hide();
	    embedVideo(
		"stable-example",
		"stable",
		"stableA");
	},

	teardown : function () {
	    jwplayer().stop();
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
	    // automatically play
	    VIDEO.onPlaylistItem = function (event) {
		debug("playing mass example");
		$("#mass-example").show();
		jwplayer().play();
	    };

	    $("#mass-example").hide();
	    embedVideo(
		"mass-example",
		"mass",
		"massA");
	},

	teardown : function () {
	    jwplayer().stop();
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
	    // Hide elements we're not ready for yet
	    $("#responses").hide();
	    $("#feedback").hide();
	    $("#video-button").hide();
	    $("#video-instructions").hide();
	    
	    // Update progress bar
	    slides.trial.updateProgress();

	    // show instructions on load
	    VIDEO.onPlaylistItem = function (event) {
		debug("showing trial");
		$("#player-float").show();
	    	$("#video-instructions").show();
	    	$("#video-button").show();
	    };
	    
	    // show response buttons on complete
	    VIDEO.onComplete = function (event) {
		debug("done showing trial");
		experiment.queryResponse();
	    };

	    $("#player-float").hide();
	    embedVideo(
	    	"player-float",
	    	experiment.stimulus, 
	    	experiment.stimulus + "-floor");
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
	    $("#video-instructions").hide();
	    $("#video-button").hide();
	    jwplayer().play();
	},

	// Show the responses to the user
	showQuery : function () {
	    debug("showing responses");
	    $("#player-float").hide();
	    $("#responses").fadeIn(fade);
	},

	showFeedback : function () {

	    var stable = experiment.textFeedback == "stable";
	    var videofb = experiment.showVideoFeedback;
	    var showTextFeedback = function () {
		if (stable) {
		    $("#feedback").html("Tower will not fall!");
		    $("#feedback").attr("class", "stable-feedback");
		} else {
		    $("#feedback").html("Tower will fall...");
		    $("#feedback").attr("class", "unstable-feedback");
		}
		$("#responses").hide();
		$("#feedback").fadeIn(fade);
	    };
	    
	    // if videofb (video feedback) is not undefined, then show a
	    // video and text
	    if (videofb != 'undefined') {

		// automaticaly start playing
		VIDEO.onPlaylistItem = function (event) {
		    debug("showing feedback");
		    $("#player-float").show();
		    jwplayer().play();
		    showTextFeedback();
		};

		// go to the next trial after it plays
		VIDEO.onComplete = function (event) {
		    debug("done showing feedback");
		    $("#feedback").fadeOut(fade, experiment.nextTrial);
		};

		embedVideo(
		    "player-float", 
		    experiment.stimulus + "-fb", 
		    experiment.stimulus + "-fbA");
	    }
            
	    // otherwise just show text
	    else {
		showTextFeedback();
		setTimeout(function () {
		    $("#feedback").fadeOut(fade, experiment.nextTrial);
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
};

// --------------------------------------------------------------------
// --------------------------------------------------------------------

preloadImages(
    ["UCSeal122x122.gif", "Bayes-500h.jpg", "scales.png"],
    function () {
	slides.show("index");
    });

