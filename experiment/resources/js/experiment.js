
// global variables
var videoFolder = "resources/video/";
var imageFolder = "resources/images/";

var videoWidth = "640";
var videoHeight = "480";
var flashVersion = "9.0.0";
//var expressInstallSwfurl = false;
var expressInstallSwfurl = "http://get.adobe.com/flashplayer/"

var pageURL = "../../index.py?page=";
var actionURL = "../../index.py?a=";

var fade = 200;

var stableVideo = videoFolder + "stable.swf";
var unstableVideo = videoFolder + "unstable.swf";
var massVideo = videoFolder + "mass.swf";

var DEBUG = true;

function debug(msg) {
    if (DEBUG) {
	console.log(msg);
    }
}

// --------------------------------------------------------------------

function videoIsLoaded(obj) {
    var elem = $("#" + obj);
    var percent;

    try {
	percent = elem[0].PercentLoaded();
    } catch (exception) {
	debug("ERROR: " + exception);
	return false;
    }
    return (percent == 100);
}

function videoIsPlaying(obj) {
    var playing = $("#" + obj)[0].IsPlaying();
    return playing;
}    

function onVideoLoaded(obj, callback) {
    if (!videoIsLoaded(obj)) {
	setTimeout(
	    function () {
		onVideoLoaded(obj, callback);
	    }, 10);
    } else {
	debug("video '" + obj + "' is loaded");
	callback();
    }
}

function onVideoFinished(obj, callback) {
    if (videoIsLoaded(obj) && !videoIsPlaying(obj)) {
	debug("video '" + obj + "' is finished");
	callback();
    } else {
	setTimeout(
	    function () {
		onVideoFinished(obj, callback);
	    }, 10);
    }
}

function embedVideo(url, div, flashvars, params, attributes, callback) {
    debug("embedding '" + url + "'");
    swfobject.embedSWF(
	url, div, videoWidth, videoHeight, flashVersion, 
	expressInstallSwfurl, flashvars, params, attributes,
	function (e) {
	    if (e.success) {
		debug("success embedding '" + url + "'");
		callback();
	    } else {
		throw e;
	    }
	});
}

function embedAndLoadVideo(url, div, flashvars, params, attributes, callback) {
    embedVideo(
	url, div, flashvars, params, attributes,
	function () {
	    onVideoLoaded(div, callback);
	});
}

function embedVideos(urls, divs, flashvars, params, attributes, callback) {
    var func = function () {
	if (urls.length == 1) {
	    callback();
	} else {
	    embedVideos(urls.slice(1, urls.length), 
			divs.slice(1, divs.length), 
			flashvars, params, 
			attributes, callback);
	}		
    };

    embedVideo(urls[0], divs[0], flashvars, params, attributes, func);
}

function preloadImages(arrayOfImages, callback) {
    var numToLoad = arrayOfImages.length;
    $(arrayOfImages).each(function () {
	debug("preloading image '" + this + "'");
	$("<img />").load(function () {
	    numToLoad--;
	    if (numToLoad == 0 && callback) { callback(); }
	}).attr("src", imageFolder + this);
    });
}

// --------------------------------------------------------------------

function showSlide(id) {
    debug("show slide '" + id + "'");
    $(".slide").hide();
    $('html, body').animate({ scrollTop: 0 }, 0);
    $("#"+id).show();
}

function showInstructions(id) {
    var cover = $("#" + id).find(".cover");
    if (cover.length != 0) {
	var player = $($("#" + id).find(".example").children()[1]).attr("id");
	cover.show();
	showSlide(id);
	onVideoLoaded(
	    player,
	    function () {
		debug("fading out cover");
		cover.fadeOut(fade);
	    }); 
    } else {
	showSlide(id);
    }
}

function error(msg) {
    showSlide("error");
    $("#error-message").html("<p>" + msg.responseText + "</p>");
}

function post(action, data, handler) {
    var request = $.ajax({
	type: "POST",
	url: actionURL + action,
	data: data,
    });
    request.done(handler);
    request.fail(error);
}

function setQuestion(question, responses) {
    // Set question and responses
    $("#question").html("<b>Question:</b> " + question);
    var resp = $("#responses");
    resp.empty();
    for (var i=0; i<responses.length; i++) {
	var text = responses[i][0];
	var color = responses[i][1];
	var val = responses[i][2];
	resp.append(
	    "<div class='response'><button type='button' " +
		"name='response-button' " +
		"class='big-option' " +
		"style='background-color: " + color + "' " +
		"onclick='experiment.submit(\"" + 
		val + "\");'>" +
		text +
		"</button></div>");
    }
    resp.append("<div class='spacer'></div>");
}

function updateProgress (index, numTrials) {
    // Update the progress bar
    var width = 2 + 98*(index / (numTrials-1.0));
    $("#progress").show();
    $("#indicator-stage").animate({"width": width + "%"}, fade);
    $("#progress-text").html(
	"Progress " + (index+1) + "/" + numTrials);
}

// --------------------------------------------------------------------

var experiment = {

    index : 0,
    numTrials : 0,
    starttime : undefined,

    curVideo : "",
    curQuestion : "",
    curResponses : [],

    validationCode : "",

    initialize : function() {
	post('initialize', {}, function (msg) {
	    var info = $.parseJSON(msg);
	    var params = { 
		wmode: "opaque",
		play: "true",
		loop: "true",
	        bgcolor: "#FFFFFF"};

	    experiment.numTraining = info.numTraining;
	    experiment.numExperiment = info.numExperiment;
	    experiment.numTrials = experiment.numTraining;
	    experiment.pid = info.pid;

	    embedVideos(
		[unstableVideo, stableVideo, massVideo],
		["unstable-example", "stable-example", "mass-example"],
		{}, params, {}, 
		function () {
		    preloadImages(
			["scales.png"], 
			function () {
			    showInstructions("instructions");
			});
		});
	});
    },

    start: function () {
	post('trialinfo', {pid: experiment.pid}, function (info) {
	    if (experiment.show(info)) {
		showSlide("trial");
	    }
	});
    },

    next: function() {
	var data = {
	    pid : experiment.pid
	};
	post('trialinfo', data, experiment.show);
    },
	

    show: function(info) {
	if (info.index == 'finished training') {
	    experiment.numTrials = experiment.numExperiment;
	    $("#question-image-container").show();
	    $("#question-image").attr("src", imageFolder + "scales.png");
	    $("#indicator-stage").animate({"width": "0%"}, 0);
	    $("#feedback").hide();
	    $("#screenshot1").hide();
	    $("#screenshot2").hide();
	    $("#player").hide();
	    showInstructions("instructions2");
	    return false;
	} else if (info.index == "finished experiment") {
	    experiment.numTrials = experiment.numTraining;
	    $("#question-image-container").hide();
	    $("#question-image").attr("src", "");
	    $("#indicator-stage").animate({"width": "0%"}, 0);
	    $("#feedback").hide();
	    $("#screenshot1").hide();
	    $("#screenshot2").hide();
	    $("#player").hide();
	    showInstructions("instructions3");
	    return false;
	} else if (info.index == 'finished posttest') {
	    experiment.validationCode = info.code;
	    experiment.end();
	    return false;
	}

	experiment.index = info.index;
	experiment.curVideo = info.stimulus + ".swf";
	experiment.curImgA = info.stimulus + "A.png";
	experiment.curImgB = info.stimulus + "B.png";
	experiment.curImgFloor = info.stimulus + "-floor.png";
	experiment.curQuestion = info.question;
	experiment.curResponses = info.responses;

	// Hide elements we're not ready for yet
	$("button[name=response-button]").attr("disabled", true);
	$("#responses").hide();
	$("#feedback").hide();
	$("#video-button").hide();
	$("#video-instructions").hide();

	// Update progress bar
	updateProgress(experiment.index, experiment.numTrials);

	// Set question and responses
	setQuestion(experiment.curQuestion, experiment.curResponses);

	// Load the images and cache them
	preloadImages(
	    [experiment.curImgFloor, experiment.curImgB],
	    function () {
		$("#screenshot1-img").attr(
		    "src", imageFolder + experiment.curImgFloor);

		// Set background image
		$("#screenshot1").fadeIn(fade, function () {
		    $("#player").hide();
		    $("#play-button").attr("disabled", false);
		    $("#video-instructions").show();
		    $("#video-button").show();

		    $("#screenshot2").hide();
		    $("#screenshot2-img").attr(
			"src", imageFolder + experiment.curImgB);

		});
	    });

	return true;
    },

    play : function () {
	var video = videoFolder + experiment.curVideo;
	var params = { wmode: "opaque",
		       play: "true",
		       loop: "false",
		       bgcolor: "#FFFFFF" };
	var attributes = { id: "player" };
	
	$("#play-button").attr("disabled", true);
	$("#video-instructions").hide();
	$("#video-button").hide();
	embedAndLoadVideo(
	    video, "player", {}, params, attributes, 
	    function () {
		$("#screenshot1").fadeOut(fade);
		onVideoFinished(
		    "player", experiment.query);
	    });
    },

    query : function () {
	// fade in the image and then remove the video when it's done
	$("#screenshot2").fadeIn(fade, function () {
	    $("#player").hide();
	    $("#player").replaceWith("<div id='player'></div>");
	    $("#responses").fadeIn(fade, function () {
		// enable the response buttons
		$("button[name=response-button]").attr("disabled", false);
	    });
	});

	// record the time so we know how long they take to answer
	experiment.starttime = new Date().getTime();
    },

    submit : function(val) {
	var time = new Date().getTime() - experiment.starttime;
	var data = {
	    pid : experiment.pid,
	    time : time / 1000,
	    response : val,
	};
	
	post("submit", data, experiment.feedback);
    },

    feedback : function(msg) {
	var stable = msg[0];
	var vfb = msg[1];

	// if the feedback is undefined, then don't display anything
	if (stable == "undefined") {
	    $("#responses").fadeOut(fade, experiment.next);

	// otherwise do give feedback
	} else {
	
	    var txtfb = function () {
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
	    
	    // if vfb (video feedback) is not undefined, then show a
	    // video and text
	    if (vfb != 'undefined') {
		var video = videoFolder + vfb + ".swf";
		var params = { wmode: "opaque",
			       play: "true",
			       loop: "false",
			       bgcolor: "#FFFFFF" };
		var attributes = { id: "player" };

		// callback to execute once the video has been loaded
		var onload = function () {
		    $("#screenshot2").fadeOut(fade);
		    txtfb();
		    onVideoFinished(
			"player", 
			function () {
			    $("#feedback").fadeOut(fade, experiment.next);
			});
		};

		embedAndLoadVideo(
		    video, "player", {}, params, attributes, onload);
	
            // otherwise just show text
	    } else {
		txtfb();
		setTimeout(function () {
		    $("#feedback").fadeOut(fade, experiment.next);
		}, 2000);
	    }
	}
    },


    end: function() {
	showSlide("finished");
	$("#code").html(experiment.validationCode);
    },
};

preloadImages(["UCSeal122x122.gif", "Bayes-500h.jpg"],
	      function () {
		  showSlide("index");
	      });

