
// global variables
var videoFolder = "resources/video/";
var imageFolder = "resources/images/";

var videoWidth = "640";
var videoHeight = "480";
var flashVersion = "9.0.0";
var expressInstallSwfurl = false;

var pageURL = "../../index.py?page=";
var actionURL = "../../index.py?a=";

var fade = 200;
var indicatorWidth = 540;

var stableVideo = videoFolder + "stable.swf";
var unstableVideo = videoFolder + "unstable.swf";
var massVideo = videoFolder + "mass.swf";

// --------------------------------------------------------------------

function videoIsLoaded(obj) {
    var percent = $("#" + obj)[0].PercentLoaded();
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
	callback();
    }
}

function onVideoFinished(obj, callback) {
    if (videoIsLoaded(obj) && !videoIsPlaying(obj)) {
	callback();
    } else {
	setTimeout(
	    function () {
		onVideoFinished(obj, callback);
	    }, 10);
    }
}

function embedVideo(url, div, flashvars, params, attributes, callback) {
    swfobject.embedSWF(
	url, div, videoWidth, videoHeight, flashVersion, 
	expressInstallSwfurl, flashvars, params, attributes,
	callback);
}

// --------------------------------------------------------------------

function showSlide(id) {
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
		cover.fadeOut(fade);
	    }); 
    } else {
	showSlide(id);
    }
}

function preloadImage(url) {
    var img = $("<img />").attr("src", url);
    return img;
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

function error(msg) {
    showSlide("error");
    $("#error-message").html(msg)
}

function setQuestion(question, responses) {
    // Set question and responses
    $("#question").html("<p><b>Question:</b> " + question + "</p>");
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

	    embedVideo(
		stableVideo, "stable-example", 
		{}, params, {}, undefined);
	    embedVideo(
		unstableVideo, "unstable-example", 
		{}, params, {}, undefined);
	    embedVideo(
		massVideo, "mass-example", 
		{}, params, {}, undefined);

	    showInstructions("instructions");
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
	if (info == 'finished training') {
	    experiment.numTrials = experiment.numExperiment;
	    $("#indicator-stage").attr("width", "2%");
	    showInstructions("instructions2");
	    return false;
	} else if (info == 'finished experiment') {
	    showSlide("finished");
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

	// Update progress bar
	updateProgress(experiment.index, experiment.numTrials);

	// Set question and responses
	setQuestion(experiment.curQuestion, experiment.curResponses);

	// Set background image
	var img = imageFolder + experiment.curImgFloor;
	$("#screenshot1-img").attr("src", img);
	$("#screenshot1").fadeIn(fade, function () {
	    $("#player").hide();
	    $("#screenshot2").hide();
	    $("#play-button").attr("disabled", false);
	    $("#video-instructions").show();
	    $("#video-button").show();
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
	embedVideo(
	    video, "player", {}, params, attributes, 
	    function (e) {
		if (e.success) {
		    onVideoLoaded(
			"player",
			function () {
			    $("#screenshot1").fadeOut(fade);
			    onVideoFinished(
				"player", experiment.query);
			});
		}});
    },

    query : function () {
	var img = imageFolder + experiment.curImgB;

	// set the end image (last frame of the video)
	$("#screenshot2-img").attr("src", img);
	// fade in the image and then remove the video when it's done
	$("#screenshot2").fadeIn(fade, function () {
	    $("#player").replaceWith("<div id='player'></div>");
	    $("#player").hide();
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
		    $("#feedback").html("Tower does not fall!");
		    $("#feedback").attr("class", "stable-feedback");
		} else {
		    $("#feedback").html("Tower is falling...");
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

		embedVideo(
		    video, "player", {}, params, attributes, 
		    function (e) {
			if (e.success) {
			    onVideoLoaded("player", onload);
			}});
	
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
	// $("#indicator-stage").animate(
	//     {"width": indicatorWidth + "px"}, fade);
    },
};

showSlide("index");

