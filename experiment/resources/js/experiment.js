
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
var indicatorWidth = 550;

var stableVideo = videoFolder + "stable.swf";
var unstableVideo = videoFolder + "unstable.swf";
var massVideo = videoFolder + "mass.swf";

// --------------------------------------------------------------------

function embedVideo(url, div, flashvars, params, attributes, callback) {
    $(".cover").show();
    swfobject.embedSWF(
	url, div, videoWidth, videoHeight, flashVersion, 
	expressInstallSwfurl, flashvars, params, attributes, callback);
}

function showSlide(id) {
    $(".slide").hide();
    $('html, body').animate({ scrollTop: 0 }, 0);
    $(".cover").show();
    $("#"+id).show();
}

function showInstructions(id) {
    showSlide(id);
    setTimeout(function () {
	$(".cover").fadeOut(fade);
    }, 300);
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
    $("#question").html("<b>Question:</b> " + question);
    $("#responses").empty();
    for (var i=0; i<responses.length; i++) {
	$("#responses").append(
	    "<button type='button' " +
		"name='response-button' " +
		"onclick='experiment.submit(\"" + 
		responses[i][1] + "\");'>" +
		responses[i][0] +
		"</button>");
    }
}

function updateProgress (index, numTrials) {
    // Update the progress bar
    var width = index * indicatorWidth / numTrials;
    $("#progress").show();
    $("#indicator-stage").animate({"width": width + "px"}, fade);
    $("#progress-text").html(
	"Progress " + (index+1) + "/" + numTrials);
}

// --------------------------------------------------------------------

showSlide("index");

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
		wmode: "direct",
		play: "true",
		loop: "true" };

	    experiment.numTrials = info.numTrials;
	    experiment.pid = info.pid;

	    embedVideo(
		stableVideo, "stable-example", 
		{}, params, {}, 
		function () {
		    embedVideo(
			unstableVideo, "unstable-example", 
			{}, params, {}, 
			function () {
			    embedVideo(
				massVideo, "mass-example", 
				{}, params, {}, 
				function () {
				    showInstructions("instructions");
				});
			});
		});

	});
    },

    start : function () {
	post('trialinfo', {pid: experiment.pid}, function (info) {
	    if (experiment.show(info)) {
		showSlide("trial");
		$("#play-button").focus();
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
	experiment.curQuestion = info.question;
	experiment.curResponses = info.responses;

	// Update progress bar
	updateProgress(experiment.index, experiment.numTrials);

	// Set question and responses
	setQuestion(experiment.curQuestion, experiment.curResponses);

	// Hide elements we're not ready for yet
	$("#player").hide();
	$("#player-img").hide();
	$("button[name=response-button]").attr("disabled", true);
	$("#responses").hide();

	// Show instructions and focus the play button
	$("#play-button").attr("disabled", false);
	$("#video-instructions").show();
	$("#play-button").focus();

	return true;
    },

    play : function () {
	var img = imageFolder + experiment.curImgA;
	var video = videoFolder + experiment.curVideo;
	var params = { wmode: "direct",
		       play: "true",
		       loop: "false" };
	var attributes = { id: "player" };
	
	$("#play-button").attr("disabled", true);
	embedVideo(video, "player", {}, params, attributes, 
		   function (e) {
		       if (e.success) {
			   $("#player-img").hide();
			   $("#video-instructions").fadeOut(fade)
			   setTimeout(experiment.query, 5000);
		       }});
    },

    query : function () {
	var img = imageFolder + experiment.curImgB;

	// set the end image (last frame of the video)
	$("#player-img").html(
	    "<img src='" + img + "' " +
		"width='" + videoWidth + "' " +
		"height='" + videoHeight + "'></img>");
	// fade in the image and then remove the video when it's done
	$("#player-img").fadeIn(fade, function () {
	    $("player").replaceWith("<div id='player'></div>");
	});

	// enable the response buttons
	$("button[name=response-button]").attr("disabled", false);
	$("#responses").slideDown();

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
	$("#responses").slideUp(fade, function () {
	    post("submit", data, experiment.next);
	});
    },

    end: function() {
	showSlide("finished");
	$("#indicator-stage").animate(
	    {"width": indicatorWidth + "px"}, fade);
    },
};

