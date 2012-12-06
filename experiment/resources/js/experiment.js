
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
    swfobject.embedSWF(
	url, div, videoWidth, videoHeight, flashVersion, 
	expressInstallSwfurl, flashvars, params, attributes, callback);
}

function showSlide(id) {
    $(".slide").fadeOut(fade);
    $('html, body').animate({ scrollTop: 0 }, 0);
    $("#"+id).fadeIn(fade);
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
    training : true,

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

	    embedVideo(stableVideo, "stable-example", 
		       {}, params, {}, undefined);
	    embedVideo(unstableVideo, "unstable-example", 
		       {}, params, {}, undefined);
	    embedVideo(massVideo, "mass-example", 
		       {}, params, {}, undefined);
	    showSlide("instructions");

	});
    },

    start : function () {
	post('trialinfo', {pid: experiment.pid}, function (info) {
	    experiment.show(info);
	    showSlide("trial");
	});
    },

    next: function() {
	var data = {
	    pid : experiment.pid
	};
	post('trialinfo', data, experiment.show);
    },

    show: function(info) {
	if (info.stimulus == "") {
	    if (info.training == true) {
		var data = {
		    pid : experiment.pid,
		    time : 0,
		    response : "",
		};
		post("submit", data, function () {
		    showSlide("instructions2");
		});
	    } else {
		showSlide("finished");
	    }
	    return;
	}

	experiment.index = info.index;
	experiment.curVideo = info.stimulus + ".swf";
	experiment.curImg = info.stimulus + ".png";
	experiment.curQuestion = info.question;
	experiment.curResponses = info.responses;
	experiment.training = info.training;

	// Set question and responses
	setQuestion(experiment.curQuestion, experiment.curResponses);

	// Hide elements we're not ready for yet
	$("#player").hide();
	$("#player-img").hide();
	$("button[name=response-button]").attr("disabled", true);
	$("#responses").hide();

	// Show instructions and focus the play button
	$("#play-button").attr("disabled", false);
	$("#play-button").focus();
	$("#video-instructions").fadeIn();

	// Update progress bar
	updateProgress(experiment.index, experiment.numTrials);
    },

    play : function () {
	var video = videoFolder + experiment.curVideo;
	var params = { wmode: "direct",
		       play: "true",
		       loop: "false" };
	var attributes = { id: "player" };
	
	embedVideo(video, "player", {}, params, attributes, 
		   function (e) {
		       if (e.success) {
			   setTimeout(experiment.query, 5000);
		       }});
	$("#video-instructions").fadeOut(fade, function () {
	    $("#play-button").attr("disabled", true);
	});
    },

    query : function () {
	var img = imageFolder + experiment.curImg;

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
	post("submit", data, experiment.next);
    },

    end: function() {
	showSlide("finished");
	$("#indicator-stage").animate(
	    {"width": indicatorWidth + "px"}, fade);
    },
};

