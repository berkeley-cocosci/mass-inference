
// global variables
var videoFolder = "resources/video/";
var imageFolder = "resources/images/";
var videoWidth = "320";
var videoHeight = "240";
var flashVersion = "9.0.0";
var expressInstallSwfurl = false;
var pageURL = "../../index.py?page=";
var actionURL = "../../index.py?a=";

var fade = 200;
var indicatorWidth = 570;

// --------------------------------------------------------------------

function embedVideo(url, div, flashvars, params, attributes, callback) {
    swfobject.embedSWF(
	url, div, videoWidth, videoHeight, flashVersion, 
	expressInstallSwfurl, flashvars, params, attributes, callback);
}

function showSlide(id) {
    $(".slide").fadeOut(fade);
    $("#"+id).fadeIn(fade);
}

function get(page, handler) {
    var request = $.ajax({
	type: "GET",
	url: pageURL + page,
    });
    request.done(handler);
    request.fail(error);
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


// --------------------------------------------------------------------

showSlide("main");
$("#progress").hide();

var experiment = {

    index : -1,
    numTrials : 0,
    start : undefined,

    curVideo : "",
    curQuestion : "",
    curResponses : [],

    start : function() {
	post('initialize', {}, experiment.initialize);
    },

    initialize : function(msg) {
	var info = $.parseJSON(msg);
	experiment.numTrials = info.numTrials;
	experiment.pid = info.pid;
	get('instructions', showInstructions);
    },

    next: function() {
	experiment.index = experiment.index + 1; 
	if (experiment.index >= experiment.numTrials) {
	    return experiment.end();
	}

	var data = {
	    pid : experiment.pid
	};
	post('trialinfo', data, experiment.show);
    },

    show: function(info) {
	experiment.index = info.index;
	experiment.curVideo = info.stimulus + ".swf";
	experiment.curImg = info.stimulus + ".png";
	experiment.curQuestion = info.question;
	experiment.curResponses = info.responses;
	get('trial', showTrial);	
    },

    play : function () {
	playStimulus();
    },

    query : function () {
	var img = imageFolder + experiment.curImg;
	$("#player-img").html(
	    "<img src='" + img + "' " +
		"width='" + videoWidth + "' " +
		"height='" + videoHeight + "'></img>");
	$("#player-img").fadeIn(fade, function () {
	    $("player").remove();
	});
	showQuestion();
	experiment.start = new Date().getTime();
    },

    submit : function(val) {
	var time = new Date().getTime() - experiment.start;
	var data = {
	    pid : experiment.pid,
	    time : time / 1000,
	    response : val,
	};
	post("submit", data, experiment.next);
    },

    end: function() {
	//setTimeout(function() { turk.submit(experiment) }, 1500);
	get('finished', function (msg) { 
	    $("#content").replaceWith(msg) 
	});
	$("#indicator-stage").animate(
	    {"width": indicatorWidth + "px"}, fade);
    },
};

function showInstructions(msg) {
    var stableVideo = videoFolder + "stable.swf";
    var unstableVideo = videoFolder + "unstable.swf";
    var params = { wmode: "direct",
		   play: "true",
		   loop: "true" };

    // var width = 360 / (experiment.numTrials + 2);
    // var step = indicatorWidth / (experiment.numTrials + 2);
    // $("#indicator-stage").animate({"width": "+=10px"}, fade);
    $("#content").fadeOut(fade, function () {
	$("#content").replaceWith(msg);
	embedVideo(stableVideo, "stable-example-video", 
		   {}, params, {}, undefined);
	embedVideo(unstableVideo, "unstable-example-video", 
		   {}, params, {}, undefined);
    });
}

function showTrial(msg) {
    // Replace content
    $("#content").fadeOut(fade, function() {
	$("#content").replaceWith(msg);

	// Set question and responses
	$("#question").html("<b>Question:</b> " + experiment.curQuestion);
	var r = experiment.curResponses;
	for (var i=0; i<r.length; i++) {
	    $("#responses").append(
		"<button type='button' " +
		    "name='response-button' " +
		    "onclick='experiment.submit(\"" + r[i][1] + "\");'>" +
		    r[i][0] +
		    "</button>");
	}		
	
	// Hide elements we're not ready for yet
	$("#player").hide();
	$("#player-img").hide();
	$("button[name=response-button]").attr("disabled", true);
	
	// Show instructions and focus the play button
	$("#content").fadeIn(fade);
	$("#play-button").focus();

	// Update the progress bar
	var width = experiment.index * indicatorWidth / experiment.numTrials;
	$("#progress").show();
	$("#indicator-stage").animate({"width": width + "px"}, fade);
	$("#progress-text").html(
	    "Progress " + (experiment.index+1) + "/" + experiment.numTrials);
    });
}

function showQuestion() {
    $("button[name=response-button]").attr("disabled", false);
}

function playStimulus() {
    var video = videoFolder + experiment.curVideo;
    var flashvars = {};
    var params = { wmode: "direct",
		   play: "true",
		   loop: "false" };
    var attributes = { id: "player" };
    
    embedVideo(video, "player", {}, params, attributes, 
	       function (e) {
		   if (e.success) {
		       setTimeout(experiment.query, 5000);
		   }});
    $("#instructions").fadeOut(fade, function () {
	$("#instructions").remove()
    });

}

