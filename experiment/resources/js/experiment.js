
// global variables
var videoFolder = "resources/video/";
var videoWidth = "320";
var videoHeight = "240";
var flashVersion = "9.0.0";
var expressInstallSwfurl = false;
var pageURL = "../../index.py?page=";
var actionURL = "../../index.py?a=";

// --------------------------------------------------------------------

function embedVideo(url, div, flashvars, params, attributes, callback) {
    swfobject.embedSWF(
	url, div, videoWidth, videoHeight, flashVersion, 
	expressInstallSwfurl, flashvars, params, attributes, callback);
}

function showSlide(id) {
    $(".slide").hide();
    $("#"+id).show();
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

function checkAnswered(name) {
    var foo = $("input[name=" + name + "]");
    var f = function (s) { return $(s).attr("checked") };
    return _.any($("input[name=" + name + "]"), f);
}

function error(msg) {
    showSlide("error");
    $("#error-message").html(msg)
}

// --------------------------------------------------------------------

showSlide("main");

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
	    index : experiment.index,
	    pid : experiment.pid
	};
	post('trialinfo', data, experiment.show);
    },

    show: function(info) {
	experiment.curVideo = info.stimulus;
	experiment.curQuestion = info.question;
	experiment.curResponses = info.responses;
	get('trial', showTrial);	
    },

    play : function () {
	playStimulus();
    },

    query : function () {
	showQuestion();
	experiment.start = new Date().getTime();
    },

    submit : function() {
	if (!checkAnswered("response")) {
	    alert("Please fill out all form fields.");
	} else {
	    var time = new Date().getTime() - experiment.start;
	    var data = {
		pid : experiment.pid,
		trial : experiment.index,
		stimulus : experiment.curVideo,
		time : time / 1000,
		response : $("input[name=response]:checked").val(),
		question : experiment.curQuestion,
	    };
	    post("submit", data, experiment.next);
	}
    },

    end: function() {
	//setTimeout(function() { turk.submit(experiment) }, 1500);
	get('finished', function (msg) { $("#content").replaceWith(msg) });
    },
};


function showInstructions(msg) {
    var stableVideo = videoFolder + "stable.swf";
    var unstableVideo = videoFolder + "unstable.swf";
    var params = { wmode: "direct",
		   play: "true",
		   loop: "true" };

    var width = 360 / (experiment.numTrials + 1);
    $("#indicator-stage").width(width + "px");
    $("#content").replaceWith(msg);

    embedVideo(stableVideo, "stable-example", {}, params, {}, undefined);
    embedVideo(unstableVideo, "unstable-example", {}, params, {}, undefined);
}

function showTrial(msg) {
    // Update the progress bar
    var width = (experiment.index + 2) * (360 / (experiment.numTrials + 1));
    $("#indicator-stage").width(120 + width + "px");

    // Replace content
    $("#content").replaceWith(msg);

    // Set question and responses
    $("#question").html(experiment.curQuestion);
    var r = experiment.curResponses;
    for (var i=0; i<r.length; i++) {
	$("#responses").append(
	    "<input type='radio' " +
		"value='" + r[i][1] + "' " + 
		"name='response' " +
		"onclick='$(\"#next-button\").show();'>" + 
		r[i][0] + 
		"</input><br />");
    }		

    // Display the video stimulus.
    $("#video-container").hide();
    $("#instructions2").hide();
    $("#question-container").hide();
    $("#next-button").hide();

    // Scroll up to the top
    $('html, body').animate({ scrollTop: 0 }, 0);
}

function showQuestion() {
    $("#instructions1").hide();
    $("#video-container").hide();
    $("#instructions2").show();
    $("#question-container").show();
}

function playStimulus() {
    var video = videoFolder + experiment.curVideo;
    var flashvars = {};
    var params = { wmode: "direct",
		   play: "true",
		   loop: "false" };
    var attributes = { id: "player" };
    
    $("#play-button").hide();
    $("#video-container").show();
    $("#player").show();
    embedVideo(video, "player", {}, params, attributes, 
	      function (e) {
		  if (e.success) {
		      setTimeout(experiment.query, 5000);
		  }});
}

