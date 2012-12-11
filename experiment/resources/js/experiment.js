
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
var indicatorWidth = 545;

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
    var resp = $("#responses");
    resp.empty();
    for (var i=0; i<responses.length; i++) {
	resp.append(
	    "<div class='response'><button type='button' " +
		"name='response-button' " +
		"onclick='experiment.submit(\"" + 
		responses[i][1] + "\");'>" +
		responses[i][0] +
		"</button></div>");
    }
    resp.append("<div class='spacer'></div>");
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
	$("#feedback").hide();

	// Show instructions and focus the play button
	$("#play-button").attr("disabled", false);
	$("#video-instructions").show();
	$("#play-button").focus();

	return true;
    },

    play : function () {
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
	    $("#responses").slideDown(fade, function () {
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
	var go = function () {
	    $("#stable-feedback").html("");
	    $("#unstable-feedback").html("");
	    experiment.next();
	};

	// if the feedback is undefined, then don't display anything
	if (stable == "undefined") {
	    $("#responses").slideUp(fade, go);

	// otherwise give feedback, then submit and go to the next
	// trial
	} else {
	
	    var txtfb = function () {
		if (stable) {
		    $("#stable-feedback").html("Tower is stable!");
		    $("#unstable-feedback").html("&nbsp;");
		} else {
		    $("#stable-feedback").html("&nbsp;");
		    $("#unstable-feedback").html("Tower is falling...");
		}
		$("#responses").hide();
		$("#feedback").show();
	    };
	    
	    if (vfb != 'undefined') {
		var video = videoFolder + vfb + ".swf";
		var params = { wmode: "direct",
			       play: "true",
			       loop: "false" };
		var attributes = { id: "player" };
		embedVideo(
		    video, "player", {}, params, attributes, 
		    function (e) {
			if (e.success) {
			    setTimeout(function () {
				$("#player-img").fadeOut(fade);
			    }, 300);
			    txtfb();
			    setTimeout(function () {
				$("#feedback").slideUp(fade, go);
			    }, 3000);
			}});
	    } else {
		txtfb();
		setTimeout(function () {
		    $("#feedback").slideUp(fade, go);
		}, 2000);
	    }
	}
    },


    end: function() {
	showSlide("finished");
	$("#indicator-stage").animate(
	    {"width": indicatorWidth + "px"}, fade);
    },
};

