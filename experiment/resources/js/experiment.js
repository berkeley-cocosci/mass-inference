
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

function embedVideo(url, div, flashvars, params, attributes, callback) {
    $(".example-cover").show();
    swfobject.embedSWF(
	url, div, videoWidth, videoHeight, flashVersion, 
	expressInstallSwfurl, flashvars, params, attributes, callback);
}

function showSlide(id) {
    $(".slide").hide();
    $('html, body').animate({ scrollTop: 0 }, 0);
    $(".example-cover").show();
    $("#"+id).show();
}

function showInstructions(id) {
    showSlide(id);
    setTimeout(function () {
	$(".example-cover").fadeOut(fade);
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

    // checkComprehension : function (block) {
    // 	var data = { pid : experiment.pid,
    // 		     time : 0;
    // 		     response : block };
    // 	post('submit', data, experiment.start);
    // },

    start: function () {
	post('trialinfo', {pid: experiment.pid}, function (info) {
	    if (experiment.show(info)) {
		showSlide("trial");
		// $("#play-button").focus();
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
	    showInstructions("instructions2");
	    return false;
	// } else if (info == 'show comprehension') {
	//     showSlide("comprehension");
	//     return false;
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
	$("#cover1").html(
	    "<img src='" + img + "' " +
		"width='" + videoWidth + "' " +
		"height='" + videoHeight + "'></img>");
	$("#cover1").fadeIn(fade, function () {
	    $("#player").hide();
	    $("#cover2").hide();
	    // Show instructions and focus the play button
	    $("#play-button").attr("disabled", false);
	    $("#video-instructions").show();
	    // $("#play-button").focus();
	});

	return true;
    },

    play : function () {
	var video = videoFolder + experiment.curVideo;
	var params = { wmode: "direct",
		       play: "true",
		       loop: "false" };
	var attributes = { id: "player" };
	
	$("#play-button").attr("disabled", true);
	$("#video-instructions").hide();//fadeOut(fade)
	embedVideo(video, "player", {}, params, attributes, 
		   function (e) {
		       if (e.success) {
			   $("#cover1").fadeOut(fade);
			   setTimeout(experiment.query, 5000);
		       }});
    },

    query : function () {
	var img = imageFolder + experiment.curImgB;

	// set the end image (last frame of the video)
	$("#cover2").html(
	    "<img src='" + img + "' " +
		"width='" + videoWidth + "' " +
		"height='" + videoHeight + "'></img>");
	// fade in the image and then remove the video when it's done
	$("#cover2").fadeIn(fade, function () {
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
	var go = function () {
	    $("#stable-feedback").html("");
	    $("#unstable-feedback").html("");
	    experiment.next();
	};

	// if the feedback is undefined, then don't display anything
	if (stable == "undefined") {
	    $("#responses").fadeOut(fade, go);

	// otherwise give feedback, then submit and go to the next
	// trial
	} else {
	
	    var txtfb = function () {
		if (stable) {
		    $("#stable-feedback").html("Tower does not fall!");
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
				$("#cover2").fadeOut(fade);
			    }, 300);
			    txtfb();
			    setTimeout(function () {
				$("#feedback").fadeOut(fade, go);
			    }, 3000);
			}});
	    } else {
		txtfb();
		setTimeout(function () {
		    $("#feedback").fadeOut(fade, go);
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

