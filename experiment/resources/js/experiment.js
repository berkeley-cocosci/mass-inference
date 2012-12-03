// Adjust this:

var videos = [1];

// --------------------------------------------------------------------

// if (!String.prototype.supplant) {
//     String.prototype.supplant = function (o) {
//         return this.replace(
// 		/{([^{}]*)}/g,
// 	    function (a, b) {
// 		var r = o[b];
// 		return typeof r === 'string' || typeof r === 'number' ? r : a;
// 	    }
// 	);
//     };
// }

var videoFolder = "resources/video/";
// var videoTemplate = "stim_{id}.swf";
var videoWidth = "320";
var videoHeight = "240";
var flashVersion = "9.0.0";
var expressInstallSwfurl = false;

function shuffle(o) {
    var j, x, i = o.length;
    while (i) { // same as while(i != 0)
	j = parseInt(Math.random() * i);
	x = o[--i];
	o[i] = o[j];
	o[j] = x;
    };
}

function setTimer(e) {
    if (e.success) {
	setTimeout(function () {
	    $("#instructions1").hide();
	    $("#video-container").hide();
	    $("#instructions2").show();
	    $("#question").show();
	}, 5000);
    }
}

function showVideo(curVideo) {
    var video = videoFolder + curVideo; //videoTemplate.supplant({id: curVideo});
    var flashvars = {};
    var params = { wmode: "direct",
		   play: "true",
		   loop: "false" };
    var attributes = { id: "player" };
    
    $("#video-container").show();
    $("#player").show();
    swfobject.embedSWF(
	video, "player", 
	videoWidth, videoHeight, flashVersion, expressInstallSwfurl, 
	flashvars, params, attributes, setTimer);
}

function checkAnswered(name) {
    var foo = $("input[name=" + name + "]");
    var f = function (s) { return $(s).attr("checked") };
    return _.any($("input[name=" + name + "]"), f);
}

function showSlide(id) {
    $(".slide").hide();
    $("#"+id).show();
}

// function random(a,b) {
//     if (typeof b == "undefined") {
// 	a = a || 2;
// 	return Math.floor(Math.random()*a);
//     } else {
// 	return Math.floor(Math.random()*(b-a+1)) + a;
//     }
// }

// --------------------------------------------------------------------


// Array.prototype.random = function() {
//     return this[random(this.length)];
// }

shuffle(videos);
var myVideoOrder = videos;

showSlide("introduction");

var experiment = {

    globalComprehension : "",
    index : 0,
    videoOrder : undefined,
    curVideo : "",
    timer : undefined,

    data: [],

    play : function() {
	$("#play-button").hide();
	showVideo(experiment.curVideo);
    },

    start: function() {
	var request = $.ajax({
	    type: "GET",
	    url: "../../index.py?f=start",
	});
	request.done(function (msg) {
	    experiment.videoOrder = msg;
	    experiment.next();
	});
	request.fail(function (msg) {
	    showSlide("error");
	});
    },

    end: function() {
	var request = $.ajax({
	    type: "POST",
	    url: "../../index.py?f=submit", 
	    data: experiment.data[0],
	});
	request.done(function (msg) { 
	    showSlide("finished");
	});
	request.fail(function (msg) {
	    showSlide("error");	    
	});
	//setTimeout(function() { turk.submit(experiment) }, 1500);
    },

    submitAndNext : function() {
	if (!checkAnswered("response")) {
	    alert("Please fill out all form fields.");
	    return false;
	} else {
	    var data = {
		index : experiment.index,
		curVideo : experiment.curVideo
	    };

	    data.response = $("input[name=response]:checked").val();
	    experiment.data.push(data);
	    return experiment.next();
	}
    },

    next: function() {
	experiment.index = experiment.index + 1;
	experiment.curVideo = experiment.videoOrder.shift();

	if (typeof experiment.curVideo == "undefined") {
	    return experiment.end();
	}

	showSlide("normal_trial");

	// Reset all select boxes
	$("input[name=response]").attr("checked", false);
	
	// Display the video stimulus.
	$("#instructions1").show();
	$("#play-button").show();
	$("#video-container").hide();
	$("#instructions2").hide();
	$("#question").hide();
	$("#next-button").hide();

	// Update the progress bar
	$("#indicator-stage").width((120 + experiment.index*60) + "px");

	$('html, body').animate({ scrollTop: 0 }, 0);
    }
};

function showInstructions() {
    var stableVideo = videoFolder + "stable.swf";
    var unstableVideo = videoFolder + "unstable.swf";
    var flashvars = {};
    var params = { wmode: "direct",
		   play: "true",
		   loop: "true" };
    var attributes = {};

    $("#introduction").hide();
    this.blur();
    
    $("#instructions").show();
    swfobject.embedSWF(
	stableVideo, "stable-example", 
	videoWidth, videoHeight, flashVersion, expressInstallSwfurl,
	flashvars, params, attributes);
    swfobject.embedSWF(
	unstableVideo, "unstable-example",
	videoWidth, videoHeight, flashVersion, expressInstallSwfurl,
	flashvars, params, attributes);
}

function startExperiment() {
    $("#instructions").hide();
    this.blur();
    return experiment.start();
}

function nextTrial() {
    this.blur();
    return experiment.submitAndNext();
}
