// --------------------------------------------------------------------
// experiment.js -- loaded after slides.js
//
// Written by Jessica Hamrick (jhamrick@berkeley.edu)
//
// --------------------------------------------------------------------


// --------------------------------------------------------------------
// Configuration

var pageUrl = "index.py?page=";
var actionUrl = "index.py?a=";

var experiment_DEBUG = true;

// --------------------------------------------------------------------
// Generic helper functions

function post(action, data, handler) {
    var request = $.ajax({
	type: "POST",
	url: actionUrl + action,
	data: data,
    });
    request.done(handler);
    request.fail(experiment.error);}

// --------------------------------------------------------------------
// Experiment

var experiment = {

    pid : undefined,
    validationCode : undefined,

    playlist : undefined,
    numTrials : undefined,

    stimulus : undefined,
    index : undefined,
    starttime : undefined,
    
    initialize : function() {
	post('initialize', {}, function (msg) {
	    var info = $.parseJSON(msg);

	    experiment.pid = info.pid;
	    experiment.playlist = info.playlist;
	    experiment.numTrials = info.numTrials;

	    slides.show("instructions1a");
	});
    },

    nextTrial : function() {
	post('trialinfo', {pid : experiment.pid}, 
	    function (info) {

		// Finished the training block
		if (info.index == 'finished training') {
		    experiment.numTrials = info.numTrials;
		    experiment.playlist = info.playlist;
		    slides.show("instructions2");
		}
		
		// Finished the experiment block
		else if (info.index == "finished experiment") {
		    experiment.numTrials = info.numTrials;
		    experiment.playlist = info.playlist;
		    slides.show("instructions3");
		}

		// Finished the post test
		else if (info.index == 'finished posttest') {
		    experiment.validationCode = info.code;
		    slides.show("finished");
		}

		// Normal trial
		else {
		    experiment.index = info.index;
		    experiment.stimulus = info.stimulus;
		    slides.show("trial");
		}
	    });
    },	

    showStimulus : function () {
	slides.trial.play();
    },

    // The stimulus is done playing, so get a response from the user
    queryResponse : function () {
	// record the time so we know how long they take to answer
	experiment.starttime = new Date().getTime();

	// show the responses
	slides.trial.showQuery();
    },

    // User submits a response
    submitResponse : function(val) {
	var time = new Date().getTime() - experiment.starttime;
	var data = {
	    pid : experiment.pid,
	    time : time / 1000,
	    response : val,
	    index : experiment.index,
	};

	// XXX: need to submit index and handle it because it's
	// possible to submit a response twice
	
	post("submit", data, experiment.getFeedback);
    },

    // Show feedback to the user after they respond
    getFeedback : function(msg) {
	experiment.textFeedback = msg.feedback;
	experiment.showVideoFeedback = msg.visual;
	slides.trial.showFeedback();
    }
};

