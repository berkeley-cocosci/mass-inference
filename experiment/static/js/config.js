/* config.js
 * 
 * This file contains the code necessary to load the configuration
 * for the experiment.
 */

// Enum-like object mapping experiment phase names to ids, in the
// order that the phases should be presented.
var EXPERIMENT = Object.freeze({
    pretest: 0,
    experiment: 1,
    posttest: 2,
    length: 3
});

// Enum-like object mapping trial phase names to ids, in the order
// that the phases should be presented.
var TRIAL = Object.freeze({
    prestim: 0,
    stim: 1,
    fall_response: 2,
    feedback: 3,
    mass_response: 4,
    length: 5
});

// Object to hold the experiment configuration. It takes as parameters
// the numeric codes representing the experimental condition and
// whether the trials are counterbalanced.
var Config = function (condition, counterbalance) {

    // These are the numeric codes for condition and counterbalancing
    this.cond_code = condition;
    this.cb_code = counterbalance;

    // String to hold the human-readable condition name
    this.condition = null;
    // Whether debug information should be printed out
    this.debug = true;
    // The amount of time to fade HTML elements in/out
    this.fade = 200;
    // List of trial information object for each experiment phase
    this.trials = new Object();

    // Lists of pages and examples for each instruction page.  We know
    // the list of pages we want to display a priori.
    this.instructions = new Object();
    this.instructions[EXPERIMENT.pretest] = {
        pages: ["instructions1a.html",
                "instructions1b.html",
                "instructions1c.html"]
    };
    this.instructions[EXPERIMENT.experiment] = {
        pages: ["instructions2.html"]
    };
    this.instructions[EXPERIMENT.posttest] = {
        pages: ["instructions3.html"],
        examples: [null]
    };

    // The list of all the HTML pages that need to be loaded
    this.pages = [
        "trial.html", 
        "submit.html"
    ].concat(
        $.map(this.instructions, 
              function(item) { 
                  return item.pages;
              })
    );

    // Parse the JSON object that we've requested and load it into the
    // configuration
    this.parse_config = function (data) {
        this.trials[EXPERIMENT.pretest] = data["pretest"];
        this.trials[EXPERIMENT.experiment] = data["experiment"]; 
        this.trials[EXPERIMENT.posttest] = data["posttest"];

        this.instructions[EXPERIMENT.pretest].examples = [
            data.unstable_example,
            data.stable_example,
            null
        ];
        
        this.instructions[EXPERIMENT.experiment].examples = [
            data.mass_example
        ];
    };

    // Load the condition name from the server
    this.load_condition = function () {
        var that = this;
        $.ajax({
            dataType: "json",
            url: "/static/json/conditions.json",
            async: false,
            success: function (data) {
                if (that.debug) {
                    console.log("Got list of conditions");
                }
                that.condition = data[that.cond_code] + 
                    "-cb" + that.cb_code;
            }
        });
    };

    // Load the experiment configuration from the server
    this.load_config = function () {
        var that = this;
        $.ajax({
            dataType: "json",
            url: "/static/json/" + this.condition + ".json",
            async: false,
            success: function (data) { 
                if (that.debug) {
                    console.log("Got configuration data");
                }
                that.parse_config(data);
            }
        });
    };

    // Request from the server configuration information for this run
    // of the experiment
    this.load_condition();
    this.load_config();
};
