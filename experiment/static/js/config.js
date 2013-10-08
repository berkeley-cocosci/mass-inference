// TODO: document this file
// TODO: document update_config function
// TODO: document load_config function
// TODO: document config object

var EXPERIMENT = Object.freeze({
    pretest: 0,
    experiment: 1,
    posttest: 2,
    length: 3
});

var TRIAL = Object.freeze({
    prestim: 0,
    stim: 1,
    fall_response: 2,
    feedback: 3,
    mass_response: 4,
    length: 5
});

// Experiment configuration object
var Config = function (condition, counterbalance) {

    // These are numeric codes representing the condition and whether
    // the trials are counterbalanced
    this.cond_code = condition;
    this.cb_code = counterbalance;

    // string to hold the human-readable condition name
    this.condition = null;
    // whether debug information should be printed out
    this.debug = true;
    // the amount of time to fade HTML elements in/out
    this.fade = 200;
    // list of trial information object for each experiment phase
    this.trials = new Object();
    // lists of pages and examples for each instruction page
    this.instructions = new Object();

    // We know the list of pages we want to display a priori
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

    // Parse the JSON object that we've requested and load it into the
    // configuration
    this.parse_config = function (data) {
        this.trials[EXPERIMENT.pretest] = data["pretest"];
        this.trials[EXPERIMENT.experiment] = data["experiment"]; 
        this.trials[EXPERIMENT.posttest] = data["posttest"];

        this.instructions[EXPERIMENT.pretest].examples = [
            data.unstable_example.stimulus,
            data.stable_example.stimulus,
            null
        ];
        
        this.instructions[EXPERIMENT.experiment].examples = [
            data.mass_example.stimulus
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
