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

var INSTRUCTIONS = new Object();
// need a function to set the instructions that will be called once $c
// is fully loaded
var update_instructions = function () {
    INSTRUCTIONS[EXPERIMENT.pretest] = {
        pages: [
            "instructions1a.html",
            "instructions1b.html",
            "instructions1c.html"
        ],
        
        examples: [
            $c.examples.unstable.stimulus,
            $c.examples.stable.stimulus,
            null
        ]
    };
    
    INSTRUCTIONS[EXPERIMENT.experiment] = {
        pages: ["instructions2.html"],
        examples: [$c.examples.mass.stimulus]
    };

    INSTRUCTIONS[EXPERIMENT.posttest] = {
        pages: ["instructions3.html"],
        examples: [null]
    };
};

// Load experiment configuration
var $c = {
    condition: null,
    debug: true,
    fade: 200,
    trials: new Object(),
    examples: new Object()
};

var load_config = function(callback) {
    var update_config = function (data) {
        debug("Got JSON configuration");
        
        $c.trials[EXPERIMENT.pretest] = data["pretest"];
        $c.trials[EXPERIMENT.experiment] = data["experiment"]; 
        $c.trials[EXPERIMENT.posttest] = data["posttest"];

        $c.examples.unstable = data.unstable_example;
        $c.examples.stable = data.stable_example;
        $c.examples.mass = data.mass_example;
        
        update_instructions();
        callback();
    };

    var get_condition = function (conditions) {
        debug("Got list of conditions");
        $c.condition = conditions[condition] + "-cb" + counterbalance;
        $.ajax({
            dataType: "json",
            url: "/static/json/" + $c.condition + ".json",
            async: false,
            success: update_config
        });
    };
        
    $.ajax({
        dataType: "json",
        url: "/static/json/conditions.json",
        async: false,
        success: get_condition
    });
};

