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

// Load experiment configuration
var $c = {
    condition: null,
    debug: true,
    fade: 200,
    trials: new Object(),
    instructions: new Object()
};

var update_config = function (data, callback) {
    debug("Got JSON configuration");
    
    $c.trials[EXPERIMENT.pretest] = data["pretest"];
    $c.trials[EXPERIMENT.experiment] = data["experiment"]; 
    $c.trials[EXPERIMENT.posttest] = data["posttest"];

    $c.instructions[EXPERIMENT.pretest] = {
        pages: [
            "instructions1a.html",
            "instructions1b.html",
            "instructions1c.html"
        ],
        
        examples: [
            data.unstable_example.stimulus,
            data.stable_example.stimulus,
            null
        ]
    };
    
    $c.instructions[EXPERIMENT.experiment] = {
        pages: ["instructions2.html"],
        examples: [data.mass_example.stimulus]
    };

    $c.instructions[EXPERIMENT.posttest] = {
        pages: ["instructions3.html"],
        examples: [null]
    };

    callback();
};


var load_config = function(callback) {
    var get_condition = function (conditions) {
        debug("Got list of conditions");
        $c.condition = conditions[condition] + "-cb" + counterbalance;
        $.ajax({
            dataType: "json",
            url: "/static/json/" + $c.condition + ".json",
            async: false,
            success: function (data) { update_config(data, callback); }
        });
    };
        
    $.ajax({
        dataType: "json",
        url: "/static/json/conditions.json",
        async: false,
        success: get_condition
    });
};

