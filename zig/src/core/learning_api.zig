const Example = @import("engine_api.zig").Example;
pub const LearnRequest = struct {
    train: []const Example = &.{},
    think: []const Example = &.{}, // the facts' answers with thinking on, learned beside `train` (one of each a step)
    held: []const Example = &.{},
    near: []const Example = &.{},
    keep: []const Example = &.{}, // answers of every kind this lesson's change must leave as they are
    asked: []const Example = &.{}, // more wordings of the fact's questions, for its router (prompts only)
    unasked: []const Example = &.{}, // questions it must not answer, for its router (prompts only)
    kept: []const Example = &.{}, // questions the last lesson took, with the answers they had (a reroute's)
    undo: bool = false, // instead take the last lesson's change back, as if it never ran
    steps: u32 = 60, // bounded steps this time at most
    more: bool = false, // more steps on the last lesson's rows, which are not captured again
    reroute: bool = false, // the last lesson's router fitted again with `unasked` among what it must not take
    drop: bool = false, // the last lesson taken out whole
    save: bool = false, // every kept lesson written beside the model's shards, each with its router
    retire: []const Neuron = &.{}, // instead zero these neurons of facts a newer fact replaces
    unretire: bool = false, // instead put back the neurons the last retire zeroed (the newer fact was not learned)
    // instead zero these neurons' input weights (a fact deleted on request), kept to put back
    forget: []const Neuron = &.{},
    // instead put back, bit for bit, what the last forget changed (a kept fact did not come back)
    unforget: bool = false,
};

/// A neuron a pocket lesson placed (Sliding Weights): its layer, its index in that layer's pocket, the row it keys on.
pub const Neuron = struct { layer: u16, neuron: u16, key: NeuronKey };
pub const NeuronKey = enum { reply, subject, value };

/// A pocket lesson's stage, as it runs: its rows read, its neurons fitted, their outputs, its held-out answers judged.
pub const LearnPhase = enum { reading, fitting, values, checking };

/// Where a pocket lesson stands: its unit `at` of `of` (0: unknown), its outputs' loss, its neurons once placed.
pub const LearnProgress = struct { phase: LearnPhase, at: u32 = 0, of: u32 = 0, loss: f32 = 0, neurons: []const Neuron = &.{} };

/// A lesson's outcome in order, ending with `done` (a message says why it failed); slices live only during the call.
pub const LearnEvent = union(enum) {
    learned: struct { recalled: bool, steps: u32, loss: f32, neurons: []const Neuron = &.{} }, // neurons: a pocket's
    progress: LearnProgress,
    done: struct { message: []const u8 = "" },
};

/// Where a learn request's events go; called on the engine's thread, so it must return at once.
pub const LearnSink = struct {
    ctx: *anyopaque,
    event: *const fn (ctx: *anyopaque, event: *const LearnEvent) void,
};

pub const LearnError = error{ Closed, Busy, Unsupported };

/// A family's learner, which LaneHost steps one bounded unit at a time while no stream decodes.
pub const Learner = struct {
    ctx: *anyopaque,
    /// Start a job; its events, `done` last, go to `sink` from later steps.
    begin: *const fn (ctx: *anyopaque, request: *const LearnRequest, sink: LearnSink) anyerror!void,
    /// One unit of the job: `changed` when weights moved (kept prompt states are then dropped), `done` when it ended.
    step: *const fn (ctx: *anyopaque) Step,
    /// End the open job now; it still sends `done`, with the reason.
    abort: *const fn (ctx: *anyopaque) void,

    /// stale: the device lacks a put-back the learner holds, so replies are refused until a later step clears it.
    pub const Step = struct { done: bool, changed: bool, stale: bool = false };
};
