//! Living Weights (Sliding Weights), family-neutral: the lesson learner.
pub const family = @import("family.zig");
pub const families = @import("families.zig");
pub const Family = family.Family;
pub const Layout = family.Layout;
pub const Layers = family.Layers;

pub const learner = @import("learner.zig");
pub const whole = @import("whole.zig");
pub const pocket = @import("pocket.zig");
pub const pocket_fit = @import("pocket_fit.zig");
pub const pocket_clock = @import("pocket_clock.zig");
pub const neuron = @import("neuron.zig");
pub const affine = @import("affine.zig");
pub const shards = @import("shards.zig");
pub const lesson_examples = @import("lesson_examples.zig");
pub const lesson_route = @import("lesson_route.zig");
pub const train_mode = @import("train_mode.zig");
pub const learn_shape = @import("learn_shape.zig");
pub const subspace = @import("subspace.zig");

test {
    _ = family;
    _ = families;
    _ = neuron;
    _ = affine;
    _ = shards;
    _ = learner;
    _ = whole;
    _ = pocket;
    _ = pocket_fit;
    _ = lesson_route;
    _ = subspace;
    _ = @import("router.zig");
    _ = @import("pocket_decouple.zig");
    _ = @import("pocket_test.zig");
    _ = @import("choice.zig");
    _ = @import("swiglu_test.zig");
    _ = @import("host_test.zig");
}
