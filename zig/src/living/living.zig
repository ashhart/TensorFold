//! Living Weights, family-neutral: the learner, its pocket of new neurons, their fits and the shard writer.
pub const family = @import("family.zig");
pub const families = @import("families.zig");
pub const Family = family.Family;
pub const Layout = family.Layout;
pub const Layers = family.Layers;

pub const learner = @import("learner.zig");
pub const whole = @import("whole.zig");
pub const choice = @import("choice.zig");
pub const editor = @import("editor.zig");
pub const probe = @import("probe.zig");
pub const router = @import("router.zig");
pub const subspace = @import("subspace.zig");
pub const train_mode = @import("train_mode.zig");
pub const learn_shape = @import("learn_shape.zig");
pub const lesson_examples = @import("lesson_examples.zig");
pub const lesson_route = @import("lesson_route.zig");
pub const neuron = @import("neuron.zig");
pub const affine = @import("affine.zig");
pub const shards = @import("shards.zig");
pub const pocket = @import("pocket.zig");
pub const pocket_clock = @import("pocket_clock.zig");
pub const pocket_dump = @import("pocket_dump.zig");
pub const pocket_fit = @import("pocket_fit.zig");
pub const pocket_forget = @import("pocket_forget.zig");
pub const pocket_joint = @import("pocket_joint.zig");
pub const pocket_load = @import("pocket_load.zig");
pub const pocket_mark = @import("pocket_mark.zig");
pub const pocket_pack = @import("pocket_pack.zig");
pub const pocket_prefix = @import("pocket_prefix.zig");
pub const pocket_quiet = @import("pocket_quiet.zig");
pub const pocket_read = @import("pocket_read.zig");
pub const pocket_values = @import("pocket_values.zig");

test {
    _ = family;
    _ = families;
    _ = neuron;
    _ = affine;
    _ = shards;
    _ = learner;
    _ = whole;
    _ = subspace;
    _ = pocket;
    _ = pocket_fit;
    _ = pocket_quiet; // an earlier fact refitted quiet only on a new fact's rows
    _ = pocket_mark; // a lesson taken out puts the pocket back bit for bit
    _ = @import("pocket_forget_test.zig"); // a fact deleted on request, and put back
    _ = router;
    _ = choice;
    _ = lesson_route;
    _ = @import("swiglu_test.zig");
    _ = @import("host_test.zig");
}
