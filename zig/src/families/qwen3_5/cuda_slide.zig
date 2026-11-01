//! Dense Living Weights adapter; the MoE adapter shares its implementation with a gated family descriptor.
const std = @import("std");
pub const For = @import("cuda_slide_family.zig").For;
const Dense = For(@import("living").families.qwen3_5);
pub const family = Dense.family;
pub const layout = Dense.layout;
pub const layoutOf = Dense.layoutOf;
pub const linked = Dense.linked;
pub const Example = Dense.Example;
pub const Lesson = Dense.Lesson;
pub const Report = Dense.Report;
pub const Step = Dense.Step;
pub const Trainer = Dense.Trainer;
pub const Model = Dense.Model;
pub const Learner = Dense.Learner;

test "the 27B's layout: every layer widened, all but the last learning, a written pocket left out of the width" {
    const lay = try layout(5120, 64, 17408 + 64, 64);
    try std.testing.expectEqual(@as(usize, 17408), lay.width);
    try std.testing.expect(lay.widen.isSet(63) and !lay.learn.isSet(63) and lay.learn.isSet(12));
}
