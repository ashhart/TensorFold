//! Clef's joint schema head on the GPU: weights uploaded once, a state per readout stream, pieces, then the finish.
pub const launch = @import("launch.zig");
pub const Fns = launch.Fns;
pub const plan = @import("core").joint_schema.plan;
pub const weights = @import("weights.zig");
pub const Head = @import("head.zig").Head;
pub const state = @import("state.zig");
pub const State = state.State;
pub const Request = state.Request;
pub const consume = @import("consume.zig").consume;
pub const finish = @import("finish.zig").finish;
pub const attention = @import("attention.zig");

test {
    _ = weights;
    _ = attention;
    _ = @import("head.zig");
    _ = state;
    _ = @import("consume.zig");
    _ = @import("finish.zig");
}
