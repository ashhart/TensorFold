//! DFlash2 native graph, operation schedule and proposal contracts over shared checkpoint contracts.
pub const generation = @import("generation.zig");
pub const runtime_backend = @import("runtime_backend.zig");
pub const runtime_model = @import("runtime_model.zig");
pub const config = @import("config.zig");
pub const weights = @import("weights.zig");
pub const operators = @import("operators.zig");
pub const execution = @import("execution.zig");
pub const session = @import("session.zig");
pub const selector = @import("selector.zig");
test {
    _ = config;
    _ = weights;
    _ = operators;
    _ = execution;
    _ = session;
    _ = selector;
    _ = @import("runtime_head.zig");
    _ = @import("runtime_context.zig");
    _ = @import("fixtures.zig");
    _ = @import("execution_test.zig");
}
