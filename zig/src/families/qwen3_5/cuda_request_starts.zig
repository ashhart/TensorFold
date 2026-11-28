//! Qwen3.5 message-start resume decision; contract pinned by native/cuda_snaps.zig's requestStepFor.
const std = @import("std");

/// Starts stay opt-in; only explicit 1 turns them on (pinned by cuda_request_starts_test.zig).
pub const default_on = false;
pub const switch_name = "TF_QWEN_REQUEST_STARTS";

pub fn on() bool {
    const v = @import("core").profile.lookup(switch_name) orelse return default_on;
    return std.mem.eql(u8, v, "1");
}
