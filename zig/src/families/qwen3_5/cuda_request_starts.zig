//! Qwen3.5 message-start resume decision: the switch, the default, and the shared-KV carve-out.
//! Pulled out of cuda_native.zig to hold that file under the project's 600-line cap; the contract
//! is the same one native/cuda_snaps.zig's requestStepFor pins, and host tests cover both sides.
const std = @import("std");

/// Starts stay opt-in: the 3896 crossover matrix measured starts-off winning every shape on the
/// 5060 Ti, so unset keeps main's behavior (off). Only explicit 1 turns them on, and only then
/// does shared KV refuse (pinned by cuda_request_starts_test.zig).
pub const default_on = false;
pub const switch_name = "TF_QWEN_REQUEST_STARTS";

pub fn on() bool {
    const v = @import("core").profile.lookup(switch_name) orelse return default_on;
    return std.mem.eql(u8, v, "1");
}
