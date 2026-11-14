//! Validate GLM's TF_GLM_* settings on every rank before its CUDA open reads ranks, NCCL or weights.
const std = @import("std");
const environment = @import("sys").environment;
const kp = @import("core").kv_prompts;

/// Every `.glm` row of the shared table, parsed exactly as its later reader parses it.
pub fn startup() !void {
    try environment.checkScope(.glm, environment.Process{});
}

/// TF_GLM_REQUEST_STARTS=1 cuts off the rows TF_KV_SHARE and --learn-conversations keep on: refused with either.
pub fn requestStartsAlone(learn_conversations: bool) !void {
    const on = if (@import("core").profile.get(.TF_GLM_REQUEST_STARTS)) |v| std.mem.eql(u8, v, "1") else false;
    if (on and learn_conversations) return error.GlmRequestStartsWithLearnConversations;
    if (on and try kp.sharing() > 0) return error.GlmRequestStartsWithSharedKv;
}

/// TF_GLM_CONTEXT_CEILING: --context past config.json's window, as the attention has no rotary positions to stretch.
pub const ceiling_switch = "TF_GLM_CONTEXT_CEILING";

/// The window TF_GLM_CONTEXT_CEILING raises --context's ceiling to, read once the open has checked every .glm row.
pub fn raisedCeiling() !?u64 {
    try startup();
    return try environment.value(?u64, ceiling_switch, environment.Process.get(.{}, ceiling_switch), null);
}

test {
    _ = @import("cuda_env_test.zig");
}
