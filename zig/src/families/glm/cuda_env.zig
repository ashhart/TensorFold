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

test {
    _ = @import("cuda_env_test.zig");
}
