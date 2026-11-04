//! One reason for every wait: the fixed code a client keys on, and the words it reads.
const std = @import("std");
const reasons = @import("engine_api").wait_reason;

/// A retryable wait (HTTP 503 with Retry-After): the code clients key on and the message they read.
pub const Wait = struct { code: []const u8, message: []const u8 };

/// The server is draining for a restart: no new work until it ends.
pub const restarting: Wait = .{ .code = "server_restarting", .message = "server restarting: retry shortly" };
/// A parked engine is still unloading, or its reload is under way.
pub const loading: Wait = .{ .code = "engine_loading", .message = "the model is loading; retry shortly" };
/// A parked engine's reload was attempted and failed; the next request tries it again.
pub const reload_failed: Wait = .{ .code = "engine_reload_failed", .message = "the model could not reload; retry shortly" };

/// The problem an idle reload reports while the engine is still unloading.
pub const unloading_problem = "the engine is still unloading; send the request again";

/// A retryable capacity refusal: the engine had no room for this request and it can be sent again.
pub const engine_busy = reasons.waitCode(.engine_busy);
pub const engine_unavailable = reasons.waitCode(.engine_unavailable);
pub const memory_pressure = reasons.waitCode(.memory_pressure);

/// Every server wait and capacity code, in one bounded set.
pub const all = [_]Wait{ restarting, loading, reload_failed };
pub const capacity_codes = [_][]const u8{ engine_busy, engine_unavailable, memory_pressure };

/// The OpenAI-shaped wait body: the message, its type and the wait's code.
pub fn body(buf: []u8, w: Wait) []const u8 {
    return std.fmt.bufPrint(buf, "{{\"error\": {{\"message\": \"{s}\", \"type\": \"server_error\", \"param\": null, \"code\": \"{s}\"}}}}", .{ w.message, w.code }) catch unreachable;
}

/// The Anthropic-shaped wait body: its envelope has no code field.
pub fn anthropicBody(buf: []u8, w: Wait) []const u8 {
    return std.fmt.bufPrint(buf, "{{\"type\": \"error\", \"error\": {{\"type\": \"overloaded_error\", \"message\": \"{s}\"}}}}", .{w.message}) catch unreachable;
}

/// The reason an idle reload's problem stands for: still unloading, or a reload that failed.
pub fn reloadReason(problem: []const u8) Wait {
    return if (std.mem.startsWith(u8, problem, unloading_problem)) loading else reload_failed;
}
