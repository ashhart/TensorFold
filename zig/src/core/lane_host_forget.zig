//! Prompt-forget requests queued by HTTP threads and answered between engine rounds.
const std = @import("std");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const prompt_forget = @import("prompt_forget.zig");
const LaneHost = @import("lane_host.zig").LaneHost;

/// Engine.forget_prompts: queued for the engine thread, which runs it before its next round.
pub fn submit(ctx: *anyopaque, span: []const u32) api.ForgetError!pc.Forgotten {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    h.lock();
    defer h.unlock();
    if (h.closing) return error.Closed;
    return h.forgets.ask(h.gpa, h.io, &h.mutex, &h.wake, span);
}

/// Runs the forgets asked since the last round (engine thread): a state a prompt pass still restores stays, counted.
pub fn take(h: *LaneHost) void {
    h.lock();
    if (!h.forgets.pending()) return h.unlock();
    var asks = h.forgets.take();
    var held: std.ArrayList(*const pc.Entry) = .empty;
    const room = if (held.ensureTotalCapacity(h.gpa, h.admitted.items.len)) |_| true else |_| false;
    if (room) for (h.admitted.items) |job| if (job.entry) |e| held.appendAssumeCapacity(e);
    for (asks.items) |ask| {
        ask.out = .{};
        for (h.queued.items) |job| if (prompt_forget.carries(job.request.prompt, ask.span)) {
            ask.out.in_flight += 1;
        };
        for (h.admitted.items) |job| if (prompt_forget.carries(job.request.prompt, ask.span)) {
            ask.out.in_flight += 1;
        };
    }
    h.unlock();
    for (asks.items) |ask| {
        const in_flight = ask.out.in_flight;
        // without room for the held list nothing is dropped: a state being restored must not be freed
        ask.out = if (!room) .{ .held = 1 } else if (h.cache) |store| store.forgetSpan(ask.span, held.items) else .{};
        ask.out.in_flight = in_flight;
    }
    h.lock();
    h.forgets.answer(h.io, asks.items);
    h.unlock();
    asks.deinit(h.gpa);
    held.deinit(h.gpa);
}
