//! Retain unchanged message prefixes before the first live-fetch request reaches the engine.
const std = @import("std");
const api = @import("engine_api");
const chat = @import("chat.zig");
const prompt = @import("prompt.zig");
const chunk_plan = @import("chunk_plan.zig");
const Server = @import("server.zig").Server;
const Cx = @import("errors.zig").Cx;

/// Only exact rendered prefixes before the message being edited may be retained.
pub fn prepare(srv: *Server, cx: *Cx, input: chat.Input, thinking: bool, effort: ?[]const u8, request: *api.Request) std.mem.Allocator.Error!void {
    const live = input.notes_live orelse return;
    if (!live.on or live.dry or input.prompt != null) return;
    if (request.structure != null or request.call != null or request.logprobs != null or request.prompt_logprobs != null or request.think_budget > 0) return;
    const list = input.messages.items() orelse return;
    const count: usize = switch (live.place) {
        .user => blk: {
            var last = list.len;
            while (last > 0) : (last -= 1) if (std.mem.eql(u8, list[last - 1].strField("role") orelse "", "user")) break;
            if (last < 2) return;
            break :blk last - 1;
        },
        .turn => list.len,
        .first => blk: {
            if (live.prepared_block.len == 0) return;
            for (list, 0..) |m, i| {
                if (!std.mem.eql(u8, m.strField("role") orelse "", "system")) continue;
                if (std.mem.indexOf(u8, m.strField("content") orelse "", live.prepared_block) != null) break :blk i;
            }
            return;
        },
    };
    if (count == 0) return;
    const before = prompt.renderIds(srv, cx, .{ .array = list[0..count] }, input.tools, thinking, effort, false) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Refused => return,
    };
    if (before.len < 2 or !std.mem.startsWith(u32, request.prompt, before)) return;
    // One token before insertion also leaves a lookahead token inside the unchanged prefix.
    request.retain_at = @intCast(before.len - 1);
    if (srv.chunks.step > 0) request.chunks = try chunk_plan.withCut(cx.a, request.chunks, request.retain_at, request.prompt.len, srv.chunks.min_chunk);
}

/// Re-rendered continuations keep the same cut only while its prefix and lookahead token remain unchanged.
pub fn continuation(srv: *const Server, a: std.mem.Allocator, base: *const api.Request, next: *api.Request) std.mem.Allocator.Error!void {
    const len = @as(usize, base.retain_at) + 1;
    next.retain_at = 0;
    if (base.retain_at == 0 or len > base.prompt.len or len > next.prompt.len) return;
    if (!std.mem.eql(u32, base.prompt[0..len], next.prompt[0..len])) return;
    next.retain_at = base.retain_at;
    if (srv.chunks.step > 0) next.chunks = try chunk_plan.withCut(a, next.chunks, next.retain_at, next.prompt.len, srv.chunks.min_chunk);
}
