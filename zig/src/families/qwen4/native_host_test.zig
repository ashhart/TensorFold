//! The public host and capability surface run through owned recording storage, with no device or CPU serving fallback.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const Api = @import("core").api;
const L = @import("core").lanes;
const support = @import("engine_test.zig");
const fixture = @import("lanes_adapter_test.zig");
const P = struct {
    const Self = @This();
    pub const Engine = @import("engine.zig").Engine(support.G);
    pub const Drawer = fixture.D;
    a: std.mem.Allocator,
    storage: support.B = .{},
    var freed: usize = 0;
    var live: ?*Self = null;
    pub fn open(mem: std.mem.Allocator, _: std.Io, _: Api.Open) !*Self {
        const p = try mem.create(Self);
        p.* = .{ .a = mem };
        live = p;
        return p;
    }
    pub fn room(_: *Self, _: std.mem.Allocator, _: std.Io) !u64 {
        return 256 << 20;
    }
    pub fn loader(_: *Self) support.Loader {
        return .{};
    }
    pub fn drawer(_: *Self, _: std.mem.Allocator, _: usize, rows: usize, _: usize) !Drawer {
        return .{ .rows = rows };
    }
    pub fn healthy(_: *const Self) bool {
        return true;
    }
    pub fn close(p: *Self) !void {
        try p.storage.drain();
        try t.expectEqual(@as(usize, 0), p.storage.held);
        const mem = p.a;
        mem.destroy(p);
        freed += 1;
        live = null;
    }
};
test "Qwen4 native host: advertised capabilities, real shared host startup and failed-completion parent recovery" {
    var f = try fixture.Fixture.open();
    defer f.close();
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const path = try std.fmt.allocPrint(arena.allocator(), ".zig-cache/tmp/{s}", .{f.tmp.sub_path});
    const H = @import("native_host.zig").Host(P, Api, L);
    var problem: @import("config.zig").Problem = .{};
    P.freed = 0;
    const opened = try H.open(a, std.testing.io, .{ .dir = path, .model_type = "qwen4", .context = 32, .lanes = 2, .lanes_fixed = true, .prompt_cache_gib = 0 }, &problem);
    const info = opened.engine.info();
    try t.expectEqualStrings("qwen4", info.name);
    try t.expectEqual(@as(u32, 2), info.lanes);
    try t.expect(!info.plain_only and !info.greedy_only and info.structures and info.loop_guard and info.answer_limit);
    try t.expectEqual(@as(u32, 16), info.prefill_step);
    try t.expect(!info.prompt_cache and !info.warm_turns);
    try t.expect(info.vision == null and !info.logprobs and !info.prompt_logprobs);
    P.live.?.storage.fail_drain = true;
    opened.close(opened.ctx);
    try t.expectEqual(@as(usize, 0), P.freed);
    try t.expect(!H.recover());
    try t.expectEqual(@as(usize, 0), P.freed);
    P.live.?.storage.fail_drain = false;
    try t.expect(H.recover());
    try t.expectEqual(@as(usize, 1), P.freed);
    try t.expect(P.live == null);
}

const Box = struct {
    done: std.atomic.Value(bool) = .init(false),
    tokens: [16]u32 = undefined,
    count: usize = 0,
    cached: u32 = 0,
    failed: bool = false,
    fn event(p: *anyopaque, _: Api.Id, e: *const Api.Event) void {
        const b: *Box = @ptrCast(@alignCast(p));
        switch (e.*) {
            .tokens => |ids| {
                if (b.count + ids.len > b.tokens.len) {
                    b.failed = true;
                    return;
                }
                @memcpy(b.tokens[b.count..][0..ids.len], ids);
                b.count += ids.len;
            },
            .prefilled => |n| b.cached = n,
            .finished => |f| {
                b.failed = b.failed or f.reason == .failed;
                b.done.store(true, .release);
            },
            else => {},
        }
    }
    fn wait(b: *Box) !void {
        const until = std.Io.Timestamp.now(std.testing.io, .awake).addDuration(.fromSeconds(15));
        while (!b.done.load(.acquire)) {
            if (std.Io.Timestamp.now(std.testing.io, .awake).nanoseconds > until.nanoseconds) return error.Qwen4HostTestTimeout;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        try t.expect(!b.failed);
    }
};
test "Qwen4 native host: public requests share the core and kept prompt state through the serving factory" {
    var f = try fixture.Fixture.open();
    defer f.close();
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const path = try std.fmt.allocPrint(arena.allocator(), ".zig-cache/tmp/{s}", .{f.tmp.sub_path});
    const H = @import("native_host.zig").Host(P, Api, L);
    var problem: @import("config.zig").Problem = .{};
    const opened = try H.open(a, std.testing.io, .{ .dir = path, .model_type = "qwen4", .context = 48, .lanes = 2, .lanes_fixed = true, .prompt_cache_gib = 0.01 }, &problem);
    defer opened.close(opened.ctx);
    try t.expect(opened.engine.info().draft_sources.copies);
    try t.expect(!opened.engine.info().draft_sources.mtp);
    try t.expect(opened.engine.info().draft_sources.drafter == null);
    var tokens: [24]u32 = undefined;
    for (&tokens, 0..) |*v, i| v.* = @intCast(i % 29);
    const request: Api.Request = .{ .prompt = &tokens, .max_tokens = 6, .history_len = 16, .drafts = true };
    var first: Box = .{};
    var second: Box = .{};
    try opened.engine.submit(1, &request, .{ .ctx = &first, .event = Box.event });
    try opened.engine.submit(2, &request, .{ .ctx = &second, .event = Box.event });
    try first.wait();
    try second.wait();
    try t.expectEqual(@as(usize, 6), first.count);
    try t.expectEqualSlices(u32, first.tokens[0..first.count], second.tokens[0..second.count]);
    var resumed: Box = .{};
    try opened.engine.submit(3, &request, .{ .ctx = &resumed, .event = Box.event });
    try resumed.wait();
    try t.expectEqual(@as(u32, 16), resumed.cached);
    try t.expectEqualSlices(u32, first.tokens[0..first.count], resumed.tokens[0..resumed.count]);
    var plain: Box = .{};
    var plain_request = request;
    plain_request.drafts = false;
    plain_request.untaught = true;
    try opened.engine.submit(4, &plain_request, .{ .ctx = &plain, .event = Box.event });
    try plain.wait();
    try t.expectEqual(@as(u32, 0), plain.cached);
    try t.expectEqualSlices(u32, first.tokens[0..first.count], plain.tokens[0..plain.count]);
}
