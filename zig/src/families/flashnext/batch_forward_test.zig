//! Real lane loop -> Backend.verify -> batch_forward. The device records widths and emits sentinel picks only.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const fz = @import("replay.zig");
const Backend = @import("backend.zig").Backend;
const Session = @import("session.zig").Session;
const Engine = @import("engine.zig").Engine;
const be = lanes.backend;
const a0 = std.testing.allocator;

fn buffers(v: anytype, b: fz.Buf) void {
    const T = @TypeOf(v.*);
    if (T == fz.Buf) {
        v.* = b;
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |s| inline for (s.field_names) |name| {
            buffers(&@field(v.*, name), b);
        },
        .array => for (v) |*x| {
            buffers(x, b);
        },
        else => {},
    }
}
fn site(r: *fz.Run, role: []const u8) !void {
    for (@import("roles_gen.zig").entries) |entry| {
        if (!std.mem.startsWith(u8, entry.site, role) or entry.site[role.len] != '|') continue;
        const text = for (@import("kernel_sources").flashnext_gen.sources) |source| {
            if (std.mem.eql(u8, source.name, entry.file)) break source.text;
        } else return error.NoSource;
        const v = try r.arena.create(fz.Variant);
        v.* = .{ .inputs = try r.arena.dupe([]const u8, entry.inputs), .outputs = try r.arena.dupe([]const u8, entry.outputs), .meta = try r.arena.dupe([]const u8, entry.meta), .name = entry.function, .text = text, .pipe = .{ .name = entry.function, .text = text } };
        try r.roles.put(r.arena, entry.site, .{ .v = v, .grid = mtl.Size.of(entry.grid[0], entry.grid[1], entry.grid[2]), .tg = mtl.Size.of(entry.tg[0], entry.tg[1], entry.tg[2]) });
    }
}
const Verify = *const fn (*anyopaque, []const be.Window, []be.Verified) anyerror!void;
const Keep = *const fn (*anyopaque, []const be.Window, []const []const u32) anyerror!void;

pub fn check(verify: Verify, keep: Keep) !void {
    var arena: std.heap.ArenaAllocator = .init(a0);
    defer arena.deinit();
    const a = arena.allocator();
    mtl.allocator = a;
    var r: fz.Run = .{ .arena = a, .device = .{}, .queue = .{}, .serial = true, .xnew = true, .fused_xsum = true, .skip = 0xff & ~@as(u32, (1 << 2) | (1 << 6)), .argmax_pipe = .{ .name = "argmax" } };
    try site(&r, "qa_embed_rows@embed");
    try site(&r, "qa_ple_lookup@ple");
    try site(&r, "q4_ple_gate@ple");
    try site(&r, "q4_ple_conv@ple");
    try site(&r, "qa_expert_gateup@moe.gate");
    try site(&r, "qa_expert_down_y@moe.down");
    r.kv_pipe = .{};
    r.router_pipe = .{};
    r.xgu_pipe = .{ .name = "fz_xgu" };
    r.xdown_pipe = .{ .name = "fz_xdown" };
    r.draw_pipe = .{ .name = "draw" };
    const dummy: fz.Buf = .{ .b = try r.device.buffer(2 << 20, fz.opts) };
    try r.shapes.put(a, "Kc_shape", dummy.b);
    try r.shapes.put(a, "IDS_shape", dummy.b);
    var m: fz.Model = undefined;
    @memset(std.mem.asBytes(&m), 0);
    buffers(&m, dummy);
    m.r = &r;
    m.cap = 128;
    m.ple.sizes = @splat(1);
    const lane: fz.Lane = .{ .six = .{ .wq = dummy, .sbt = dummy } };
    m.head = lane;
    m.ple.kv = lane;
    for (&m.layers, 0..) |*l, i| {
        l.linear = i % 4 != 3;
        l.proj = lane;
        l.out = lane;
    }
    inline for (@typeInfo(fz.Tmp).@"struct".field_names, @typeInfo(fz.Tmp).@"struct".field_types) |name, T| @field(m.t, name) = if (T == fz.Buf) .{ .b = try r.device.buffer(512 << 10, fz.opts) } else .{ dummy, dummy };
    var e: Engine = undefined;
    e.r = &r;
    e.m = &m;
    e.o_cs = dummy.b;
    e.o_so = dummy.b;
    e.draw_at = dummy;
    var sessions = [_]Session{ .{ .m = m }, .{ .m = m } };
    for (&sessions) |*s| {
        s.m.t.rows = .{ .b = try r.device.buffer(64, fz.opts) };
        s.m.t.mdims = .{ .b = try r.device.buffer(64, fz.opts) };
        s.m.t.pos8 = .{ .b = try r.device.buffer(64, fz.opts) };
        s.m.t.nk8 = .{ .b = try r.device.buffer(64, fz.opts) };
        s.m.t.kvmeta = .{ .b = try r.device.buffer(64, fz.opts) };
        s.m.t.ple_ids = .{ .b = try r.device.buffer(1024, fz.opts) };
    }
    var b: Backend = .{ .gpa = a, .e = &e, .primary = &m, .sessions = &sessions, .hidden = dummy, .grammar = .{ .pipe = .{}, .bits = dummy } };
    // Null widths preserve the recorded 6/32 dispatch. Non-null widths include mixed routed/shared widths.
    for ([_][4]usize{ .{ 4, 64, 4, 64 }, .{ 6, 32, 6, 32 }, .{ 4, 32, 8, 128 } }) |w| {
        m.embed_w = if (w[0] == 6) null else .{ w[0], w[1] };
        m.ple.w = m.embed_w;
        m.ple.scale = .{ .pipe = .{ .name = "scale" }, .value = 0.5 };
        for (&m.layers) |*l| l.xw = if (w[0] == 6) null else .{ .gu = w, .down = w };
        mtl.expected = w;
        mtl.wrong = 0;
        mtl.calls = @splat(0);
        mtl.scale_calls = 0;
        r.enc = .{};
        m.windowMeta(1);
        try m.windowEncode(1, m.t.ids8);
        const legacy = m.t.picks.b.slice(u32, 1)[0];
        try std.testing.expectEqual(@as(u32, 7), legacy);
        try std.testing.expectEqual(@as(usize, 1), mtl.scale_calls);
        for ([_]u8{ 0, 1, 2, 3 }) |mode| {
            for ([_]bool{ false, true }) |drafted| {
                // The lane core owns acceptance. A known first draft matches, a later draft deliberately misses.
                var cfg = try lanes.Config.init(a, .{ .exact_width = 16, .mtp = true, .drafts = 3, .hidden_rows = true, .batch_rows = 16, .max_streams = 2 }, 16, 15);
                defer cfg.deinit(a);
                var clock: lanes.fake.FixedClock = .{};
                const Bridge = struct {
                    back: *Backend,
                    verify_: Verify,
                    keep_fn: Keep,
                    fn nofill(_: *anyopaque, _: *lanes.Stream) anyerror!void {
                        unreachable;
                    }
                    fn first(_: *anyopaque, _: *lanes.Stream, _: u64) anyerror!u64 {
                        unreachable;
                    }
                    fn queue(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
                        unreachable;
                    }
                    fn read(_: *anyopaque, _: u64) anyerror!u32 {
                        unreachable;
                    }
                    fn check_(p: *anyopaque, ws: []const be.Window, os: []be.Verified) anyerror!void {
                        const x: *@This() = @ptrCast(@alignCast(p));
                        try x.verify_(x.back, ws, os);
                    }
                    fn keep_(p: *anyopaque, ws: []const be.Window, ps: []const []const u32) anyerror!void {
                        const x: *@This() = @ptrCast(@alignCast(p));
                        try x.keep_fn(x.back, ws, ps);
                    }
                    fn draft(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {}
                    fn release(_: *anyopaque, _: *lanes.Stream) void {}
                };
                var bridge: Bridge = .{ .back = &b, .verify_ = verify, .keep_fn = keep };
                var core = lanes.Engine.init(a, &cfg, .{ .ptr = &bridge, .vtable = &.{ .prefill = Bridge.nofill, .first = Bridge.first, .queue = Bridge.queue, .read = Bridge.read, .verify = Bridge.check_, .keep = Bridge.keep_, .draft = Bridge.draft, .release = Bridge.release } }, clock.clock());
                defer core.deinit();
                const sampling: ?lanes.Sampling = if (mode % 2 == 0) null else .{ .seed = 123, .temperature = 0 };
                var stream = try lanes.Stream.init(a, .{ .id = "width", .prompt = &.{1}, .max_new = 8, .drafts = drafted, .sampling = sampling });
                var neighbor = try lanes.Stream.init(a, .{ .id = "neighbor", .prompt = &.{2}, .max_new = 8, .drafts = false, .sampling = sampling });
                defer neighbor.deinit(a);
                defer stream.deinit(a);
                stream.cache_len = 1;
                stream.pending = 7;
                for (&sessions, 0..) |*s, i| {
                    const metadata = s.m.t;
                    s.m = m;
                    s.m.t = metadata;
                    s.m.pos = 1;
                    s.m.state = 0;
                    s.m.state_row = 0;
                    s.round = .{};
                    s.used = i == 0 or mode >= 2;
                }
                sessions[0].held = @splat(0);
                sessions[0].held[0..2].* = .{ 7, 8 };
                sessions[0].held_n = 2;
                if (drafted) stream.next = .{ .count = 2 };
                try b.by.put(a, &stream, 0);
                defer _ = b.by.remove(&stream);
                try core.adopt(&stream);
                if (mode >= 2) {
                    neighbor.cache_len = 1;
                    neighbor.pending = 7;
                    try b.by.put(a, &neighbor, 1);
                    try core.adopt(&neighbor);
                }
                defer _ = b.by.remove(&neighbor);
                mtl.draws = 0;
                mtl.wrong = 0;
                mtl.scale_calls = 0;
                mtl.calls = @splat(0);
                try core.step();
                try std.testing.expect(stream.emitted().len > 0);
                try std.testing.expectEqual(legacy, stream.emitted()[0]);
                try std.testing.expectEqual(@as(usize, 0), mtl.wrong);
                try std.testing.expectEqual(@as(usize, 0), mtl.draws);
                try std.testing.expectEqual(@as(usize, 1), mtl.scale_calls);
                try std.testing.expectEqual([4]usize{ 1, 1, fz.LAYERS, fz.LAYERS }, mtl.calls);
                try std.testing.expectEqual(@as(usize, if (drafted) 3 else 1), sessions[0].round.rows);
                if (mode >= 2) {
                    try std.testing.expectEqual(legacy, neighbor.emitted()[0]);
                    try std.testing.expectEqual(@as(usize, if (drafted) 3 else 1), sessions[1].row0);
                }
                if (drafted) {
                    try std.testing.expectEqualSlices(u32, &.{ 7, 7 }, stream.emitted());
                    try std.testing.expectEqual(@as(usize, 3), sessions[0].m.pos);
                    try std.testing.expectEqual(@as(usize, 1), sessions[0].m.state_row);
                    try std.testing.expect(!sessions[0].round.pending);
                }
                stream.drafts = false;
                stream.dropHeld(a);
                mtl.first_blit = null;
                mtl.blits = 0;
                try core.step();
                try std.testing.expectEqual(@as(usize, if (drafted) fz.CS_ROW else 0), mtl.first_blit.?.offset);
                try std.testing.expectEqual(fz.CS_ROW, mtl.first_blit.?.bytes);
                try std.testing.expectEqual(@as(u32, 7), stream.emitted()[stream.emitted().len - 1]);
            }
        }
    }
}
