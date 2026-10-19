//! Cleared-environment child fixture exercises completed lone and shared rounds with a history-dependent CPU target.
const std = @import("std");
const lanes = @import("lanes");
const be = lanes.backend;
const timing = @import("qwen35").native.round_timing;
const Target = struct {
    fake: lanes.fake.Fake,
    table: be.Backend.VTable = undefined,
    trace: ?*timing.Trace,
    io: std.Io,
    rounds: usize = 0,
    offered: usize = 0,
    frame_limit: usize = 128,

    fn backend(t: *Target) be.Backend {
        t.table = t.fake.backend().vtable.*;
        t.table.verify = verify;
        return .{ .ptr = &t.fake, .vtable = &t.table, .round_observer = if (t.trace) |x| x.observer() else null };
    }
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const fake: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const t: *Target = @fieldParentPtr("fake", fake);
        if (t.trace) |x| x.reset();
        var at: usize = 0;
        while (at < windows.len) {
            var n: usize = 0;
            var rows: usize = 0;
            while (at + n < windows.len and rows + windows[at + n].rows() <= t.frame_limit) : (n += 1)
                rows += windows[at + n].rows();
            const start = if (t.trace != null) std.Io.Clock.awake.now(t.io) else std.Io.Timestamp.zero;
            try fake.backend().verify(windows[at..][0..n], out[at..][0..n]);
            if (t.trace) |x| {
                const ns = start.durationTo(std.Io.Clock.awake.now(t.io)).toNanoseconds();
                x.frame(at, n, rows, timing.Shape.init(rows, n, true, false, 1).total, @as(f64, @floatFromInt(ns)) / 1e6);
            }
            t.rounds += 1;
            t.offered += rows;
            at += n;
        }
    }
};

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.Arguments;
    const split = std.mem.eql(u8, args[1], "split");
    const shared = std.mem.eql(u8, args[1], "shared") or split;
    const partial = std.mem.eql(u8, args[1], "partial");
    const trace = try timing.Trace.create(a, timing.Trace.enabled());
    defer if (trace) |x| a.destroy(x);
    var target: Target = .{ .fake = .{ .gpa = a }, .trace = trace, .io = init.io, .frame_limit = if (split) 16 else 128 };
    defer target.fake.deinit();
    var cfg = try lanes.Config.init(a, .{ .exact_width = 16, .hidden_rows = true, .streams_exact = true, .batch_rows = 128, .max_streams = 8 }, 16, 15);
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var streams: [2]lanes.Stream = undefined;
    var count: usize = 0;
    defer for (streams[0..count]) |*s| s.deinit(a);
    for (streams[0..if (shared) @as(usize, 2) else 1]) |*s| {
        s.* = try lanes.Stream.init(a, .{ .id = "timing", .prompt = &.{ 1, 2, 3 }, .max_new = 20, .eos = &.{} });
        count += 1;
        try core.addStream(s);
        if (partial) {
            s.next = .{ .count = 3, .tokens = try a.dupe(u32, &.{ 0, 0, 0 }) };
        } else if (split) {
            try s.force.appendSlice(a, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });
        } else try s.force.appendSlice(a, &.{ 11, 12, 13, 14 });
    }
    while (core.activeCount() > 0) try core.step();
    var kept: usize = 0;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    for (streams[0..count]) |*s| {
        kept += s.cache_len - s.prompt().len;
        hash.update(std.mem.sliceAsBytes(s.emitted()));
    }
    var bytes: [256]u8 = undefined;
    const text = try std.fmt.bufPrint(&bytes, "{{\"rounds\":{d},\"offered\":{d},\"kept\":{d},\"sha\":\"{s}\"}}\n", .{ target.rounds, target.offered, kept, &std.fmt.bytesToHex(hash.finalResult(), .lower) });
    if (@import("builtin").os.tag == .windows) return std.Io.File.stdout().writeStreamingAll(init.io, text);
    if (std.c.write(1, text.ptr, text.len) != text.len) return error.Output;
}
