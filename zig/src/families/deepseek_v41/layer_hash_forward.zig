//! The existing trace readback, with optional persistent request/window receipts.
const std = @import("std");
const cuda = @import("cuda");
const Forward = @import("forward.zig").Forward;
const hash = @import("layer_hash.zig");
const calls = @import("block_calls.zig");
const run = @import("run.zig");
const rowtab = @import("rowtab.zig");

pub fn attach(f: *Forward, io: std.Io, path: ?[]const u8, rank: u32) !void {
    if (f.layer_hash != null) return error.HashAlreadyOpen;
    f.layer_hash = try hash.Writer.open(f.gpa, io, path, rank);
    if (f.layer_hash != null) f.opts.trace = true;
}

pub fn begin(f: *Forward, slot: u32, ids: []const u32) !void {
    const w = f.layer_hash orelse return;
    const req = try w.begin(slot, ids);
    f.hash_slot = slot;
    if (f.leads()) try f.link.?.send(&hash.frame(slot, req));
}

pub fn follow(f: *Forward, msg: []const i64) !void {
    const req = try hash.readFrame(msg);
    if (f.layer_hash) |w| {
        try w.accept(req.slot, req.request);
        f.hash_slot = req.slot;
    }
}

pub fn trace(f: *Forward, r: *run.Runner, c: *const calls.Call, phase: hash.Phase) !void {
    return read(f, r, c, phase, null, &.{});
}

pub fn batch(f: *Forward, r: *run.Runner, c: *const calls.Call, mix: []const rowtab.Seg, ids: []const u32) !void {
    return read(f, r, c, .decode, mix, ids);
}

fn read(f: *Forward, r: *run.Runner, c: *const calls.Call, phase: hash.Phase, mix: ?[]const rowtab.Seg, ids: []const u32) !void {
    if (f.trace == null and f.layer_hash == null) return;
    const t = c.args[0].arg.t;
    var count: usize = 1;
    for (t.shape) |n| count = try std.math.mul(usize, count, @intCast(n));
    const host = try f.gpa.alloc(u8, try std.math.mul(usize, 2, count));
    defer f.gpa.free(host);
    try r.stream.synchronize();
    try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(t), .len = host.len }, 0, host);
    const L: u32 = @intCast(c.args[1].arg.i);
    if (f.trace) |map| {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(host, &digest, .{});
        try map.put(f.gpa, L, std.fmt.bytesToHex(digest, .lower));
    }
    const w = f.layer_hash orelse return;
    const cols: usize = @intCast(t.shape[1]);
    if (mix) |segments| {
        var at: u32 = 0;
        for (segments) |sg| {
            try w.layer(phase, L, .{ .slot = sg.slot, .start = sg.start, .row0 = at, .rows = sg.rows, .ids = ids[at..][0..sg.rows] }, cols, host);
            at += sg.rows;
        }
    } else if (f.multi) |m| {
        for (m.segs, m.row0) |sg, at| try w.layer(phase, L, .{ .slot = sg.slot, .start = sg.start, .row0 = @intCast(at), .rows = @intCast(sg.ids.len), .ids = sg.ids }, cols, host);
    } else {
        const pending = f.slot.pending orelse return error.HashWithoutWindow;
        try w.layer(phase, L, .{ .slot = f.hash_slot, .start = pending.start, .row0 = 0, .rows = pending.n, .ids = f.ids }, cols, host);
    }
}
