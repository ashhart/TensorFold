//! Nemotron's kept states on disk on Metal (--learn-conversations): their image, copies out and in of shared buffers.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const nemotron = tf.nemotron;
const rows = api.prompt_cache.rows;
const Snap = nemotron.snapshot.Snap;
const Metal = nemotron.backend.Metal;
const Allocator = std.mem.Allocator;

/// The store's view of a kept state: its buffer as snapshot.zig lays it out (Mamba states, then rows by head).
pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter };

fn metal(ptr: *anyopaque) *Metal {
    return @ptrCast(@alignCast(ptr));
}

fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
    const b = metal(ptr);
    const sh = nemotron.snapshot.diskShape(b.m.config, b.head != null);
    try out.append(a, .{ .bytes = sh.mamba });
    for (0..sh.planes) |_| try out.append(a, .{ .bytes = @as(u64, at) * sh.row, .row = sh.row });
}

/// Ranges of a state's buffer, once the copy into it has landed (the queue drained).
fn gather(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    try metal(ptr).drain();
    const snap: *const Snap = @ptrCast(@alignCast(saved));
    var o: usize = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        @memcpy(out[o..][0..len], snap.buf.contents()[@intCast(g.off)..][0..len]);
        o += len;
    }
}

fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
    const b = metal(ptr);
    const head = b.head != null;
    const n = nemotron.snapshot.bytes(b.m.config, at, head);
    const snap = try b.gpa.create(Snap);
    errdefer b.gpa.destroy(snap);
    snap.* = .{ .at = at, .head = head, .buf = try b.m.device.buffer(n, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked), .bytes = n };
    return snap;
}

fn scatter(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    const snap: *Snap = @ptrCast(@alignCast(saved));
    var o: usize = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        @memcpy(snap.buf.contents()[@intCast(g.off)..][0..len], in[o..][0..len]);
        o += len;
    }
}
