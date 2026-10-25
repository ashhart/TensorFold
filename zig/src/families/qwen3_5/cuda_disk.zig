//! Qwen3.5 CUDA kept states on disk (--learn-conversations): the host copy's image, a blank copy, and the probe.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const api = @import("engine_api");
const reuse = @import("cuda_reuse.zig");
const engine = @import("cuda_engine.zig");
const rows = api.prompt_cache.rows;
const Target = reuse.Target;
const Copy = reuse.Copy;
const Allocator = std.mem.Allocator;

/// The store's view of a kept state: its words, then the host copy as cuda_reuse.zig lays it out.
pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter, .whole = whole };

const words = 16; // the position, the rotary shift and the cache format
const Words = extern struct { at: u32, delta: i32, format: u32, pad: u32 = 0 };

fn target(ptr: *anyopaque) *Target {
    const t: *Target = @ptrCast(@alignCast(ptr));
    if (t.ctx) |c| c.makeCurrent() catch {};
    return t;
}

/// The words, the recurrences and tails whole, then each (layer, key/value, head) piece's rows, as the copy holds them.
fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    try partsOf(t.e, at, out, a);
}

pub fn partsOf(e: *const engine.Engine, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) !void {
    const l = reuse.layoutOf(e, at);
    const one = reuse.layoutOf(e, 1); // a piece's row: its bytes at one position
    try out.append(a, .{ .bytes = words });
    for (0..l.count()) |i| try out.append(a, .{ .bytes = l.piece(i).len, .row = if (i == 0) 0 else @intCast(one.piece(i).len) });
}

fn wordsOf(c: *const Copy) Words {
    return .{ .at = @intCast(c.at), .delta = c.delta, .format = @backingInt(c.format) };
}

fn gather(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    const c: *const Copy = @ptrCast(@alignCast(saved));
    const w = wordsOf(c);
    var o: usize = 0;
    for (ranges) |g| {
        var off: usize = @intCast(g.off);
        var len: usize = @intCast(g.len);
        if (off + len > words + c.len) return error.BadKeptRange;
        if (off < words) {
            const n = @min(len, words - off);
            @memcpy(out[o..][0..n], std.mem.asBytes(&w)[off..][0..n]);
            o += n;
            off += n;
            len -= n;
        }
        if (len == 0) continue;
        @memcpy(out[o..][0..len], c.host.bytes[off - words ..][0..len]);
        o += len;
    }
}

/// A host copy of `at` positions in the engine's cache format, filled by `scatter`, restored as any other.
fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
    const t = target(ptr);
    return @ptrCast(try blankCopy(t.e, at));
}

pub fn blankCopy(e: *engine.Engine, at: u32) !*Copy {
    const n = reuse.layoutOf(e, at).total();
    const copy = try e.gpa.create(Copy);
    errdefer e.gpa.destroy(copy);
    const images = try e.gpa.alloc(reuse.Seen, 0);
    errdefer e.gpa.free(images);
    copy.* = .{ .host = try cuda.HostBuffer.alloc(e.d, n), .len = n, .at = at, .delta = 0, .format = reuse.formatOf(e), .images = images };
    return copy;
}

fn scatter(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    const c: *Copy = @ptrCast(@alignCast(saved));
    var w = wordsOf(c);
    var o: usize = 0;
    for (ranges) |g| {
        var off: usize = @intCast(g.off);
        var len: usize = @intCast(g.len);
        if (off + len > words + c.len) return error.BadKeptRange;
        if (off < words) {
            const n = @min(len, words - off);
            @memcpy(std.mem.asBytes(&w)[off..][0..n], in[o..][0..n]);
            o += n;
            off += n;
            len -= n;
        }
        if (len == 0) continue;
        @memcpy(c.host.bytes[off - words ..][0..len], in[o..][0..len]);
        o += len;
    }
    // a state read back stands where it was kept, in the format it was kept in: else it restores nowhere
    if (w.at != c.at or w.format != @backingInt(c.format)) return error.KeptWordsDiffer;
    c.delta = w.delta;
}

/// A state whose prompt holds images names them in RAM only: on disk it would lose them, so it stays off disk.
fn whole(_: *anyopaque, saved: api.prompt_cache.Saved) bool {
    const c: *const Copy = @ptrCast(@alignCast(saved));
    return c.images.len == 0;
}

/// The context the disk tier's copies make current (they may run off the engine's thread).
pub fn bind(t: *Target, ctx: *const cuda.Context) void {
    t.ctx = ctx;
}

/// A fixed prompt's state at the first chunk cut, hashed (a kernel or driver change moves it), and its prompt rate.
pub fn probe(a: Allocator, io: std.Io, backend: lanes.backend.Backend, t: *Target) !struct { hash: u64, rate: f64 } {
    var toks: [engine.max_rows + 40 + 7]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = @intCast(1000 + i * 7919 % 50000);
    const Hook = struct {
        t: *Target,
        saved: ?*anyopaque = null,
        failed: ?anyerror = null,
        fn at(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
            const p: *@This() = @ptrCast(@alignCast(ptr));
            p.saved = reuse.save(p.t, s, mark) catch |e| {
                p.failed = e;
                return;
            };
        }
    };
    var hook: Hook = .{ .t = t };
    const marks = [_]u32{engine.max_rows};
    var s = try lanes.Stream.init(a, .{ .id = "probe", .prompt = &toks, .max_new = 1, .reuse = .{ .marks = &marks, .hook = .{ .ptr = &hook, .at = Hook.at } } });
    defer s.deinit(a);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    backend.vtable.prefill(backend.ptr, &s) catch |e| {
        backend.vtable.release(backend.ptr, &s);
        return e;
    };
    try t.e.stream.synchronize();
    const ns = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    backend.vtable.release(backend.ptr, &s);
    const saved = hook.saved orelse return hook.failed orelse error.ProbeUnmarked;
    defer reuse.drop(t, saved);
    const c: *const Copy = @ptrCast(@alignCast(saved));
    var h = std.hash.Wyhash.init(0x7135);
    h.update(std.mem.asBytes(&wordsOf(c)));
    h.update(c.host.bytes[0..c.len]);
    return .{ .hash = h.final(), .rate = @as(f64, @floatFromInt(toks.len)) / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9) };
}

/// What else the states depend on: the cache format and whether a drafter set the target's taps.
pub fn facts(t: *const Target) [2]u64 {
    return .{ @backingInt(reuse.formatOf(t.e)), @intFromBool(t.e.n_taps > 0) };
}

test {
    _ = @import("cuda_disk_test.zig");
}
