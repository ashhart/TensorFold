//! Flash Next's learned prompt states (--learn): a kept state's file on disk, and the probe keying this build's bits.
const std = @import("std");
const ks = @import("kernel_sources");
const snapshot = @import("snapshot.zig");
const roles_gen = @import("roles_gen.zig");
const engine = @import("engine.zig");
const Engine = engine.Engine;
const State = snapshot.State;

const FILE_MAGIC: u32 = 0x46584e53; // "FXNS"

/// A learned state's file header: its layout and position, its bytes, and the n-gram history at it.
const Head = extern struct { magic: u32, layout: u32, at: u64, bytes: u64, hist: [2]i64 };

/// A learned state's file: <dir>/<key>.bin.
pub fn path(buf: []u8, dir: []const u8, key: u64) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.bin", .{ dir, key }, 0);
}

/// `st` to `file` (its header, then its bytes), through a temporary file renamed into place.
pub fn writeFile(st: *const State, file: [:0]const u8) !void {
    var tmp_buf: [1100]u8 = undefined;
    const tmp = try std.fmt.bufPrintSentinel(&tmp_buf, "{s}.part", .{file}, 0);
    const fd = std.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.SnapshotWrite;
    errdefer _ = std.c.unlink(tmp);
    {
        defer _ = std.c.close(fd);
        const head: Head = .{ .magic = FILE_MAGIC, .layout = @backingInt(st.layout), .at = st.at, .bytes = st.bytes, .hist = st.hist };
        try put(fd, std.mem.asBytes(&head));
        try put(fd, st.buf.contents()[0..st.bytes]);
        if (std.c.fsync(fd) != 0) return error.SnapshotWrite;
    }
    if (std.c.rename(tmp, file) != 0) return error.SnapshotWrite;
}

/// `file`'s state after `at` tokens, in a buffer from the engine's pool; another position, layout or size is refused.
pub fn readFile(e: *Engine, gpa: std.mem.Allocator, at: usize, file: [:0]const u8) !*State {
    const fd = std.c.open(file, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.SnapshotRead;
    defer _ = std.c.close(fd);
    var head: Head = undefined;
    try get(fd, std.mem.asBytes(&head), 0);
    const layout: u32 = @backingInt(snapshot.layoutOf(e));
    if (head.magic != FILE_MAGIC or head.at != at or head.layout != layout or head.bytes != snapshot.bytes(at)) return error.SnapshotRead;
    const st = try snapshot.blank(e, gpa, at);
    errdefer snapshot.drop(gpa, st);
    try get(fd, st.buf.contents()[0..st.bytes], @sizeOf(Head));
    st.hist = head.hist;
    return st;
}

fn put(fd: c_int, b: []const u8) !void {
    var done: usize = 0;
    while (done < b.len) {
        const n = std.c.write(fd, b.ptr + done, @min(b.len - done, 1 << 30));
        if (n <= 0) return error.SnapshotWrite;
        done += @intCast(n);
    }
}

fn get(fd: c_int, b: []u8, at: u64) !void {
    var done: usize = 0;
    while (done < b.len) {
        const n = std.c.pread(fd, b.ptr + done, @min(b.len - done, 1 << 30), @intCast(at + done));
        if (n <= 0) return error.SnapshotRead;
        done += @intCast(n);
    }
}

/// The checked-in kernel sources and the roles that run them, hashed: a change compiles other kernels.
pub fn sourceHash() u64 {
    var h = std.hash.Wyhash.init(0x6b73);
    for (ks.flashnext_gen.sources) |s| {
        h.update(s.name);
        h.update(s.text);
    }
    for (&roles_gen.entries) |*r| {
        h.update(r.site);
        h.update(r.file);
        h.update(r.function);
    }
    return h.final();
}

/// This build's prompt-pass bits: fixed tokens over a full call, a middle and a short one, two marks' states hashed.
pub fn probe(e: *Engine, gpa: std.mem.Allocator) !u64 {
    var toks: [4096 + 40 + 7]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(1000 + i * 7919 % 50000);
    const Probe = struct {
        e: *Engine,
        gpa: std.mem.Allocator,
        h: std.hash.Wyhash = .init(0x6678),
        marks: usize = 0,
        err: ?anyerror = null,
        fn prefilled(_: *anyopaque) void {}
        fn tokens(_: *anyopaque, _: []const u32) bool {
            return true; // the first token ends it
        }
        fn cancelled(_: *anyopaque) bool {
            return false;
        }
        fn marked(ctx: *anyopaque, at: usize) void {
            const p: *@This() = @ptrCast(@alignCast(ctx));
            const st = snapshot.save(p.e, p.gpa, at) catch |err| {
                p.err = err;
                return;
            };
            defer snapshot.drop(p.gpa, st);
            p.h.update(st.buf.contents()[0..st.bytes]);
            p.h.update(std.mem.asBytes(&st.hist));
            p.marks += 1;
        }
    };
    var p: Probe = .{ .e = e, .gpa = gpa };
    const marks = [_]u32{ 4096, toks.len - 8 };
    const out: engine.Out = .{ .ctx = &p, .prefilled = Probe.prefilled, .tokens = Probe.tokens, .cancelled = Probe.cancelled, .marked = Probe.marked };
    _ = try e.generateFrom(&toks, 0, &marks, 1, &.{}, 0, out);
    if (p.err) |err| return err;
    if (p.marks != marks.len) return error.ProbeUnmarked;
    return p.h.final();
}
