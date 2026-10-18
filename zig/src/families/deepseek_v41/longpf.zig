//! Long prompts on the prefill, up to the 1M limit: stream top-k selection and split-KV attention in row blocks.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const run = @import("run.zig");

/// stream_topk.SPLIT: keys a program scans (positions mode)
pub const split_keys: i64 = 16384;
/// stream_topk.BUDGET: bytes of selection scratch a launch (ENGINE-PLAN 5), not the index budget
pub const budget: i64 = 256 << 20;
/// the runtime's own role for a row block's first position (Runner.external)
pub const pos_role = "s.longpf.pos";

/// stream_topk.splits: programs a row over `keys` positions.
pub fn splits(keys: i64) i64 {
    return @max(1, @divFloor(keys + split_keys - 1, split_keys));
}

/// stream_topk.plan: rows a launch so its scratch (rows x splits x 2 K int64) fits the budget.
pub fn plan(rows: i64, keys: i64, k: i64) i64 {
    const per = splits(keys) * 2 * k * 8;
    return @max(1, @min(rows, @divFloor(budget, per)));
}

/// TF_DSV41_STREAM_LAZY=1: the stream top-k's scratch held only while prompts that name it prefill (256 MiB at 1M).
pub const lazy_role = "L.ix.stream";

pub fn lazyEnv() bool {
    const v = std.c.getenv("TF_DSV41_STREAM_LAZY") orelse return false;
    return std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "1");
}

/// The on-demand scratch: the plan's size for it, and its buffer while a prompt holds it.
pub const Lazy = struct { bytes: u64, buf: ?cuda.DeviceBuffer = null, allocs: u64 = 0 };

/// Whether a program names `role` (the stream select's scratch, or any other).
pub fn names(cs: []const calls.Call, role: []const u8) bool {
    for (cs) |c| for (c.args) |x| if (namesArg(x.arg, role)) return true;
    return false;
}

fn namesArg(x: calls.Arg, role: []const u8) bool {
    return switch (x) {
        .list => |items| for (items) |y| {
            if (namesArg(y, role)) break true;
        } else false,
        .t, .opaque_table => |t| t.role == .buf and std.mem.eql(u8, t.role.buf, role),
        else => false,
    };
}

/// The runtime's buffers (forward_prefill.State.long).
pub const State = struct {
    /// TF_DSV41_STREAM_LAZY: the stream scratch on demand (null: the plan holds it)
    lazy: ?Lazy = null,
    /// uint32 0, 2, 4, ...: the merge's gather rows (a split's first K slots are every other K-slot row)
    even: ?cuda.DeviceBuffer = null,
    even_host: std.ArrayList(u32) = .empty,
    /// int32 [1]: a row block's first position (the attention's POS)
    pos: ?cuda.DeviceBuffer = null,

    pub fn deinit(s: *State, gpa: std.mem.Allocator) void {
        if (s.lazy) |*z| if (z.buf) |*b| b.free();
        if (s.even) |*b| b.free();
        if (s.pos) |*b| b.free();
        s.even_host.deinit(gpa);
    }

    /// The gather rows for `n` (row, split) pairs, grown on the compute stream (the host copy stays alive).
    fn evenRows(s: *State, gpa: std.mem.Allocator, r: *run.Runner, n: usize) !u64 {
        if (s.even_host.items.len < n) {
            const want = std.math.ceilPowerOfTwo(usize, n) catch n;
            const old = s.even_host.items.len;
            try s.even_host.resize(gpa, want);
            for (s.even_host.items[old..], old..) |*v, i| v.* = @intCast(2 * i);
            if (s.even) |*b| b.free();
            s.even = null;
        }
        if (s.even == null) {
            s.even = try cuda.DeviceBuffer.alloc(r.d, 4 * s.even_host.items.len);
            try cuda.copy_source.enqueue(r.d, s.even.?.ptr, std.mem.sliceAsBytes(s.even_host.items), r.stream);
        }
        return s.even.?.ptr;
    }
};

fn addr(r: *run.Runner, x: calls.Arg) !u64 {
    return r.tensorAddr(x.t);
}

/// glue.stream_merge(buf int64 [R, ns, 2 K], out int32 [R, K], K, keys int64 [R, ns K]): stream_topk.merge.
pub fn merge(s: *State, gpa: std.mem.Allocator, r: *run.Runner, c: *const calls.Call) !void {
    const x = c.args;
    if (x.len != 4) return error.BadGlue;
    const buf = x[0].arg.t;
    const R: usize = @intCast(buf.shape[0]);
    const ns: usize = @intCast(buf.shape[1]);
    const cap: usize = @intCast(buf.shape[2]);
    const k: usize = @intCast(x[2].arg.i);
    if (cap != 2 * k or x[1].arg.t.shape[1] != k) return error.BadGlue;
    const ops = r.kernels.others(r.stream);
    if (ns == 1) return ops.topPositions(try addr(r, x[0].arg), cap, R, k, k, try addr(r, x[1].arg));
    const keys = try addr(r, x[3].arg);
    try ops.kvGather(try addr(r, x[0].arg), @intCast(8 * k), try s.evenRows(gpa, r, R * ns), @intCast(R * ns), keys);
    return ops.topPositions(keys, ns * k, R, ns * k, k, try addr(r, x[1].arg));
}

fn named(c: *const calls.Call, name: []const u8) ?calls.Tensor {
    for (c.args) |x| if (std.mem.eql(u8, x.name, name)) return switch (x.arg) {
        .t => |t| t,
        else => null,
    };
    return null;
}

/// A `_fused` attention reading split KV's exchanged rows.
pub fn readsExchange(c: *const calls.Call) bool {
    if (!c.triton or !std.mem.eql(u8, c.name, "_fused")) return false;
    const cv = named(c, "CV") orelse return false;
    return cv.role == .buf and std.mem.eql(u8, cv.role.buf, "w.kx.recv");
}

/// `c` (a `_fused` launch) over its rows [a, b): row-indexed tensors moved by a, POS its own, grid rows b - a.
pub fn blockCall(al: std.mem.Allocator, c: *const calls.Call, a: i64, b: i64) !calls.Call {
    const args = try al.dupe(calls.Named, c.args);
    for (args) |*x| {
        const t = switch (x.arg) {
            .t => |t| t,
            else => continue,
        };
        const nm = x.name;
        const rowed = std.mem.eql(u8, nm, "Q") or std.mem.eql(u8, nm, "TOK") or std.mem.eql(u8, nm, "CNT") or std.mem.eql(u8, nm, "LO");
        if (rowed) {
            const shape = try al.dupe(i64, t.shape);
            shape[0] = b - a;
            x.arg = .{ .t = .{ .role = t.role, .dt = t.dt, .shape = shape, .stride = t.stride, .offset = t.offset + a * t.stride[0] * dtSize(t.dt) } };
        } else if (std.mem.eql(u8, nm, "OUT")) {
            // group-major [groups, OR, HG x 512]: row r of every group at (g OR + r) rows; the base moves by a rows
            x.arg = .{ .t = .{ .role = t.role, .dt = t.dt, .shape = t.shape, .stride = t.stride, .offset = t.offset + a * t.stride[1] * dtSize(t.dt) } };
        } else if (std.mem.eql(u8, nm, "POS")) {
            x.arg = .{ .t = .{ .role = .{ .buf = pos_role }, .dt = .i32, .shape = &.{1}, .stride = &.{1} } };
        }
    }
    var out = c.*;
    out.args = args;
    out.grid[0] = b - a;
    return out;
}

fn dtSize(dt: calls.Dt) i64 {
    return switch (dt) {
        .bool, .i8, .u8 => 1,
        .bf16, .f16, .i16 => 2,
        .f32, .i32 => 4,
        .f64, .i64 => 8,
    };
}

/// A prefill segment's calls, as Runner.window, but a union kept in row blocks attends block by block (`k`: ?*Kv).
pub fn window(s: *State, gpa: std.mem.Allocator, r: *run.Runner, k: anytype, start: u64, cs: []const calls.Call) !void {
    try r.guard(cs);
    if (k) |kx| if (kx.blocks.items.len > 0) for (cs) |*c| if (readsExchange(c)) try guardBlocks(gpa, r, kx, c);
    if (s.lazy) |*z| if (z.buf == null and names(cs, lazy_role)) try acquire(z, r);
    const p = r.prof;
    if (p) |x| x.beginPrefill();
    for (cs) |*c| {
        if (p) |x| try x.before(r.stream, c);
        if (k) |kx| if (kx.blocks.items.len > 0 and readsExchange(c)) {
            try attendBlocks(s, gpa, r, kx, start, c);
            if (p) |x| try x.after(r.stream);
            continue;
        };
        try r.issue(c);
        if (p) |x| try x.after(r.stream);
    }
    if (p) |x| try x.end(r.stream);
}

/// The scratch for a prompt that streams, zeroed as the plan's arena is (every rank in the same order).
fn acquire(z: *Lazy, r: *run.Runner) !void {
    const b = try cuda.DeviceBuffer.alloc(r.d, @max(z.bytes, 256));
    errdefer {
        var x = b;
        x.free();
    }
    try b.fill8(0, r.stream.handle);
    try r.external(lazy_role, b.ptr);
    z.buf = b;
    z.allocs += 1;
    if (z.allocs <= 4 or z.allocs % 64 == 0) std.log.scoped(.dsv41).info("stream top-k scratch: {d} MiB allocated (prompt {d}), held while its prompts prefill (TF_DSV41_STREAM_LAZY)", .{ z.bytes >> 20, z.allocs });
}

/// Its prompts prefilled: the scratch freed once the stream is past its last use.
pub fn release(s: *State, r: *run.Runner) !void {
    const z = &(s.lazy orelse return);
    var b = z.buf orelse return;
    try r.stream.synchronize();
    r.forget(lazy_role);
    b.free();
    z.buf = null;
}

/// The calls attendBlocks issues for `c`, one a block, probed before the window's first launch (Runner.guard).
fn guardBlocks(gpa: std.mem.Allocator, r: *run.Runner, k: anytype, c: *const calls.Call) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const subs = try arena.allocator().alloc(calls.Call, k.blocks.items.len);
    for (k.blocks.items, subs) |b, *sub| sub.* = try blockCall(arena.allocator(), c, b.a, b.b);
    try r.guard(subs);
}

fn attendBlocks(s: *State, gpa: std.mem.Allocator, r: *run.Runner, k: anytype, start: u64, c: *const calls.Call) !void {
    if (s.pos == null) {
        s.pos = try cuda.DeviceBuffer.alloc(r.d, 256);
        try r.external(pos_role, s.pos.?.ptr);
    }
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    for (k.blocks.items, 0..) |b, j| {
        try k.fetchBlock(r, j);
        const p: i32 = @intCast(start + b.a);
        try r.d.check(r.d.api.cuMemsetD32Async(s.pos.?.ptr, @bitCast(p), 1, r.stream.handle), "cuMemsetD32Async");
        const sub = try blockCall(arena.allocator(), c, b.a, b.b);
        try r.issue(&sub);
    }
}
