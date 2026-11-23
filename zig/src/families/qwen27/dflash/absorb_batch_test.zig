//! Host arithmetic over real bytes: a stacked absorb leaves every stream's caches byte-equal to its own absorb.
const std = @import("std");
const ck = @import("../checkpoint.zig");
const op = @import("operators.zig");
const execution = @import("execution.zig");
const fixtures = @import("fixtures.zig");
const absorb_batch = @import("absorb_batch.zig");
const c = fixtures.toy;
const max_streams = 4;
const layers = 5;
const Row = struct { width: u32, data: []u16 };

/// A per-row hash stands in for each kernel; non-finite bf16 inputs pass through so a planted value reaches the caches.
const Cpu = struct {
    arena: std.heap.ArenaAllocator,
    views: std.ArrayList(Row) = .empty,
    keys: [max_streams][layers][]u16 = undefined,
    values: [max_streams][layers][]u16 = undefined,
    fusion_reads: u32 = 0,
    leak: bool = false, // the planted control: a row's result depends on how many rows share its pass
    fn init(gpa: std.mem.Allocator) !*Cpu {
        const f = try gpa.create(Cpu);
        f.* = .{ .arena = std.heap.ArenaAllocator.init(gpa) };
        const a = f.arena.allocator();
        for (&f.keys, &f.values) |*ks, *vs| for (ks, vs) |*k, *v| {
            k.* = try a.alloc(u16, @as(usize, c.window) * c.kvWidth());
            v.* = try a.alloc(u16, @as(usize, c.window) * c.kvWidth());
            @memset(k.*, 0);
            @memset(v.*, 0);
        };
        return f;
    }
    fn deinit(f: *Cpu, gpa: std.mem.Allocator) void {
        f.arena.deinit();
        gpa.destroy(f);
    }
    fn self(ptr: *anyopaque) *Cpu {
        return @ptrCast(@alignCast(ptr));
    }
    fn add(f: *Cpu, rows: u32, width: u32) !op.View {
        const data = try f.arena.allocator().alloc(u16, @as(usize, rows) * width);
        try f.views.append(f.arena.allocator(), .{ .width = width, .data = data });
        return .{ .handle = f.views.items.len, .rows = rows, .width = width, .stride = width, .dtype = .bf16 };
    }
    fn bytes(f: *Cpu, v: op.View) []u16 {
        return f.views.items[v.handle - 1].data;
    }
    fn special(x: u16) bool {
        return x & 0x7f80 == 0x7f80;
    }
    fn mix(seed: u64, x: u16, column: usize) u16 {
        if (special(x)) return x;
        const h = std.hash.Wyhash.hash(seed +% column, std.mem.asBytes(&x));
        const out: u16 = @truncate(h);
        return if (special(out)) out & 0xbfff else out;
    }
    fn map(f: *Cpu, x: op.View, width: u32, seed: u64) !op.View {
        const y = try f.add(x.rows, width);
        const from = f.bytes(x);
        const to = f.bytes(y);
        const salt = if (f.leak) seed +% x.rows else seed;
        for (0..x.rows) |r| for (0..width) |col| {
            to[r * width + col] = mix(salt, from[r * x.width + col % x.width], col);
        };
        return y;
    }
    fn beginFn(_: *anyopaque) !u64 {
        return 1;
    }
    fn releaseFn(_: *anyopaque, _: u64) void {}
    fn projectFn(ptr: *anyopaque, _: u64, x: op.View, t: ck.Tensor, _: op.Mode) !op.View {
        const f = self(ptr);
        if (t.shape[1] == c.tapWidth()) f.fusion_reads += 1;
        return f.map(x, @intCast(t.shape[0]), @intFromPtr(t.bytes.ptr));
    }
    fn normFn(ptr: *anyopaque, _: u64, x: op.View, n: op.Norm) !op.View {
        return self(ptr).map(x, x.width, @intFromPtr(n.weight.bytes.ptr));
    }
    fn ropeFn(ptr: *anyopaque, _: u64, x: op.View, a: op.Rotary) !op.View {
        const f = self(ptr);
        const y = try f.add(x.rows, x.width);
        for (0..x.rows) |r| for (0..x.width) |col| {
            f.bytes(y)[r * x.width + col] = mix(a.positions[r], f.bytes(x)[r * x.width + col], col);
        };
        return y;
    }
    fn maskFn(ptr: *anyopaque, _: u64, x: op.View, first: u32, rows: u32) !op.View {
        const f = self(ptr);
        const y = try f.add(rows, x.width);
        @memcpy(f.bytes(y), f.bytes(x)[@as(usize, first) * x.width ..][0 .. @as(usize, rows) * x.width]);
        return y;
    }
    fn appendFn(ptr: *anyopaque, _: u64, stream: u32, layer: u32, k: op.View, v: op.View, positions: []const u64) !void {
        const f = self(ptr);
        for (positions, 0..) |p, r| {
            const slot: usize = @intCast(p % c.window);
            @memcpy(f.keys[stream][layer][slot * k.width ..][0..k.width], f.bytes(k)[r * k.width ..][0..k.width]);
            @memcpy(f.values[stream][layer][slot * v.width ..][0..v.width], f.bytes(v)[r * v.width ..][0..v.width]);
        }
    }
    fn embedFn(_: *anyopaque, _: u64, _: ck.Linear, _: []const u32) !op.View {
        return error.NotInAbsorb;
    }
    fn convFn(_: *anyopaque, _: u64, _: op.View, _: op.Conv) !op.View {
        return error.NotInAbsorb;
    }
    fn attentionFn(_: *anyopaque, _: u64, _: op.Attention) !op.View {
        return error.NotInAbsorb;
    }
    fn swigluFn(_: *anyopaque, _: u64, _: op.View, _: op.View) !op.View {
        return error.NotInAbsorb;
    }
    fn headFn(_: *anyopaque, _: u64, _: op.View) !op.Head {
        return error.NotInAbsorb;
    }
    fn latticeFn(_: *anyopaque, _: u64, _: op.Head, _: op.View, _: @import("config.zig").Config) !op.Lattice {
        return error.NotInAbsorb;
    }
    fn latticesFn(_: *anyopaque, _: u64, _: op.Head, _: op.View, _: @import("config.zig").Config, _: u32, _: []op.Lattice) !void {
        return error.NotInAbsorb;
    }
    fn contextBeginFn(_: *anyopaque, _: op.Context) !u64 {
        return error.NotInAbsorb;
    }
    fn contextAppendFn(_: *anyopaque, _: u64, _: u64, _: u32, _: op.View, _: op.View, _: []const u64) !void {
        return error.NotInAbsorb;
    }
    fn contextCommitFn(_: *anyopaque, _: u64, _: u64, _: op.Context) !void {
        return error.NotInAbsorb;
    }
    fn contextAbortFn(_: *anyopaque, _: u64) void {}
    fn ops(f: *Cpu) op.Ops {
        return .{ .ptr = f, .vtable = &.{ .begin = beginFn, .release = releaseFn, .project = projectFn, .norm = normFn, .embed = embedFn, .conv = convFn, .rope = ropeFn, .attention = attentionFn, .swiglu = swigluFn, .mask_rows = maskFn, .head = headFn, .lattice = latticeFn, .lattices = latticesFn, .context_begin = contextBeginFn, .context_append = contextAppendFn, .context_commit = contextCommitFn, .context_stage = contextCommitFn, .context_abort = contextAbortFn, .stream_append = appendFn } };
    }
};

const Input = struct { rows: []const u32, starts: []const u64 };

/// A tap value for a stream's row and column, with NaN, +Inf and -Inf planted in every stream.
fn tap(stream: usize, row: usize, col: usize) u16 {
    if (col == stream and row == 0) return 0x7fc0;
    if (col == stream + 1 and row == 0) return 0x7f80;
    if (col == stream + 2 and row == 0) return 0xff80;
    return @truncate((stream *% 257 +% row *% 31 +% col *% 7 +% 0x3f00) & 0x3fff);
}

fn run(gpa: std.mem.Allocator, g: anytype, input: Input, stacked: bool, leak: bool) !*Cpu {
    const f = try Cpu.init(gpa);
    errdefer f.deinit(gpa);
    f.leak = leak;
    const preparation = op.Preparation{ .mode = .bf16_reference };
    const width = c.tapWidth();
    if (stacked) {
        var spans: [max_streams]absorb_batch.Span = undefined;
        // production keeps a lone stream's own pass; the cell still stacks its one span to compare the bytes
        const lone = [_]absorb_batch.Span{.{ .stream = 0, .first = 0, .rows = input.rows[0] }};
        const plan = if (input.rows.len == 1) lone[0..] else absorb_batch.plan(input.rows, &spans) orelse return error.NoPlan;
        const last = plan[plan.len - 1];
        const total = last.first + last.rows;
        const taps = try f.add(total, width);
        var positions: [absorb_batch.max_rows]u64 = undefined;
        for (plan) |s| for (0..s.rows) |r| {
            positions[s.first + r] = input.starts[s.stream] + r;
            for (0..width) |col| f.bytes(taps)[(s.first + r) * width + col] = tap(s.stream, r, col);
        };
        try execution.encodeStacked(f.ops(), g, preparation, 1, .{ .taps = taps, .positions = positions[0..total], .spans = plan });
        return f;
    }
    var absorbs: [max_streams]?execution.Absorb = @splat(null);
    var positions: [max_streams][absorb_batch.max_rows]u64 = undefined;
    for (input.rows, 0..) |rows, k| {
        if (rows == 0) continue;
        const taps = try f.add(rows, width);
        for (0..rows) |r| {
            positions[k][r] = input.starts[k] + r;
            for (0..width) |col| f.bytes(taps)[r * width + col] = tap(k, r, col);
        }
        absorbs[k] = .{ .taps = taps, .positions = positions[k][0..rows] };
    }
    try execution.encodeAbsorbs(f.ops(), g, preparation, 1, absorbs[0..input.rows.len]);
    return f;
}

fn same(x: *const Cpu, y: *const Cpu) bool {
    for (x.keys, y.keys, x.values, y.values) |xk, yk, xv, yv| for (xk, yk, xv, yv) |a, b, d, e| {
        if (!std.mem.eql(u16, a, b) or !std.mem.eql(u16, d, e)) return false;
    };
    return true;
}

fn holds(x: *const Cpu, value: u16) bool {
    for (x.keys) |stream| for (stream) |layer| if (std.mem.indexOfScalar(u16, layer, value) != null) return true;
    return false;
}

const cells = [_]Input{
    .{ .rows = &.{6}, .starts = &.{13} },
    .{ .rows = &.{ 3, 1 }, .starts = &.{ 5, 14 } },
    .{ .rows = &.{ 1, 7 }, .starts = &.{ 0, 9 } },
    .{ .rows = &.{ 2, 5, 1, 4 }, .starts = &.{ 3, 12, 0, 7 } },
    .{ .rows = &.{ 6, 0, 3, 1 }, .starts = &.{ 10, 0, 15, 2 } },
};

test "qwen27 window: stacked absorb leaves each stream's caches byte-equal at 1, 2 and 4 streams with NaN and Inf planted" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.init(a);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    for (cells) |cell| {
        const apart = try run(a, &g, cell, false, false);
        defer apart.deinit(a);
        const together = try run(a, &g, cell, true, false);
        defer together.deinit(a);
        try std.testing.expect(same(apart, together));
        try std.testing.expect(holds(together, 0x7fc0) and holds(together, 0x7f80) and holds(together, 0xff80));
        try std.testing.expectEqual(@as(u32, 1), together.fusion_reads);
        var absorbing: u32 = 0;
        for (cell.rows) |r| absorbing += @intFromBool(r > 0);
        try std.testing.expectEqual(absorbing, apart.fusion_reads);
    }
}

test "qwen27 window: one absorbing stream keeps its own pass" {
    var spans: [max_streams]absorb_batch.Span = undefined;
    try std.testing.expectEqual(@as(?[]const absorb_batch.Span, null), absorb_batch.plan(&.{ 0, 4, 0, 0 }, &spans));
    try std.testing.expectEqual(@as(?[]const absorb_batch.Span, null), absorb_batch.plan(&.{4}, &spans));
}

test "qwen27 window: the planted control, a row that depends on its pass, fails the byte comparison" {
    const a = std.testing.allocator;
    var store = try fixtures.Store.init(a);
    defer store.deinit();
    var g = try store.graph(a);
    defer g.deinit();
    const apart = try run(a, &g, cells[3], false, true);
    defer apart.deinit(a);
    const together = try run(a, &g, cells[3], true, true);
    defer together.deinit(a);
    try std.testing.expect(!same(apart, together));
}
