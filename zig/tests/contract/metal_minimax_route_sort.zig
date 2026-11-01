//! MiniMax-M3.1-Flash's mm_sort, mm_rows_by_order and mm_index_select under the contract, each by its own rule.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const r = @import("metal_minimax_route.zig");

const c = contract.spec;
const g = contract.gates;
const Args = r.Args;

// --- mm_sort: a prompt chunk's picks by expert, each expert's in slot order ---------------------------------------

pub const Sort = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    slots: usize = 0,
    picks: []i32 = &.{},
    in: rig_.Ref = undefined,
    offsets: rig_.Ref = undefined,
    order: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    /// Rows: slots (4 a token); n: experts.
    pub fn spec() c.Spec {
        return .{
            .name = "metal minimax_m3 mm_sort experts 128",
            .accum = .none,
            .depth = r.noSums,
            .out = .raw,
            .nan = .{ .stated = "the host's stable sort: offsets its exclusive prefix counts, order each expert's slots ascending inside its own offsets" },
            .tiles = .{ .{ .size = 1024, .step = 4, .min = 4 * (r.batch_rows + 1) }, .{}, .{} },
            // the prompt path's chunks: 33 rows (past a shared decode forward) to 4,096
            .served = &.{ .{ .rows = r.prompt_rows * r.topk, .n = r.experts, .k = 1 }, .{ .rows = (r.batch_rows + 1) * r.topk, .n = r.experts, .k = 1 } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Sort {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try r.gluePipe(device, "mm_sort") };
    }
    pub fn deinit(t: *Sort) void {
        t.a.free(t.picks);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Sort) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Sort);

    pub fn prepare(t: *Sort, shape: c.Shape, _: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows % r.topk != 0 or shape.n != r.experts) return false;
        t.rig.reset();
        t.a.free(t.picks);
        t.slots = shape.rows;
        t.picks = try t.a.alloc(i32, shape.rows);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // each token's 4 distinct experts; a few hot experts take most tokens, so empty and crowded experts both occur
        for (0..shape.rows / r.topk) |tok| {
            var used: [r.experts]bool = @splat(false);
            for (t.picks[tok * r.topk ..][0..r.topk]) |*p| {
                var e: u32 = if (rng.boolean()) rng.uintLessThan(u32, 12) else rng.uintLessThan(u32, r.experts);
                while (used[e]) e = (e + 1) % r.experts;
                used[e] = true;
                p.* = @intCast(e);
            }
        }
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.picks));
        t.offsets = try t.rig.output(r.experts * 4);
        t.order = try t.rig.output(shape.rows * 4);
        return true;
    }

    pub fn run(t: *Sort) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.in.buf, t.in.off, 0);
        p.e.setValue(Args{ .rows = @intCast(t.slots), .width = r.experts }, 1);
        p.e.setBuffer(t.offsets.buf, t.offsets.off, 2);
        p.e.setBuffer(t.order.buf, t.order.off, 3);
        p.e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1024, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Sort) usize {
        return (r.experts + t.slots) * 4;
    }
    pub fn reference(_: *Sort, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    pub fn check(t: *Sort, _: g.Gate) !?[]const u8 {
        const offs = std.mem.bytesAsSlice(i32, t.rig.bytes(t.offsets, r.experts * 4));
        const order = std.mem.bytesAsSlice(i32, t.rig.bytes(t.order, t.slots * 4));
        var want: [r.experts + 1]i32 = @splat(0);
        for (t.picks) |e| want[@intCast(e + 1)] += 1;
        for (1..r.experts + 1) |e| want[e] += want[e - 1];
        for (0..r.experts) |e| if (offs[e] != want[e]) return try std.fmt.bufPrint(&t.msg, "expert {d}'s offset is {d} where the host counts {d}", .{ e, offs[e], want[e] });
        var i: usize = 0;
        for (0..r.experts) |e| for (t.picks, 0..) |p, slot| if (p == e) {
            if (order[i] != slot) return try std.fmt.bufPrint(&t.msg, "order[{d}] = {d} where the host's stable sort gives slot {d} (expert {d})", .{ i, order[i], slot, e });
            i += 1;
        };
        return null;
    }
};

// --- mm_rows_by_order: each sorted slot's token row, bit for bit --------------------------------------------------

pub const RowsByOrder = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    order: []i32 = &.{},
    ins: [2]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    /// Rows: sorted slots (4 a token); n: the hidden width.
    pub fn spec() c.Spec {
        return .{
            .name = "metal minimax_m3 mm_rows_by_order top 4",
            .accum = .none,
            .depth = r.noSums,
            .out = .raw,
            .nan = .{ .stated = "each sorted slot's row is its token's input row, bit for bit" },
            .tiles = .{ .{ .size = 4, .step = 4, .min = 4 }, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            .served = &.{ .{ .rows = 1024 * r.topk, .n = r.hidden, .k = 1 }, .{ .rows = r.prompt_rows * r.topk, .n = r.hidden, .k = 1 }, .{ .rows = (r.batch_rows + 1) * r.topk, .n = r.hidden, .k = 1 } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !RowsByOrder {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try r.gluePipe(device, "mm_rows_by_order") };
    }
    pub fn deinit(t: *RowsByOrder) void {
        t.a.free(t.x);
        t.a.free(t.order);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *RowsByOrder) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(RowsByOrder);

    pub fn prepare(t: *RowsByOrder, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows % r.topk != 0 or shape.n == 0) return false;
        t.rig.reset();
        t.a.free(t.x);
        t.a.free(t.order);
        t.shape = shape;
        const tokens = shape.rows / r.topk;
        t.x = try t.a.alloc(u16, tokens * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.x) |*v| v.* = rng.int(u16);
        if (fill == .nonfinite) for ([_]f32{ r.nan32, r.inf32, -r.inf32 }, 0..) |v, i| {
            t.x[i] = c.toBf16(v);
        };
        t.order = try t.a.alloc(i32, shape.rows);
        for (t.order, 0..) |*o, i| o.* = @intCast(i);
        rng.shuffle(i32, t.order);
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(t.x));
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.order));
        t.out = try t.rig.output(shape.rows * shape.n * 2);
        return true;
    }

    pub fn run(t: *RowsByOrder) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins[0].buf, t.ins[0].off, 0);
        p.e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
        p.e.setValue(Args{ .rows = @intCast(t.shape.rows), .width = @intCast(t.shape.n), .heads = r.topk }, 2);
        p.e.setBuffer(t.out.buf, t.out.off, 3);
        p.e.dispatchThreads(mtl.Size.of(t.shape.n, t.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *RowsByOrder) usize {
        return t.shape.rows * t.shape.n * 2;
    }
    pub fn reference(_: *RowsByOrder, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    pub fn check(t: *RowsByOrder, _: g.Gate) !?[]const u8 {
        const n = t.shape.n;
        const got = t.rig.bytes(t.out, t.count());
        for (t.order, 0..) |o, i| {
            const tok: usize = @intCast(@divTrunc(o, r.topk));
            if (!std.mem.eql(u8, got[i * n * 2 ..][0 .. n * 2], std.mem.sliceAsBytes(t.x[tok * n ..][0..n]))) return try std.fmt.bufPrint(&t.msg, "sorted slot {d} is not token {d}'s row", .{ i, tok });
        }
        return null;
    }
};

// --- mm_index_select: each (row, key head)'s kept blocks ascending, and the prompt's bitmap -----------------------

/// mm_index_args (glue.metal).
pub const IArgs = extern struct { rows: u32, pos: u32, heads: u32, blocks: u32, first_row: u32 = 0 };

/// A row's position past `pos`: the last row ends 7 keys short of block `blocks`' end (blocks(pos + rows) = blocks).
pub fn startOf(rows: usize, blocks: usize) ?usize {
    const end = blocks * r.block - 7;
    if (end < rows or blocks == 0) return null;
    return end - rows;
}

pub const Select = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    pos: usize = 0,
    scores: []f32 = &.{},
    in: rig_.Ref = undefined,
    sel: rig_.Ref = undefined,
    bits: rig_.Ref = undefined,
    msg: [200]u8 = undefined,

    /// Rows: query rows; n: the index blocks the chunk's keys span; k: bitmap words a (row, head) (0: decode).
    pub fn spec() c.Spec {
        return .{
            .name = "metal minimax_m3 mm_index_select top 16 blocks, key heads 4",
            .accum = .none,
            .depth = r.noSums,
            .out = .raw,
            .nan = .{ .stated = "the host's selection: every block while a row sees at most 16, else its own block and the best 15 by score (ties to the lower block, +Inf first), NaN and -Inf never kept; ascending, -1 past the count; the bitmap marks the same blocks" },
            .tiles = .{ .{ .size = 32 }, .{}, .{} },
            // decode windows at the longest window and just past 16 blocks; prompt chunks with their bitmaps
            .served = &.{ .{ .rows = 16, .n = r.context_blocks, .k = 0 }, .{ .rows = 1, .n = r.topk_blocks + 1, .k = 0 }, .{ .rows = r.prompt_rows, .n = r.context_blocks, .k = r.context_blocks / 32 }, .{ .rows = 64, .n = 5, .k = r.context_blocks / 32 }, .{ .rows = 200, .n = 300, .k = 10 } },
        };
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Select {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try r.gluePipe(device, "mm_index_select") };
    }
    pub fn deinit(t: *Select) void {
        t.a.free(t.scores);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Select) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Select);

    pub fn prepare(t: *Select, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const pos = startOf(shape.rows, shape.n) orelse return false;
        if (shape.rows == 0 or shape.k * 32 < shape.n and shape.k != 0) return false;
        t.rig.reset();
        t.a.free(t.scores);
        t.shape = shape;
        t.pos = pos;
        t.scores = try t.a.alloc(f32, shape.rows * r.kv_heads * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // scores from a small set, so ties are common and the lower block must win them
        for (t.scores) |*v| v.* = @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -64, 64))) / 8;
        if (fill == .nonfinite) {
            const put = [_]f32{ r.nan32, r.inf32, -r.inf32, r.nan32, -r.inf32 };
            for (put, 0..) |v, i| if (i < shape.rows) {
                const row = t.scores[(i * r.kv_heads + i % r.kv_heads) * shape.n ..][0..shape.n];
                row[(i * 37) % shape.n] = v;
                row[0] = v;
            };
        }
        t.in = try t.rig.input(std.mem.sliceAsBytes(t.scores));
        t.sel = try t.rig.output(shape.rows * r.kv_heads * r.topk_blocks * 4);
        t.bits = try t.rig.output(@max(shape.rows * r.kv_heads * shape.k, 1) * 4);
        return true;
    }

    pub fn run(t: *Select) !void {
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.pipe);
        e.setBuffer(t.in.buf, t.in.off, 0);
        e.setValue(IArgs{ .rows = @intCast(t.shape.rows), .pos = @intCast(t.pos), .heads = r.kv_heads, .blocks = @intCast(t.shape.n) }, 1);
        e.setValue(@as(u32, r.topk_blocks), 2);
        e.setBuffer(t.sel.buf, t.sel.off, 3);
        e.setBuffer(t.bits.buf, t.bits.off, 4);
        e.setValue(@as(u32, @intCast(t.shape.k)), 5);
        e.dispatchGroups(mtl.Size.of(r.kv_heads, t.shape.rows, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Select) usize {
        return t.shape.rows * r.kv_heads * (r.topk_blocks + t.shape.k) * 4;
    }
    pub fn reference(_: *Select, _: []f64, _: []f64) !void {
        return error.CheckedByItsOwnRule;
    }

    /// Whether (v, b) comes before (w, d) in the selection's order: score descending, then block ascending.
    fn before(v: f32, b: usize, w: f32, d: usize) bool {
        return v > w or (v == w and b < d);
    }

    /// Row `row`, head `h`'s kept blocks, ascending; the count.
    fn hostList(t: *Select, row: usize, h: usize, out: *[r.topk_blocks]i32) usize {
        const nb = (t.pos + row) / r.block + 1;
        if (nb <= r.topk_blocks) {
            for (0..nb) |b| out[b] = @intCast(b);
            return nb;
        }
        const s = t.scores[(row * r.kv_heads + h) * t.shape.n ..][0..t.shape.n];
        var n: usize = 1;
        out[0] = @intCast(nb - 1);
        var taken: [r.context_blocks]bool = @splat(false);
        while (n < r.topk_blocks) : (n += 1) {
            var best: ?usize = null;
            for (0..nb - 1) |b| {
                if (taken[b] or std.math.isNan(s[b]) or s[b] == -std.math.inf(f32)) continue;
                if (best == null or before(s[b], b, s[best.?], best.?)) best = b;
            }
            const b = best orelse break;
            taken[b] = true;
            out[n] = @intCast(b);
        }
        std.mem.sort(i32, out[0..n], {}, std.sort.asc(i32));
        return n;
    }

    pub fn check(t: *Select, _: g.Gate) !?[]const u8 {
        const rows = t.shape.rows;
        const words = t.shape.k;
        const sel = std.mem.bytesAsSlice(i32, t.rig.bytes(t.sel, rows * r.kv_heads * r.topk_blocks * 4));
        const bits = std.mem.bytesAsSlice(u32, t.rig.bytes(t.bits, @max(rows * r.kv_heads * words, 1) * 4));
        for (0..rows) |row| for (0..r.kv_heads) |h| {
            var list: [r.topk_blocks]i32 = undefined;
            const n = t.hostList(row, h, &list);
            const at = (row * r.kv_heads + h) * r.topk_blocks;
            for (0..r.topk_blocks) |i| {
                const want: i32 = if (i < n) list[i] else -1;
                if (sel[at + i] != want) return try std.fmt.bufPrint(&t.msg, "row {d} head {d}'s kept block {d} is {d} where the host keeps {d}", .{ row, h, i, sel[at + i], want });
            }
            for (0..words) |wd| {
                var m: u32 = 0;
                for (list[0..n]) |b| if (@as(usize, @intCast(b)) / 32 == wd) {
                    m |= @as(u32, 1) << @intCast(@as(usize, @intCast(b)) % 32);
                };
                const got = bits[(row * r.kv_heads + h) * words + wd];
                if (got != m) return try std.fmt.bufPrint(&t.msg, "row {d} head {d}'s bitmap word {d} is {x} where the host has {x}", .{ row, h, wd, got, m });
            }
        };
        return null;
    }
};

/// This file's kernels in the suite.
pub const Group = struct {
    sort: ?*Sort = null,
    rows: ?*RowsByOrder = null,
    select: ?*Select = null,

    pub fn collect(s: *Group, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, kernels: *std.ArrayList(g.Kernel)) !void {
        inline for (.{ .{ "sort", Sort }, .{ "rows", RowsByOrder }, .{ "select", Select } }) |e| {
            if (only == null or std.mem.indexOf(u8, e[1].spec().name, only.?) != null) {
                const x = try a.create(e[1]);
                x.* = try e[1].init(a, device, queue);
                @field(s, e[0]) = x;
                try kernels.append(a, x.kernel());
            }
        }
    }
    pub fn deinit(s: *Group, a: std.mem.Allocator) void {
        inline for (.{ "sort", "rows", "select" }) |f| if (@field(s, f)) |x| {
            x.deinit();
            a.destroy(x);
        };
    }
};
