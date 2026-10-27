//! MiniMax-M3.1-Flash's decode attention (mm_decode_vec, mm_merge_vec) under the contract, in attention_range's range.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const ar = contract.attention_range;
const rig_ = @import("rig.zig");
const r = @import("metal_minimax_route.zig");
const sel = @import("metal_minimax_route_sort.zig");

const c = contract.spec;
const g = contract.gates;
const D = r.head_dim;
const G = r.heads / r.kv_heads;
/// Partials a (head, row): four quarters of each kept block.
const parts = r.topk_blocks * 4;

/// mm_vec_args (glue.metal).
const VArgs = extern struct { rows: u32, pos: u32, heads: u32, groups: u32, topk: u32, scale: f32, sel: u32, all: u32 };

/// fp32 roundings: a score 10, 10 fast::exps a weight, o's and l's chains (8 rescales, 64 partials), a fast division.
const score_ops = 10;
const num_ops = 8 * 9 + 2 * parts + 2;
const den_ops = 8 * 5 + 2 * parts + 2;

const form: ar.Form = .{ .score = score_ops, .fast_exp = false, .chunk_exp = false, .p_rel = 0, .num = num_ops, .den = den_ops, .div = 3, .exp_ulps = .{ 4 * 10, 2 } };

pub const Decode = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    part: mtl.Pipeline,
    merge: mtl.Pipeline,
    rows: usize = 0,
    blocks: usize = 0,
    pos: usize = 0,
    all: bool = false,
    stress: bool = false,
    q: []u16 = &.{},
    k: []u16 = &.{},
    v: []u16 = &.{},
    picks: []i32 = &.{},
    ins: [4]rig_.Ref = undefined,
    scratch: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    point: []f64 = &.{},
    half: []f64 = &.{},

    /// Rows: decode rows; n: the index blocks their keys span (every block kept while at most 16); k: the head dim.
    pub fn spec() c.Spec {
        return .{
            .name = "metal minimax_m3 mm_decode_vec + mm_merge_vec 64/4 heads of 128, top 16 blocks",
            .accum = .f32,
            .depth = depth,
            .ops = 4,
            .out = .bf16,
            .tiles = .{ .{ .size = 16 }, .{}, .{} },
            // every block kept (gate 4's), the 147,456-token window, past 16 blocks, 32 shared rows, one row at key 0
            .served = &.{ .{ .rows = 16, .n = r.topk_blocks, .k = D }, .{ .rows = 16, .n = 1152, .k = D }, .{ .rows = 1, .n = r.topk_blocks + 1, .k = D }, .{ .rows = r.batch_rows, .n = 300, .k = D }, .{ .rows = 1, .n = 1, .k = D } },
        };
    }

    fn depth(_: c.Shape) usize {
        return num_ops;
    }

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue) !Decode {
        const part = try r.gluePipe(device, "mm_decode_vec");
        errdefer part.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .part = part, .merge = try r.gluePipe(device, "mm_merge_vec") };
    }
    fn free(t: *Decode) void {
        inline for (.{ "q", "k", "v", "picks" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
        t.a.free(t.point);
        t.a.free(t.half);
        t.point = &.{};
        t.half = &.{};
    }
    pub fn deinit(t: *Decode) void {
        t.free();
        t.rig.deinit();
        t.part.deinit();
        t.merge.deinit();
    }
    pub fn kernel(t: *Decode) g.Kernel {
        return .{ .spec = spec(), .ctx = t, .vt = &vt };
    }
    const vt = g.hooks(Decode);

    const ld = r.kv_heads * D;

    pub fn prepare(t: *Decode, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > r.batch_rows or shape.k != D) return false;
        const pos = sel.startOf(shape.rows, shape.n) orelse return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.blocks = shape.n;
        t.pos = pos;
        t.all = shape.n <= r.topk_blocks;
        t.stress = fill == .stress;
        const keys = shape.n * r.block;
        t.q = try t.a.alloc(u16, shape.rows * r.heads * D);
        t.k = try t.a.alloc(u16, keys * ld);
        t.v = try t.a.alloc(u16, keys * ld);
        t.picks = try t.a.alloc(i32, shape.rows * r.kv_heads * r.topk_blocks);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (t.q) |*x| x.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (t.k) |*x| x.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (t.v) |*x| x.* = c.toBf16(rng.float(f32) * 2 - 1);
        if (fill == .stress) {
            // keys alike (weights exactly 1), integer values: a 2,000-key sum is exact in fp32 and passes fp16's range
            for (0..keys) |j| @memcpy(t.k[j * ld ..][0..ld], t.k[0..ld]);
            for (0..keys) |j| for (0..ld) |e| {
                t.v[j * ld + e] = c.toBf16(@floatFromInt(1 + (j * 7 + e * 13) % 250));
            };
        }
        // a row keeps its own block and up to 15 earlier ones (a few rows fewer: -1 past them), ascending
        for (0..shape.rows) |row| for (0..r.kv_heads) |h| {
            const list = t.picks[(row * r.kv_heads + h) * r.topk_blocks ..][0..r.topk_blocks];
            @memset(list, -1);
            const own = (pos + row) / r.block;
            const want: usize = if (row % 5 == 3) 9 else r.topk_blocks;
            var n: usize = 0;
            list[n] = @intCast(own);
            n += 1;
            while (n < @min(want, own + 1)) {
                const b: i32 = @intCast(rng.uintLessThan(usize, own));
                if (std.mem.indexOfScalar(i32, list[0..n], b) == null) {
                    list[n] = b;
                    n += 1;
                }
            }
            std.mem.sort(i32, list[0..n], {}, std.sort.asc(i32));
        };
        if (fill == .nonfinite) {
            // NaN in row 0's head 0; +Inf in row 1's own value (key head 1, dim 7); -Inf in row 2's (key head 2, dim 9)
            t.q[3] = c.toBf16(r.nan32);
            if (shape.rows > 1) t.v[(pos + 1) * ld + 1 * D + 7] = c.toBf16(r.inf32);
            if (shape.rows > 2) t.v[(pos + 2) * ld + 2 * D + 9] = c.toBf16(-r.inf32);
        }
        for ([_][]const u8{ std.mem.sliceAsBytes(t.q), std.mem.sliceAsBytes(t.k), std.mem.sliceAsBytes(t.v), std.mem.sliceAsBytes(t.picks) }, &t.ins) |b, *ref| ref.* = try t.rig.input(b);
        const slots = r.heads * shape.rows * parts;
        t.scratch[0] = try t.rig.scratch(slots * 4);
        t.scratch[1] = try t.rig.scratch(slots * 4);
        t.scratch[2] = try t.rig.scratch(slots * D * 4);
        t.out = try t.rig.output(t.count() * 2);
        t.point = try t.a.alloc(f64, t.count());
        t.half = try t.a.alloc(f64, t.count());
        try t.ranges();
        return true;
    }

    fn args(t: *const Decode) VArgs {
        const scale: f32 = @floatCast(1 / @sqrt(@as(f64, D)));
        return .{ .rows = @intCast(t.rows), .pos = @intCast(t.pos), .heads = r.heads, .groups = r.kv_heads, .topk = parts, .scale = scale, .sel = r.topk_blocks, .all = @intFromBool(t.all) };
    }

    pub fn run(t: *Decode) !void {
        const p = t.rig.begin();
        const e = p.e;
        e.setPipeline(t.part);
        for (t.ins[0..3], 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setValue(t.args(), 3);
        e.setBuffer(t.ins[3].buf, t.ins[3].off, 4);
        for (t.scratch, 5..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.dispatchGroups(mtl.Size.of(r.kv_heads, t.rows, parts), mtl.Size.of(32, G, 1));
        e.setPipeline(t.merge);
        for (t.scratch, 0..) |ref, i| e.setBuffer(ref.buf, ref.off, i);
        e.setValue(t.args(), 3);
        e.setBuffer(t.out.buf, t.out.off, 4);
        e.dispatchGroups(mtl.Size.of(r.heads, t.rows, 1), mtl.Size.of(32, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Decode) usize {
        return t.rows * r.heads * D;
    }

    /// Row `row`, key head `kv`'s kept keys in block order (the merge's).
    fn kept(t: *const Decode, row: usize, kv: usize, out: *std.ArrayList(usize)) !void {
        out.clearRetainingCapacity();
        const p = t.pos + row;
        for (0..r.topk_blocks) |i| {
            const b: i64 = if (t.all) (if (i <= p / r.block) @intCast(i) else -1) else t.picks[(row * r.kv_heads + kv) * r.topk_blocks + i];
            if (b < 0) continue;
            const from: usize = @intCast(b * r.block);
            for (from..@min(p + 1, from + r.block)) |key| try out.append(t.a, key);
        }
    }

    /// Every output's float64 point and half-range, from the kept keys of its (row, head).
    fn ranges(t: *Decode) !void {
        var keys: std.ArrayList(usize) = .empty;
        defer keys.deinit(t.a);
        const scale: f64 = t.args().scale;
        var qf: [D]f64 = undefined;
        for (0..t.rows) |row| for (0..r.kv_heads) |kv| {
            try t.kept(row, kv, &keys);
            const n = keys.items.len;
            const kf = try t.a.alloc(f64, 2 * n * D + 2 * n);
            defer t.a.free(kf);
            const vf = kf[n * D .. 2 * n * D];
            for (keys.items, 0..) |key, j| for (0..D) |e| {
                kf[j * D + e] = c.fromBf16(t.k[key * ld + kv * D + e]);
                vf[j * D + e] = c.fromBf16(t.v[key * ld + kv * D + e]);
            };
            for (0..G) |gi| {
                const hd = kv * G + gi;
                for (&qf, 0..) |*x, e| x.* = c.fromBf16(t.q[(row * r.heads + hd) * D + e]);
                const at = (row * r.heads + hd) * D;
                ar.head(&qf, kf[0 .. n * D], vf, n, D, scale, form, kf[2 * n * D ..][0..n], kf[2 * n * D + n ..][0..n], t.point[at..][0..D], t.half[at..][0..D]);
            }
        };
    }

    pub fn reference(t: *Decode, want: []f64, mag: []f64) !void {
        @memcpy(want, t.point);
        for (mag, want) |*m, w| m.* = if (std.math.isFinite(w)) @abs(w) else 0;
    }

    pub fn intervals(t: *Decode, low: []f64, high: []f64) !void {
        for (low, high, t.point, t.half) |*l, *h, p, hf| {
            l.* = if (std.math.isFinite(p)) r.bf(p - hf) else 0;
            h.* = if (std.math.isFinite(p)) r.bf(p + hf) else 0;
        }
    }

    pub fn bounds(t: *Decode, want: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, want, t.point, t.half) |*l, w, p, hf| l.* = if (std.math.isFinite(w)) r.spread(w, .{ .lo = r.bf(p - hf), .hi = r.bf(p + hf) }) else 0;
    }

    /// Gate 4's sums: every weight is 1, so o is V's column sum over the kept keys' count, rounded once.
    pub fn simulate(t: *Decode, acc: c.Accum, order: g.Order, out: []f64) !void {
        var keys: std.ArrayList(usize) = .empty;
        defer keys.deinit(t.a);
        for (0..t.rows) |row| for (0..r.kv_heads) |kv| {
            try t.kept(row, kv, &keys);
            const col = try t.a.alloc(f64, keys.items.len);
            defer t.a.free(col);
            for (0..D) |e| {
                for (keys.items, col) |key, *x| x.* = c.fromBf16(t.v[key * ld + kv * D + e]);
                const o = contract.rows.sumIn(acc, order, col) / @as(f64, @floatFromInt(keys.items.len));
                for (0..G) |gi| out[(row * r.heads + kv * G + gi) * D + e] = o;
            }
        };
    }
};

pub const Group = struct {
    decode: ?*Decode = null,

    pub fn collect(s: *Group, a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, kernels: *std.ArrayList(g.Kernel)) !void {
        if (only != null and std.mem.indexOf(u8, Decode.spec().name, only.?) == null) return;
        const x = try a.create(Decode);
        x.* = try Decode.init(a, device, queue);
        s.decode = x;
        try kernels.append(a, x.kernel());
    }
    pub fn deinit(s: *Group, a: std.mem.Allocator) void {
        if (s.decode) |x| {
            x.deinit();
            a.destroy(x);
        }
    }
};
