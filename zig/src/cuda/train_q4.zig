//! train_q4.cu's input gradients for MLX affine 4-bit (group 32) models, with train.cu's loss, sort and pair kernels.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;
const Args = launch_.Args;

/// A 4-bit matrix [n, k] in MLX's layout: words [n, k / 8], bf16 scales and biases [n, k / 32].
pub const Q4 = struct {
    w: u64,
    s: u64,
    b: u64,
    n: usize,
    k: usize,
};

/// The routed experts' stacks [routed, n, k] (gate, up, down as words, scales, biases) and the shared expert's own.
pub const Experts = struct {
    gate: [3]u64,
    up: [3]u64,
    down: [3]u64,
    shared_gate: [3]u64,
    shared_up: [3]u64,
    shared_down: [3]u64,
    routed: usize,
    width: usize,
    dim: usize,
};

/// Experts as experts_pack.cu packs them (group 32): `up` gate | up block by block, `down` alone; `routed` the shared.
pub const PackedExperts = struct { up: u64, down: u64, routed: usize, width: usize, dim: usize };

/// What the MoE layer kept: x [rows, d] bf16, ids [rows, slots] (top_k, then the shared one), wts [rows, slots] fp32.
pub const MoeKept = struct { x: u64, ids: u64, wts: u64 };

/// Scratch for the MoE backward of `rows` rows (pairs = rows slots).
pub const MoeScratch = struct {
    logits: u64, // [rows, routed + 1] fp32
    order: u64, // [pairs] i32
    inv: u64, // [pairs] i32
    offsets: u64, // [routed + 1] i32
    tiles: u64, // [2 tileCap] u32
    gu: u64, // [pairs, 2 width] fp32
    dact: u64, // [pairs, width] fp32
    dgu: u64, // [pairs, 2 width] fp32
    dxp: u64, // [pairs, d] fp32
    a: u64, // [pairs] fp32: dL/d each slot's weight
    dlog: u64, // [rows, slots] fp32 or 0
};

const q4_names = [_][:0]const u8{
    "tf_q4_back",        "tf_q4_back_bf",           "tf_q4_sum_parts",         "tf_q4_experts_mm", "tf_q4_experts_nt",
    "tf_q4_swiglu_back", "tf_q4_experts_nt_packed", "tf_q4_experts_mm_packed", "tf_q4_route_back", "tf_q4_axpy",
};

const train_names = [_][:0]const u8{ "tf_train_softmax", "tf_train_logits", "tf_train_sort_pairs", "tf_train_experts_tiles", "tf_train_pairs_out", "tf_train_zero" };

fn index(comptime list: []const [:0]const u8, comptime name: []const u8) usize {
    inline for (list, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return i;
    @compileError("no kernel " ++ name);
}

pub const Kernels = struct {
    q4: Module,
    train: Module,
    fq: [q4_names.len]Function,
    ft: [train_names.len]Function,

    pub fn load(d: *const Driver) !Kernels {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = .{ .q4 = try Module.load(d, kernels.train_q4), .train = undefined, .fq = undefined, .ft = undefined };
        errdefer k.q4.unload();
        k.train = try Module.load(d, kernels.train);
        errdefer k.train.unload();
        for (q4_names, &k.fq) |n, *f| f.* = try k.q4.function(n);
        for (train_names, &k.ft) |n, *f| f.* = try k.train.function(n);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.q4.unload();
        k.train.unload();
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

fn up(n: usize, by: usize) u32 {
    return @intCast((n + by - 1) / by);
}

/// The tiles of 16 pairs `pairs` pairs cut into at most, each expert's last one part full.
pub fn tileCap(pairs: usize, experts: usize) usize {
    return pairs / 16 + experts;
}

/// K splits for a product of m rows by n columns over k: enough blocks to fill the GPU, each split 1024 rows or more.
pub fn splitsFor(m: usize, n: usize, k: usize) usize {
    const tiles = ((m + 63) / 64) * ((n + 63) / 64);
    const want = (192 + tiles - 1) / tiles;
    return @max(1, @min(want, k / 1024, 64));
}

/// The routing arrays hold 16 picks; the sorter holds 1024 experts including the shared one.
fn routeShape(top_k: usize, routed: usize) !void {
    if (routed == 0 or routed >= 1024 or top_k == 0 or top_k > @min(16, routed)) return error.UnsupportedShape;
}
fn expertShape(width: usize, dim: usize) !void {
    if (width == 0 or dim == 0 or width % 64 != 0 or dim % 64 != 0) return error.UnsupportedShape;
}

/// Launches on one stream.
pub const Ops = struct {
    k: *const Kernels,
    s: Stream,

    fn go(o: Ops, comptime name: []const u8, grid: [3]u32, block: u32, args: *Args) !void {
        const f = if (comptime std.mem.startsWith(u8, name, "tf_q4_")) o.k.fq[index(&q4_names, name)] else o.k.ft[index(&train_names, name)];
        try launch_.launch(f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = block } }, o.s, args);
    }

    pub fn zero(o: Ops, x: u64, n: usize) !void {
        var a: Args = .{};
        a.add(x);
        a.add(@as(c_longlong, @intCast(n)));
        try o.go("tf_train_zero", .{ up(n, 256), 1, 1 }, 256, &a);
    }

    /// c [m, w.k] (+)= a [m, w.n] (fp32, or bf16; rows lda apart) times w; `parts`: splitsFor's partials (0: none).
    pub fn product(o: Ops, a_: u64, a_bf16: bool, lda: usize, w: Q4, c: u64, ldc: usize, m: usize, accumulate: bool, parts: u64) !void {
        const n = w.k;
        const k = w.n;
        if (n % 32 != 0) return error.UnsupportedShape;
        const splits = if (parts != 0) splitsFor(m, n, k) else 1;
        const kchunk = (((k + splits - 1) / splits) + 31) / 32 * 32;
        const z = (k + kchunk - 1) / kchunk;
        var a: Args = .{};
        a.add(a_);
        for ([_]u64{ w.w, w.s, w.b, if (z > 1) parts else c }) |v| a.add(v);
        for ([_]usize{ m, n, k, lda, ldc }) |v| a.add(int(v));
        a.add(@as(c_int, @intFromBool(accumulate)));
        a.add(int(kchunk));
        if (a_bf16) try o.go("tf_q4_back_bf", .{ up(n, 64), up(m, 64), u(z) }, 128, &a) else try o.go("tf_q4_back", .{ up(n, 64), up(m, 64), u(z) }, 128, &a);
        if (z == 1) return;
        var b: Args = .{};
        b.add(parts);
        for ([_]usize{ z, m, n }) |v| b.add(int(v));
        b.add(c);
        b.add(int(ldc));
        b.add(@as(c_int, @intFromBool(accumulate)));
        try o.go("tf_q4_sum_parts", .{ up(m * n, 256), 1, 1 }, 256, &b);
    }

    /// Each row's softmax over its bf16 logits: (loss, target probability) into stats, then (p - onehot) w in place.
    pub fn softmax(o: Ops, logits: u64, targets: u64, weights: u64, stats: u64, rows: usize, vocab: usize) !void {
        var a: Args = .{};
        for ([_]u64{ logits, targets, weights, stats }) |v| a.add(v);
        a.add(int(vocab));
        try o.go("tf_train_softmax", .{ u(rows), 1, 1 }, 1024, &a);
    }

    /// dx [rows, head.k] = the softmax's bf16 rows [rows, head.n] times the 4-bit head (split over the vocabulary).
    pub fn headBack(o: Ops, dlogits: u64, head: Q4, dx: u64, rows: usize, parts: u64) !void {
        try o.product(dlogits, true, head.n, head, dx, head.k, rows, false, parts);
    }

    /// y[0 .. n] += x[0 .. n] (fp32).
    pub fn axpy(o: Ops, y: u64, x: u64, n: usize) !void {
        var a: Args = .{};
        a.add(y);
        a.add(x);
        a.add(int(n));
        try o.go("tf_q4_axpy", .{ up(n, 256), 1, 1 }, 256, &a);
    }

    /// The router's logits again, then the pairs sorted by expert into tiles of 16 (the shared expert last).
    fn sortPairs(o: Ops, router: u64, kept: MoeKept, t: MoeScratch, rows: usize, top_k: usize, routed: usize, dim: usize) !void {
        try routeShape(top_k, routed);
        const slots = top_k + 1;
        const pairs = rows * slots;
        const total = routed + 1;
        var a: Args = .{};
        for ([_]u64{ kept.x, router, t.logits }) |v| a.add(v);
        for ([_]usize{ rows, total, dim }) |v| a.add(int(v));
        try o.go("tf_train_logits", .{ up(32 * total, 256), u(rows), 1 }, 256, &a);
        var b: Args = .{};
        b.add(kept.ids);
        for ([_]usize{ rows, slots, slots, total }) |v| b.add(int(v));
        for ([_]u64{ t.order, t.inv, t.offsets }) |v| b.add(v);
        try o.go("tf_train_sort_pairs", .{ 1, 1, 1 }, 256, &b);
        var c: Args = .{};
        c.add(t.offsets);
        c.add(t.tiles);
        for ([_]usize{ pairs, tileCap(pairs, total), total }) |v| c.add(int(v));
        try o.go("tf_train_experts_tiles", .{ 1, 1, 1 }, 1, &c);
    }

    fn experts(o: Ops, comptime name: []const u8, x: u64, ldx: usize, order: u64, slots: usize, w: [3]u64, w2: [3]u64, t: MoeScratch, y: u64, extra: anytype, pairs: usize, grid_x: usize, n_experts: usize) !void {
        var a: Args = .{};
        a.add(x);
        a.add(int(ldx));
        a.add(order);
        a.add(int(slots));
        for (w) |v| a.add(v);
        for (w2) |v| a.add(v);
        a.add(t.tiles);
        a.add(t.offsets);
        a.add(y);
        inline for (extra) |v| a.add(v);
        try o.go(name, .{ u(grid_x), u(tileCap(pairs, n_experts)), 1 }, 64, &a);
    }

    fn packed_(o: Ops, comptime name: []const u8, x: u64, ldx: usize, order: u64, slots: usize, w: u64, stack: usize, half: usize, t: MoeScratch, y: u64, extra: anytype, pairs: usize, grid_x: usize, n_experts: usize) !void {
        var a: Args = .{};
        a.add(x);
        a.add(int(ldx));
        a.add(order);
        a.add(int(slots));
        a.add(w);
        a.add(int(stack));
        a.add(int(half));
        a.add(t.tiles);
        a.add(t.offsets);
        a.add(y);
        inline for (extra) |v| a.add(v);
        try o.go(name, .{ u(grid_x), u(tileCap(pairs, n_experts)), 1 }, 64, &a);
    }

    /// Each pair's SwiGLU undone: t.a its dL/dw, t.dgu its gate and up rows' gradients.
    fn swigluPairs(o: Ops, kept: MoeKept, t: MoeScratch, pairs: usize, width: usize) !void {
        var s: Args = .{};
        for ([_]u64{ t.gu, t.dact, t.order, kept.wts, t.a, t.dgu }) |v| s.add(v);
        s.add(int(width));
        try o.go("tf_q4_swiglu_back", .{ u(pairs), 1, 1 }, 128, &s);
    }

    fn pairsOut(o: Ops, dx: u64, t: MoeScratch, rows: usize, dim: usize, slots: usize) !void {
        var p: Args = .{};
        for ([_]u64{ t.inv, t.dxp, dx }) |v| p.add(v);
        p.add(int(dim));
        p.add(int(slots));
        try o.go("tf_train_pairs_out", .{ up(dim, 256), u(rows), 1 }, 256, &p);
    }

    /// The experts undone from g = dL/d the MoE output [rows, d]: dx [rows, d] += their input gradients, t.a = dL/dw.
    pub fn expertsBack(o: Ops, e: Experts, router: u64, kept: MoeKept, g: u64, dx: u64, t: MoeScratch, rows: usize, top_k: usize) !void {
        try routeShape(top_k, e.routed);
        try expertShape(e.width, e.dim);
        const slots = top_k + 1;
        const pairs = rows * slots;
        const total = e.routed + 1;
        const W = e.width;
        const D = e.dim;
        try o.sortPairs(router, kept, t, rows, top_k, e.routed, D);
        // each pair's gate and up rows again, then its down projection's input gradient (before its weight)
        inline for (.{ .{ e.gate, e.shared_gate, 0 }, .{ e.up, e.shared_up, W } }) |m|
            try o.experts("tf_q4_experts_nt", kept.x, D, t.order, slots, m[0], m[1], t, t.gu, .{ int(2 * W), int(m[2]), int(pairs), int(D), int(W), int(e.routed) }, pairs, W / 64, total);
        try o.experts("tf_q4_experts_mm", g, D, t.order, slots, e.down, e.shared_down, t, t.dact, .{ int(pairs), int(D), int(W), int(e.routed), @as(c_int, 0) }, pairs, W / 64, total);
        try o.swigluPairs(kept, t, pairs, W);
        try o.experts("tf_q4_experts_mm", t.dgu, 2 * W, 0, slots, e.gate, e.shared_gate, t, t.dxp, .{ int(pairs), int(W), int(D), int(e.routed), @as(c_int, 0) }, pairs, D / 64, total);
        try o.experts("tf_q4_experts_mm", t.dgu + W * 4, 2 * W, 0, slots, e.up, e.shared_up, t, t.dxp, .{ int(pairs), int(W), int(D), int(e.routed), @as(c_int, 1) }, pairs, D / 64, total);
        try o.pairsOut(dx, t, rows, D, slots);
    }

    /// expertsBack over packed experts: the same gradients, the weights read where the engine keeps them.
    pub fn expertsBackPacked(o: Ops, e: PackedExperts, router: u64, kept: MoeKept, g: u64, dx: u64, t: MoeScratch, rows: usize, top_k: usize) !void {
        try routeShape(top_k, e.routed);
        try expertShape(e.width, e.dim);
        const slots = top_k + 1;
        const pairs = rows * slots;
        const total = e.routed + 1;
        const W = e.width;
        const D = e.dim;
        if (W % 64 != 0 or D % 64 != 0) return error.UnsupportedShape;
        try o.sortPairs(router, kept, t, rows, top_k, e.routed, D);
        inline for (.{ .{ 0, 0 }, .{ 1, W } }) |m|
            try o.packed_("tf_q4_experts_nt_packed", kept.x, D, t.order, slots, e.up, 2, m[0], t, t.gu, .{ int(2 * W), int(m[1]), int(pairs), int(D), int(W), int(e.routed) }, pairs, W / 64, total);
        try o.packed_("tf_q4_experts_mm_packed", g, D, t.order, slots, e.down, 1, 0, t, t.dact, .{ int(pairs), int(D), int(W), int(e.routed), @as(c_int, 0) }, pairs, W / 64, total);
        try o.swigluPairs(kept, t, pairs, W);
        try o.packed_("tf_q4_experts_mm_packed", t.dgu, 2 * W, 0, slots, e.up, 2, 0, t, t.dxp, .{ int(pairs), int(W), int(D), int(e.routed), @as(c_int, 0) }, pairs, D / 64, total);
        try o.packed_("tf_q4_experts_mm_packed", t.dgu + W * 4, 2 * W, 0, slots, e.up, 2, 1, t, t.dxp, .{ int(pairs), int(W), int(D), int(e.routed), @as(c_int, 1) }, pairs, D / 64, total);
        try o.pairsOut(dx, t, rows, D, slots);
    }

    /// The routing undone from t.a: dx += the top_k softmax weights' and shared gate's gradients through the router.
    pub fn routeBack(o: Ops, router: u64, kept: MoeKept, dx: u64, t: MoeScratch, rows: usize, top_k: usize, routed: usize, dim: usize) !void {
        try routeShape(top_k, routed);
        var r: Args = .{};
        r.add(t.logits);
        r.add(int(routed + 1));
        r.add(kept.ids);
        r.add(int(top_k + 1));
        for ([_]u64{ t.a, router, dx }) |v| r.add(v);
        for ([_]usize{ top_k, routed, dim }) |v| r.add(int(v));
        r.add(t.dlog);
        try o.go("tf_q4_route_back", .{ u(rows), 1, 1 }, 256, &r);
    }

    /// The MoE layer undone from g: expertsBack, then routeBack.
    pub fn moeBack(o: Ops, e: Experts, router: u64, kept: MoeKept, g: u64, dx: u64, t: MoeScratch, rows: usize, top_k: usize) !void {
        try o.expertsBack(e, router, kept, g, dx, t, rows, top_k);
        try o.routeBack(router, kept, dx, t, rows, top_k, e.routed, e.dim);
    }

    /// moeBack over packed experts.
    pub fn moeBackPacked(o: Ops, e: PackedExperts, router: u64, kept: MoeKept, g: u64, dx: u64, t: MoeScratch, rows: usize, top_k: usize) !void {
        try o.expertsBackPacked(e, router, kept, g, dx, t, rows, top_k);
        try o.routeBack(router, kept, dx, t, rows, top_k, e.routed, e.dim);
    }
};

test "a product splits only past 1024 inputs a split, and tiles cover every pair" {
    try std.testing.expectEqual(@as(usize, 1), splitsFor(256, 2560, 640));
    try std.testing.expect(splitsFor(64, 2560, 248320) > 1);
    try std.testing.expectEqual(@as(usize, 513 + 256 * 11 / 16), tileCap(256 * 11, 513));
}

test {
    _ = @import("train_q4_test.zig");
}
