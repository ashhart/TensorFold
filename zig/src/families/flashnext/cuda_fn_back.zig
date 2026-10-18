//! Flash Next's input gradients on one stream: its hyper-connections and pocket, and the core's products and layers.
const std = @import("std");
const cuda = @import("cuda");

const q4 = cuda.train_q4;
const ta = cuda.train_attn;

pub const Q4 = q4.Q4;
pub const Experts = q4.Experts;
pub const tileCap = q4.tileCap;
pub const splitsFor = q4.splitsFor;
pub const max_keys = ta.max_keys;

/// Neurons a pocket kernel takes at most: also the row stride of the forward's activations (Pocket.act).
pub const max_pocket = 128;
/// Whether pocketRows takes its two kernels' tiled forms: where they ran faster on a GB10 (tf-cuda-test fn-pocket).
pub fn pocketTiled(rows: usize, n: usize) [2]bool {
    return .{ rows * n >= 4096, rows >= 64 and n >= 32 };
}

const names = [_][:0]const u8{
    "tf_fn_hc_branch_back", "tf_fn_hc_mix_back", "tf_fn_hc_act_back",     "tf_fn_hc_norm_back", "tf_fn_pocket_in",
    "tf_fn_pocket_in_rows", "tf_fn_pocket_out",  "tf_fn_pocket_out_rows", "tf_fn_pocket_back",  "tf_fn_pocket_grad",
};

fn index(comptime name: []const u8) usize {
    inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return i;
    @compileError("no kernel " ++ name);
}

pub const Kernels = struct {
    q4: q4.Kernels,
    attn: ta.Kernels,
    fn_mod: cuda.Module,
    fns: [names.len]cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = .{ .q4 = try q4.Kernels.load(d), .attn = undefined, .fn_mod = undefined, .fns = undefined };
        errdefer k.q4.deinit();
        k.attn = try ta.Kernels.load(d);
        errdefer k.attn.deinit();
        k.fn_mod = try cuda.Module.load(d, cuda.kernels.flashnext_train);
        errdefer k.fn_mod.unload();
        for (names, &k.fns) |n, *f| f.* = try k.fn_mod.function(n);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.fn_mod.unload();
        k.attn.deinit();
        k.q4.deinit();
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

/// Launches on one stream.
pub const Ops = struct {
    k: *const Kernels,
    s: cuda.Stream,

    pub const MoeKept = q4.MoeKept;
    pub const MoeScratch = q4.MoeScratch;
    pub const PackedExperts = q4.PackedExperts;
    pub const Attn = ta.Attn;
    pub const AttnKept = ta.Kept;
    pub const AttnScratch = ta.Scratch;

    fn go(o: Ops, comptime name: []const u8, grid: [3]u32, block: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(o.k.fns[index(name)], .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = block } }, o.s, args);
    }

    /// The core's product launches on this stream.
    pub fn core(o: Ops) q4.Ops {
        return .{ .k = &o.k.q4, .s = o.s };
    }

    fn attnOps(o: Ops) ta.Ops {
        return .{ .k = &o.k.attn, .q = o.core() };
    }

    pub fn zero(o: Ops, x: u64, n: usize) !void {
        try o.core().zero(x, n);
    }

    pub fn product(o: Ops, a_: u64, a_bf16: bool, lda: usize, w: Q4, c: u64, ldc: usize, m: usize, accumulate: bool, parts: u64) !void {
        try o.core().product(a_, a_bf16, lda, w, c, ldc, m, accumulate, parts);
    }

    pub fn softmax(o: Ops, logits: u64, targets: u64, weights: u64, stats: u64, rows: usize, vocab: usize) !void {
        try o.core().softmax(logits, targets, weights, stats, rows, vocab);
    }

    pub fn headBack(o: Ops, dlogits: u64, head: Q4, dmixed: u64, rows: usize, parts: u64) !void {
        try o.core().headBack(dlogits, head, dmixed, rows, parts);
    }

    pub fn axpy(o: Ops, y: u64, x: u64, n: usize) !void {
        try o.core().axpy(y, x, n);
    }

    // ---- the hyper-connections -----------------------------------------------------------------------------------

    pub const Hc = struct {
        down: Q4, // input_mix_weight_down [low, S d]
        inject: ?Q4, // block_inject_weight [S, S d]; null on the final mixer
        up: Q4, // input_mix_weight_up [S d, low]
        scale: u64, // hc_norm's fp32 1 + w [S d]
        streams: usize,
        d: usize,
        low: usize,
        eps: f32,
    };

    /// A read-out's kept h and normed [rows, S d] bf16, v [rows, low (+ S)] fp32, up [rows, S d] and inject [rows, S].
    pub const HcKept = struct { h: u64, normed: u64, v: u64, up: u64, inject: u64 };

    /// Scratch for one read-out's backward: dn and du [rows, S d], dact [rows, low], dv [rows, low + S] (fp32).
    pub const HcScratch = struct { dn: u64, du: u64, dact: u64, dv: u64 };

    /// The write-back h' = h + bf16(branch inject) undone from g = dL/dh': dbranch [rows, d], dinject [rows, S].
    pub fn hcBranchBack(o: Ops, hc: Hc, g: u64, inject: u64, branch: u64, dbranch: u64, dinject: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ g, inject, branch, dbranch, dinject }) |v| a.add(v);
        a.add(int(hc.d));
        a.add(int(hc.streams));
        try o.go("tf_fn_hc_branch_back", .{ u(rows), 1, 1 }, 256, &a);
    }

    /// The read-out undone: g [rows, S d] (holding the residual's gradient) += dL/dh from dmix and dinject (0: none).
    pub fn hcReadoutBack(o: Ops, hc: Hc, kept: HcKept, dmix: u64, dinject: u64, g: u64, t: HcScratch, rows: usize) !void {
        const wide = hc.streams * hc.d;
        const nv = hc.low + if (hc.inject != null) hc.streams else 0;
        var a: cuda.Args = .{};
        for ([_]u64{ dmix, kept.up, kept.normed, t.dn, t.du }) |v| a.add(v);
        a.add(int(hc.d));
        a.add(int(hc.streams));
        try o.go("tf_fn_hc_mix_back", .{ up(hc.d, 256), u(rows), 1 }, 256, &a);
        try o.product(t.du, false, wide, hc.up, t.dact, hc.low, rows, false, 0);
        var b: cuda.Args = .{};
        for ([_]u64{ t.dact, kept.v, if (hc.inject != null) dinject else 0, t.dv }) |v| b.add(v);
        for ([_]usize{ hc.low, hc.streams, nv }) |v| b.add(int(v));
        try o.go("tf_fn_hc_act_back", .{ u(rows), 1, 1 }, 256, &b);
        try o.product(t.dv, false, nv, hc.down, t.dn, wide, rows, true, 0);
        if (hc.inject) |inj| try o.product(t.dv + hc.low * 4, false, nv, inj, t.dn, wide, rows, true, 0);
        var c: cuda.Args = .{};
        for ([_]u64{ kept.h, hc.scale, t.dn, g }) |v| c.add(v);
        c.add(int(hc.d));
        c.add(int(hc.streams));
        c.add(hc.eps);
        try o.go("tf_fn_hc_norm_back", .{ u(hc.streams), u(rows), 1 }, 256, &c);
    }

    // ---- the MoE layer -------------------------------------------------------------------------------------------

    /// SwiGLU neurons fp32 [n, d] behind the shared gate; `scale` (a row's, 0: none) from `from` on; `act` the scratch.
    pub const Pocket = struct {
        gate: u64,
        up: u64,
        down: u64,
        n: usize,
        scale: u64 = 0,
        from: usize = 0,
        act: u64 = 0,
    };

    /// The MoE layer undone from g = dL/d its output [rows, d]: the core's experts, the pocket, then the core's router.
    pub fn moeBack(o: Ops, e: Experts, router: u64, kept: MoeKept, pocket: ?Pocket, g: u64, dx: u64, t: MoeScratch, rows: usize, top_k: usize) !void {
        try o.core().expertsBack(e, router, kept, g, dx, t, rows, top_k);
        if (pocket) |pk| if (pk.n > 0) try o.pocketBack(pk, kept, g, dx, t.a, rows, e.dim, top_k);
        try o.core().routeBack(router, kept, dx, t, rows, top_k, e.routed, e.dim);
    }

    /// moeBack over the engine's packed experts.
    pub fn moeBackPacked(o: Ops, e: PackedExperts, router: u64, kept: MoeKept, pocket: ?Pocket, g: u64, dx: u64, t: MoeScratch, rows: usize, top_k: usize) !void {
        try o.core().expertsBackPacked(e, router, kept, g, dx, t, rows, top_k);
        if (pocket) |pk| if (pk.n > 0) try o.pocketBack(pk, kept, g, dx, t.a, rows, e.dim, top_k);
        try o.core().routeBack(router, kept, dx, t, rows, top_k, e.routed, e.dim);
    }

    /// The pocket undone: dx += its input gradient (times the shared gate), a's shared slot += its part of dL/dgate.
    pub fn pocketBack(o: Ops, pk: Pocket, kept: MoeKept, g: u64, dx: u64, a_: u64, rows: usize, d: usize, top_k: usize) !void {
        if (pk.n > max_pocket) return error.UnsupportedShape;
        var a: cuda.Args = .{};
        a.add(kept.x);
        a.add(int(d));
        for ([_]u64{ pk.gate, pk.up, pk.down }) |v| a.add(v);
        a.add(int(pk.n));
        a.add(kept.wts);
        a.add(int(top_k + 1));
        a.add(int(top_k));
        for ([_]u64{ g, dx, a_, pk.scale }) |v| a.add(v);
        a.add(int(@min(pk.from, pk.n)));
        try o.go("tf_fn_pocket_back", .{ u(rows), 1, 1 }, 256, &a);
    }

    /// out [rows, d] (+)= the pocket's output (kept.wts 0: no gate), tf_fn_pocket_in then _out with _rows' bits.
    pub fn pocketRows(o: Ops, pk: Pocket, kept: MoeKept, out: u64, out_f32: bool, rows: usize, d: usize, top_k: usize) !void {
        if (pk.n > max_pocket) return error.UnsupportedShape;
        if (pk.n == 0 or rows == 0) return; // no neuron in use: nothing launched, the output keeps its bits
        if (pk.act == 0) return error.UnsupportedShape;
        var a: cuda.Args = .{};
        a.add(kept.x);
        a.add(int(d));
        a.add(pk.gate);
        a.add(pk.up);
        a.add(int(pk.n));
        a.add(kept.wts);
        a.add(int(top_k + 1));
        a.add(int(top_k));
        a.add(pk.act);
        a.add(int(rows));
        a.add(pk.scale);
        a.add(int(@min(pk.from, pk.n)));
        // a block row a row until the tiled forms (two neurons by four rows a warp, four rows a thread) run faster
        const tiled = pocketTiled(rows, pk.n);
        if (tiled[0]) {
            try o.go("tf_fn_pocket_in_rows", .{ up(pk.n, 16), up(rows, 4), 1 }, 256, &a);
        } else {
            try o.go("tf_fn_pocket_in", .{ up(pk.n, 4), u(rows), 1 }, 128, &a);
        }
        var b: cuda.Args = .{};
        b.add(pk.act);
        b.add(pk.down);
        b.add(int(pk.n));
        b.add(int(d));
        b.add(out);
        b.add(@as(c_int, @intFromBool(out_f32)));
        b.add(int(rows));
        if (tiled[1]) {
            try o.go("tf_fn_pocket_out_rows", .{ up(d, 128), up(rows, 4), 1 }, 128, &b);
        } else {
            try o.go("tf_fn_pocket_out", .{ up(d, 128), u(rows), 1 }, 128, &b);
        }
    }

    /// grad [n, d] += w gate act_j push[row_j], neuron j at row row_of[j] (negative: none): the outputs' gradient.
    pub fn pocketGrad(o: Ops, pk: Pocket, kept: MoeKept, push: u64, row_of: u64, w: f32, grad: u64, d: usize, top_k: usize) !void {
        var a: cuda.Args = .{};
        a.add(kept.x);
        a.add(int(d));
        a.add(pk.gate);
        a.add(pk.up);
        a.add(int(pk.n));
        a.add(kept.wts);
        a.add(int(top_k + 1));
        a.add(int(top_k));
        a.add(push);
        a.add(row_of);
        a.add(w);
        a.add(grad);
        try o.go("tf_fn_pocket_grad", .{ u(pk.n), 1, 1 }, 256, &a);
    }

    // ---- the attention layer -------------------------------------------------------------------------------------

    pub fn attnPrep(o: Ops, at: Attn, kept: AttnKept, qkv: u64, rows: usize) !void {
        try o.attnOps().prep(at, kept, qkv, rows);
    }

    pub fn attnBack(o: Ops, at: Attn, kept: AttnKept, g: u64, dx: u64, t: AttnScratch, rows: usize, accumulate: bool) !void {
        try o.attnOps().back(at, kept, g, dx, t, rows, accumulate);
    }
};
