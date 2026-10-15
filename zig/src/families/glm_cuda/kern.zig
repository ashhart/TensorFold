//! GLM-5.3-Flash on CUDA: every launch the forward makes; a row's bits depend on its own inputs alone (``Rows``).

const std = @import("std");
const cuda = @import("cuda");

pub fn cdiv(a: i64, b: i64) i64 {
    return @divFloor(a + b - 1, b);
}

pub fn nextPow2(x: u64) u64 {
    return std.math.ceilPowerOfTwo(u64, @max(x, 1)) catch unreachable;
}

pub const Ext = enum(u8) { experts, experts_prefill, kda, sample, ops, experts_sym, experts_sym_prefill, vision, experts_sym3, experts_sym3_prefill, mm, rows, latent, index, exl3, dflash };

/// The launch context: the stream and the modules' functions by name.
pub const K = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,
    names: [16][]const [:0]const u8 = @splat(&.{}),
    funcs: [16][]cuda.Function = @splat(&.{}),
    sms: i64 = 48,
    cache: std.AutoHashMap(u64, cuda.Function) = undefined,
    gpa: std.mem.Allocator,
    /// Development: synchronize after every launch and name the one that faulted.
    sync_each: bool = false,
    /// Development: each kernel's time by name (every launch synchronized first and after: proportions only).
    prof: ?*std.StringHashMap(f64) = null,
    io: ?std.Io = null,
    /// The routed experts packed as symmetric groups (scales alone: glm/experts_sym*).
    experts_sym: bool = false,
    /// ... with 3-bit codes in planes (glm/experts_sym* built with GLM_SYM3).
    experts3: bool = false,
    /// The routed experts' current product is a prompt's: the prefill kernels for every prompt row (resumed == fresh).
    prompt: bool = false,

    fn after(self: *K, what: []const u8) !void {
        if (!self.sync_each) return;
        self.stream.synchronize() catch |e| {
            std.log.err("glm: launch {s} faulted", .{what});
            return e;
        };
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, ctx: *const cuda.Context, stream: cuda.Stream, modules: []const cuda.Module) !K {
        var k: K = .{ .d = d, .stream = stream, .gpa = gpa };
        k.cache = .init(gpa);
        k.sms = try ctx.attribute(.multiprocessor_count);
        for (modules, 0..) |m, i| {
            var n: c_uint = 0;
            try d.check(d.api.cuModuleGetFunctionCount(&n, m.handle), "cuModuleGetFunctionCount");
            const hs = try gpa.alloc(cuda.abi.Function, n);
            defer gpa.free(hs);
            try d.check(d.api.cuModuleEnumerateFunctions(hs.ptr, n, m.handle), "cuModuleEnumerateFunctions");
            const names = try gpa.alloc([:0]const u8, n);
            const fs = try gpa.alloc(cuda.Function, n);
            for (hs, 0..) |h, j| {
                var s: ?[*:0]const u8 = null;
                try d.check(d.api.cuFuncGetName(&s, h), "cuFuncGetName");
                names[j] = try gpa.dupeSentinel(u8, std.mem.span(s.?), 0);
                fs[j] = .{ .d = d, .handle = h };
            }
            k.names[i] = names;
            k.funcs[i] = fs;
        }
        return k;
    }

    /// A kernel of module ``m`` whose mangled name contains ``frag``: the exact match, else the one containing it.
    pub fn ext(self: *K, m: Ext, frag: []const u8) !cuda.Function {
        const key = std.hash.Wyhash.hash(@backingInt(m), frag);
        if (self.cache.get(key)) |f| return f;
        var found: ?cuda.Function = null;
        for (self.names[@backingInt(m)], 0..) |n, j| {
            if (std.mem.eql(u8, n, frag)) {
                found = self.funcs[@backingInt(m)][j];
                break;
            }
            if (std.mem.indexOf(u8, n, frag) != null) {
                if (found != null) {
                    std.log.err("glm: two kernels in {s} match {s}", .{ @tagName(m), frag });
                    return error.AmbiguousKernel;
                }
                found = self.funcs[@backingInt(m)][j];
            }
        }
        const f = found orelse {
            std.log.err("glm: no kernel in {s} matches {s}", .{ @tagName(m), frag });
            return error.MissingKernel;
        };
        try self.cache.put(key, f);
        return f;
    }

    pub fn go(self: *K, f: cuda.Function, grid: cuda.Dim3, block: u32, shared: u32, args: *cuda.Args) !void {
        return self.goWith(f, grid, block, shared, args, false);
    }

    /// A launch overlapping the one before (PDL): it waits at griddepcontrol.wait; TF_GLM_PDL=0 launches plainly.
    pub fn goDependent(self: *K, f: cuda.Function, grid: cuda.Dim3, block: u32, shared: u32, args: *cuda.Args) !void {
        return self.goWith(f, grid, block, shared, args, pdl_on);
    }

    fn goWith(self: *K, f: cuda.Function, grid: cuda.Dim3, block: u32, shared: u32, args: *cuda.Args, pdl: bool) !void {
        if (self.prof) |m| {
            try self.stream.synchronize();
            const t0 = std.Io.Timestamp.now(self.io.?, .awake);
            try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = block }, .shared = shared }, self.stream, args);
            try self.stream.synchronize();
            const ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(self.io.?, .awake)).toNanoseconds())) / 1e6;
            var nm: ?[*:0]const u8 = null;
            _ = self.d.api.cuFuncGetName(&nm, f.handle);
            const name = if (nm) |x| std.mem.span(x) else "kernel";
            const gop = try m.getOrPut(name);
            if (!gop.found_existing) gop.value_ptr.* = 0;
            gop.value_ptr.* += ms;
            return;
        }
        try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = block }, .shared = shared, .pdl = pdl }, self.stream, args);
        if (self.sync_each) {
            var nm: ?[*:0]const u8 = null;
            _ = self.d.api.cuFuncGetName(&nm, f.handle);
            try self.after(if (nm) |x| std.mem.span(x) else "kernel");
        }
    }

    pub fn memset32(self: *K, dst: u64, v: u32, n: usize) !void {
        try self.d.check(self.d.api.cuMemsetD32Async(dst, v, n, self.stream.handle), "memset32");
    }
    pub fn copy(self: *K, dst: u64, src: u64, n: usize) !void {
        try self.d.check(self.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, self.stream.handle), "copy");
    }
    pub fn upload(self: *K, dst: u64, bytes: []const u8) !void {
        try self.d.check(self.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, self.stream.handle), "upload");
    }
};

// -- the model's fixed dimensions (GLM-5.3-Flash, one rank of two) ---------------------------------------------------
pub const D: i64 = 4096; // hidden
pub const S: i64 = 4; // hyper-connection streams
pub const WIDE: i64 = D * S;
pub const EPS: f32 = 1e-5; // rms_norm_eps
pub const HC_EPS: f32 = 1e-6;
pub const HC_ITERS: i64 = 20;
pub const LIMIT: f32 = 10.0; // swiglu_limit
pub const LOWER: f32 = -5.0; // KDA gate lower bound
pub const ROUTED_SCALE: f32 = 2.5;
pub const HL: i64 = 32; // a rank's attention heads (64 / 2)
pub const LL: i64 = 32; // a rank's KDA heads
pub const QK: i64 = 256; // NoPE: the whole head is the nope part
pub const DV: i64 = 256;
pub const Q_LORA: i64 = 1536;
pub const KV_LORA: i64 = 512;
pub const EXPERTS: i64 = 288;
pub const TOPK: i64 = 8;
pub const SLOTS: i64 = TOPK + 1; // the shared expert's slot last
pub const IDX_HEADS: i64 = 32;
pub const IDX_DIM: i64 = 128;
pub const POOL: i64 = 4;
pub const TOPK_POOLS: i64 = 512;
pub const DENSE_LIMIT: i64 = 2051; // index_topk + kpool - 1
pub const CHUNK: i64 = 512; // keys a latent attention chunk program reads
pub const DENSE_CHUNKS: i64 = 5; // chunks a dense row (position < DENSE_LIMIT) reads at most
pub const SEL_W: i64 = TOPK_POOLS * POOL + POOL - 1; // 2051: a sparse row's attended tokens
pub const SPARSE_CHUNKS: i64 = 5; // cdiv(SEL_W, CHUNK)
pub const ATT_ROWS: i64 = 512; // rows one attention call takes (its partials' scratch)
pub const SELECT_ROWS: i64 = 512; // rows scored at once (the scores' scratch)
pub const SCALE: f32 = 0.0625; // qk_dim ** -0.5

/// A step's per-row tables: positions (i32) and per latent cache ``ci`` the rows' cache pointers; ``at``: a row block.
pub const Rows = struct {
    pos: u64,
    lc: u64,
    ik: u64,
    ig: u64,
    pk: u64,
    cap: i64,
    r0: i64 = 0,

    pub fn at(t: Rows, r0: i64) Rows {
        var x = t;
        x.r0 += r0;
        return x;
    }
    pub fn posAt(t: Rows) u64 {
        return t.pos + @as(u64, @intCast(t.r0 * 4));
    }
    fn table(t: Rows, base: u64, ci: usize) u64 {
        return base + (@as(u64, @intCast(ci)) * @as(u64, @intCast(t.cap)) + @as(u64, @intCast(t.r0))) * 8;
    }
    pub fn lcAt(t: Rows, ci: usize) u64 {
        return t.table(t.lc, ci);
    }
    pub fn ikAt(t: Rows, ci: usize) u64 {
        return t.table(t.ik, ci);
    }
    pub fn igAt(t: Rows, ci: usize) u64 {
        return t.table(t.ig, ci);
    }
    pub fn pkAt(t: Rows, ci: usize) u64 {
        return t.table(t.pk, ci);
    }
};

// -- rows.cu ---------------------------------------------------------------------------------------------------
pub fn embed(k: *K, ids: u64, table: u64, out: u64, R: i64, copies: i64) !void {
    var a: cuda.Args = .{};
    a.add(ids);
    a.add(table);
    a.add(out);
    a.add(@as(i32, @intCast(D)));
    a.add(@as(i32, @intCast(copies)));
    try k.go(try k.ext(.rows, "glm_embed"), .{ .x = @intCast(R) }, 256, 0, &a);
}

/// rmsnorm(x [R, d] (row stride x_stride), w) -> out (row stride o_stride), with xs the 64-groups' sums when given.
pub fn rmsnorm(k: *K, x: u64, x_stride: i64, w: u64, out: u64, o_stride: i64, xs: ?u64, R: i64, d: i64) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(x_stride);
    a.add(w);
    a.add(out);
    a.add(o_stride);
    a.add(xs orelse @as(u64, 0));
    a.add(EPS);
    a.add(@as(i32, @intCast(d)));
    try k.go(try k.ext(.rows, "glm_rmsnorm"), .{ .x = @intCast(R) }, 256, 0, &a);
}

/// hc_pre: x [R, S*D] -> out [R, D] normed collapsed rows + xs, post [R, S], comb [R, S, S] (part: [R, 16, 25] fp32).
pub fn hcPre(k: *K, x: u64, fn_: u64, base: u64, scale: u64, norm_w: u64, out: u64, xs: u64, post: u64, comb: u64, part: u64, R: i64) !void {
    var p: cuda.Args = .{};
    p.add(x);
    p.add(fn_);
    p.add(part);
    try k.go(try k.ext(.rows, "glm_hc_part"), .{ .x = @intCast(R), .y = 16 }, 128, 0, &p);
    var a: cuda.Args = .{};
    a.add(x);
    a.add(part);
    a.add(base);
    a.add(scale);
    a.add(norm_w);
    a.add(out);
    a.add(xs);
    a.add(post);
    a.add(comb);
    a.add(EPS);
    a.add(HC_EPS);
    a.add(@as(i32, @intCast(HC_ITERS)));
    try k.go(try k.ext(.rows, "glm_hc_finish"), .{ .x = @intCast(R) }, 512, 0, &a);
}

/// hc_post: x (in place) from the gathered [2, R, D] fp32 partials.
pub fn hcPost(k: *K, x: u64, gathered: u64, post: u64, comb: u64, R: i64) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(gathered);
    a.add(post);
    a.add(comb);
    a.add(@as(i32, @intCast(R)));
    try k.go(try k.ext(.rows, "glm_hc_post"), .{ .x = @intCast(R), .y = @intCast(@divExact(D, 1024)) }, 256, 0, &a);
}

pub fn residualAdd(k: *K, x: u64, gathered: u64, R: i64) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(gathered);
    a.add(@as(i32, @intCast(R)));
    try k.go(try k.ext(.rows, "glm_residual_add"), .{ .x = @intCast(R), .y = @intCast(@divExact(D, 1024)) }, 256, 0, &a);
}

pub fn streamMean(k: *K, x: u64, out: u64, R: i64) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(out);
    try k.go(try k.ext(.rows, "glm_stream_mean"), .{ .x = @intCast(R), .y = @intCast(@divExact(D, 1024)) }, 256, 0, &a);
}

pub fn swiglu(k: *K, gu: u64, out: u64, xs: u64, R: i64, w: i64) !void {
    var a: cuda.Args = .{};
    a.add(gu);
    a.add(out);
    a.add(xs);
    a.add(LIMIT);
    a.add(@as(i32, @intCast(w)));
    try k.go(try k.ext(.rows, "glm_swiglu"), .{ .x = @intCast(R), .y = @intCast(@divExact(w, 512)) }, 256, 0, &a);
}

/// router logits: x [m, D] (row stride x_stride) by w [ne, D] -> out [m, ne] fp32.
pub fn router(k: *K, x: u64, x_stride: i64, w: u64, out: u64, m: i64, ne: i64) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(x_stride);
    a.add(w);
    a.add(out);
    a.add(@as(i32, @intCast(m)));
    a.add(@as(i32, @intCast(ne)));
    a.add(@as(i32, @intCast(D)));
    try k.go(try k.ext(.rows, "glm_router"), .{ .x = @intCast(cdiv(ne, 2)), .y = @intCast(cdiv(m, 16)) }, 256, 0, &a);
}

/// each row's top-8 experts and weights, then the shared expert (id 288, weight 1) in slot 8.
pub fn topk(k: *K, logits: u64, bias: u64, pick: u64, wts: u64, R: i64) !void {
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(bias);
    a.add(pick);
    a.add(wts);
    a.add(ROUTED_SCALE);
    a.add(@as(i32, @intCast(EXPERTS)));
    try k.go(try k.ext(.rows, "glm_topk"), .{ .x = @intCast(R) }, 32, 0, &a);
}

/// combine: y [R, SLOTS, D] bf16 (prompt) or fp32 (windows) by wts -> out [R, D] fp32.
pub fn combine(k: *K, y: u64, y_f32: bool, wts: u64, out: u64, R: i64) !void {
    var a: cuda.Args = .{};
    a.add(y);
    a.add(@as(i32, @intFromBool(y_f32)));
    a.add(wts);
    a.add(out);
    try k.go(try k.ext(.rows, "glm_combine"), .{ .x = @intCast(R), .y = @intCast(@divExact(D, 1024)) }, 256, 0, &a);
}

/// conv [n, 3, C] in place takes rows keep .. keep + 2 of [conv; proj rows] (proj [n, R, W >= C]), element strides.
pub fn convShift(k: *K, conv: u64, proj: u64, keep: i64, conv_layer: i64, proj_layer: i64, proj_row: i64, n: i64, c: i64) !void {
    var a: cuda.Args = .{};
    a.add(conv);
    a.add(proj);
    a.add(@as(i32, @intCast(keep)));
    a.add(conv_layer);
    a.add(proj_layer);
    a.add(proj_row);
    a.add(@as(i32, @intCast(c)));
    try k.go(try k.ext(.rows, "glm_conv_shift"), .{ .x = @intCast(n), .y = @intCast(cdiv(c, 1024)) }, 256, 0, &a);
}

// -- mm.cu: the dense linears ----------------------------------------------------------------------------------

/// A dense linear [n, k] tiled at load as glm/mm.cu reads it: ``bits`` 0 (bf16) or 4, 6, 8 (fp16 scales ``s`` / 64).
pub const B16 = struct { w: u64, n: i64, k: i64, s: u64 = 0, bits: u8 = 0 };

pub const SPLIT_TARGET: i64 = 192;

/// K slices fixed by the shape (column tiles times slices reach SPLIT_TARGET), in groups of 64 inputs.
var pdl_on: bool = true;

/// TF_GLM_PDL=0 turns the matmuls' dependent launches off (read once at engine start).
pub fn readPdl() void {
    if (std.c.getenv("TF_GLM_PDL")) |v| pdl_on = !std.mem.eql(u8, std.mem.span(v), "0");
}

pub fn splitK(n: i64, kk: i64) i64 {
    const tiles = cdiv(n, 64);
    const steps = @divExact(kk, 64);
    var sk: i64 = 1;
    while (sk < 8 and tiles * sk < SPLIT_TARGET and @mod(steps, sk * 2) == 0 and @divFloor(steps, sk * 2) >= 4) sk *= 2;
    return sk;
}

fn mmName(bits: u8, bm: i64, buf: []u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "glm_mm{d}_{d}", .{ if (bits == 0) @as(u8, 16) else bits, bm });
}

fn mmSmem(bits: u8, bm: i64) u32 {
    const wpc: i64 = if (bits == 0) 8 else @divExact(@as(i64, bits), 2);
    return @intCast(3 * (bm * 128 + 8 * 32 * wpc * 4 + 128));
}

/// x [m, k] (stride x_stride) by w -> out [m, n], bf16 or fp32; ``part``: sk * m * n fp32 when K splits; 128 rows a go.
pub fn mm(k: *K, x: u64, x_stride: i64, w: B16, out: u64, f32_out: bool, m: i64, part: u64) !void {
    const bm: i64 = if (m <= 16) 16 else if (m <= 32) 32 else if (m <= 64) 64 else 128;
    const sk = splitK(w.n, w.k);
    var nb: [32]u8 = undefined;
    const f = try k.ext(.mm, try mmName(w.bits, bm, &nb));
    const smem = mmSmem(w.bits, bm);
    try f.allowDynamicShared(smem);
    var a: cuda.Args = .{};
    a.add(x);
    a.add(x_stride);
    a.add(w.w);
    a.add(w.s);
    a.add(out);
    a.add(part);
    a.add(@as(i32, @intFromBool(f32_out)));
    a.add(@as(i32, @intCast(m)));
    a.add(@as(i32, @intCast(w.n)));
    a.add(@as(i32, @intCast(w.k)));
    a.add(@as(i32, @intCast(sk)));
    try k.goDependent(f, .{ .x = @intCast(cdiv(w.n, 64)), .y = @intCast(cdiv(m, bm)), .z = @intCast(sk) }, if (bm <= 32) 128 else 256, smem, &a);
    if (sk > 1) {
        const total = m * w.n;
        var r: cuda.Args = .{};
        r.add(part);
        r.add(out);
        r.add(total);
        r.add(@as(i32, @intCast(sk)));
        r.add(@as(i32, @intFromBool(f32_out)));
        try k.go(try k.ext(.mm, "glm_mm_reduce"), .{ .x = @intCast(cdiv(total, 256)) }, 256, 0, &r);
    }
}

/// The loader's row-major linear (bf16 or codes and scales) at ``src`` -> its tiled copy at ``dst`` (scales ss -> ds).
pub fn mmTile(k: *K, bits: u8, src: u64, ss: u64, dst: u64, ds: u64, n: i64, kk: i64) !void {
    const chunks = @divExact(n, 8) * @divExact(kk, 64) * 32;
    var a: cuda.Args = .{};
    var nb: [32]u8 = undefined;
    if (bits == 0) {
        a.add(src);
        a.add(dst);
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intCast(kk)));
        try k.go(try k.ext(.mm, "glm_mm_tile16"), .{ .x = @intCast(cdiv(chunks, 256)) }, 256, 0, &a);
        return;
    }
    a.add(src);
    a.add(ss);
    a.add(dst);
    a.add(ds);
    a.add(@as(i32, @intCast(n)));
    a.add(@as(i32, @intCast(kk)));
    try k.go(try k.ext(.mm, try std.fmt.bufPrint(&nb, "glm_mm_tile{d}", .{bits})), .{ .x = @intCast(cdiv(chunks, 256)) }, 256, 0, &a);
}

// -- latent.cu and index.cu (kern_attn.zig) ---------------------------------------------------------------------
const attn = @import("kern_attn.zig");
pub const absorbQ = attn.absorbQ;
pub const expandV = attn.expandV;
pub const latentWrite = attn.latentWrite;
pub const denseAttention = attn.denseAttention;
pub const sparseAttention = attn.sparseAttention;
pub const indexUpdate = attn.indexUpdate;
pub const poolBucket = attn.poolBucket;
pub const scores = attn.scores;
pub const selectRows = attn.selectRows;
pub const sparseTokens = attn.sparseTokens;

// -- kda.cu (kern_kda.zig) -----------------------------------------------------------------------------------------
const kda_k = @import("kern_kda.zig");
pub const kdaChainSeg = kda_k.kdaChainSeg;
pub const KdaScratch = kda_k.KdaScratch;
pub const kdaChain = kda_k.kdaChain;
pub const kdaReplayLayers = kda_k.kdaReplayLayers;

// -- the grouped experts (cuda/experts.py, shared with Flash Next's fatbins) ----------------------------------------
pub const promptTile = @import("kern_exl3.zig").promptTile;
pub const promptOldKernel = @import("kern_exl3.zig").promptOld;
pub const promptSplit = @import("kern_exl3.zig").promptSplit;
const moe_k = @import("kern_moe.zig");
pub const maxItems = moe_k.maxItems;
pub const Plan = moe_k.Plan;
pub const expertsPlan = moe_k.expertsPlan;
pub const expertsCall = moe_k.expertsCall;

// -- our small kernels (glm/ops.cu) ----------------------------------------------------------------------------

/// x rows (``copies`` streams each) <- feats rows: image features in place of placeholder embeddings.
pub fn injectRows(k: *K, feats: u64, x: u64, rows: i64, copies: i64) !void {
    var a: cuda.Args = .{};
    a.add(feats);
    a.add(x);
    a.add(@as(i32, @intCast(rows)));
    a.add(@as(i32, @intCast(copies)));
    try k.go(try k.ext(.ops, "glm_inject_rows"), .{ .x = @intCast(@min(4096, cdiv(rows * copies * 4096, 256))) }, 256, 0, &a);
}

pub fn castBf16(k: *K, src: u64, src_stride: i64, dst: u64, dst_stride: i64, rows: i64, cols: i64) !void {
    var a: cuda.Args = .{};
    a.add(src);
    a.add(src_stride);
    a.add(dst);
    a.add(dst_stride);
    a.add(@as(i32, @intCast(cols)));
    try k.go(try k.ext(.ops, "glm_cast_bf16"), .{ .x = @intCast(cdiv(cols, 256)), .y = @intCast(rows) }, 256, 0, &a);
}

/// Rows of ``bytes`` (a multiple of 4) from src to dst, row pitches in bytes (torch's strided copy_).
pub fn copyRows(k: *K, dst: u64, dpitch: usize, src: u64, spitch: usize, bytes: usize, rows: usize) !void {
    if (rows == 0) return;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(@as(i64, @intCast(spitch / 4)));
    a.add(dst);
    a.add(@as(i64, @intCast(dpitch / 4)));
    a.add(@as(i32, @intCast(bytes / 4)));
    try k.go(try k.ext(.ops, "glm_copy_rows"), .{ .x = @intCast(@min(64, cdiv(@intCast(bytes / 4), 256))), .y = @intCast(rows) }, 256, 0, &a);
}

/// fn_rows_top (Flash Next's sample.cu): each bf16 row's first maximum as a global id, its value and log-sum-exp.
pub fn rowsTop(k: *K, logits: u64, cols: i64, id_map: u64, offset: i64, out: u64, rows: i64) !void {
    const f = try k.ext(.sample, "fn_rows_top");
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(@as(i32, @intCast(cols)));
    a.add(@as(i32, @intCast(cols)));
    a.add(id_map);
    a.add(@as(i32, @intCast(offset)));
    a.add(out);
    try k.go(f, .{ .x = @intCast(rows) }, 1024, 0, &a);
}

// -- exl3.cu: EXL3 routed experts (kern_exl3.zig) ---------------------------------------------------------------
const x3 = @import("kern_exl3.zig");
pub const Exl3 = x3.Exl3;
pub const Exl3Scratch = x3.Exl3Scratch;
pub const EXL3_SK = x3.EXL3_SK;
pub const exl3Routed = x3.exl3Routed;
