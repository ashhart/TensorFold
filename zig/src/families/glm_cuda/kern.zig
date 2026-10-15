//! GLM-5.3-Flash on CUDA: every launch the forward makes, on this family's own kernels (glm/mm.cu, rows.cu,
//! latent.cu, index.cu, kda.cu, ops.cu, vision.cu) and the shared grouped experts. Each is written so that a row's
//! bits depend on its own inputs alone: a window's rows, sixteen streams' windows and a prompt chunk's rows go through
//! the same arithmetic. The attention and indexer launches read a step's per-row tables (``Rows``): each row's
//! position and its stream's caches, so one launch serves any mix of streams.

const std = @import("std");
const cuda = @import("cuda");

fn cdiv(a: i64, b: i64) i64 {
    return @divFloor(a + b - 1, b);
}

fn nextPow2(x: u64) u64 {
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
    /// The routed experts' current product is a prompt's: the prefill kernels (resumed == fresh: a prompt row always
    /// takes them, however few rows its chunk has).
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

    /// A kernel of module ``m`` whose (mangled) name contains ``frag``: the one exact match, else the one containing it.
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

    /// A launch the kernel before may overlap (programmatic dependent launch): the kernel waits at
    /// griddepcontrol.wait before it reads what that one writes. TF_GLM_PDL=0: plain launches.
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

/// A step's per-row tables (device arrays): each row's position (i32) and, for each latent cache ``ci``, the row's
/// stream's latent rows, index key and gate rings (offset so that position p is row p) and pooled keys (u64 each, the
/// ``ci``th array at ci * cap). ``at`` moves to a row block.
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

// -- rows.cu -------------------------------------------------------------------------------------------------------------
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

/// conv [n, 3, C] (in place) takes rows keep .. keep + 2 of [conv; proj rows] (proj [n, R, W >= C]); strides in elements.
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

// -- mm.cu: the dense linears ------------------------------------------------------------------------------------------
/// A dense linear [n, k] as glm/mm.cu reads it, tiled at load: ``bits`` 0 (bf16 values) or 4, 6, 8 (symmetric codes
/// with fp16 scales ``s`` per 64 inputs).
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

/// x [m, k] (row stride x_stride) by w -> out [m, n] contiguous, bf16 or unrounded fp32; ``part`` holds sk * m * n fp32
/// when the shape splits K. Rows go 128 at a time when there are more (the same arithmetic for every row).
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

/// The loader's row-major linear (bf16 [n, k], or packed codes and scales) at ``src`` -> its tiled copy at ``dst``
/// (the same bytes; scales ``ss`` -> ``ds``).
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

// -- latent.cu ----------------------------------------------------------------------------------------------------------
pub fn absorbQ(k: *K, q: u64, wk: u64, qa: u64, R: i64) !void {
    var a: cuda.Args = .{};
    a.add(q);
    a.add(wk);
    a.add(qa);
    a.add(@as(i32, @intCast(R)));
    try k.go(try k.ext(.latent, "glm_absorb_q"), .{ .x = @intCast(HL), .y = @intCast(@divExact(KV_LORA, 64)), .z = @intCast(cdiv(R, 16)) }, 64, 0, &a);
}

pub fn expandV(k: *K, ol: u64, wv: u64, out: u64, R: i64) !void {
    var a: cuda.Args = .{};
    a.add(ol);
    a.add(wv);
    a.add(out);
    a.add(@as(i32, @intCast(R)));
    try k.go(try k.ext(.latent, "glm_expand_v"), .{ .x = @intCast(HL), .y = @intCast(@divExact(DV, 8)), .z = @intCast(cdiv(R, 16)) }, 256, 0, &a);
}

/// each row's latent into its stream's cache ``ci`` at its position.
pub fn latentWrite(k: *K, lat: u64, lat_stride: i64, t: Rows, ci: usize, R: i64) !void {
    var a: cuda.Args = .{};
    a.add(lat);
    a.add(lat_stride);
    a.add(t.lcAt(ci));
    a.add(t.posAt());
    try k.go(try k.ext(.latent, "glm_lat_write"), .{ .x = @intCast(R) }, 64, 0, &a);
}

const ATT_SMEM: u32 = 32 * 1024 + 2 * 16 * 1024 + 2 * 32 * 16 * 4 + 32 * 16 * 2 + 3 * 32 * 4;

/// Dense causal attention of qa [R, H, 512] (rows below the dense limit) over ``nch`` 512-key chunks -> out.
pub fn denseAttention(k: *K, qa: u64, t: Rows, ci: usize, po: u64, pm: u64, pl: u64, out: u64, R: i64, nch: i64) !void {
    const f = try k.ext(.latent, "glm_attn_dense");
    try f.allowDynamicShared(ATT_SMEM);
    var a: cuda.Args = .{};
    a.add(qa);
    a.add(t.lcAt(ci));
    a.add(t.posAt());
    a.add(po);
    a.add(pm);
    a.add(pl);
    a.add(@as(i32, @intCast(R)));
    a.add(SCALE);
    try k.go(f, .{ .x = @intCast(R), .y = @intCast(nch) }, 256, ATT_SMEM, &a);
    try merge(k, po, pm, pl, out, t.posAt(), 0, R, nch, false);
}

/// Attention of the rows with counts > 0 over their selected tokens, written into out (others untouched).
pub fn sparseAttention(k: *K, qa: u64, t: Rows, ci: usize, tokens: u64, counts: u64, po: u64, pm: u64, pl: u64, out: u64, R: i64) !void {
    const f = try k.ext(.latent, "glm_attn_sparse");
    try f.allowDynamicShared(ATT_SMEM);
    var a: cuda.Args = .{};
    a.add(qa);
    a.add(t.lcAt(ci));
    a.add(tokens);
    a.add(counts);
    a.add(po);
    a.add(pm);
    a.add(pl);
    a.add(@as(i32, @intCast(R)));
    a.add(SCALE);
    try k.go(f, .{ .x = @intCast(R), .y = @intCast(SPARSE_CHUNKS) }, 256, ATT_SMEM, &a);
    try merge(k, po, pm, pl, out, 0, counts, R, SPARSE_CHUNKS, true);
}

fn merge(k: *K, po: u64, pm: u64, pl: u64, out: u64, pos: u64, counts: u64, R: i64, nch: i64, sparse: bool) !void {
    var a: cuda.Args = .{};
    a.add(po);
    a.add(pm);
    a.add(pl);
    a.add(out);
    a.add(pos);
    a.add(counts);
    a.add(@as(i32, @intCast(R)));
    a.add(@as(i32, @intCast(nch)));
    a.add(@as(i32, @intFromBool(sparse)));
    try k.go(try k.ext(.latent, "glm_attn_merge"), .{ .x = @intCast(R), .y = @intCast(HL) }, 128, 0, &a);
}

// -- index.cu: the DSA indexer -------------------------------------------------------------------------------------------
/// The rows' index keys (LayerNorm) and gates into their streams' rings for cache ``ci``, then the pools they complete.
pub fn indexUpdate(k: *K, kr: u64, k_stride: i64, gate: u64, ln_w: u64, ln_b: u64, ape: u64, t: Rows, ci: usize, R: i64) !void {
    var a: cuda.Args = .{};
    a.add(kr);
    a.add(k_stride);
    a.add(gate);
    a.add(ln_w);
    a.add(ln_b);
    a.add(t.ikAt(ci));
    a.add(t.igAt(ci));
    a.add(t.posAt());
    a.add(@as(f32, 1e-6));
    try k.go(try k.ext(.index, "glm_index_write"), .{ .x = @intCast(R) }, 128, 0, &a);
    var b: cuda.Args = .{};
    b.add(t.ikAt(ci));
    b.add(t.igAt(ci));
    b.add(ape);
    b.add(t.pkAt(ci));
    b.add(t.posAt());
    try k.go(try k.ext(.index, "glm_pool_keys"), .{ .x = @intCast(R) }, 128, 0, &b);
}

/// Pools to score for rows up to position ``last`` (the visible ones rounded up to a power of two, at least 1024, at
/// most ``np_max``).
pub fn poolBucket(last: i64, np_max: i64) i64 {
    const visible = @divFloor(last + 1, POOL) + 1;
    return @min(np_max, @max(1024, @as(i64, @intCast(nextPow2(@intCast(visible))))));
}

/// Each of n rows' scores over np pools -> scores [n, np] fp32.
pub fn scores(k: *K, qi: u64, w: u64, w_stride: i64, t: Rows, ci: usize, out: u64, n: i64, np: i64) !void {
    var a: cuda.Args = .{};
    a.add(qi);
    a.add(w);
    a.add(w_stride);
    a.add(t.pkAt(ci));
    a.add(out);
    a.add(t.posAt());
    a.add(@as(i32, @intCast(np)));
    a.add(@as(f32, 0.08838834764831845));
    a.add(@as(f32, 1.0 / 5.656854249492381));
    try k.go(try k.ext(.index, "glm_index_scores"), .{ .x = @intCast(n), .y = @intCast(cdiv(np, 64)) }, 128, 0, &a);
}

/// Each row's 512 best visible pools, ascending (int64).
pub fn selectRows(k: *K, sc: u64, out: u64, np: i64, t: Rows, n: i64) !void {
    var a: cuda.Args = .{};
    a.add(sc);
    a.add(out);
    a.add(@as(i32, @intCast(np)));
    a.add(t.posAt());
    try k.go(try k.ext(.index, "glm_index_select"), .{ .x = @intCast(n) }, 1024, 0, &a);
}

/// Each row's attended tokens and their count (0: a dense row).
pub fn sparseTokens(k: *K, pools: u64, t: Rows, rows: i64, tokens: u64, counts: u64) !void {
    var a: cuda.Args = .{};
    a.add(pools);
    a.add(t.posAt());
    a.add(tokens);
    a.add(counts);
    try k.go(try k.ext(.index, "glm_index_tokens"), .{ .x = @intCast(rows) }, 256, 0, &a);
}

/// A batched step's KDA layer ``li``: every stream's window rows through its own states (segs: [n] device KdaSeg).
pub fn kdaChainSeg(k: *K, segs: u64, nseg: i64, li: usize, p: u64, p_stride: i64, b_off: i64, a_: u64, a_stride: i64, g: u64, g_stride: i64, cw: u64, a_log: u64, dt_bias: u64, norm_w: u64, out: u64, max_rows: i64, proj_w: i64) !void {
    var a: cuda.Args = .{};
    a.add(segs);
    a.add(@as(i32, @intCast(li)));
    a.add(@as(i32, @intCast(LL)));
    a.add(p);
    a.add(@as(i32, @intCast(p_stride)));
    a.add(@as(i32, @intCast(b_off)));
    a.add(a_);
    a.add(@as(i32, @intCast(a_stride)));
    a.add(g);
    a.add(@as(i32, @intCast(g_stride)));
    a.add(cw);
    a.add(a_log);
    a.add(dt_bias);
    a.add(norm_w);
    a.add(EPS);
    a.add(LOWER);
    a.add(out);
    a.add(@as(i32, @intCast(max_rows)));
    a.add(@as(i32, @intCast(proj_w)));
    try k.go(try k.ext(.kda, "glm_kda_chain_seg"), .{ .x = @intCast(LL), .y = @intCast(nseg) }, 1024, 0, &a);
}

// -- kda.py: the KDA extension (glm/kda.cu) --------------------------------------------------------------------------
pub const KdaScratch = struct { out: u64, k: u64, v: u64, g: u64, b: u64 };

/// kda.chain: projection rows p [q | k | v | f_a | g_a | b at b_off], bf16 gate rows a and g, the conv window and
/// state -> out rows, state_out, and the replay inputs; windows of 64 rows or more take the three-kernel path.
pub fn kdaChain(k: *K, p: u64, p_stride: i64, b_off: i64, a: u64, a_stride: i64, g: u64, g_stride: i64, cs: u64, cw: u64, state_in: u64, a_log: u64, dt_bias: u64, norm_w: u64, rows: i64, sc: KdaScratch, state_out: u64, q_tmp: u64, y_tmp: u64) !void {
    const H: i32 = @intCast(LL);
    if (rows >= 64) {
        var pa: cuda.Args = .{};
        pa.add(H);
        pa.add(p);
        pa.add(@as(i32, @intCast(p_stride)));
        pa.add(@as(i32, @intCast(b_off)));
        pa.add(a);
        pa.add(@as(i32, @intCast(a_stride)));
        pa.add(cs);
        pa.add(cw);
        pa.add(a_log);
        pa.add(dt_bias);
        pa.add(LOWER);
        pa.add(q_tmp);
        pa.add(sc.k);
        pa.add(sc.v);
        pa.add(sc.g);
        pa.add(sc.b);
        try k.go(try k.ext(.kda, "11prep_kernel"), .{ .x = @intCast(rows), .y = @intCast(H) }, 512, 0, &pa);
        var sa: cuda.Args = .{};
        sa.add(H);
        sa.add(state_in);
        sa.add(q_tmp);
        sa.add(sc.k);
        sa.add(sc.v);
        sa.add(sc.g);
        sa.add(sc.b);
        sa.add(@as(i32, @intCast(rows)));
        sa.add(y_tmp);
        sa.add(state_out);
        try k.go(try k.ext(.kda, "11step_kernelILi4ELi16E"), .{ .x = @intCast(H), .y = 128 / 4 / 4 }, 4 * 32, 0, &sa);
        var oa: cuda.Args = .{};
        oa.add(H);
        oa.add(y_tmp);
        oa.add(g);
        oa.add(@as(i32, @intCast(g_stride)));
        oa.add(norm_w);
        oa.add(EPS);
        oa.add(sc.out);
        try k.go(try k.ext(.kda, "10out_kernel"), .{ .x = @intCast(rows), .y = @intCast(H) }, 128, 0, &oa);
        return;
    }
    var ca: cuda.Args = .{};
    ca.add(H);
    ca.add(p);
    ca.add(@as(i32, @intCast(p_stride)));
    ca.add(@as(i32, @intCast(b_off)));
    ca.add(a);
    ca.add(@as(i32, @intCast(a_stride)));
    ca.add(g);
    ca.add(@as(i32, @intCast(g_stride)));
    ca.add(cs);
    ca.add(cw);
    ca.add(state_in);
    ca.add(a_log);
    ca.add(dt_bias);
    ca.add(norm_w);
    ca.add(EPS);
    ca.add(LOWER);
    ca.add(@as(i32, @intCast(rows)));
    ca.add(sc.out);
    ca.add(state_out);
    ca.add(sc.k);
    ca.add(sc.v);
    ca.add(sc.g);
    ca.add(sc.b);
    try k.go(try k.ext(.kda, "12chain_kernel"), .{ .x = @intCast(H) }, 1024, 0, &ca);
}

/// kda.replay_layers: every layer's state after its first ``rows`` rows (states [layers, H, 128, 128]).
pub fn kdaReplayLayers(k: *K, state_in: u64, sk: u64, sv: u64, sg: u64, sb: u64, layers: i64, rows_cap: i64, rows: i64, state_out: u64) !void {
    const H: i32 = @intCast(LL);
    var a: cuda.Args = .{};
    a.add(H);
    a.add(state_in);
    a.add(@as(u64, @intCast(LL * 128 * 128)));
    a.add(sk);
    a.add(sv);
    a.add(sg);
    a.add(sb);
    a.add(@as(u64, @intCast(rows_cap * LL * 128)));
    a.add(@as(u64, @intCast(rows_cap * LL)));
    a.add(@as(i32, @intCast(rows)));
    a.add(state_out);
    try k.go(try k.ext(.kda, "20replay_layers_kernel"), .{ .x = @intCast(layers * LL) }, 1024, 0, &a);
}

// -- the grouped experts (cuda/experts.py, shared with Flash Next's fatbins) ----------------------------------------
pub const promptTile = @import("kern_exl3.zig").promptTile;
pub const promptOldKernel = @import("kern_exl3.zig").promptOld;
pub const promptSplit = @import("kern_exl3.zig").promptSplit;

pub fn maxItems(pairs: i64, experts: i64, tile: i64) i64 {
    return @min(pairs, experts) + @divFloor(pairs, tile);
}

pub const Plan = struct { members: u64, items: u64, counts: u64, rank: u64, hist: u64, tile: i64 };

/// experts_v7.plan: picks [R, 8] (the routed slots alone) -> the plan over the 288 experts.
pub fn expertsPlan(k: *K, picks: u64, pairs: i64, plan: Plan) !void {
    var a: cuda.Args = .{};
    const E: i32 = @intCast(EXPERTS);
    if (pairs <= 1024) {
        a.add(picks);
        a.add(@as(i32, @intCast(pairs)));
        a.add(E);
        a.add(@as(i32, @intCast(plan.tile)));
        a.add(plan.members);
        a.add(plan.items);
        a.add(plan.counts);
        try k.go(try k.ext(.experts, "11plan_kernelE"), .{ .x = 1 }, 1024, 0, &a);
    } else {
        const nblk = cdiv(pairs, 1024);
        a.add(picks);
        a.add(@as(i32, @intCast(pairs)));
        a.add(E);
        a.add(plan.rank);
        a.add(plan.hist);
        try k.go(try k.ext(.experts, "9plan_rankE"), .{ .x = @intCast(nblk) }, 1024, 0, &a);
        var b: cuda.Args = .{};
        b.add(@as(i32, @intCast(nblk)));
        b.add(E);
        b.add(@as(i32, @intCast(plan.tile)));
        b.add(plan.hist);
        b.add(plan.items);
        b.add(plan.counts);
        try k.go(try k.ext(.experts, "12plan_offsetsE"), .{ .x = 1 }, 1024, 0, &b);
        var s: cuda.Args = .{};
        s.add(picks);
        s.add(@as(i32, @intCast(pairs)));
        s.add(E);
        s.add(plan.rank);
        s.add(plan.hist);
        s.add(plan.members);
        try k.go(try k.ext(.experts, "12plan_scatterE"), .{ .x = @intCast(cdiv(pairs, 256)) }, 256, 0, &s);
    }
}

/// experts_v7.prefill / run: gate-up (epi 2) or down (epi 3 bf16 out / 0 fp32 out) over the plan; ``slots``: the
/// pairs a row of x feeds (0 for down: one pair a row).
pub fn expertsCall(k: *K, prefill: bool, epi: i64, x: u64, x_stride: i64, slots: i64, w: u64, kb: i64, nb: i64, plan: Plan, out: u64, width: i64, items: i64, limit: f32) !void {
    const gs: i64 = 64;
    const m: i64 = if (epi == 2) 2 else 1;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i32, @intCast(x_stride)));
    a.add(@as(i32, @intCast(slots)));
    a.add(w);
    a.add(@as(i32, @intCast(kb)));
    a.add(@as(i32, @intCast(nb)));
    a.add(plan.items);
    a.add(plan.counts);
    a.add(plan.members);
    a.add(out);
    a.add(@as(i32, @intCast(width)));
    a.add(limit);
    var buf: [128]u8 = undefined;
    const sbv: i64 = if (k.experts_sym) 1 else 2; // a quad's scale (and bias) uint4s a group
    if (prefill) {
        const f = try k.ext(if (k.experts3) .experts_sym3_prefill else if (k.experts_sym) .experts_sym_prefill else .experts_prefill, try std.fmt.bufPrint(&buf, "14prefill_kernelILi{d}ELi{d}ELi{d}ELi2ELi2ELi4E", .{ gs, m, epi }));
        const block: i64 = @as(i64, if (k.experts3) 24 else 32) * @divExact(4 * gs, 128) + 4 * sbv;
        const su: i64 = 64 * 8 + 4 * @divExact(64, gs) * m * block;
        const sb = su * 16;
        const stages: i64 = if (sb * 4 <= 49152) 4 else if (sb * 3 <= 49152) 3 else 2;
        const smem: u32 = @intCast(stages * sb);
        try f.allowDynamicShared(smem);
        const grid = items * cdiv(nb, 4);
        if (grid >= 1) try k.go(f, .{ .x = @intCast(grid) }, 256, smem, &a);
    } else {
        const f = try k.ext(if (k.experts3) .experts_sym3 else if (k.experts_sym) .experts_sym else .experts, try std.fmt.bufPrint(&buf, "13expert_kernelILi{d}ELi{d}ELi{d}ELi2ELi4E", .{ gs, m, epi }));
        const per_sm: i64 = @max(1, try f.occupancy(128, 0));
        const grid = @min(cdiv(items, 4), per_sm * k.sms);
        if (grid >= 1) try k.go(f, .{ .x = @intCast(grid) }, 128, 0, &a);
    }
}

// -- our small kernels (glm/ops.cu) ---------------------------------------------------------------------------------
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

// -- exl3.cu: EXL3 routed experts (kern_exl3.zig) -----------------------------------------------------------------------
const x3 = @import("kern_exl3.zig");
pub const Exl3 = x3.Exl3;
pub const Exl3Scratch = x3.Exl3Scratch;
pub const EXL3_SK = x3.EXL3_SK;
pub const exl3Routed = x3.exl3Routed;
