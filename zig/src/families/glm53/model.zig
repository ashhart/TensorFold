//! Full GLM-5.3 (glm_moe_dsa, orcarouter MLX 6-bit): shapes, a rank's split, and its weights read by parallel preads.
const std = @import("std");
const mtl = @import("metal");

pub const D = 6144;
pub const QL = 2048; // q_lora
pub const KVL = 512; // kv_lora
pub const ROPE = 64;
pub const NOPE = 192;
pub const VD = 256; // v head dim
pub const QH = NOPE + ROPE; // 256: a query head in q_b's output
pub const H = 64;
pub const IH = 32; // indexer heads
pub const ID = 128; // indexer head dim
pub const E = 256;
pub const TOP = 8;
pub const MI = 2048; // moe intermediate
pub const DI = 12288; // dense intermediate
pub const V = 154880;
pub const LAYERS = 78;
pub const MTP = LAYERS; // the MTP draft head's decoder layer (index 78, a full-indexer MoE layer)
pub const DENSE = 3;
pub const KEYS = 2048; // index_topk
pub const CROW = KVL + ROPE; // 576 floats a latent cache row
pub const LOG2_THETA: f32 = @floatCast(std.math.log2(@as(f64, 8000000.0)));

/// Which layers carry their own indexer ("full"); the rest reuse the last full layer's picks.
pub fn fullIndexer(i: usize) bool {
    return i < 3 or (i % 4 == 2);
}

/// G53_LSPLIT=1 = the layer-split latent cache. Layer i's latent rows (and its attention, all 64 heads)
/// live on rank i % ranks only; the indexer keys stay on every rank. Each rank then holds about a quarter of the latent
/// cache (2M positions: ~46 GB of latent + ~11 GB of indexer keys a rank instead of ~190 GB).
pub fn lsplitOn() bool {
    return if (std.c.getenv("G53_LSPLIT")) |v| v[0] == '1' else false;
}

/// The rank that owns layer i under G53_LSPLIT (shares are equal head ranges: rank = heads[0] / heads a share): i % ranks, or
/// G53_LSPLIT_OWNER=<rank> = every layer on that rank (put it on the Mac with the most spare memory: the same per-layer cost
/// as the round-robin split, and with a 512 GB owner beside three 256 GB Macs a ~3.1M cap instead of ~2.0M).
pub fn lsOwnerRank(i: usize, ranks: usize) usize {
    if (std.c.getenv("G53_LSPLIT_OWNER")) |v| {
        const r = std.fmt.parseInt(usize, std.mem.span(v), 10) catch return i % ranks;
        if (r < ranks) return r;
    }
    return i % ranks;
}

/// Does share `sh` own layer i (one Mac with every head: always)?
pub fn lsOwns(i: usize, sh: Share) bool {
    const nh = sh.heads[1] - sh.heads[0];
    if (nh >= H) return true;
    return lsOwnerRank(i, H / nh) == sh.heads[0] / nh;
}

/// Part `i` of `parts` of `units` (whole units; the first parts take one more).
pub fn part(units: usize, parts: usize, i: usize) [2]usize {
    const base = units / parts;
    const extra = units % parts;
    const lo = i * base + @min(i, extra);
    return .{ lo, lo + base + @as(usize, @intFromBool(i < extra)) };
}

/// A rank's (or a canonical slice's) share: heads, intermediate rows (dense and MoE, whole groups of 64), vocab rows.
pub const Share = struct {
    heads: [2]usize,
    dense: [2]usize, // dense intermediate rows
    moe: [2]usize, // moe intermediate rows
    vocab: [2]usize,

    pub fn of(i: usize, n: usize) Share {
        const d = part(DI / 64, n, i);
        const m = part(MI / 64, n, i);
        return .{ .heads = part(H, n, i), .dense = .{ d[0] * 64, d[1] * 64 }, .moe = .{ m[0] * 64, m[1] * 64 }, .vocab = part(V, n, i) };
    }

    pub fn whole() Share {
        return .{ .heads = .{ 0, H }, .dense = .{ 0, DI }, .moe = .{ 0, MI }, .vocab = .{ 0, V } };
    }
};

pub const Ref = struct {
    buf: mtl.Buffer,
    off: usize = 0,

    pub fn at(r: Ref, bytes: usize) Ref {
        return .{ .buf = r.buf, .off = r.off + bytes };
    }
    pub fn addr(r: Ref) [*]u8 {
        return r.buf.contents() + r.off;
    }
};

/// An affine-quantized matrix (or a stack): n rows of k values, `bits` wide in groups of 64, f16 scales and biases.
pub const Q = struct {
    w: Ref,
    s: Ref,
    b: Ref,
    bits: u8,
    n: u32,
    k: u32,

    pub fn ldw(q: Q) u32 {
        return q.k * q.bits / 8;
    }
    pub fn lds(q: Q) u32 {
        return q.k / 64;
    }
};

fn qBytes(n: usize, k: usize, bits: usize) [2]usize {
    return .{ n * k * bits / 8, n * (k / 64) * 2 };
}

pub const Attn = struct {
    q_a: Q,
    q_a_norm: Ref,
    kv_a: Q,
    kv_a_norm: Ref,
    q_b: Q, // this share's heads' rows
    embed_q: Q, // [heads][512][192]
    unembed: Q, // [heads][256][512]
    o_proj: Q, // [6144][heads*256] (this share's columns)
    // indexer (full layers only)
    wq_b: ?Ref = null, // bf16 [4096][2048]
    wk: ?Ref = null, // bf16 [128][6144]
    k_norm_w: ?Ref = null,
    k_norm_b: ?Ref = null,
    wproj: ?Ref = null, // bf16 [32][6144]
    // G53_LSPLIT=1: the layer's owner rank also holds every head's q_b / embed_q / unembed (the same bytes
    // the other ranks hold for their heads) and computes the layer's attention for all 64 heads
    q_b_all: ?Q = null,
    embed_q_all: ?Q = null,
    unembed_all: ?Q = null,
};

pub const Mlp = union(enum) {
    dense: struct { gate: Q, up: Q, down: Q },
    moe: struct { router: Ref, bias: Ref, gate: Q, up: Q, down: Q, sh_gate: Q, sh_up: Q, sh_down: Q },
};

pub const Layer = struct {
    in_norm: Ref,
    post_norm: Ref,
    attn: Attn,
    mlp: Mlp,
};

/// The MTP head's extras around its decoder layer: x = eh_proj([enorm(embed(next)), hnorm(h)]), out norm before lm_head.
pub const Mtp = struct {
    eh_proj: Ref, // bf16 [D][2D]
    enorm: Ref,
    hnorm: Ref,
    norm: Ref, // shared_head.norm
};

pub const Weights = struct {
    layers: [LAYERS + 1]Layer = undefined, // [MTP] is the draft head's layer when `mtp` is loaded
    mtp: ?Mtp = null,
    first: usize,
    last: usize, // layers [first, last)
    embed: Ref = undefined, // bf16 [V][D]
    norm: Ref = undefined,
    head: Ref = undefined, // bf16 [vocab share][D]
    share: Share,
    buffers: std.ArrayList(mtl.Buffer) = .empty,
    bytes: usize = 0,
    gpa: std.mem.Allocator,
    resident: ?mtl.ResidencySet = null, // every weight buffer, wired as it is allocated (the queue keeps it resident)

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |b| b.deinit();
        w.buffers.deinit(w.gpa);
    }
};

/// Bytes of layer i's dense buffer for share `sh` (everything but routed experts), rounded up generously.
fn layerBytes(i: usize, sh: Share) usize {
    const nh = sh.heads[1] - sh.heads[0];
    const q = struct {
        fn b(n: usize, k: usize, bits: usize) usize {
            const x = qBytes(n, k, bits);
            return x[0] + 2 * x[1] + 3 * 256;
        }
    };
    var n: usize = 64 * 1024 + 2 * D * 2 + QL * 2 + KVL * 2;
    n += q.b(QL, D, 8) + q.b(CROW, D, 8) + q.b(nh * QH, QL, 8) + nh * (q.b(KVL, NOPE, 8) + q.b(VD, KVL, 8)) + q.b(D, nh * VD, 8);
    if (fullIndexer(i)) n += (IH * ID * QL + ID * D + 2 * ID + IH * D) * 2 + 5 * 256;
    if (i < DENSE) {
        const r = sh.dense[1] - sh.dense[0];
        n += 2 * q.b(r, D, 6) + q.b(D, r, 6);
    } else {
        const r = sh.moe[1] - sh.moe[0];
        n += E * D * 2 + E * 4 + 2 * q.b(r, D, 8) + q.b(D, r, 8) + 2 * 256;
    }
    return n;
}

/// A Metal buffer over anonymous host memory (mmap, no copy): Metal-allocated buffers past ~110 GB got the process
/// killed during load on a 256 GB M3 Ultra; plain VM memory wrapped for the GPU does not count against that limit.
extern "c" fn mlock(addr: *const anyopaque, len: usize) c_int;

pub fn hostBuffer(device: mtl.Device, len: usize) !mtl.Buffer {
    const page = 16384;
    const n = std.mem.alignForward(usize, @max(len, page), page);
    const mem = std.posix.mmap(null, n, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0) catch return error.NoBuffer;
    // wired: the compressor never takes it, so the kernel's low-swap kill (1 GB of swap) never picks this process
    if (mlock(mem.ptr, n) != 0) std.log.warn("glm53: mlock of {d} bytes failed (vm.user_wire_limit?)", .{n});
    return device.bufferNoCopy(@ptrCast(mem.ptr), n, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
}

const Src = struct { shard: u32, off: u64, len: u64, dtype: []const u8, shape: [4]usize, rank: u8 };

/// A read: `len` bytes at `off` of a shard into `dst`, or (rows > 0) `rows` strided rows: each row's [col_off, +col_len)
/// of a stored row `row_len` bytes long, packed.
const Job = struct { shard: u32, off: u64, len: u64, dst: [*]u8, rows: u64 = 0, row_len: u64 = 0, col_off: u64 = 0, col_len: u64 = 0 };

const Loader = struct {
    gpa: std.mem.Allocator,
    arena: std.mem.Allocator,
    device: mtl.Device,
    w: *Weights,
    shards: std.ArrayList([:0]const u8) = .empty,
    names: std.StringHashMapUnmanaged(Src) = .empty,
    jobs: std.ArrayList(Job) = .empty,
    cur: ?mtl.Buffer = null,
    cur_len: usize = 0,
    cur_at: usize = 0,

    fn src(l: *Loader, comptime fmt: []const u8, args: anytype) !Src {
        var name: [256]u8 = undefined;
        const full = try std.fmt.bufPrint(&name, fmt, args);
        return l.names.get(full) orelse {
            std.log.err("glm53: no tensor {s}", .{full});
            return error.MissingTensor;
        };
    }

    fn begin(l: *Loader, len: usize) !void {
        const b = try hostBuffer(l.device, len);
        try l.w.buffers.append(l.gpa, b);
        // wire it now: unwired weights get compressed under pressure, and with 1 GB of swap the kernel then kills us
        if (l.w.resident) |set| {
            set.add(b);
            set.commit();
            set.requestResidency();
        }
        l.cur = b;
        l.cur_len = len;
        l.cur_at = 0;
        l.w.bytes += len;
    }

    fn take(l: *Loader, len: usize) Ref {
        const at = l.cur_at;
        l.cur_at = std.mem.alignForward(usize, at + len, 256);
        if (l.cur_at > l.cur_len) std.debug.panic("glm53: buffer of {d} too small for {d}", .{ l.cur_len, l.cur_at });
        return .{ .buf = l.cur.?, .off = at };
    }

    fn check(s: Src, dtype: []const u8, shape: []const usize, what: []const u8) !void {
        if (std.mem.eql(u8, s.dtype, dtype) and s.rank == shape.len and std.mem.eql(usize, s.shape[0..s.rank], shape)) return;
        std.log.err("glm53: {s} is {s} {any}, expected {s} {any}", .{ what, s.dtype, s.shape[0..s.rank], dtype, shape });
        return error.UnexpectedTensor;
    }

    /// A whole tensor as stored.
    fn plain(l: *Loader, comptime fmt: []const u8, args: anytype, dtype: []const u8, shape: []const usize) !Ref {
        const s = try l.src(fmt, args);
        try check(s, dtype, shape, fmt);
        const r = l.take(s.len);
        try l.jobs.append(l.gpa, .{ .shard = s.shard, .off = s.off, .len = s.len, .dst = r.addr() });
        return r;
    }

    /// Rows [r0, r1) of a stored [n, k] matrix (bf16 or quantized parts), contiguous on disk.
    fn rowsOf(l: *Loader, s: Src, row_bytes: usize, r0: usize, r1: usize, dst: Ref) !void {
        try l.jobs.append(l.gpa, .{ .shard = s.shard, .off = s.off + r0 * row_bytes, .len = (r1 - r0) * row_bytes, .dst = dst.addr() });
    }

    /// Every row's byte columns [c0, c0 + clen) of a stored matrix with rows `row_bytes` long, rows [r0, r1).
    fn colsOf(l: *Loader, s: Src, row_bytes: usize, r0: usize, r1: usize, c0: usize, clen: usize, dst: Ref) !void {
        if (c0 == 0 and clen == row_bytes) return l.rowsOf(s, row_bytes, r0, r1, dst);
        try l.jobs.append(l.gpa, .{ .shard = s.shard, .off = s.off + r0 * row_bytes, .len = (r1 - r0) * row_bytes, .dst = dst.addr(), .rows = r1 - r0, .row_len = row_bytes, .col_off = c0, .col_len = clen });
    }

    /// A quantized [n, k] matrix `base` (bits), keeping rows [r0, r1) and input columns [k0, k1) (whole groups).
    fn quant(l: *Loader, base: []const u8, bits: u8, n: usize, k: usize, r0: usize, r1: usize, k0: usize, k1: usize) !Q {
        var nm: [256]u8 = undefined;
        const ws = try l.src("{s}.weight", .{base});
        try check(ws, "U32", &.{ n, k * bits / 32 }, try std.fmt.bufPrint(&nm, "{s}.weight", .{base}));
        const ss = try l.src("{s}.scales", .{base});
        try check(ss, "F16", &.{ n, k / 64 }, try std.fmt.bufPrint(&nm, "{s}.scales", .{base}));
        const bs = try l.src("{s}.biases", .{base});
        try check(bs, "F16", &.{ n, k / 64 }, try std.fmt.bufPrint(&nm, "{s}.biases", .{base}));
        const rows = r1 - r0;
        const kk = k1 - k0;
        const sz = qBytes(rows, kk, bits);
        const q: Q = .{ .w = l.take(sz[0]), .s = l.take(sz[1]), .b = l.take(sz[1]), .bits = bits, .n = @intCast(rows), .k = @intCast(kk) };
        try l.colsOf(ws, k * bits / 8, r0, r1, k0 * bits / 8, kk * bits / 8, q.w);
        try l.colsOf(ss, k / 64 * 2, r0, r1, k0 / 64 * 2, kk / 64 * 2, q.s);
        try l.colsOf(bs, k / 64 * 2, r0, r1, k0 / 64 * 2, kk / 64 * 2, q.b);
        return q;
    }

    /// Experts 0..255 of `proj` stacked [256][rows][k-cols] (one take per part, experts back to back).
    fn experts(l: *Loader, i: usize, proj: []const u8, bits: u8, n: usize, k: usize, r0: usize, r1: usize, k0: usize, k1: usize) !Q {
        const rows = r1 - r0;
        const kk = k1 - k0;
        const sz = qBytes(rows, kk, bits);
        const q: Q = .{ .w = l.take(E * sz[0]), .s = l.take(E * sz[1]), .b = l.take(E * sz[1]), .bits = bits, .n = @intCast(rows), .k = @intCast(kk) };
        for (0..E) |e| {
            var nm: [256]u8 = undefined;
            const base = try std.fmt.bufPrint(&nm, "model.layers.{d}.mlp.experts.{d}.{s}", .{ i, e, proj });
            const ws = try l.src("{s}.weight", .{base});
            try check(ws, "U32", &.{ n, k * bits / 32 }, base);
            const ss = try l.src("{s}.scales", .{base});
            const bs = try l.src("{s}.biases", .{base});
            try l.colsOf(ws, k * bits / 8, r0, r1, k0 * bits / 8, kk * bits / 8, q.w.at(e * sz[0]));
            try l.colsOf(ss, k / 64 * 2, r0, r1, k0 / 64 * 2, kk / 64 * 2, q.s.at(e * sz[1]));
            try l.colsOf(bs, k / 64 * 2, r0, r1, k0 / 64 * 2, kk / 64 * 2, q.b.at(e * sz[1]));
        }
        return q;
    }

    fn layer(l: *Loader, i: usize, sh: Share) !Layer {
        const hd = sh.heads;
        var name: [128]u8 = undefined;
        const a = try std.fmt.bufPrint(&name, "model.layers.{d}.self_attn.", .{i});
        var n2: [160]u8 = undefined;
        // the dense buffer: attention, norms, router, shared expert or dense MLP
        try l.begin(layerBytes(i, sh));
        var out: Layer = .{
            .in_norm = try l.plain("model.layers.{d}.input_layernorm.weight", .{i}, "BF16", &.{D}),
            .post_norm = try l.plain("model.layers.{d}.post_attention_layernorm.weight", .{i}, "BF16", &.{D}),
            .attn = .{
                .q_a = try l.quant(try std.fmt.bufPrint(&n2, "{s}q_a_proj", .{a}), 8, QL, D, 0, QL, 0, D),
                .q_a_norm = try l.plain("model.layers.{d}.self_attn.q_a_layernorm.weight", .{i}, "BF16", &.{QL}),
                .kv_a = try l.quant(try std.fmt.bufPrint(&n2, "{s}kv_a_proj_with_mqa", .{a}), 8, CROW, D, 0, CROW, 0, D),
                .kv_a_norm = try l.plain("model.layers.{d}.self_attn.kv_a_layernorm.weight", .{i}, "BF16", &.{KVL}),
                .q_b = try l.quant(try std.fmt.bufPrint(&n2, "{s}q_b_proj", .{a}), 8, H * QH, QL, hd[0] * QH, hd[1] * QH, 0, QL),
                .embed_q = try l.packHeads(i, "embed_q", KVL, NOPE, hd),
                .unembed = try l.packHeads(i, "unembed_out", VD, KVL, hd),
                .o_proj = try l.quant(try std.fmt.bufPrint(&n2, "{s}o_proj", .{a}), 8, D, H * VD, 0, D, hd[0] * VD, hd[1] * VD),
            },
            .mlp = undefined,
        };
        if (fullIndexer(i)) {
            out.attn.wq_b = try l.plain("model.layers.{d}.self_attn.indexer.wq_b.weight", .{i}, "BF16", &.{ IH * ID, QL });
            out.attn.wk = try l.plain("model.layers.{d}.self_attn.indexer.wk.weight", .{i}, "BF16", &.{ ID, D });
            out.attn.k_norm_w = try l.plain("model.layers.{d}.self_attn.indexer.k_norm.weight", .{i}, "BF16", &.{ID});
            out.attn.k_norm_b = try l.plain("model.layers.{d}.self_attn.indexer.k_norm.bias", .{i}, "BF16", &.{ID});
            out.attn.wproj = try l.plain("model.layers.{d}.self_attn.indexer.weights_proj.weight", .{i}, "BF16", &.{ IH, D });
        }
        var m: [128]u8 = undefined;
        const mp = try std.fmt.bufPrint(&m, "model.layers.{d}.mlp.", .{i});
        if (i < DENSE) {
            const dr = sh.dense;
            out.mlp = .{ .dense = .{
                .gate = try l.quant(try std.fmt.bufPrint(&n2, "{s}gate_proj", .{mp}), 6, DI, D, dr[0], dr[1], 0, D),
                .up = try l.quant(try std.fmt.bufPrint(&n2, "{s}up_proj", .{mp}), 6, DI, D, dr[0], dr[1], 0, D),
                .down = try l.quant(try std.fmt.bufPrint(&n2, "{s}down_proj", .{mp}), 6, D, DI, 0, D, dr[0], dr[1]),
            } };
        } else {
            const mr = sh.moe;
            var moe: @FieldType(Mlp, "moe") = .{
                .router = try l.plain("model.layers.{d}.mlp.gate.weight", .{i}, "BF16", &.{ E, D }),
                .bias = try l.plain("model.layers.{d}.mlp.gate.e_score_correction_bias", .{i}, "F32", &.{E}),
                .sh_gate = try l.quant(try std.fmt.bufPrint(&n2, "{s}shared_experts.gate_proj", .{mp}), 8, MI, D, mr[0], mr[1], 0, D),
                .sh_up = try l.quant(try std.fmt.bufPrint(&n2, "{s}shared_experts.up_proj", .{mp}), 8, MI, D, mr[0], mr[1], 0, D),
                .sh_down = try l.quant(try std.fmt.bufPrint(&n2, "{s}shared_experts.down_proj", .{mp}), 8, D, MI, 0, D, mr[0], mr[1]),
                .gate = undefined,
                .up = undefined,
                .down = undefined,
            };
            const rows = mr[1] - mr[0];
            const per = qBytes(rows, D, 6)[0] + 2 * qBytes(rows, D, 6)[1];
            const per_d = qBytes(D, rows, 8)[0] + 2 * qBytes(D, rows, 8)[1];
            try l.begin(E * (2 * per + per_d) + 16 * 256);
            moe.gate = try l.experts(i, "gate_proj", 6, MI, D, mr[0], mr[1], 0, D);
            moe.up = try l.experts(i, "up_proj", 6, MI, D, mr[0], mr[1], 0, D);
            moe.down = try l.experts(i, "down_proj", 8, D, MI, 0, D, mr[0], mr[1]);
            out.mlp = .{ .moe = moe };
        }
        if (lsplitOn() and hd[1] - hd[0] < H and lsOwns(i, sh)) { // the owner's every-head attention weights (own buffer, after every take from the layer's buffers)
            const qb = qBytes(H * QH, QL, 8);
            const eq = qBytes(KVL, NOPE, 8);
            const ue = qBytes(VD, KVL, 8);
            try l.begin(qb[0] + 2 * qb[1] + H * (eq[0] + 2 * eq[1] + ue[0] + 2 * ue[1]) + 16 * 256);
            const every = [2]usize{ 0, H };
            out.attn.q_b_all = try l.quant(try std.fmt.bufPrint(&n2, "{s}q_b_proj", .{a}), 8, H * QH, QL, 0, H * QH, 0, QL);
            out.attn.embed_q_all = try l.packHeads(i, "embed_q", KVL, NOPE, every);
            out.attn.unembed_all = try l.packHeads(i, "unembed_out", VD, KVL, every);
        }
        return out;
    }

    /// The pack's per-head matrices [64][n][k] (8-bit g64), heads [h0, h1).
    fn packHeads(l: *Loader, i: usize, comptime which: []const u8, n: usize, k: usize, hd: [2]usize) !Q {
        const ws = try l.src("layers.{d}." ++ which ++ ".weight", .{i});
        try check(ws, "U32", &.{ H, n, k / 4 }, which);
        const ss = try l.src("layers.{d}." ++ which ++ ".scales", .{i});
        try check(ss, "F16", &.{ H, n, k / 64 }, which);
        const bs = try l.src("layers.{d}." ++ which ++ ".biases", .{i});
        const nh = hd[1] - hd[0];
        const sz = qBytes(n, k, 8);
        const q: Q = .{ .w = l.take(nh * sz[0]), .s = l.take(nh * sz[1]), .b = l.take(nh * sz[1]), .bits = 8, .n = @intCast(n), .k = @intCast(k) };
        try l.rowsOf(ws, sz[0], hd[0], hd[1], q.w);
        try l.rowsOf(ss, sz[1], hd[0], hd[1], q.s);
        try l.rowsOf(bs, sz[1], hd[0], hd[1], q.b);
        return q;
    }

    /// One safetensors file's header into the name map (shard index `si`).
    fn header(l: *Loader, full: [:0]const u8, si: u32) !void {
        const fd = std.c.open(full, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.OpenFailed;
        defer _ = std.c.close(fd);
        var head: [8]u8 = undefined;
        try readAll(fd, &head, 0);
        const hlen = std.mem.readInt(u64, &head, .little);
        const text = try l.arena.alloc(u8, hlen);
        try readAll(fd, text, 8);
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, l.arena, text, .{});
        var it = doc.object.iterator();
        while (it.next()) |kv| {
            if (std.mem.eql(u8, kv.key_ptr.*, "__metadata__")) continue;
            const o = kv.value_ptr.object;
            const dims = o.get("shape").?.array.items;
            var shape: [4]usize = @splat(1);
            for (dims, 0..) |d, j| shape[j] = @intCast(d.integer);
            const offs = o.get("data_offsets").?.array.items;
            const b: u64 = @intCast(offs[0].integer);
            const e2: u64 = @intCast(offs[1].integer);
            try l.names.put(l.arena, kv.key_ptr.*, .{ .shard = si, .off = 8 + hlen + b, .len = e2 - b, .dtype = o.get("dtype").?.string, .shape = shape, .rank = @intCast(dims.len) });
        }
    }

    /// One more safetensors file's tensors (the MTP head's).
    fn extra(l: *Loader, path: []const u8) !void {
        const pk = try l.arena.dupeSentinel(u8, path, 0);
        try l.shards.append(l.gpa, pk);
        try l.header(pk, @intCast(l.shards.items.len - 1));
    }

    fn index(l: *Loader, dir: []const u8, pack: []const u8) !void {
        const path = try std.fmt.allocPrintSentinel(l.arena, "{s}/model.safetensors.index.json", .{dir}, 0);
        const f = try mtl.MappedFile.open(path);
        defer f.deinit();
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, l.arena, f.bytes[0..f.size], .{ .allocate = .alloc_always });
        var files: std.StringArrayHashMapUnmanaged(void) = .empty;
        var it = doc.object.get("weight_map").?.object.iterator();
        while (it.next()) |kv| try files.put(l.arena, kv.value_ptr.string, {});
        for (files.keys()) |name| {
            const full = try std.fmt.allocPrintSentinel(l.arena, "{s}/{s}", .{ dir, name }, 0);
            try l.shards.append(l.gpa, full);
            try l.header(full, @intCast(l.shards.items.len - 1));
        }
        const pk = try l.arena.dupeSentinel(u8, pack, 0);
        try l.shards.append(l.gpa, pk);
        try l.header(pk, @intCast(l.shards.items.len - 1));
    }
};

fn readAll(fd: std.c.fd_t, dest: []u8, at: u64) !void {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

const Pool = struct {
    fds: []std.c.fd_t,
    jobs: []const Job,
    gpa: std.mem.Allocator,
    next: std.atomic.Value(usize) = .init(0),
    failed: std.atomic.Value(bool) = .init(false),

    fn run(p: *Pool) void {
        while (true) {
            const i = p.next.fetchAdd(1, .monotonic);
            if (i >= p.jobs.len) return;
            const j = p.jobs[i];
            if (j.rows == 0) {
                readAll(p.fds[j.shard], j.dst[0..j.len], j.off) catch p.failed.store(true, .release);
                continue;
            }
            // strided: read the stored rows in pieces of up to 64 MiB, keep each row's columns
            const tmp = p.gpa.alloc(u8, @min(j.len, 64 << 20)) catch {
                p.failed.store(true, .release);
                continue;
            };
            defer p.gpa.free(tmp);
            const per = @max(1, tmp.len / j.row_len);
            var r: u64 = 0;
            while (r < j.rows) {
                const n = @min(per, j.rows - r);
                readAll(p.fds[j.shard], tmp[0 .. n * j.row_len], j.off + r * j.row_len) catch {
                    p.failed.store(true, .release);
                    break;
                };
                for (0..n) |x| @memcpy(j.dst[(r + x) * j.col_len ..][0..j.col_len], tmp[x * j.row_len + j.col_off ..][0..j.col_len]);
                r += n;
            }
        }
    }
};

/// Layers [first, last) of share `sh`, the embedding, final norm and the share's head rows.
pub fn load(gpa: std.mem.Allocator, device: mtl.Device, dir: []const u8, pack: []const u8, sh: Share, first: usize, last: usize, threads: usize, mtp: []const u8) !*Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const w = try gpa.create(Weights);
    w.* = .{ .gpa = gpa, .first = first, .last = last, .share = sh };
    errdefer {
        w.deinit();
        gpa.destroy(w);
    }
    w.resident = device.residencySet(4096) catch null;
    var l: Loader = .{ .gpa = gpa, .arena = arena_state.allocator(), .device = device, .w = w };
    defer l.shards.deinit(gpa);
    defer l.jobs.deinit(gpa);
    try l.index(dir, pack);
    const vr = sh.vocab[1] - sh.vocab[0];
    try l.begin((V + vr) * D * 2 + D * 2 + 4 * 256);
    w.embed = try l.plain("model.embed_tokens.weight", .{}, "BF16", &.{ V, D });
    w.norm = try l.plain("model.norm.weight", .{}, "BF16", &.{D});
    const hs = try l.src("lm_head.weight", .{});
    try Loader.check(hs, "BF16", &.{ V, D }, "lm_head");
    w.head = l.take(vr * D * 2);
    try l.rowsOf(hs, D * 2, sh.vocab[0], sh.vocab[1], w.head);
    for (first..last) |i| w.layers[i] = try l.layer(i, sh);
    if (mtp.len > 0) { // the draft head: its layer sharded like the trunk's, its extras whole
        try l.extra(mtp);
        w.layers[MTP] = try l.layer(MTP, sh);
        try l.begin(D * 2 * D * 2 + 3 * D * 2 + 8 * 256);
        w.mtp = .{
            .eh_proj = try l.plain("model.layers.{d}.eh_proj.weight", .{MTP}, "BF16", &.{ D, 2 * D }),
            .enorm = try l.plain("model.layers.{d}.enorm.weight", .{MTP}, "BF16", &.{D}),
            .hnorm = try l.plain("model.layers.{d}.hnorm.weight", .{MTP}, "BF16", &.{D}),
            .norm = try l.plain("model.layers.{d}.shared_head.norm.weight", .{MTP}, "BF16", &.{D}),
        };
    }
    // run every read on `threads` threads, the shards opened uncached
    const fds = try gpa.alloc(std.c.fd_t, l.shards.items.len);
    defer gpa.free(fds);
    for (l.shards.items, fds) |p, *fd| {
        fd.* = std.c.open(p, .{ .ACCMODE = .RDONLY });
        if (fd.* < 0) return error.OpenFailed;
        _ = std.c.fcntl(fd.*, 48, @as(c_int, 1)); // F_NOCACHE
    }
    defer for (fds) |fd| {
        _ = std.c.close(fd);
    };
    var pool: Pool = .{ .fds = fds, .jobs = l.jobs.items, .gpa = gpa };
    const workers = try gpa.alloc(?std.Thread, threads);
    defer gpa.free(workers);
    for (workers) |*t| t.* = std.Thread.spawn(.{}, Pool.run, .{&pool}) catch null;
    pool.run();
    for (workers) |t| if (t) |th| th.join();
    if (pool.failed.load(.acquire)) return error.ShortRead;
    return w;
}
