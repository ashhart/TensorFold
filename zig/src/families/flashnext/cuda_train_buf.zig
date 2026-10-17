//! The Flash Next trainer's device buffers beside the engine: MLX copies of small linears, keep lists, scratch.
const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const fb = @import("cuda_fn_back.zig");

const batch = prompt.batch;
const gt = cuda.gdn_train;

const layers_n = wt.layers_n;
const dims = wt.dims;
const wide = wt.wide;
const streams = wt.streams;
const low_rank = wt.low;
const vocab = wt.vocab;
const slots = wt.slots;
const top_k = slots - 1;
const moe_w = wt.moe_w;
const experts_n = wt.experts_n;
const q_heads = wt.q_heads;
const kv_heads = wt.kv_heads;
const head_dim = wt.head_dim;
const out_k = wt.out_k;
const eps = wt.eps;

/// Rows a sequence learns from at most (a question, its answer and the template around them).
pub const max_rows = 512;
/// Rows a pass holds at most, its sequences side by side at even rows.
pub const pack_rows = 1024;
/// Sequences a pass holds at most.
pub const max_seqs = 32;
/// The lowest layer a pass keeps what a backward reads: a pocket's neurons sit at layers / 5 and up.
pub const keep_low = layers_n / 5;

const per_row_kept = 2 * (3 * wide * 2 + (low_rank + streams) * 4 + streams * 2) + 3 * dims * 2 + 2 * slots * 4;

fn up(n: usize, by: usize) usize {
    return (n + by - 1) / by * by;
}

/// The device bytes `init` takes at most, beside the engine (its batched pass's scratch is the engine's own).
pub fn memory() usize {
    const kept_layers = layers_n - keep_low;
    var n: usize = 0;
    n += pack_rows * (kept_layers * (per_row_kept + 33 * 1024) + layers_n * (dims * 2 + slots * 8) + 3 * wide * 2 + low_rank * 4);
    n += pack_rows * vocab * 2; // every row's logits, then their gradient
    n += scratchBytes();
    n += weightBytes();
    n += prompt.seqBytes(max_rows);
    return n + (256 << 20);
}

fn scratchBytes() usize {
    const pairs = pack_rows * slots;
    var n: usize = 0;
    n += pack_rows * wide * 4 * 3; // g, dn, du
    n += pack_rows * dims * 4 * 3 + pack_rows * (low_rank + streams) * 4 * 2 + pack_rows * streams * 4;
    n += pack_rows * (experts_n + 1) * 4 + pairs * 4 * 3 + 2 * fb.tileCap(pairs, experts_n + 1) * 4 + (experts_n + 2) * 4;
    n += pairs * 2 * moe_w * 4 * 2 + pairs * moe_w * 4 + pairs * dims * 4;
    n += max_rows * ((q_heads + 2 * kv_heads) * head_dim * 2 + q_heads * head_dim * 4 * 3 + q_heads * 2 * head_dim * 4 + 3 * kv_heads * head_dim * 4);
    n += 2 * q_heads * max_rows * max_rows * 4;
    n += gt.Scratch.bytes(max_rows, gdn.nk, gdn.nv, dims);
    n += (pack_rows + 512) * dims * 4; // the head's split partials
    return n;
}

fn weightBytes() usize {
    const q = struct {
        fn f(n: usize, k: usize) usize {
            return n * k / 2 + n * k / 16 + 1024;
        }
    }.f;
    var n: usize = q(vocab, dims) + q(low_rank, wide) + q(wide, low_rank);
    for (keep_low..layers_n) |l| {
        n += 2 * (q(low_rank, wide) + q(streams, wide) + q(wide, low_rank));
        n += if (wt.linearLayer(l)) q(gdn.proj_width, dims) + q(dims, gdn.value_dim) else q(q_heads * 2 * head_dim, dims) + 2 * q(kv_heads * head_dim, dims) + q(dims, out_k);
    }
    return n;
}

/// Host memory the GPU maps: kernels write it at `dev`, the host reads it through `slice`.
pub const Mapped = struct {
    host: cuda.HostBuffer,
    dev: u64,

    pub fn init(d: *const cuda.Driver, bytes: usize) !Mapped {
        var h = try cuda.HostBuffer.allocMapped(d, @max(bytes, 16));
        errdefer h.free();
        @memset(h.bytes, 0);
        return .{ .host = h, .dev = try h.device() };
    }

    pub fn deinit(m: *Mapped) void {
        m.host.free();
    }

    pub fn slice(m: Mapped, comptime T: type, n: usize) []T {
        return m.host.slice(T)[0..n];
    }
};

/// One layer's MLX copies its backward's products read, and the engine's own small tensors.
pub const LayerW = struct {
    attn_hc: fb.Ops.Hc,
    mlp_hc: fb.Ops.Hc,
    gdn: ?gt.Layer = null,
    attn: ?fb.Ops.Attn = null,
};

/// Every buffer a trainer owns besides the engine's: weights, keep lists, scratch.
pub const Buffers = struct {
    gpa: std.mem.Allocator,
    d: *cuda.Driver,
    owned: std.ArrayList(cuda.DeviceBuffer) = .empty,
    w: [layers_n]LayerW = undefined,
    mixer: fb.Ops.Hc = undefined,
    head: fb.Q4 = undefined,
    /// What the forward keeps: every layer's MoE input and weights, and from keep_low on all a backward reads.
    full: batch.Keep = .{},
    light: batch.Keep = .{},
    logits: u64 = 0, // bf16 [pack_rows, vocab]: the logits, then their gradient
    g: u64 = 0,
    dbranch: u64 = 0,
    dmix: u64 = 0,
    dinj: u64 = 0,
    parts: u64 = 0,
    hs: fb.Ops.HcScratch = undefined,
    ms: fb.Ops.MoeScratch = undefined,
    as: fb.Ops.AttnScratch = undefined,
    gs: gt.Scratch = undefined,
    /// The stream `carve` zeroes on.
    zero_on: ?cuda.Stream = null,

    /// The small linears from the checkpoint `e` loaded, then the keep lists and scratch, zeroed.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, e: *prompt.Engine) !Buffers {
        var t: Buffers = .{ .gpa = gpa, .d = e.driver };
        errdefer t.deinit();
        try t.loadWeights(e, io);
        try t.carve(e.stream.*);
        return t;
    }

    pub fn deinit(t: *Buffers) void {
        for (t.owned.items) |*b| b.free();
        t.owned.deinit(t.gpa);
    }

    /// Bytes the buffers hold on the device.
    pub fn bytes(t: *const Buffers) usize {
        var n: usize = 0;
        for (t.owned.items) |b| n += b.len;
        return n;
    }

    fn device(t: *Buffers, n: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(t.d, @max(n, 256));
        errdefer b.free();
        try t.owned.append(t.gpa, b);
        return b.ptr;
    }

    fn devBytes(t: *Buffers, data: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(t.d, data);
        errdefer b.free();
        try t.owned.append(t.gpa, b);
        return b.ptr;
    }

    /// `name` (.weight, .scales, .biases under `prefix`) in MLX's layout on the device, [n, k].
    fn q4(t: *Buffers, store: *wt.Store, prefix: []const u8, name: []const u8, n: usize, k: usize) !fb.Q4 {
        var buf: [200]u8 = undefined;
        const got = try wt.join3(t.gpa, store, &buf, prefix, name);
        defer t.gpa.free(got.w);
        defer t.gpa.free(got.s);
        defer t.gpa.free(got.b);
        if (got.w.len != n * k / 2 or got.s.len != n * k / 16) return error.UnexpectedTensor;
        return .{ .w = try t.devBytes(got.w), .s = try t.devBytes(got.s), .b = try t.devBytes(got.b), .n = n, .k = k };
    }

    fn hc(t: *Buffers, store: *wt.Store, prefix: []const u8, inject: bool, scale: u64) !fb.Ops.Hc {
        return .{
            .down = try t.q4(store, prefix, ".input_mix_weight_down", low_rank, wide),
            .inject = if (inject) try t.q4(store, prefix, ".block_inject_weight", streams, wide) else null,
            .up = try t.q4(store, prefix, ".input_mix_weight_up", wide, low_rank),
            .scale = scale,
            .streams = streams,
            .d = dims,
            .low = low_rank,
            .eps = eps,
        };
    }

    fn gq(m: fb.Q4) gt.Q4 {
        return .{ .w = m.w, .s = m.s, .b = m.b, .n = m.n, .k = m.k };
    }

    /// The small linears a backward reads, from the checkpoint the engine loaded, for layers keep_low and up.
    fn loadWeights(t: *Buffers, e: *prompt.Engine, io: std.Io) !void {
        var store = try wt.Store.open(t.gpa, io, e.model_dir);
        defer store.deinit();
        var name: [160]u8 = undefined;
        const inv = e.batch_s.?.inv;
        for (keep_low..layers_n) |l| {
            const lw = &e.layers[l];
            const root = try std.fmt.bufPrint(&name, "language_model.model.layers.{d}", .{l});
            var pre: [200]u8 = undefined;
            var lw_t: LayerW = .{
                .attn_hc = try t.hc(&store, try std.fmt.bufPrint(&pre, "{s}.attn_hyper_connection", .{root}), true, lw.attn_hc.scale.ptr),
                .mlp_hc = undefined,
            };
            lw_t.mlp_hc = try t.hc(&store, try std.fmt.bufPrint(&pre, "{s}.mlp_hyper_connection", .{root}), true, lw.mlp_hc.scale.ptr);
            if (wt.linearLayer(l)) {
                const lp = try std.fmt.bufPrint(&pre, "{s}.linear_attn", .{root});
                var parts: [4][]const u8 = .{ ".in_proj_qkv", ".in_proj_z", ".in_proj_b", ".in_proj_a" };
                const st = try wt.stitch(t.gpa, &store, lp, &parts);
                defer t.gpa.free(st.w);
                defer t.gpa.free(st.s);
                defer t.gpa.free(st.b);
                if (st.w.len != gdn.proj_width * dims / 2) return error.UnexpectedTensor;
                const in_proj: fb.Q4 = .{ .w = try t.devBytes(st.w), .s = try t.devBytes(st.s), .b = try t.devBytes(st.b), .n = gdn.proj_width, .k = dims };
                const out = try t.q4(&store, lp, ".out_proj", dims, gdn.value_dim);
                lw_t.gdn = .{ .nk = gdn.nk, .nv = gdn.nv, .hidden = dims, .eps = eps, .conv_w = lw.cw.ptr, .a_log = lw.a_log.ptr, .dt_bias = lw.dt.ptr, .norm_w = lw.norm.ptr, .in_proj = gq(in_proj), .out_proj = gq(out) };
            } else {
                const ap = try std.fmt.bufPrint(&pre, "{s}.self_attn", .{root});
                var buf2: [200]u8 = undefined;
                const apc = buf2[0..ap.len];
                @memcpy(apc, ap);
                lw_t.attn = .{
                    .q = try t.q4(&store, apc, ".q_proj", q_heads * 2 * head_dim, dims),
                    .k = try t.q4(&store, apc, ".k_proj", kv_heads * head_dim, dims),
                    .v = try t.q4(&store, apc, ".v_proj", kv_heads * head_dim, dims),
                    .o = try t.q4(&store, apc, ".o_proj", dims, out_k),
                    .qscale = lw.q_scale.ptr,
                    .kscale = lw.k_scale.ptr,
                    .inv = inv,
                    .heads = q_heads,
                    .kv_heads = kv_heads,
                    .dim = head_dim,
                    .half = wt.half,
                    .eps = eps,
                };
            }
            store.closeFile();
            t.w[l] = lw_t;
        }
        t.mixer = try t.hc(&store, "language_model.model.hyper_connection_mixer", false, e.mixer.scale.ptr);
        t.head = try t.q4(&store, "language_model.lm_head", "", vocab, dims);
    }

    /// The keep lists and the backward's scratch, each zeroed on `s` (rows between sequences read zeros).
    fn carve(t: *Buffers, s: cuda.Stream) !void {
        defer s.synchronize() catch {};
        const R = pack_rows;
        const pairs = R * slots;
        t.zero_on = s;
        const z = struct {
            fn get(tt: *Buffers, n: usize) !u64 {
                const p = try tt.device(n);
                try tt.d.check(tt.d.api.cuMemsetD8Async(p, 0, @max(n, 256), tt.zero_on.?.handle), "cuMemsetD8Async");
                return p;
            }
        }.get;
        for (0..layers_n) |l| {
            var k: batch.LayerKeep = .{};
            k.moe_x = try z(t, R * dims * 2);
            k.moe_wts = try z(t, R * slots * 4);
            k.moe_ids = try z(t, R * slots * 4);
            t.light.layers[l] = k;
            if (l >= keep_low) {
                inline for (.{ "attn_hc", "mlp_hc" }) |f| @field(k, f) = .{ .h = try z(t, R * wide * 2), .normed = try z(t, R * wide * 2), .v = try z(t, R * (low_rank + streams) * 4), .up = try z(t, R * wide * 2), .inject = try z(t, R * streams * 2) };
                k.mix_branch = try z(t, R * dims * 2);
                k.moe_branch = try z(t, R * dims * 2);
                if (wt.linearLayer(l)) {
                    k.proj = try z(t, R * gdn.proj_width * 2);
                } else {
                    k.pq = try z(t, R * q_heads * 2 * head_dim * 2);
                    k.pk = try z(t, R * kv_heads * head_dim * 2);
                    k.pv = try z(t, R * kv_heads * head_dim * 2);
                    k.gated = try z(t, R * out_k * 2);
                }
            }
            t.full.layers[l] = k;
        }
        t.full.mixer = .{ .h = try z(t, R * wide * 2), .normed = try z(t, R * wide * 2), .v = try z(t, R * low_rank * 4), .up = try z(t, R * wide * 2) };
        t.logits = try z(t, R * vocab * 2);
        t.g = try z(t, R * wide * 4);
        t.dbranch = try z(t, R * dims * 4);
        t.dmix = try z(t, R * dims * 4);
        t.dinj = try z(t, R * streams * 4);
        t.parts = try z(t, (R + 512) * dims * 4);
        t.hs = .{ .dn = try z(t, R * wide * 4), .du = try z(t, R * wide * 4), .dact = try z(t, R * low_rank * 4), .dv = try z(t, R * (low_rank + streams) * 4) };
        const total = experts_n + 1;
        t.ms = .{
            .logits = try z(t, R * total * 4),
            .order = try z(t, pairs * 4),
            .inv = try z(t, pairs * 4),
            .offsets = try z(t, (total + 1) * 4),
            .tiles = try z(t, 2 * fb.tileCap(pairs, total) * 4),
            .gu = try z(t, pairs * 2 * moe_w * 4),
            .dact = try z(t, pairs * moe_w * 4),
            .dgu = try z(t, pairs * 2 * moe_w * 4),
            .dxp = try z(t, pairs * dims * 4),
            .a = try z(t, pairs * 4),
            .dlog = 0,
        };
        const M = max_rows;
        const qw = q_heads * head_dim;
        const kw = kv_heads * head_dim;
        t.as = .{
            .qkv = try z(t, M * (qw + 2 * kw) * 2),
            .dg = try z(t, M * qw * 4),
            .dout = try z(t, M * qw * 4),
            .dpq = try z(t, M * 2 * qw * 4),
            .dq = try z(t, M * qw * 4),
            .dk = try z(t, M * kw * 4),
            .dv = try z(t, M * kw * 4),
            .dpk = try z(t, M * kw * 4),
            .probs = try z(t, q_heads * M * M * 4),
            .dscores = try z(t, q_heads * M * M * 4),
        };
        t.gs = gt.Scratch.carve(try z(t, gt.Scratch.bytes(M, gdn.nk, gdn.nv, dims)), M, gdn.nk, gdn.nv, dims);
    }
};
