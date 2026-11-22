//! Inkling 2 Mini's text checkpoint on one CUDA rank. Every scheduled visit shares these physical layer weights.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const plan_ = @import("plan.zig");
const modelopt = @import("modelopt.zig");
const bf12 = @import("bf12.zig");

pub const Controller = struct {
    down: u64,
    forget: u64,
    gain: u64,
};

pub const Attention = struct {
    wq: u64,
    wk: u64,
    wv: u64,
    wr: u64,
    wo: u64,
    gate: u64,
    sink: u64,
    q_scale: u64,
    rel_scale: u64,
    rel_proj: u64,
    /// rel_proj transposed, BF16 [extent][d_rel]: the prompt kernels read one key distance's weights as 16-byte pieces
    rel_proj_t: u64,
    k_conv: u64,
    v_conv: u64,
    k_dynamic: u64 = 0,
    v_dynamic: u64 = 0,
    /// TF_INKLING2_BF12=1: the rows that decode reads through a matmul, packed (their bf16 pointer above is 0 unless
    /// the packer refused the matrix)
    wo_b12: ?bf12.Mat = null,
    wr_b12: ?bf12.Mat = null,
    wq_b12: ?bf12.Mat = null,
    wk_b12: ?bf12.Mat = null,
    wv_b12: ?bf12.Mat = null,
    gate_b12: ?bf12.Mat = null,
    /// TF_INKLING2_BF12=1: wq, wr and gate concatenated rows-first ([q, rel, q] x h), one matmul + split.
    /// Set exactly when the fused pack wins; then wq/wr/gate fall back to their separate unpacked copies.
    wq_wr_gate_b12: ?bf12.Mat = null,
    k_dynamic_b12: ?bf12.Mat = null,
    v_dynamic_b12: ?bf12.Mat = null,
    /// Sliding layers use entry zero for q, k and v. Global layers use the three separate norms.
    norms: [3]u64,
};

/// Controller order is q, k, v and MLP, matching loop_math_cuda.messages' message layout.
pub const Layer = struct {
    attn: Attention,
    attn_conv: u64,
    mlp_norm: u64,
    mlp_conv: u64,
    router: u64,
    router_bias: u64,
    experts: modelopt.Layer,
    controllers: [4]Controller,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    c: config.Config,
    p: plan_.Plan,
    embed: u64 = 0,
    embed_norm: u64 = 0,
    norm: u64 = 0,
    unembed: u64 = 0,
    unembed_b12: ?bf12.Mat = null,
    /// bf16 bytes of the widest packed matrix a wide call unpacks (not the output head: it never has wide calls), and the
    /// split-K partial floats a decode call of window rows needs
    bf12_scratch_bytes: usize = 0,
    bf12_part_floats: usize = 0,
    output_query: u64 = 0,
    attn_queries: []u64 = &.{},
    mlp_queries: []u64 = &.{},
    cross_qm: u64 = 0,
    cross_kv: u64 = 0,
    layers: []Layer = &.{},
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: usize = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.gpa.free(w.attn_queries);
        w.gpa.free(w.mlp_queries);
        w.* = undefined;
    }
};

/// All text tensors occupy one aligned device arena; expert uploads use this same bump allocator.
const Loader = struct {
    w: *Weights,
    d: *const cuda.Driver,
    arena: cuda.DeviceBuffer,
    used: usize = 0,
    pack_bf12: bool = false,

    pub fn alloc(l: *Loader, len: usize) !u64 {
        const at = std.mem.alignForward(usize, l.used, 256);
        const end = try std.math.add(usize, at, len);
        if (end > l.arena.len) return error.LoopWeightsOverflow;
        l.used = end;
        l.w.bytes += len;
        return l.arena.ptr + at;
    }

    fn put(l: *Loader, bytes: []const u8) !u64 {
        const p = try l.alloc(bytes.len);
        try l.arena.upload(p - l.arena.ptr, bytes);
        return p;
    }

    /// A bf16 [rows, cols] tensor uploaded transposed, [cols, rows] (checkpointBytes counts its room a second time).
    fn transposed(l: *Loader, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, rows: usize, cols: usize) !u64 {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        const bytes = (try ck.expect(name, .bf16, &.{ rows, cols })).bytes;
        const t = try l.w.gpa.alloc(u8, rows * cols * 2);
        defer l.w.gpa.free(t);
        for (0..rows) |r| for (0..cols) |c| {
            t[(c * rows + r) * 2 ..][0..2].* = bytes[(r * cols + c) * 2 ..][0..2].*;
        };
        return l.put(t);
    }

    /// A bf16 [n, k] dense tensor: packed BF12 when TF_INKLING2_BF12 asks and it fits (`mat` set, the pointer 0), else
    /// plain in the arena. Only the first `used_rows` rows are packed: the rest is padding no call reads.
    const Dense = struct { ptr: u64, mat: ?bf12.Mat };
    fn dense(l: *Loader, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dims: []const usize, used_rows: usize, scratch: bool) !Dense {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        const bytes = (try ck.expect(name, .bf16, dims)).bytes;
        if (l.pack_bf12) { // the arena holds no room for these: a matrix that is not packed gets a buffer of its own
            if (dims.len == 2 and bf12.eligible(used_rows, dims[1])) {
                const k = dims[1];
                if (bf12.pack(l.w.gpa, l.d, &l.w.buffers, bytes[0 .. used_rows * k * 2], used_rows, k)) |m| {
                    l.w.bytes += m.bytes();
                    if (scratch) l.w.bf12_scratch_bytes = @max(l.w.bf12_scratch_bytes, m.bf16Bytes());
                    l.w.bf12_part_floats = @max(l.w.bf12_part_floats, bf12.partFloats(used_rows, k, if (scratch) 64 else window_rows));
                    return .{ .ptr = 0, .mat = m };
                } else |err| switch (err) {
                    error.Bf12EntriesOverflow => {}, // too many escapes to be worth packing
                    else => return err,
                }
            }
            var own = try cuda.DeviceBuffer.alloc(l.d, bytes.len);
            errdefer own.free();
            try own.upload(0, bytes);
            try l.w.buffers.append(l.w.gpa, own);
            l.w.bytes += bytes.len;
            return .{ .ptr = own.ptr, .mat = null };
        }
        return .{ .ptr = try l.put(bytes), .mat = null };
    }

    fn tensor(l: *Loader, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dtype: core.safetensors.DType, dims: []const usize) !u64 {
        var buf: [256]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        return l.put((try ck.expect(name, dtype, dims)).bytes);
    }

    /// Concatenated wq|wr|gate ([q, rel, q] x h, rows wq then wr then gate) packed as one BF12 matrix, or
    /// null when BF12 is off, the shape is ineligible, or escapes overflow (the caller falls back to three
    /// separate dense() loads). The fused bytes live in w.buffers like separate packs (zero extra VRAM).
    fn fusedQrg(l: *Loader, ck: *core.Checkpoint, layer: u32, q: usize, rel: usize, h: usize) !?bf12.Mat {
        if (!l.pack_bf12) return null;
        const n = q + rel + q;
        if (!bf12.eligible(n, h)) return null;
        var wb: [96]u8 = undefined;
        const wq = (try ck.expect(try std.fmt.bufPrint(&wb, "model.llm.layers.{d}.attn.wq_du.weight", .{layer}), .bf16, &.{ q, h })).bytes;
        const wr = (try ck.expect(try std.fmt.bufPrint(&wb, "model.llm.layers.{d}.attn.wr_du.weight", .{layer}), .bf16, &.{ rel, h })).bytes;
        const gate = (try ck.expect(try std.fmt.bufPrint(&wb, "model.llm.layers.{d}.attn.gate_attn.weight", .{layer}), .bf16, &.{ q, h })).bytes;
        const cat = try l.w.gpa.alloc(u8, n * h * 2);
        defer l.w.gpa.free(cat);
        @memcpy(cat[0 .. q * h * 2], wq[0 .. q * h * 2]);
        @memcpy(cat[q * h * 2 .. (q + rel) * h * 2], wr[0 .. rel * h * 2]);
        @memcpy(cat[(q + rel) * h * 2 ..], gate[0 .. q * h * 2]);
        const m = bf12.pack(l.w.gpa, l.d, &l.w.buffers, cat, n, h) catch |err| switch (err) {
            error.Bf12EntriesOverflow => return null, // too many escapes to be worth packing
            else => return err,
        };
        l.w.bytes += m.bytes();
        l.w.bf12_scratch_bytes = @max(l.w.bf12_scratch_bytes, m.bf16Bytes());
        l.w.bf12_part_floats = @max(l.w.bf12_part_floats, bf12.partFloats(n, h, 64));
        return m;
    }
};

/// The most rows a decode call takes: the matmul's limit, and the calls that read packed rows without unpacking them.
pub const window_rows: usize = 16;

/// A tensor TF_INKLING2_BF12=1 packs elsewhere (its bf16 bytes never enter the arena; a refused one gets its own buffer).
fn packedName(name: []const u8) bool {
    if (std.mem.eql(u8, name, "model.llm.unembed.weight")) return true;
    return std.mem.endsWith(u8, name, ".attn.wo_ud.weight") or std.mem.endsWith(u8, name, ".attn.wr_du.weight") or
        std.mem.endsWith(u8, name, ".attn.wq_du.weight") or std.mem.endsWith(u8, name, ".attn.wk_dv.weight") or
        std.mem.endsWith(u8, name, ".attn.wv_dv.weight") or std.mem.endsWith(u8, name, ".attn.gate_attn.weight") or
        std.mem.endsWith(u8, name, "_sconv.weight_proj");
}

fn checkpointBytes(ck: *const core.Checkpoint, skip_packed: bool) !usize {
    var total: usize = 0;
    for (ck.files.items) |file| for (file.names.keys(), file.names.values()) |name, entry| {
        if (!std.mem.startsWith(u8, name, "model.llm.")) continue;
        if (skip_packed and packedName(name)) continue;
        const padded = std.mem.alignForward(usize, entry.end - entry.begin, 256);
        total = try std.math.add(usize, total, padded);
        // the relative-bias projection is kept twice: as stored and transposed (Attention.rel_proj_t)
        if (std.mem.endsWith(u8, name, ".attn.rel_logits_proj.proj")) total = try std.math.add(usize, total, padded);
    };
    return total;
}

fn controllers(l: *Loader, ck: *core.Checkpoint, c: *const config.Text, i: u32, rank: u32, out: *[4]Controller) !void {
    const h: usize = c.hidden;
    const names = [4][]const u8{ "attn", "k", "v", "mlp" };
    for (names, out) |role, *g| {
        var prefix: [160]u8 = undefined;
        const p = try std.fmt.bufPrint(&prefix, "model.llm.layers.{d}.controllers.{s}", .{ i, role });
        var down: [224]u8 = undefined;
        var forget: [224]u8 = undefined;
        var gain: [224]u8 = undefined;
        g.* = .{
            .down = try l.put((try ck.expect(try std.fmt.bufPrint(&down, "{s}.down_weight", .{p}), .bf16, &.{ rank, 2 * h })).bytes),
            .forget = try l.put((try ck.expect(try std.fmt.bufPrint(&forget, "{s}.forget_weight", .{p}), .bf16, &.{ h, rank })).bytes),
            .gain = try l.put((try ck.expect(try std.fmt.bufPrint(&gain, "{s}.gain_weight", .{p}), .bf16, &.{ h, rank })).bytes),
        };
    }
}

/// Load every text tensor, including all four controller triplets and both cross-pass query matrices.
pub fn load(gpa: std.mem.Allocator, d: *const cuda.Driver, ck: *core.Checkpoint, c: *const config.Config, p: plan_.Plan) !Weights {
    if (c.text.loop == null) return error.Inkling2LoopRequired;
    var formats: [config.max_layers]modelopt.Format = undefined;
    for (p.layers[0..c.text.layers], 0..) |layer, k| {
        formats[k] = switch (layer.precision) {
            .nvfp4 => .nvfp4,
            .mxfp8 => .mxfp8,
        };
    }
    const pack_dense = bf12.enabled();
    const omma = modelopt.ommaEnabled();
    const total = try checkpointBytes(ck, pack_dense);
    var arena = try cuda.DeviceBuffer.alloc(d, @max(total + 1024 * 1024, 256));
    var w: Weights = .{ .gpa = gpa, .c = c.*, .p = p };
    errdefer w.deinit();
    w.buffers.append(gpa, arena) catch |err| {
        arena.free();
        return err;
    };
    w.layers = try gpa.alloc(Layer, c.text.layers);
    w.attn_queries = try gpa.alloc(u64, c.text.layers);
    w.mlp_queries = try gpa.alloc(u64, c.text.layers);
    var l: Loader = .{ .w = &w, .d = d, .arena = arena, .pack_bf12 = pack_dense };
    const h: usize = c.text.hidden;
    w.embed = try l.tensor(ck, "model.llm.embed.weight", .{}, .bf16, &.{ c.text.vocab, h });
    w.embed_norm = try l.tensor(ck, "model.llm.embed_norm.weight", .{}, .bf16, &.{h});
    w.norm = try l.tensor(ck, "model.llm.norm.weight", .{}, .bf16, &.{h});
    {
        const head = try l.dense(ck, "model.llm.unembed.weight", .{}, &.{ c.text.vocab, h }, c.text.unpadded_vocab, false);
        w.unembed = head.ptr;
        w.unembed_b12 = head.mat;
    }
    w.output_query = try l.tensor(ck, "model.llm.attn_res.output_query", .{}, .f32, &.{h});
    w.cross_qm = try l.tensor(ck, "model.llm.middle_cycle_cross_pass_queries", .{}, .f32, &.{ 2 * (c.text.loop.?.core_end - c.text.loop.?.core_start), h });
    w.cross_kv = try l.tensor(ck, "model.llm.middle_cycle_cross_pass_kv_queries", .{}, .f32, &.{ 2 * (c.text.loop.?.core_end - c.text.loop.?.core_start), h });

    for (w.layers, 0..) |*layer, k| {
        const i: u32 = @intCast(k);
        const L = .{i};
        const hd = c.text.heads(i);
        const q: usize = @as(usize, hd.q) * hd.dim;
        const kv: usize = @as(usize, hd.kv) * hd.dim;
        w.mlp_queries[k] = try l.tensor(ck, "model.llm.attn_res.mlp_queries.{d}", L, .f32, &.{h});
        if (i == 0) {
            w.attn_queries[k] = 0;
        } else if (config.isGlobal(i)) {
            w.attn_queries[k] = try l.tensor(ck, "model.llm.attn_res.attn_queries.{d}", .{i - 1}, .f32, &.{ 3, h });
        } else {
            w.attn_queries[k] = try l.tensor(ck, "model.llm.attn_res.attn_queries.{d}", .{i - 1}, .f32, &.{h});
        }
        const A = struct {
            fn put(loader: *Loader, checkpoint: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dims: []const usize) !u64 {
                return loader.tensor(checkpoint, fmt, args, .bf16, dims);
            }
        };
        const wo = try l.dense(ck, "model.llm.layers.{d}.attn.wo_ud.weight", L, &.{ h, q }, h, true);
        const rel: usize = @as(usize, hd.q) * c.text.d_rel;
        const qrg = try l.fusedQrg(ck, i, q, rel, h);
        const wr = if (qrg == null) try l.dense(ck, "model.llm.layers.{d}.attn.wr_du.weight", L, &.{ rel, h }, rel, true) else Loader.Dense{ .ptr = 0, .mat = null };
        const wq = if (qrg == null) try l.dense(ck, "model.llm.layers.{d}.attn.wq_du.weight", L, &.{ q, h }, q, true) else Loader.Dense{ .ptr = 0, .mat = null };
        const wk = try l.dense(ck, "model.llm.layers.{d}.attn.wk_dv.weight", L, &.{ kv, h }, kv, true);
        const wv = try l.dense(ck, "model.llm.layers.{d}.attn.wv_dv.weight", L, &.{ kv, h }, kv, true);
        const gate = if (qrg == null) try l.dense(ck, "model.llm.layers.{d}.attn.gate_attn.weight", L, &.{ q, h }, q, true) else Loader.Dense{ .ptr = 0, .mat = null };
        layer.attn = .{
            .wo_b12 = wo.mat,
            .wr_b12 = wr.mat,
            .wq_b12 = wq.mat,
            .wk_b12 = wk.mat,
            .wv_b12 = wv.mat,
            .gate_b12 = gate.mat,
            .wq_wr_gate_b12 = qrg,
            .wq = wq.ptr,
            .wk = wk.ptr,
            .wv = wv.ptr,
            .wr = wr.ptr,
            .wo = wo.ptr,
            .gate = gate.ptr,
            .sink = try A.put(&l, ck, "model.llm.layers.{d}.attn.attn_sink_proj.weight", L, &.{ hd.q, h }),
            .q_scale = try A.put(&l, ck, "model.llm.layers.{d}.attn.q_l2_clip.scale", L, &.{ hd.q, hd.dim }),
            .rel_scale = try A.put(&l, ck, "model.llm.layers.{d}.attn.rel_l2_clip.scale", L, &.{ hd.q, c.text.d_rel }),
            .rel_proj = try A.put(&l, ck, "model.llm.layers.{d}.attn.rel_logits_proj.proj", L, &.{ c.text.d_rel, c.text.extent(i) }),
            .rel_proj_t = try l.transposed(ck, "model.llm.layers.{d}.attn.rel_logits_proj.proj", L, c.text.d_rel, c.text.extent(i)),
            .k_conv = undefined,
            .v_conv = undefined,
            .norms = undefined,
        };
        inline for ([2][]const u8{ "k", "v" }, 0..) |kind, j| {
            const conv = if (config.isGlobal(i))
                try A.put(&l, ck, "model.llm.layers.{d}.attn." ++ kind ++ "_sconv.static_w", L, &.{ config.conv_taps, kv })
            else
                try A.put(&l, ck, "model.llm.layers.{d}.attn." ++ kind ++ "_sconv.weight", L, &.{ kv, 1, config.conv_taps });
            if (j == 0) layer.attn.k_conv = conv else layer.attn.v_conv = conv;
            if (config.isGlobal(i)) {
                const dyn_rows = @as(usize, hd.kv) * config.conv_taps;
                const dyn = try l.dense(ck, "model.llm.layers.{d}.attn." ++ kind ++ "_sconv.weight_proj", L, &.{ dyn_rows, h }, dyn_rows, true);
                if (j == 0) {
                    layer.attn.k_dynamic = dyn.ptr;
                    layer.attn.k_dynamic_b12 = dyn.mat;
                } else {
                    layer.attn.v_dynamic = dyn.ptr;
                    layer.attn.v_dynamic_b12 = dyn.mat;
                }
            }
            const norm = if (config.isGlobal(i))
                try A.put(&l, ck, "model.llm.layers.{d}." ++ kind ++ "_input_norm.weight", L, &.{h})
            else
                try A.put(&l, ck, "model.llm.layers.{d}.attn_norm.weight", L, &.{h});
            layer.attn.norms[j + 1] = norm;
        }
        layer.attn.norms[0] = if (config.isGlobal(i))
            try A.put(&l, ck, "model.llm.layers.{d}.q_input_norm.weight", L, &.{h})
        else
            layer.attn.norms[1];
        layer.attn_conv = try A.put(&l, ck, "model.llm.layers.{d}.attn_sconv.weight", L, &.{ h, 1, config.conv_taps });
        layer.mlp_norm = try A.put(&l, ck, "model.llm.layers.{d}.mlp_norm.weight", L, &.{h});
        layer.mlp_conv = try A.put(&l, ck, "model.llm.layers.{d}.mlp_sconv.weight", L, &.{ h, 1, config.conv_taps });
        layer.router = try A.put(&l, ck, "model.llm.layers.{d}.mlp.gate.weight", L, &.{ c.text.experts, h });
        layer.router_bias = try l.tensor(ck, "model.llm.layers.{d}.mlp.gate.bias", L, .f32, &.{c.text.experts});
        var prefix: [96]u8 = undefined;
        layer.experts = try modelopt.load(&l, ck, try std.fmt.bufPrint(&prefix, "model.llm.layers.{d}.mlp.experts.", L), formats[k], c.text.experts, h, c.text.expert_inter, omma);
        @memset(&layer.controllers, .{ .down = 0, .forget = 0, .gain = 0 });
        const lp = c.text.loop.?;
        if (i >= lp.core_start and i < lp.core_end) try controllers(&l, ck, &c.text, i, lp.gate_rank, &layer.controllers);
    }
    if (ck.unused() != 0) return error.UnusedCheckpointTensor;
    return w;
}
