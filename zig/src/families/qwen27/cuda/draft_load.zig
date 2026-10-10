//! DFlash2's weights as dflash2 holds them: projections quantize4'd and tiled, codebooks fp32 on the host.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const kern = @import("kernels.zig");
const st = @import("state.zig");

pub const QLinear = kern.QLinear;
const Tensor = core.checkpoint.Tensor;

pub const hidden = 5120;
pub const layers = 5;
pub const heads = 32;
pub const kv_heads = 8;
pub const head_dim = 128;
pub const conv_group = 16;
pub const mask_id = 248070;
pub const window = 2047; // sliding_window - 1: context rows each layer keeps
pub const taps = [_]usize{ 5, 19, 33, 47, 61 }; // dflash_config.target_layer_ids
pub const selector_rank = 256;
pub const theta: f64 = 10000000;
/// The head's rows the drafter reads: the first 98,304 tokens and the last 288 (dflash2.vocab_spans).
pub const spans = [2][2]usize{ .{ 0, 98304 }, .{ 248032, 248320 } };
pub const head_rows = spans[0][1] - spans[0][0] + spans[1][1] - spans[1][0];

pub const Layer = struct {
    in_norm: u64,
    post_norm: u64,
    attn_proj: QLinear, // attention_conv.kernel_projection: the conv's dynamic taps (2 x 2 x 320)
    attn_base: u64, // attention_conv.base_kernel (2, 2, 5120) bf16
    mlp_proj: QLinear,
    mlp_base: u64,
    qkv: QLinear, // [q | k | v] stacked
    kv: QLinear, // [k | v] stacked: the context's projection
    o: QLinear,
    q_norm: u64,
    k_norm: u64,
    gate: QLinear,
    up: QLinear,
    down: QLinear,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    layers: [layers]Layer = undefined,
    fc: QLinear = undefined, // five taps (25,600) to the hidden width
    hidden_norm: u64 = 0,
    norm: u64 = 0,
    select: QLinear = undefined, // candidate_selector.hidden_projection
    head_tail: QLinear = undefined, // the target head's last 288 rows, tiled on their own (rows() copies them)
    pred: []f32 = &.{}, // candidate_selector.predecessor_codebook (vocab, 256)
    succ: []f32 = &.{},
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.pred);
        w.gpa.free(w.succ);
        w.* = undefined;
    }
};

fn bf16Value(word: u16) f32 {
    return @bitCast(@as(u32, word) << 16);
}

fn toBf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @intCast((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

/// torch.round: halves to even.
fn roundEven(x: f32) f32 {
    const r = @round(x);
    return if (@abs(x - @trunc(x)) == 0.5) 2.0 * @round(x / 2.0) else r;
}

/// dflash2.quantize4 on the host, bit for bit: (n, k) bf16 to MLX affine 4-bit words, bf16 scales and biases.
pub fn quantize4(src: []const u8, n: usize, k: usize, words: []u32, scales: []u16, biases: []u16) void {
    const kg = k / 64;
    for (0..n) |r| for (0..kg) |gi| {
        var vals: [64]f32 = undefined;
        for (&vals, 0..) |*v, j| v.* = bf16Value(std.mem.readInt(u16, src[(r * k + gi * 64 + j) * 2 ..][0..2], .little));
        var lo = vals[0];
        var hi = vals[0];
        for (vals[1..]) |v| {
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        const s = toBf16(@max((hi - lo) / 15.0, 1e-8));
        const b = toBf16(lo);
        scales[r * kg + gi] = s;
        biases[r * kg + gi] = b;
        const sf = bf16Value(s);
        const bf = bf16Value(b);
        for (0..8) |wi| {
            var word: u32 = 0;
            for (0..8) |j| {
                const q = std.math.clamp(roundEven((vals[wi * 8 + j] - bf) / sf), 0.0, 15.0);
                word |= @as(u32, @intFromFloat(q)) << @intCast(4 * j);
            }
            words[r * (k / 8) + gi * 8 + wi] = word;
        }
    };
}

const Loader = struct {
    gpa: std.mem.Allocator,
    ops: kern.Ops,
    w: *Weights,
    words: std.ArrayList(u32) = .empty,
    scales: std.ArrayList(u16) = .empty,
    biases: std.ArrayList(u16) = .empty,
    staging: ?cuda.DeviceBuffer = null,

    fn alloc(L: *Loader, len: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(L.ops.k.d, len);
        try L.w.buffers.append(L.gpa, b);
        return b.ptr;
    }

    fn raw(L: *Loader, t: Tensor) !u64 {
        const ptr = try L.alloc(t.bytes.len);
        try L.ops.upload(ptr, t.bytes);
        try L.ops.s.synchronize();
        return ptr;
    }

    /// qmm_fast.tile of host MLX words, scales and biases (n rows of k inputs): the lane matmul's layout.
    fn tile(L: *Loader, words: []const u32, scales: []const u16, biases: []const u16, n: usize, k: usize) !QLinear {
        const k8 = k / 8;
        const kg = k / 64;
        const npad = (n + 127) / 128 * 128;
        const sizes = [3]usize{ n * k8 * 4, n * kg * 2, n * kg * 2 };
        const total_in = sizes[0] + sizes[1] + sizes[2];
        if (L.staging == null or L.staging.?.len < total_in) {
            if (L.staging) |*b| b.free();
            L.staging = try cuda.DeviceBuffer.alloc(L.ops.k.d, total_in);
        }
        const base = L.staging.?.ptr;
        try L.ops.upload(base, std.mem.sliceAsBytes(words));
        try L.ops.upload(base + sizes[0], std.mem.sliceAsBytes(scales));
        try L.ops.upload(base + sizes[0] + sizes[1], std.mem.sliceAsBytes(biases));
        const q: QLinear = .{ .w = try L.alloc(npad * k / 2), .s = try L.alloc(kg * npad * 2), .b = try L.alloc(kg * npad * 2), .n = n, .k = k, .npad = npad };
        const total: u64 = @as(u64, npad / 64) * kg * 512;
        var a: cuda.Args = .{};
        a.add(base);
        for ([_]usize{ n, k8, kg }) |v| a.add(@as(c_int, @intCast(v)));
        a.add(q.w);
        a.add(@as(c_longlong, @intCast(total)));
        try cuda.launch.launch(L.ops.k.pack_dense, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, L.ops.s, &a);
        for ([_]u64{ q.s, q.b }, [_]usize{ sizes[0], sizes[0] + sizes[1] }) |out, off| {
            var t: cuda.Args = .{};
            t.add(base + off);
            for ([_]usize{ n, kg, npad }) |v| t.add(@as(c_int, @intCast(v)));
            t.add(out);
            const cells = @as(u64, kg) * npad;
            try cuda.launch.launch(L.ops.k.transpose16, .{ .grid = .{ .x = @intCast((cells + 255) / 256) }, .block = .{ .x = 256 } }, L.ops.s, &t);
        }
        try L.ops.s.synchronize(); // the staging buffer and host arrays are reused next
        return q;
    }

    /// One bf16 (n, k) checkpoint tensor (or several stacked by rows) quantized and tiled.
    fn quantized(L: *Loader, parts: []const Tensor, k: usize) !QLinear {
        var n: usize = 0;
        for (parts) |t| {
            if (t.dtype != .bf16 or t.rank != 2 or t.dim(1) != k) return error.UnexpectedDraftTensor;
            n += t.dim(0);
        }
        try L.words.resize(L.gpa, n * k / 8);
        try L.scales.resize(L.gpa, n * k / 64);
        try L.biases.resize(L.gpa, n * k / 64);
        var row: usize = 0;
        for (parts) |t| {
            const r = t.dim(0);
            quantize4(t.bytes, r, k, L.words.items[row * k / 8 ..][0 .. r * k / 8], L.scales.items[row * k / 64 ..][0 .. r * k / 64], L.biases.items[row * k / 64 ..][0 .. r * k / 64]);
            row += r;
        }
        return L.tile(L.words.items, L.scales.items, L.biases.items, n, k);
    }

    fn get(ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype) !Tensor {
        var buf: [160]u8 = undefined;
        return ck.get(try std.fmt.bufPrint(&buf, fmt, args));
    }

    /// The codebook as fp32 on the host (dflash2 keeps tensor.float().numpy()).
    fn codebook(L: *Loader, t: Tensor) ![]f32 {
        if (t.dtype != .bf16 or t.rank != 2 or t.dim(1) != selector_rank) return error.UnexpectedDraftTensor;
        const out = try L.gpa.alloc(f32, t.dim(0) * selector_rank);
        for (out, 0..) |*o, i| o.* = bf16Value(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
        return out;
    }
};

/// The drafter at `dir`, and the target head's last span tiled from the target checkpoint at `target_dir`.
pub fn load(gpa: std.mem.Allocator, io: std.Io, ops: kern.Ops, dir: []const u8, target_dir: []const u8) !Weights {
    var w: Weights = .{ .gpa = gpa };
    errdefer w.deinit();
    var L: Loader = .{ .gpa = gpa, .ops = ops, .w = &w };
    defer {
        L.words.deinit(gpa);
        L.scales.deinit(gpa);
        L.biases.deinit(gpa);
        if (L.staging) |*b| b.free();
    }
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    for (&w.layers, 0..) |*l, i| {
        const q = try Loader.get(&ck, "layers.{d}.self_attn.q_proj.weight", .{i});
        const k = try Loader.get(&ck, "layers.{d}.self_attn.k_proj.weight", .{i});
        const v = try Loader.get(&ck, "layers.{d}.self_attn.v_proj.weight", .{i});
        l.qkv = try L.quantized(&.{ q, k, v }, hidden);
        l.kv = try L.quantized(&.{ k, v }, hidden);
        l.o = try L.quantized(&.{try Loader.get(&ck, "layers.{d}.self_attn.o_proj.weight", .{i})}, heads * head_dim);
        l.q_norm = try L.raw(try Loader.get(&ck, "layers.{d}.self_attn.q_norm.weight", .{i}));
        l.k_norm = try L.raw(try Loader.get(&ck, "layers.{d}.self_attn.k_norm.weight", .{i}));
        l.in_norm = try L.raw(try Loader.get(&ck, "layers.{d}.input_layernorm.weight", .{i}));
        l.post_norm = try L.raw(try Loader.get(&ck, "layers.{d}.post_attention_layernorm.weight", .{i}));
        l.attn_proj = try L.quantized(&.{try Loader.get(&ck, "layers.{d}.attention_conv.kernel_projection.weight", .{i})}, hidden);
        l.attn_base = try L.raw(try Loader.get(&ck, "layers.{d}.attention_conv.base_kernel", .{i}));
        l.mlp_proj = try L.quantized(&.{try Loader.get(&ck, "layers.{d}.mlp_conv.kernel_projection.weight", .{i})}, hidden);
        l.mlp_base = try L.raw(try Loader.get(&ck, "layers.{d}.mlp_conv.base_kernel", .{i}));
        l.gate = try L.quantized(&.{try Loader.get(&ck, "layers.{d}.mlp.gate_proj.weight", .{i})}, hidden);
        l.up = try L.quantized(&.{try Loader.get(&ck, "layers.{d}.mlp.up_proj.weight", .{i})}, hidden);
        const down = try Loader.get(&ck, "layers.{d}.mlp.down_proj.weight", .{i});
        l.down = try L.quantized(&.{down}, down.dim(1));
    }
    const fc = try ck.get("fc.weight");
    w.fc = try L.quantized(&.{fc}, fc.dim(1));
    w.hidden_norm = try L.raw(try ck.get("hidden_norm.weight"));
    w.norm = try L.raw(try ck.get("norm.weight"));
    w.select = try L.quantized(&.{try ck.get("candidate_selector.hidden_projection.weight")}, hidden);
    w.pred = try L.codebook(try ck.get("candidate_selector.predecessor_codebook"));
    w.succ = try L.codebook(try ck.get("candidate_selector.successor_codebook"));
    if (ck.unused() != 0) return error.UnusedDraftTensors;
    // dflash2's sub-head: rows() of the tiled head copies a span off the 64-row tile grid, packed on its own
    var tk = try core.Checkpoint.openModelPrefix(gpa, io, target_dir, "language_model.lm_head");
    defer tk.close();
    const hw = try tk.get("language_model.lm_head.weight");
    const hs = try tk.get("language_model.lm_head.scales");
    const hb = try tk.get("language_model.lm_head.biases");
    const a = spans[1][0];
    const n = spans[1][1] - a;
    const k8 = hidden / 8;
    const kg = hidden / 64;
    // mapped tensors may sit at odd offsets: the rows go through aligned host arrays
    try L.words.resize(gpa, n * k8);
    try L.scales.resize(gpa, n * kg);
    try L.biases.resize(gpa, n * kg);
    @memcpy(std.mem.sliceAsBytes(L.words.items), hw.bytes[a * k8 * 4 ..][0 .. n * k8 * 4]);
    @memcpy(std.mem.sliceAsBytes(L.scales.items), hs.bytes[a * kg * 2 ..][0 .. n * kg * 2]);
    @memcpy(std.mem.sliceAsBytes(L.biases.items), hb.bytes[a * kg * 2 ..][0 .. n * kg * 2]);
    w.head_tail = try L.tile(L.words.items, L.scales.items, L.biases.items, n, hidden);
    return w;
}

test "quantize4 rounds as torch does" {
    // one group: values 0..63 / 4 -> lo 0, scale bf16(63/4/15 = 1.05), each q round-half-even of v / 1.046875
    var src: [64 * 2]u8 = undefined;
    for (0..64) |j| std.mem.writeInt(u16, src[2 * j ..][0..2], toBf16(@as(f32, @floatFromInt(j)) / 4.0), .little);
    var words: [8]u32 = undefined;
    var scales: [1]u16 = undefined;
    var biases: [1]u16 = undefined;
    quantize4(&src, 1, 64, &words, &scales, &biases);
    try std.testing.expectEqual(toBf16(63.0 / 4.0 / 15.0), scales[0]);
    try std.testing.expectEqual(@as(u16, 0), biases[0]);
    try std.testing.expectEqual(@as(u32, 15), words[7] >> 28);
    try std.testing.expectEqual(roundEven(2.5), 2.0);
    try std.testing.expectEqual(roundEven(3.5), 4.0);
}
