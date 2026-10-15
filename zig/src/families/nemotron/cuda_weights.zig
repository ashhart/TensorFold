//! Nemotron-H weights on the GPU: tiled projections, packed expert blocks, fp32 Mamba; MLX here, ModelOpt in its file.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Config = @import("config.zig").Config;
const Kind = @import("config.zig").Kind;
const kern = @import("cuda_kernels.zig");
const draft_ids = @import("draft_ids.zig");
const Source = @import("cuda_source.zig").Source;
const host4 = core.affine4_host;
const modelopt = @import("cuda_modelopt.zig");

pub const QLinear = kern.QLinear;
pub const Experts = kern.Experts;
const Tensor = core.checkpoint.Tensor;

pub const mtp_file = "mtp-4bit.safetensors";

/// The token table as stored (MLX words, scales and biases, or a bf16 table at w): a row lookup, not a matmul.
pub const Embed = struct { w: u64, s: u64 = 0, b: u64 = 0, n: usize, k: usize, bf16: bool = false };
/// A layer's learned change while lessons run (cuda_sites.zig): y += scale (x a^T) b over the blocks open on each row.
pub const Adapter = struct { a: u64, b: u64, tau: u64, xa: u64, xn: u64, gates: u64, need: u64, latch: u64, score: u64, whole: bool, in: usize, out: usize, rank: usize, scale: f32, unit: f32 };

/// A `*_rest` is a lesson's change past its projection's 4-bit codes ([out, in] bf16), added after it; 0 when none.
pub const Mamba = struct { in_proj: QLinear, out_proj: QLinear, conv_w: u64, conv_b: u64, a: u64, d: u64, dt_bias: u64, gnorm: u64, out_rest: u64 = 0, adapter: ?Adapter = null };
pub const Attention = struct { qkv: QLinear, o: QLinear, o_rest: u64 = 0, adapter: ?Adapter = null };
/// The pocket kernels' neurons (config's pocket): input weights and outputs, [n, D] fp32 each, from the 4-bit codes.
pub const Pocket = kern.Pocket;
/// The shared expert's down projection rest in two halves (experts E and E + 1); its learned change reads both.
pub const MoE = struct { router: u64, bias: u64, experts: Experts, rest: [2]u64 = .{ 0, 0 }, adapter: ?Adapter = null, pocket: ?Pocket = null, latent: ?Latent = null };

/// LatentMoE's parts (Bolt): latent-width projections around the routed `experts`; `shared` reads full-width rows.
pub const Latent = struct { fc1: QLinear, fc2: QLinear, shared: Experts };
pub const Block = struct { kind: Kind, norm: u64, mamba: Mamba = undefined, attn: Attention = undefined, moe: MoE = undefined };
pub const Mtp = struct { enorm: u64, hnorm: u64, eh_proj: QLinear, attn_norm: u64, attn: Attention, moe_norm: u64, moe: MoE, final_norm: u64 };

/// An MLX 4-bit projection's tiles: what Sliding Weights' learner reads (it serves MLX checkpoints alone).
pub fn affine(q: QLinear) !kern.Affine4 {
    return switch (q) {
        .affine4 => |a| a,
        else => error.SlideNeedsMlx,
    };
}

/// A loaded tensor by the Python engine's dotted name, for digests against its weights.
pub const Named = struct { name: []u8, ptr: u64, len: usize };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: Config,
    embed: Embed = undefined,
    blocks: []Block = &.{},
    norm_f: u64 = 0,
    head: QLinear = undefined,
    mtp: ?Mtp = null,
    draft_head: ?QLinear = null,
    draft_ids: u64 = 0,
    draft_count: usize = 0,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    named: std.ArrayList(Named) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        for (w.named.items) |n| w.gpa.free(n.name);
        w.buffers.deinit(w.gpa);
        w.named.deinit(w.gpa);
        w.gpa.free(w.blocks);
        w.* = undefined;
    }
};

pub const Loader = struct {
    gpa: std.mem.Allocator,
    ops: kern.Ops,
    w: *Weights,
    src: *Source,
    scratch: cuda.DeviceBuffer,
    host: std.ArrayList(u8) = .empty,
    held: std.ArrayList([]align(4) u8) = .empty, // learned projections' made codes, kept until the load's uploads finish
    learned: usize = 0, // projections loaded from a lesson's bf16

    pub fn alloc(L: *Loader, name: []const u8, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(L.ops.k.d, bytes);
        try L.w.buffers.append(L.gpa, b);
        try L.w.named.append(L.gpa, .{ .name = try L.gpa.dupe(u8, name), .ptr = b.ptr, .len = bytes });
        return b.ptr;
    }

    /// Device scratch of at least `bytes`; growing waits for the stream so pending packs keep their inputs.
    pub fn tmp(L: *Loader, bytes: usize) !u64 {
        if (L.scratch.len < bytes) {
            try L.src.flush();
            try L.ops.s.synchronize();
            L.scratch.free();
            L.scratch = try cuda.DeviceBuffer.alloc(L.ops.k.d, bytes);
        }
        return L.scratch.ptr;
    }

    /// The host staging buffer, once the stream has read its last contents.
    pub fn staging(L: *Loader, bytes: usize) ![]u8 {
        try L.src.flush();
        try L.ops.s.synchronize();
        try L.host.resize(L.gpa, bytes);
        return L.host.items;
    }

    pub fn raw(L: *Loader, name: []const u8, t: Tensor) !u64 {
        const ptr = try L.alloc(name, t.bytes.len);
        try L.src.upload(ptr, t.bytes);
        return ptr;
    }

    /// `.float()` of a bf16 tensor into the host staging buffer: exact.
    pub fn widened(L: *Loader, t: Tensor) ![]u8 {
        if (t.dtype != .bf16) return error.UnexpectedTensor;
        const n = t.bytes.len / 2;
        const out = std.mem.bytesAsSlice(u32, try L.staging(n * 4));
        const in = try L.src.view(t.bytes);
        for (out, 0..) |*o, i| o.* = @as(u32, std.mem.readInt(u16, in[2 * i ..][0..2], .little)) << 16;
        return L.host.items;
    }

    pub fn widen(L: *Loader, name: []const u8, t: Tensor) !u64 {
        const host = try L.widened(t);
        const ptr = try L.alloc(name, host.len);
        try L.ops.upload(ptr, host);
        return ptr;
    }

    /// qmm_fast.tile of row-stacked parts (one projection, or q, k, v as attention stacks them).
    pub fn tile(L: *Loader, name: []const u8, parts: []const [3]Tensor) !kern.Affine4 {
        var n: usize = 0;
        const k8 = parts[0][0].dim(1);
        for (parts) |pt| {
            if (pt[0].dtype != .u32 or pt[1].dtype != .bf16 or pt[2].dtype != .bf16) return error.UnsupportedQuantization;
            if (pt[0].dim(1) != k8 or pt[1].dim(1) * 8 != k8) return error.UnexpectedTensor;
            n += pt[0].dim(0);
        }
        const k = k8 * 8;
        const kg = k / 64;
        const lay = cuda.qlinear.Affine4.layout(n, k);
        const sizes = [3]usize{ @as(usize, n) * k8 * 4, @as(usize, n) * kg * 2, @as(usize, n) * kg * 2 };
        const base = try L.tmp(sizes[0] + sizes[1] + sizes[2]);
        var at = [3]usize{ 0, sizes[0], sizes[0] + sizes[1] };
        for (parts) |pt| for (0..3) |j| {
            try L.src.upload(base + at[j], pt[j].bytes);
            at[j] += pt[j].bytes.len;
        };
        var buf: [128]u8 = undefined;
        const q: kern.Affine4 = .{
            .w = try L.alloc(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), lay.words),
            .s = try L.alloc(try std.fmt.bufPrint(&buf, "{s}.scales", .{name}), lay.scales),
            .b = try L.alloc(try std.fmt.bufPrint(&buf, "{s}.biases", .{name}), lay.scales),
            .n = n,
            .k = k,
            .npad = lay.npad,
        };
        try L.src.flush();
        try L.ops.k.affine.pack(L.ops.s, base, base + sizes[0], base + sizes[0] + sizes[1], q);
        return q;
    }

    /// Host memory the loader keeps until its uploads are done.
    pub fn keep(L: *Loader, comptime T: type, n: usize) ![]T {
        const bytes = try L.gpa.alignedAlloc(u8, .fromByteUnits(4), n * @sizeOf(T));
        L.held.append(L.gpa, bytes) catch |e| {
            L.gpa.free(bytes);
            return e;
        };
        return @alignCast(std.mem.bytesAsSlice(T, bytes));
    }

    /// A projection a lesson changed (bf16 in the shards) as 4-bit words, scales and biases, and the rest they miss.
    fn lesson(L: *Loader, ck: *core.Checkpoint, pre: []const u8, mid: []const u8) !?Learned {
        var buf: [192]u8 = undefined;
        const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ pre, mid }));
        if (w.dtype != .bf16) return null;
        if (w.rank != 2 or w.dim(1) % host4.group != 0) return error.UnexpectedTensor;
        const n = w.dim(0);
        const k = w.dim(1);
        const count = n * k;
        const values = try L.gpa.alloc(f32, count);
        defer L.gpa.free(values);
        for (values, 0..) |*v, i| v.* = host4.f32of(std.mem.readInt(u16, w.bytes[2 * i ..][0..2], .little));
        const words = try L.keep(u32, count / 8);
        const scales = try L.keep(u16, count / host4.group);
        const biases = try L.keep(u16, count / host4.group);
        host4.quantize(values, words, scales, biases);
        const back = try L.gpa.alloc(f32, count);
        defer L.gpa.free(back);
        host4.dequantize(words, scales, biases, back);
        const left = try L.keep(u16, count);
        for (left, values, back) |*r, v, q| r.* = host4.bf16of(v - q);
        L.learned += 1;
        return .{
            .parts = .{ made(.u32, n, k / 8, std.mem.sliceAsBytes(words)), made(.bf16, n, k / host4.group, std.mem.sliceAsBytes(scales)), made(.bf16, n, k / host4.group, std.mem.sliceAsBytes(biases)) },
            .rest = left,
            .n = n,
            .k = k,
        };
    }

    /// Columns [first, first + width) of a learned rest on the GPU.
    fn rest(L: *Loader, name: []const u8, l: Learned, first: usize, width: usize) !u64 {
        const host = std.mem.bytesAsSlice(u16, try L.staging(l.n * width * 2));
        for (0..l.n) |r| @memcpy(host[r * width ..][0..width], l.rest[r * l.k + first ..][0..width]);
        var buf: [128]u8 = undefined;
        const ptr = try L.alloc(try std.fmt.bufPrint(&buf, "{s}.rest", .{name}), l.n * width * 2);
        try L.ops.upload(ptr, L.host.items);
        return ptr;
    }

    /// A projection of row-stacked parts in the checkpoint's format; a ModelOpt one reads its format from its tensors.
    pub fn dense(L: *Loader, ck: *core.Checkpoint, name: []const u8, prefixes: []const []const u8) !QLinear {
        if (L.w.config.format == .modelopt) return modelopt.dense(L, ck, name, prefixes);
        return L.denseRest(ck, name, prefixes, null);
    }

    /// `dense`, with `rest` taking a lesson's change when an MLX weight is bf16 (else that is refused).
    fn denseRest(L: *Loader, ck: *core.Checkpoint, name: []const u8, prefixes: []const []const u8, rest_out: ?*u64) !QLinear {
        if (L.w.config.format == .modelopt) return modelopt.dense(L, ck, name, prefixes);
        if (rest_out) |out| if (try L.lesson(ck, prefixes[0], "")) |l| {
            out.* = try L.rest(name, l, 0, l.k);
            return .{ .affine4 = try L.tile(name, &.{l.parts}) };
        };
        var parts: [3][3]Tensor = undefined;
        for (prefixes, 0..) |pre, i| parts[i] = try three(ck, pre, "");
        return .{ .affine4 = try L.tile(name, parts[0..prefixes.len]) };
    }

    /// weights.fold_shared, experts.make: the shared halves become experts E and E + 1; the pocket goes to `pocket`.
    fn experts(L: *Loader, ck: *core.Checkpoint, name: []const u8, pre: []const u8, c: Config, shared_down: ?[3]Tensor) !Experts {
        var buf: [128]u8 = undefined;
        const own = 2 * c.expert_width;
        const s_up = try three(ck, pre, "shared_experts.up_proj");
        if (s_up[0].dim(0) != own + c.pocket) return error.SharedExpertWidth;
        if (c.pocket > 0 and shared_down != null) return error.UnexpectedTensor;
        const s_down = if (c.pocket > 0) try L.leading(try three(ck, pre, "shared_experts.down_proj"), c.hidden, own, own + c.pocket) else shared_down orelse try three(ck, pre, "shared_experts.down_proj");
        const e = c.experts + 2;
        return .{
            .format = .affine4,
            .up = try L.packExperts(try std.fmt.bufPrint(&buf, "{s}.up", .{name}), try three(ck, pre, "switch_mlp.fc1"), firstRows(s_up, own), false, e, c.expert_width, c.hidden),
            .down = try L.packExperts(try std.fmt.bufPrint(&buf, "{s}.down", .{name}), try three(ck, pre, "switch_mlp.fc2"), s_down, true, e, c.hidden, c.expert_width),
            .experts = e,
            .width = c.expert_width,
            .dims = c.hidden,
        };
    }

    /// The first `first` of `cols` columns of a quantized matrix's `n` rows, kept on the host until uploads finish.
    fn leading(L: *Loader, t: [3]Tensor, n: usize, first: usize, cols: usize) ![3]Tensor {
        var out: [3]Tensor = undefined;
        for (t, &out) |x, *o| {
            const per = x.bytes.len / n;
            const part = per * first / cols;
            const full = try L.gpa.alloc(u8, x.bytes.len);
            defer L.gpa.free(full);
            try L.src.read(full, x.bytes);
            const kept = try L.keep(u8, n * part);
            for (0..n) |r| @memcpy(kept[r * part ..][0..part], full[r * per ..][0..part]);
            o.* = x;
            o.shape[1] = x.shape[1] * first / cols;
            o.bytes = kept;
        }
        return out;
    }

    /// The pocket's neurons: up rows and down columns past the shared expert's own, dequantized to fp32 and uploaded.
    fn pocket(L: *Loader, ck: *core.Checkpoint, name: []const u8, pre: []const u8, c: Config) !Pocket {
        const own = 2 * c.expert_width;
        const n = c.pocket;
        if (n > kern.pocket_width) return error.PocketTooWide;
        const d = c.hidden;
        const g = host4.group;
        const values = try L.gpa.alloc(f32, n * d);
        defer L.gpa.free(values);
        var buf: [128]u8 = undefined;
        var out: Pocket = .{ .up = 0, .down = 0, .n = n };
        // up: rows own..own + n, each d inputs (d / 8 words, d / 64 scales and biases)
        const up = try three(ck, pre, "shared_experts.up_proj");
        var part: [3][]align(4) u8 = @splat(&.{});
        defer for (part) |p| L.gpa.free(p);
        for (up, &part) |x, *p| {
            const per = x.bytes.len / (own + n);
            p.* = try L.gpa.alignedAlloc(u8, .fromByteUnits(4), n * per);
            try L.src.read(p.*, x.bytes[own * per ..][0 .. n * per]);
        }
        host4.dequantize(std.mem.bytesAsSlice(u32, part[0]), std.mem.bytesAsSlice(u16, part[1]), std.mem.bytesAsSlice(u16, part[2]), values);
        out.up = try L.alloc(try std.fmt.bufPrint(&buf, "{s}.pocket_up", .{name}), n * d * 4);
        try L.ops.upload(out.up, std.mem.sliceAsBytes(values));
        // the neurons in use: through the last whose up row is not all zero, as the learner's pocketRead counts them
        for (0..n) |j| if (!std.mem.allEqual(f32, values[j * d ..][0..d], 0)) {
            out.used = j + 1;
        };
        const used: u32 = @intCast(out.used);
        out.count = try L.alloc(try std.fmt.bufPrint(&buf, "{s}.pocket_used", .{name}), 4);
        try L.ops.upload(out.count, std.mem.asBytes(&used));
        try L.ops.s.synchronize();
        // down: d rows, each own + n inputs; the last n of each row (its last n / 64 groups), stored neuron by neuron
        const down = try three(ck, pre, "shared_experts.down_proj");
        var whole: [3][]align(4) u8 = @splat(&.{});
        defer for (whole) |p| L.gpa.free(p);
        for (down, &whole) |x, *p| {
            p.* = try L.gpa.alignedAlloc(u8, .fromByteUnits(4), x.bytes.len);
            try L.src.read(p.*, x.bytes);
        }
        const words = std.mem.bytesAsSlice(u32, whole[0]);
        const scales = std.mem.bytesAsSlice(u16, whole[1]);
        const biases = std.mem.bytesAsSlice(u16, whole[2]);
        const wide = own + n;
        var row: [64]f32 = undefined;
        for (0..d) |r| {
            host4.dequantize(words[r * wide / 8 + own / 8 ..][0 .. n / 8], scales[r * wide / g + own / g ..][0 .. n / g], biases[r * wide / g + own / g ..][0 .. n / g], row[0..n]);
            for (row[0..n], 0..) |v, j| values[j * d + r] = v;
        }
        out.down = try L.alloc(try std.fmt.bufPrint(&buf, "{s}.pocket_down", .{name}), n * d * 4);
        try L.ops.upload(out.down, std.mem.sliceAsBytes(values));
        try L.ops.s.synchronize();
        return out;
    }

    /// One expert matrix (n outputs of k inputs) with the shared halves appended; `split` halves the shared columns.
    pub fn packExperts(L: *Loader, name: []const u8, routed: [3]Tensor, shared: [3]Tensor, split: bool, e: usize, n: usize, k: usize) !u64 {
        const kg = k / 64;
        const sizes = [3]usize{ @as(usize, e) * n * (k / 8) * 4, @as(usize, e) * n * kg * 2, @as(usize, e) * n * kg * 2 };
        const base = try L.tmp(sizes[0] + sizes[1] + sizes[2]);
        const at = [3]usize{ 0, sizes[0], sizes[0] + sizes[1] };
        for (0..3) |j| {
            if (routed[j].bytes.len + shared[j].bytes.len != sizes[j]) return error.UnexpectedTensor;
            try L.src.upload(base + at[j], routed[j].bytes);
            if (!split) try L.src.upload(base + at[j] + routed[j].bytes.len, shared[j].bytes);
        }
        // the output's allocation runs while the reads queued above go on
        const ptr = try L.alloc(name, cuda.experts.Affine4Experts.bytes(e, n, k));
        if (split) for (0..3) |j| {
            const row = shared[j].bytes.len / n;
            const half = row / 2;
            const host = try L.staging(shared[j].bytes.len);
            const in = try L.src.view(shared[j].bytes);
            for (0..2) |h| for (0..n) |r| @memcpy(host[(h * n + r) * half ..][0..half], in[r * row + h * half ..][0..half]);
            try L.ops.upload(base + at[j] + routed[j].bytes.len, host);
        };
        try L.src.flush();
        try L.ops.k.experts.pack(L.ops.s, base, base + sizes[0], base + sizes[0] + sizes[1], ptr, e, n, k);
        return ptr;
    }

    fn mamba(L: *Loader, ck: *core.Checkpoint, name: []const u8, pre: []const u8) !Mamba {
        var a: [128]u8 = undefined;
        var b: [128]u8 = undefined;
        var m: Mamba = undefined;
        m.in_proj = try L.dense(ck, try join(&a, name, ".in_proj"), &.{try join(&b, pre, "in_proj")});
        m.out_rest = 0;
        m.adapter = null;
        m.out_proj = try L.denseRest(ck, try join(&a, name, ".out_proj"), &.{try join(&b, pre, "out_proj")}, &m.out_rest);
        const conv = try ck.get(try join(&b, pre, "conv1d.weight"));
        // MLX stores [C, 4, 1], torch [C, 1, 4]: the same bytes, channel-major
        if (conv.rank != 3 or conv.dim(1) * conv.dim(2) != 4 or @min(conv.dim(1), conv.dim(2)) != 1 or conv.dtype != .bf16) return error.UnexpectedTensor;
        const ch = conv.dim(0);
        const taps = std.mem.bytesAsSlice(u32, try L.staging(ch * 4 * 4));
        const in = try L.src.view(conv.bytes);
        for (0..ch) |c| for (0..4) |t| {
            taps[t * ch + c] = @as(u32, std.mem.readInt(u16, in[(c * 4 + t) * 2 ..][0..2], .little)) << 16;
        };
        m.conv_w = try L.alloc(try join(&a, name, ".conv_w"), L.host.items.len);
        try L.ops.upload(m.conv_w, L.host.items);
        m.conv_b = try L.widen(try join(&a, name, ".conv_b"), try ck.get(try join(&b, pre, "conv1d.bias")));
        const alog = try L.widened(try ck.get(try join(&b, pre, "A_log")));
        const src = try L.tmp(alog.len);
        try L.ops.upload(src, alog);
        m.a = try L.alloc(try join(&a, name, ".a"), alog.len);
        try L.ops.torch().mambaA(src, m.a, alog.len / 4);
        m.d = try L.widen(try join(&a, name, ".d"), try ck.get(try join(&b, pre, "D")));
        m.dt_bias = try L.widen(try join(&a, name, ".dt_bias"), try ck.get(try join(&b, pre, "dt_bias")));
        m.gnorm = try L.raw(try join(&a, name, ".gnorm"), try ck.get(try join(&b, pre, "norm.weight")));
        return m;
    }

    fn attention(L: *Loader, ck: *core.Checkpoint, name: []const u8, pre: []const u8) !Attention {
        var a: [128]u8 = undefined;
        var q: [128]u8 = undefined;
        var k: [128]u8 = undefined;
        var v: [128]u8 = undefined;
        const qkv = try L.dense(ck, try join(&a, name, ".qkv"), &.{ try join(&q, pre, "q_proj"), try join(&k, pre, "k_proj"), try join(&v, pre, "v_proj") });
        var att: Attention = .{ .qkv = qkv, .o = undefined };
        att.o = try L.denseRest(ck, try join(&a, name, ".o"), &.{try join(&q, pre, "o_proj")}, &att.o_rest);
        return att;
    }

    fn moe(L: *Loader, ck: *core.Checkpoint, name: []const u8, pre: []const u8, c: Config) !MoE {
        if (c.format == .modelopt) return modelopt.moe(L, ck, name, pre, c);
        var a: [128]u8 = undefined;
        var b: [128]u8 = undefined;
        const router = try ck.expect(try join(&b, pre, "gate.weight"), .bf16, &.{ c.experts, c.hidden });
        const bias = try ck.expect(try join(&b, pre, "gate.e_score_correction_bias"), .f32, &.{c.experts});
        var m: MoE = .{ .router = try L.raw(try join(&a, name, ".router"), router), .bias = try L.raw(try join(&a, name, ".bias"), bias), .experts = undefined };
        const down = try L.lesson(ck, pre, "shared_experts.down_proj");
        if (down) |l| for (&m.rest, 0..) |*r, h| {
            r.* = try L.rest(try std.fmt.bufPrint(&b, "{s}.shared_down{d}", .{ name, h }), l, h * c.expert_width, c.expert_width);
        };
        m.experts = try L.experts(ck, try join(&a, name, ".experts"), pre, c, if (down) |l| l.parts else null);
        if (c.pocket > 0) m.pocket = try L.pocket(ck, name, pre, c);
        return m;
    }
};

/// The first `rows` rows of a quantized matrix's words, scales and biases: views of the same bytes.
fn firstRows(t: [3]Tensor, rows: usize) [3]Tensor {
    var out = t;
    for (&out) |*x| {
        const per = x.bytes.len / x.dim(0);
        x.shape[0] = rows;
        x.bytes = x.bytes[0 .. rows * per];
    }
    return out;
}

fn join(buf: *[128]u8, a: []const u8, b: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}{s}", .{ a, b });
}

/// A quantized matrix's words, scales and biases ("{pre}{mid}.weight" ...), refused unless 4-bit with bf16 scales.
fn three(ck: *core.Checkpoint, pre: []const u8, mid: []const u8) ![3]Tensor {
    var out: [3]Tensor = undefined;
    var buf: [192]u8 = undefined;
    for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |s, i| out[i] = try ck.get(try std.fmt.bufPrint(&buf, "{s}{s}.{s}", .{ pre, mid, s }));
    if (out[0].dtype != .u32 or out[1].dtype != .bf16 or out[2].dtype != .bf16) return error.UnsupportedQuantization;
    return out;
}

/// A lesson's projection made 4-bit on the host: the parts `three` would read, and the bf16 rest [n, k] they miss.
const Learned = struct { parts: [3]Tensor, rest: []const u16, n: usize, k: usize };

/// A 2-d tensor over host memory.
pub fn made(dtype: core.checkpoint.DType, d0: usize, d1: usize, bytes: []const u8) Tensor {
    var t: Tensor = .{ .dtype = dtype, .rank = 2, .shape = undefined, .bytes = bytes };
    @memset(&t.shape, 1);
    t.shape[0] = d0;
    t.shape[1] = d1;
    return t;
}

/// weights.load: the model folder's checkpoint (and its MTP head when `with_mtp`) on the GPU, leftovers refused.
pub fn load(gpa: std.mem.Allocator, io: std.Io, ops: kern.Ops, dir: []const u8, c: Config, with_mtp: bool) !Weights {
    if (c.format == .mlx and (c.group_size != 64 or c.bits != 4)) return error.UnsupportedQuantization;
    var w: Weights = .{ .gpa = gpa, .config = c };
    errdefer w.deinit();
    var src = try Source.init(gpa, ops);
    defer src.deinit();
    var L: Loader = .{ .gpa = gpa, .ops = ops, .w = &w, .src = &src, .scratch = try cuda.DeviceBuffer.alloc(ops.k.d, 1 << 20) };
    defer {
        src.flush() catch {};
        ops.s.synchronize() catch {};
        L.scratch.free();
        L.host.deinit(gpa);
        for (L.held.items) |h| gpa.free(h);
        L.held.deinit(gpa);
    }
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    try src.add(&ck);
    if (c.omni) {
        // Bolt: the language model under language_model., its vision tower and projector beside it
        try ck.skip("vision_model.");
        try ck.skip("mlp1.");
        ck.scope = "language_model.";
    }
    w.blocks = try gpa.alloc(Block, c.layers);
    for (c.kinds[0..c.layers], 0..) |kind, i| {
        var nm: [128]u8 = undefined;
        var nb: [128]u8 = undefined;
        var pb: [128]u8 = undefined;
        const name = try std.fmt.bufPrint(&nm, "blocks.{d}", .{i});
        const norm = try ck.expect(try std.fmt.bufPrint(&pb, "backbone.layers.{d}.norm.weight", .{i}), .bf16, &.{c.hidden});
        var b: Block = .{ .kind = kind, .norm = try L.raw(try join(&nb, name, ".norm"), norm) };
        const mixer = try std.fmt.bufPrint(&pb, "backbone.layers.{d}.mixer.", .{i});
        switch (kind) {
            .mamba => b.mamba = try L.mamba(&ck, try join(&nb, name, ".mamba"), mixer),
            .attention => b.attn = try L.attention(&ck, try join(&nb, name, ".attn"), mixer),
            .moe => b.moe = if (c.latent > 0) try modelopt.latent(&L, &ck, try join(&nb, name, ".moe"), mixer, c, i) else try L.moe(&ck, try join(&nb, name, ".moe"), mixer, c),
        }
        w.blocks[i] = b;
    }
    if (c.format == .modelopt) w.embed = try modelopt.embed(&L, &ck, c) else {
        const emb = try ck.get("backbone.embeddings.weight");
        w.embed = .{
            .w = try L.raw("embed.weight", emb),
            .s = try L.raw("embed.scales", try ck.get("backbone.embeddings.scales")),
            .b = try L.raw("embed.biases", try ck.get("backbone.embeddings.biases")),
            .n = emb.dim(0),
            .k = emb.dim(1) * 8,
        };
    }
    w.norm_f = try L.raw("norm_f", try ck.expect("backbone.norm_f.weight", .bf16, &.{c.hidden}));
    w.head = try L.dense(&ck, "head", &.{"lm_head"});
    // a ModelOpt checkpoint keeps its MTP head in its own shards
    if (c.format == .modelopt) if (with_mtp) try modelopt.mtp(&L, &ck, c) else try ck.skip("mtp.");
    if (ck.unused() != 0) return error.UnusedCheckpointTensors;
    if (L.learned > 0) std.log.info("{d} output projections carry what lessons taught this model", .{L.learned});
    if (with_mtp and c.format == .mlx) try loadMtp(gpa, io, &L, &ck, dir, c);
    try src.flush();
    try ops.s.synchronize();
    return w;
}

fn loadMtp(gpa: std.mem.Allocator, io: std.Io, L: *Loader, main: *core.Checkpoint, dir: []const u8, c: Config) !void {
    var ck: core.Checkpoint = .{ .gpa = gpa, .io = io };
    defer ck.close();
    try ck.add(dir, mtp_file);
    try L.src.add(&ck);
    L.w.mtp = .{
        .enorm = try L.raw("mtp.enorm", try ck.get("layers.0.enorm.weight")),
        .hnorm = try L.raw("mtp.hnorm", try ck.get("layers.0.hnorm.weight")),
        .eh_proj = try L.dense(&ck, "mtp.eh_proj", &.{"layers.0.eh_proj"}),
        .attn_norm = try L.raw("mtp.attn_norm", try ck.get("layers.0.norm.weight")),
        .attn = try L.attention(&ck, "mtp.attn", "layers.0.mixer."),
        .moe_norm = try L.raw("mtp.moe_norm", try ck.get("layers.1.norm.weight")),
        .moe = try L.moe(&ck, "mtp.moe", "layers.1.mixer.", c),
        .final_norm = try L.raw("mtp.final_norm", try ck.get("layers.1.final_layernorm.weight")),
    };
    if (ck.unused() != 0) return error.UnusedMtpTensors;
    const ids = try draft_ids.load(gpa, c.vocab);
    defer gpa.free(ids);
    // head_rows: the vocabulary head's rows for the draft ids, tiled as a head of their own
    var parts = try three(main, "lm_head", "");
    var total: usize = 0;
    for (parts) |t| total += ids.len * (t.bytes.len / t.dim(0));
    const host = try L.staging(total);
    var at: usize = 0;
    for (&parts) |*t| {
        const row = t.bytes.len / t.dim(0);
        const whole = try gpa.alloc(u8, t.bytes.len);
        defer gpa.free(whole);
        try L.src.read(whole, t.bytes);
        for (ids, 0..) |id, r| @memcpy(host[at + r * row ..][0..row], whole[id * row ..][0..row]);
        t.shape[0] = ids.len;
        t.bytes = host[at..][0 .. ids.len * row];
        at += ids.len * row;
    }
    L.w.draft_head = .{ .affine4 = try L.tile("draft_head", &.{parts}) };
    try draftIds(L, ids);
}

/// The vocabulary ids the draft head's rows are, on the GPU.
pub fn draftIds(L: *Loader, ids: []const u32) !void {
    const wide = std.mem.bytesAsSlice(i64, try L.staging(ids.len * 8));
    for (ids, wide) |id, *x| x.* = id;
    L.w.draft_ids = try L.alloc("draft_ids", ids.len * 8);
    try L.ops.upload(L.w.draft_ids, L.host.items);
    L.w.draft_count = ids.len;
}
