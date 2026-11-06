//! Ripple's DFlash2 drafter weights on CUDA: the bf16 release packed for the lane matmul, values as stored.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const spec = @import("dflash2_spec.zig");
const cfg = @import("config.zig");

pub const Config = spec.Config;
const Checkpoint = core.checkpoint.Checkpoint;
const Tensor = core.checkpoint.Tensor;
const Weight = cuda.qmmf.Weight;
const DeviceBuffer = cuda.DeviceBuffer;

/// Bytes a small tensor's device copy starts on (the glue kernels read 16-byte chunks).
const small_align = 256;

/// Shapes the packer takes: every matmul's k in whole 64s. Rows are padded to 128 by the packer.
pub fn gate(c: *const Config) !void {
    if (c.layers == 0 or c.layers > spec.max_layers) return error.UnsupportedRicursiveDFlash2Shape;
    const q = @as(usize, c.heads) * cfg.head_dim;
    for ([_]usize{ c.hidden, q, c.ffn, @as(usize, c.hidden) * spec.max_taps }) |k| if (k % cuda.qmmf.group != 0) return error.UnsupportedRicursiveDFlash2Shape;
    if (c.rank % 32 != 0) return error.UnsupportedRicursiveDFlash2Shape; // the host walk's 32-rank partial sums
}

fn queryWidth(c: Config) usize {
    return @as(usize, c.heads) * cfg.head_dim;
}

fn kvWidth(c: Config) usize {
    return @as(usize, c.kv_heads) * cfg.head_dim;
}

/// Bytes of one packed bf16 matrix [n, k]: rows padded to whole 128s.
fn packedBytes(n: usize, k: usize) usize {
    return cuda.modelopt.padded(n) * k * 2;
}

/// Bytes of a layer's packed matrices.
fn layerPacked(c: Config) usize {
    const h: usize = c.hidden;
    const qw = queryWidth(c);
    const kv = kvWidth(c);
    return packedBytes(qw + 2 * kv, h) + packedBytes(2 * kv, h) + packedBytes(h, qw) + packedBytes(2 * c.ffn, h) + packedBytes(h, c.ffn) + 2 * packedBytes(c.dyn(), h);
}

/// The small tensors' device image in a fixed order (norm, hidden_norm, each layer's norms, conv bases), 256-aligned.
fn smallBytes(c: Config) usize {
    var at: usize = 0;
    const sizes = [_]usize{ @as(usize, c.hidden) * 2, @as(usize, c.hidden) * 2 };
    for (sizes) |n| at = std.mem.alignForward(usize, at, small_align) + n;
    for (0..c.layers) |_| {
        const per = [_]usize{ @as(usize, c.hidden) * 2, @as(usize, c.hidden) * 2, cfg.head_dim * 2, cfg.head_dim * 2, 2 * @as(usize, c.conv) * c.hidden * 2, 2 * @as(usize, c.conv) * c.hidden * 2 };
        for (per) |n| at = std.mem.alignForward(usize, at, small_align) + n;
    }
    return at;
}

/// Device bytes `load` puts on the GPU for `c`: each packed matrix and the small image; codebooks stay on the host.
pub fn deviceBytes(c: Config) usize {
    return packedBytes(c.hidden, @as(usize, c.hidden) * spec.max_taps) + packedBytes(c.rank, c.hidden) + c.layers * layerPacked(c) + smallBytes(c);
}

/// A convolution's device weights: its base kernel [2, conv, H] bf16 and the dynamic projection [dyn, H] packed.
pub const Conv = struct { base: u64, proj: Weight };

pub const Layer = struct {
    qkv: Weight, // bf16 [q + kv + kv, H]: q rows, k rows, v rows
    kv: Weight, // bf16 [kv + kv, H]: k rows, v rows (the absorb's projection)
    o: Weight, // bf16 [H, q]
    gate_up: Weight, // bf16 [2 * ffn, H]: gate rows, then up rows
    down: Weight, // bf16 [H, ffn]
    attn_conv: Conv,
    mlp_conv: Conv,
    in_norm: u64, // bf16 [H]
    post_norm: u64,
    q_norm: u64, // bf16 [128]
    k_norm: u64,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: Config,
    fc: Weight, // bf16 [H, 5H]
    hproj: Weight, // bf16 [rank, H]: the selector's hidden projection
    hidden_norm: u64, // bf16 [H]
    norm: u64,
    layers: [spec.max_layers]Layer,
    pred: []u16, // host, bf16 bits [vocab, rank]: the predecessor codebook
    succ: []u16, // host, bf16 bits [vocab, rank]: the successor codebook
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    bytes: usize = 0,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.pred);
        w.gpa.free(w.succ);
        w.* = undefined;
    }
};

/// Where `load`'s packed bytes go: device buffers it owns until the Weights take them.
const Device = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    bytes: usize = 0,

    fn put(s: *Device, bytes: []const u8) !u64 {
        var b = try DeviceBuffer.fromHost(s.d, bytes);
        errdefer b.free();
        try s.buffers.append(s.gpa, b);
        s.bytes += bytes.len;
        return b.ptr;
    }
};

/// The small tensors in one host image, each at a 256-byte boundary.
const Small = struct {
    bytes: std.ArrayList(u8) = .empty,

    fn put(s: *Small, gpa: std.mem.Allocator, t: Tensor) !usize {
        const at = std.mem.alignForward(usize, s.bytes.items.len, small_align);
        try s.bytes.appendNTimes(gpa, 0, at - s.bytes.items.len);
        try s.bytes.appendSlice(gpa, t.bytes);
        return at;
    }
};

/// bf16 matrices one under another ([sum n, k]) packed as one projection, then handed to `sink`.
fn dense(gpa: std.mem.Allocator, sink: anytype, parts: []const Tensor, k: usize) !Weight {
    var n: usize = 0;
    for (parts) |p| n += p.bytes.len / 2 / k;
    const src = try gpa.alloc(u16, n * k);
    defer gpa.free(src);
    var at: usize = 0;
    for (parts) |p| {
        @memcpy(std.mem.sliceAsBytes(src)[at..][0..p.bytes.len], p.bytes);
        at += p.bytes.len;
    }
    const out = try gpa.alloc(u16, cuda.modelopt.padded(n) * k);
    defer gpa.free(out);
    cuda.qmmf.packBf16(out, src, n, k);
    return cuda.modelopt.bf16Weight(try sink.put(std.mem.sliceAsBytes(out)), n, k);
}

/// The release's tensor, checked by dtype and shape.
fn bf16(ck: *Checkpoint, comptime fmt: []const u8, args: anytype, shape: []const usize) !Tensor {
    var buf: [160]u8 = undefined;
    return ck.expect(try std.fmt.bufPrint(&buf, fmt, args), .bf16, shape);
}

/// Every tensor of the release from `ck`, packed and put to `sink` in a fixed order; the codebooks copied to the host.
fn build(gpa: std.mem.Allocator, ck: *Checkpoint, c: Config, sink: anytype) !Weights {
    try gate(&c);
    const h: usize = c.hidden;
    const qw = queryWidth(c);
    const kvw = kvWidth(c);
    const dyn: usize = c.dyn();
    var w: Weights = .{ .gpa = gpa, .config = c, .fc = undefined, .hproj = undefined, .hidden_norm = 0, .norm = 0, .layers = undefined, .pred = &.{}, .succ = &.{} };
    errdefer w.deinit();
    w.fc = try dense(gpa, sink, &.{try bf16(ck, "fc.weight", .{}, &.{ h, h * spec.max_taps })}, h * spec.max_taps);
    w.hproj = try dense(gpa, sink, &.{try bf16(ck, "candidate_selector.hidden_projection.weight", .{}, &.{ c.rank, h })}, h);
    var small: Small = .{};
    defer small.bytes.deinit(gpa);
    const at_norm = try small.put(gpa, try bf16(ck, "norm.weight", .{}, &.{h}));
    const at_hidden = try small.put(gpa, try bf16(ck, "hidden_norm.weight", .{}, &.{h}));
    var offsets: [spec.max_layers][6]usize = undefined;
    for (w.layers[0..c.layers], offsets[0..c.layers], 0..) |*l, *o, i| {
        const q = try bf16(ck, "layers.{d}.self_attn.q_proj.weight", .{i}, &.{ qw, h });
        const k = try bf16(ck, "layers.{d}.self_attn.k_proj.weight", .{i}, &.{ kvw, h });
        const v = try bf16(ck, "layers.{d}.self_attn.v_proj.weight", .{i}, &.{ kvw, h });
        const gate_w = try bf16(ck, "layers.{d}.mlp.gate_proj.weight", .{i}, &.{ c.ffn, h });
        const up_w = try bf16(ck, "layers.{d}.mlp.up_proj.weight", .{i}, &.{ c.ffn, h });
        const a_base = try bf16(ck, "layers.{d}.attention_conv.base_kernel", .{i}, &.{ 2, c.conv, h });
        const m_base = try bf16(ck, "layers.{d}.mlp_conv.base_kernel", .{i}, &.{ 2, c.conv, h });
        l.* = .{
            .qkv = try dense(gpa, sink, &.{ q, k, v }, h),
            .kv = try dense(gpa, sink, &.{ k, v }, h),
            .o = try dense(gpa, sink, &.{try bf16(ck, "layers.{d}.self_attn.o_proj.weight", .{i}, &.{ h, qw })}, qw),
            .gate_up = try dense(gpa, sink, &.{ gate_w, up_w }, h),
            .down = try dense(gpa, sink, &.{try bf16(ck, "layers.{d}.mlp.down_proj.weight", .{i}, &.{ h, c.ffn })}, c.ffn),
            .attn_conv = .{ .base = 0, .proj = try dense(gpa, sink, &.{try bf16(ck, "layers.{d}.attention_conv.kernel_projection.weight", .{i}, &.{ dyn, h })}, h) },
            .mlp_conv = .{ .base = 0, .proj = try dense(gpa, sink, &.{try bf16(ck, "layers.{d}.mlp_conv.kernel_projection.weight", .{i}, &.{ dyn, h })}, h) },
            .in_norm = 0,
            .post_norm = 0,
            .q_norm = 0,
            .k_norm = 0,
        };
        const smalls = [_]Tensor{
            try bf16(ck, "layers.{d}.input_layernorm.weight", .{i}, &.{h}),
            try bf16(ck, "layers.{d}.post_attention_layernorm.weight", .{i}, &.{h}),
            try bf16(ck, "layers.{d}.self_attn.q_norm.weight", .{i}, &.{cfg.head_dim}),
            try bf16(ck, "layers.{d}.self_attn.k_norm.weight", .{i}, &.{cfg.head_dim}),
            a_base,
            m_base,
        };
        for (smalls, o) |x, *at| at.* = try small.put(gpa, x);
    }
    const base = try sink.put(small.bytes.items);
    w.norm = base + at_norm;
    w.hidden_norm = base + at_hidden;
    for (w.layers[0..c.layers], offsets[0..c.layers]) |*l, o| {
        l.in_norm = base + o[0];
        l.post_norm = base + o[1];
        l.q_norm = base + o[2];
        l.k_norm = base + o[3];
        l.attn_conv.base = base + o[4];
        l.mlp_conv.base = base + o[5];
    }
    const vr: usize = c.vocab;
    const pred = try bf16(ck, "candidate_selector.predecessor_codebook", .{}, &.{ vr, c.rank });
    const succ = try bf16(ck, "candidate_selector.successor_codebook", .{}, &.{ vr, c.rank });
    w.pred = try gpa.alloc(u16, vr * c.rank);
    @memcpy(std.mem.sliceAsBytes(w.pred), pred.bytes);
    w.succ = try gpa.alloc(u16, vr * c.rank);
    @memcpy(std.mem.sliceAsBytes(w.succ), succ.bytes);
    return w;
}

/// The drafter's config.json through #2260's qualified parse beside `target`, every tensor checked, then onto the GPU.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, target: *const cfg.Config) !Weights {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(bytes);
    const c = try spec.parse(gpa, bytes, target);
    var ck = try Checkpoint.openModel(gpa, io, dir);
    defer core.checkpoint.cache.close(&ck);
    var dev: Device = .{ .d = d, .gpa = gpa };
    errdefer {
        for (dev.buffers.items) |*b| b.free();
        dev.buffers.deinit(gpa);
    }
    var w = try build(gpa, &ck, c, &dev);
    w.buffers = dev.buffers;
    w.bytes = dev.bytes;
    return w;
}

const testing = std.testing;
const tio = testing.io;

fn release(a: std.mem.Allocator) !Config {
    const target = try cfg.parse(a, @embedFile("fixtures/config.json"));
    return spec.parse(a, @embedFile("fixtures/dflash2_config.json"), &target);
}

test "ricursive dflash2 config: the imported parse reads the release, a wrong one is refused by name" {
    const a = testing.allocator;
    const target = try cfg.parse(a, @embedFile("fixtures/config.json"));
    const bytes = @embedFile("fixtures/dflash2_config.json");
    const c = try spec.parse(a, bytes, &target);
    try testing.expectEqualSlices(u32, &.{ 1, 20, 39, 58, 77 }, &c.taps);
    try testing.expectEqual(@as(u32, 16), c.block);
    try testing.expectEqual(@as(u32, 32), c.top);
    try testing.expectEqual(@as(u32, 4224), c.ring(0));
    try testing.expectEqual(@as(u32, 16512), c.ring(4));
    try testing.expect(!c.rope[4] and c.rope[3] and c.global[4] and !c.global[0]);
    const other = try std.mem.replaceOwned(u8, a, bytes, "\"block_size\": 16", "\"block_size\": 8");
    defer a.free(other);
    try testing.expectError(error.UnqualifiedRippleDFlash2, spec.parse(a, other, &target));
    const not = try std.mem.replaceOwned(u8, a, bytes, "DFlash2DraftModel", "Qwen3ForCausalLM");
    defer a.free(not);
    try testing.expectError(error.NotRippleDFlash2, spec.parse(a, not, &target));
}

test "ricursive dflash2 weights: the release's device bytes, about 8.08 GB beside 7.97 GB of unfused matrices" {
    const c = try release(testing.allocator);
    const b = deviceBytes(c);
    const kv_copy = c.layers * packedBytes(2 * kvWidth(c), c.hidden);
    try testing.expect(b - kv_copy > 7_960_000_000 and b - kv_copy < 8_000_000_000);
    try testing.expect(b > 8_070_000_000 and b < 8_110_000_000);
}

/// A tiny drafter: 2 layers, hidden 128, 2 query heads over 1 key head, whole-128 outputs and a padded one (hproj).
fn tiny() Config {
    return .{ .layers = 2, .heads = 2, .kv_heads = 1, .hidden = 128, .ffn = 256, .vocab = 128, .window = 16, .block = 16, .mask = 7, .rank = 64, .top = 32, .group = 4, .conv = 2, .taps = .{ 1, 2, 3, 4, 5 }, .theta = 10000, .eps = 1e-6 };
}

const Entry = struct { name: []const u8, shape: []const usize };

/// Every tensor `c` names, in the release's names.
fn entries(a: std.mem.Allocator, c: Config) ![]Entry {
    var list: std.ArrayList(Entry) = .empty;
    const h: usize = c.hidden;
    const add = struct {
        fn f(l: *std.ArrayList(Entry), al: std.mem.Allocator, comptime fmt: []const u8, args: anytype, shape: []const usize) !void {
            try l.append(al, .{ .name = try std.fmt.allocPrint(al, fmt, args), .shape = try al.dupe(usize, shape) });
        }
    }.f;
    try add(&list, a, "fc.weight", .{}, &.{ h, h * 5 });
    try add(&list, a, "hidden_norm.weight", .{}, &.{h});
    try add(&list, a, "norm.weight", .{}, &.{h});
    try add(&list, a, "candidate_selector.hidden_projection.weight", .{}, &.{ c.rank, h });
    try add(&list, a, "candidate_selector.predecessor_codebook", .{}, &.{ c.vocab, c.rank });
    try add(&list, a, "candidate_selector.successor_codebook", .{}, &.{ c.vocab, c.rank });
    for (0..c.layers) |i| {
        try add(&list, a, "layers.{d}.input_layernorm.weight", .{i}, &.{h});
        try add(&list, a, "layers.{d}.post_attention_layernorm.weight", .{i}, &.{h});
        try add(&list, a, "layers.{d}.self_attn.q_norm.weight", .{i}, &.{cfg.head_dim});
        try add(&list, a, "layers.{d}.self_attn.k_norm.weight", .{i}, &.{cfg.head_dim});
        try add(&list, a, "layers.{d}.self_attn.q_proj.weight", .{i}, &.{ queryWidth(c), h });
        try add(&list, a, "layers.{d}.self_attn.k_proj.weight", .{i}, &.{ kvWidth(c), h });
        try add(&list, a, "layers.{d}.self_attn.v_proj.weight", .{i}, &.{ kvWidth(c), h });
        try add(&list, a, "layers.{d}.self_attn.o_proj.weight", .{i}, &.{ h, queryWidth(c) });
        try add(&list, a, "layers.{d}.mlp.gate_proj.weight", .{i}, &.{ c.ffn, h });
        try add(&list, a, "layers.{d}.mlp.up_proj.weight", .{i}, &.{ c.ffn, h });
        try add(&list, a, "layers.{d}.mlp.down_proj.weight", .{i}, &.{ h, c.ffn });
        for ([_][]const u8{ "attention_conv", "mlp_conv" }) |n| {
            try add(&list, a, "layers.{d}.{s}.base_kernel", .{ i, n }, &.{ 2, c.conv, h });
            try add(&list, a, "layers.{d}.{s}.kernel_projection.weight", .{ i, n }, &.{ c.dyn(), h });
        }
    }
    return list.items;
}

/// A safetensors file of every tensor of `c`, each bf16 with its own pseudo-random bits.
fn writeFixture(a: std.mem.Allocator, dir: std.Io.Dir, c: Config) !void {
    const list = try entries(a, c);
    var header: std.Io.Writer.Allocating = .init(a);
    try header.writer.writeAll("{");
    var at: usize = 0;
    for (list, 0..) |e, j| {
        var n: usize = 2;
        for (e.shape) |x| n *= x;
        try header.writer.print("{s}\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[", .{ if (j == 0) "" else ",", e.name });
        for (e.shape, 0..) |x, i| try header.writer.print("{s}{d}", .{ if (i == 0) "" else ",", x });
        try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ at, at + n });
        at += n;
    }
    try header.writer.writeAll("}");
    const hd = header.written();
    const file = try a.alloc(u8, 8 + hd.len + at);
    std.mem.writeInt(u64, file[0..8], hd.len, .little);
    @memcpy(file[8..][0..hd.len], hd);
    var prng: std.Random.DefaultPrng = .init(0xd2);
    prng.random().bytes(file[8 + hd.len ..]);
    // small finite bf16 everywhere: the sign and a low exponent, the mantissa as drawn
    for (std.mem.bytesAsSlice(u16, file[8 + hd.len ..])) |*x| x.* = (x.* & 0x807f) | 0x3c00;
    try dir.writeFile(tio, .{ .sub_path = "model.safetensors", .data = file });
}

fn tmpPath(a: std.mem.Allocator, tmp: testing.TmpDir) ![]u8 {
    const cwd = try std.process.currentPathAlloc(tio, a);
    return std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
}

/// A sink that keeps what `load` would upload, at fake addresses one 4 KiB apart.
const Recorder = struct {
    gpa: std.mem.Allocator,
    images: std.ArrayList([]u8) = .empty,
    bytes: usize = 0,

    fn put(r: *Recorder, bytes: []const u8) !u64 {
        try r.images.append(r.gpa, try r.gpa.dupe(u8, bytes));
        r.bytes += bytes.len;
        return (r.images.items.len) << 12;
    }

    fn deinit(r: *Recorder) void {
        for (r.images.items) |i| r.gpa.free(i);
        r.images.deinit(r.gpa);
    }

    fn at(r: *const Recorder, ptr: u64) []const u16 {
        return @alignCast(std.mem.bytesAsSlice(u16, r.images.items[(ptr >> 12) - 1]));
    }
};

fn tensorOf(ck: *Checkpoint, comptime fmt: []const u8, args: anytype) !Tensor {
    var buf: [160]u8 = undefined;
    return ck.get(try std.fmt.bufPrint(&buf, fmt, args));
}

test "ricursive dflash2 weights: each fused packing equals its parts packed one by one, the bytes equal the formula" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = tiny();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeFixture(a, tmp.dir, c);
    var ck = try Checkpoint.openModel(testing.allocator, tio, try tmpPath(a, tmp));
    defer ck.close();
    var rec: Recorder = .{ .gpa = testing.allocator };
    defer rec.deinit();
    var w = try build(testing.allocator, &ck, c, &rec);
    defer w.deinit();
    try testing.expectEqual(deviceBytes(c), rec.bytes);
    const h: usize = c.hidden;
    for (w.layers[0..c.layers], 0..) |l, i| {
        const q = try tensorOf(&ck, "layers.{d}.self_attn.q_proj.weight", .{i});
        const k = try tensorOf(&ck, "layers.{d}.self_attn.k_proj.weight", .{i});
        const v = try tensorOf(&ck, "layers.{d}.self_attn.v_proj.weight", .{i});
        const g = try tensorOf(&ck, "layers.{d}.mlp.gate_proj.weight", .{i});
        const u = try tensorOf(&ck, "layers.{d}.mlp.up_proj.weight", .{i});
        try expectFused(a, &rec, l.qkv, &.{ q, k, v }, h);
        try expectFused(a, &rec, l.kv, &.{ k, v }, h);
        try expectFused(a, &rec, l.gate_up, &.{ g, u }, h);
        try expectFused(a, &rec, l.o, &.{try tensorOf(&ck, "layers.{d}.self_attn.o_proj.weight", .{i})}, queryWidth(c));
        try expectFused(a, &rec, l.down, &.{try tensorOf(&ck, "layers.{d}.mlp.down_proj.weight", .{i})}, c.ffn);
        try expectFused(a, &rec, l.attn_conv.proj, &.{try tensorOf(&ck, "layers.{d}.attention_conv.kernel_projection.weight", .{i})}, h);
        try expectFused(a, &rec, l.mlp_conv.proj, &.{try tensorOf(&ck, "layers.{d}.mlp_conv.kernel_projection.weight", .{i})}, h);
        // norms and bases are plain bf16 at 256-byte boundaries
        try testing.expectEqual(@as(u64, 0), l.in_norm % small_align);
        const norm = try tensorOf(&ck, "layers.{d}.input_layernorm.weight", .{i});
        try testing.expectEqualSlices(u8, norm.bytes, smallAt(&rec, l.in_norm)[0..norm.bytes.len]);
        const base = try tensorOf(&ck, "layers.{d}.attention_conv.base_kernel", .{i});
        try testing.expectEqualSlices(u8, base.bytes, smallAt(&rec, l.attn_conv.base)[0..base.bytes.len]);
    }
    try expectFused(a, &rec, w.fc, &.{try tensorOf(&ck, "fc.weight", .{})}, h * 5);
    try expectFused(a, &rec, w.hproj, &.{try tensorOf(&ck, "candidate_selector.hidden_projection.weight", .{})}, h);
    const pred = try tensorOf(&ck, "candidate_selector.predecessor_codebook", .{});
    try testing.expectEqualSlices(u8, pred.bytes, std.mem.sliceAsBytes(w.pred));
}

/// The small image holds the last put; a pointer's bytes start `ptr % 4096`... the recorder's address of that image.
fn smallAt(r: *const Recorder, ptr: u64) []const u8 {
    // addresses are image index << 12 plus an offset into the small image, which is the last image
    const image = r.images.items[r.images.items.len - 1];
    const base: u64 = r.images.items.len << 12;
    return image[ptr - base ..];
}

/// A fused weight's packed words equal each part packed on its own and laid one after another.
fn expectFused(a: std.mem.Allocator, r: *const Recorder, wt: Weight, parts: []const Tensor, k: usize) !void {
    const got = r.at(wt.codes);
    var at: usize = 0;
    var rows: usize = 0;
    for (parts) |p| {
        const n = p.bytes.len / 2 / k;
        const src = try a.alloc(u16, n * k);
        @memcpy(std.mem.sliceAsBytes(src), p.bytes);
        const one = try a.alloc(u16, cuda.modelopt.padded(n) * k);
        cuda.qmmf.packBf16(one, src, n, k);
        try testing.expectEqualSlices(u16, one, got[at..][0..one.len]);
        at += one.len;
        rows += n;
    }
    try testing.expectEqual(got.len, at);
    try testing.expectEqual(@as(u32, @intCast(rows)), wt.n);
}

test "ricursive dflash2 weights: the device loader type-checks" {
    _ = &load;
}
