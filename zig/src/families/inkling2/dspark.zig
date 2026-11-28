//! Inkling 2 Mini DSpark checkpoint weights and committed per-stream caches.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const wts = @import("loop_weights.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Lane = cuda.qmmf.Weight;

pub const hidden = 2304;
pub const query = 3072;
pub const kv = 1024;
pub const inter = 9216;
pub const rank = 256;
pub const markov_vocab = 201024;
/// The block's rows: the pending token, then seven masks; each mask row drafts one step.
pub const rows = 8;
pub const depth = rows - 1;
/// Context rows one absorb launch set takes.
pub const absorb_rows = 64;
pub const taps = [_]u32{ 5, 23, 41, 48 };
/// The local rings retain the last 1,024 committed keys plus eight transient draft keys.
pub const local_ring = 1024 + rows;

/// The global ring retains `capacity` committed keys plus eight transient draft keys.
pub fn globalRing(capacity: u32) u32 {
    return capacity + rows;
}

pub const Proj = struct { qkv: Lane, ctx_kv: Lane, o: Lane, gu: Lane, down: Lane };
const Norms = struct { input: u64, post: u64, q: u64, k: u64, k_ctx: u64 };

pub const Weights = struct {
    proj: [3]Proj,
    norms: [3]Norms,
    alpha: u64, // fp32 softmax of the checkpoint's four rows of fuse_weight
    dflare: u64, // bf16 [4][H]: the layers' then the Markov head's
    mask: u64,
    norm: u64,
    hidden_w: u64, // bf16 [H][R]
    hidden_b: u64,
    markov: u64, // bf16 [201024][R]
    rnn: u64, // bf16 [768][5120]
    rnn_b: u64,
    unembed: Lane, // packed BF16 copy of the target's output table
    buffers: std.ArrayList(DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights, gpa: std.mem.Allocator) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(gpa);
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    ck: *core.checkpoint.Checkpoint,
    w: *Weights,

    fn tensor(l: Loader, name: []const u8, dims: []const usize) !core.checkpoint.Tensor {
        const t = l.ck.get(name) catch |e| {
            std.log.err("inkling2 dspark: mtp.safetensors has no {s}", .{name});
            return e;
        };
        if (t.dtype != .bf16 or t.rank != dims.len) return error.DSparkTensorShape;
        for (dims, 0..) |n, i| if (t.dim(i) != n) return error.DSparkTensorShape;
        return t;
    }

    fn upload(l: Loader, bytes: []const u8) !u64 {
        var b = try DeviceBuffer.fromHost(l.d, bytes);
        errdefer b.free();
        try l.w.buffers.append(l.gpa, b);
        return b.ptr;
    }

    fn raw(l: Loader, name: []const u8, dims: []const usize) !u64 {
        return l.upload((try l.tensor(name, dims)).bytes);
    }

    const Piece = struct { suffix: []const u8, rows: usize };

    /// Join checkpoint projections before packing so the drafter projects all streams in one GEMM.
    fn joined(l: Loader, layer: usize, pieces: []const Piece, k: usize) !Lane {
        var n: usize = 0;
        for (pieces) |piece| n += piece.rows;
        const src = try l.gpa.alloc(u16, n * k);
        defer l.gpa.free(src);
        var at: usize = 0;
        for (pieces) |piece| {
            var name: [96]u8 = undefined;
            const key = try std.fmt.bufPrint(&name, "layers.{d}.{s}", .{ layer, piece.suffix });
            const t = try l.tensor(key, &.{ piece.rows, k });
            @memcpy(std.mem.sliceAsBytes(src[at..][0 .. piece.rows * k]), t.bytes);
            at += piece.rows * k;
        }
        const dest = try l.gpa.alloc(u16, cuda.modelopt.padded(n) * k);
        defer l.gpa.free(dest);
        cuda.qmmf.packBf16(dest, src, n, k);
        return cuda.modelopt.bf16Weight(try l.upload(std.mem.sliceAsBytes(dest)), n, k);
    }
};

/// Load mtp.safetensors and a packed target unembedding for the shared BF16 matmul lane.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, target_dir: []const u8, dir: []const u8, vocab: usize, valid: usize) !Weights {
    var ck: core.checkpoint.Checkpoint = .{ .gpa = gpa, .io = io };
    defer ck.close();
    ck.add(dir, "mtp.safetensors") catch |e| {
        std.log.err("inkling2 dspark: no mtp.safetensors in {s}", .{dir});
        return e;
    };
    var w: Weights = undefined;
    w.buffers = .empty;
    errdefer w.deinit(gpa);
    const l: Loader = .{ .gpa = gpa, .d = d, .ck = &ck, .w = &w };
    var dflare: [4 * hidden]u16 = undefined;
    for (0..3) |i| {
        var b: [96]u8 = undefined;
        const n = struct {
            fn at(buf: []u8, layer: usize, s: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ layer, s });
            }
        }.at;
        w.proj[i] = .{
            .qkv = try l.joined(i, &.{
                .{ .suffix = "self_attn.q_proj.weight", .rows = query },
                .{ .suffix = "self_attn.k_proj.weight", .rows = kv },
                .{ .suffix = "self_attn.v_proj.weight", .rows = kv },
            }, hidden),
            .ctx_kv = try l.joined(i, &.{
                .{ .suffix = "self_attn.k_proj_ctx.weight", .rows = kv },
                .{ .suffix = "self_attn.v_proj_ctx.weight", .rows = kv },
            }, hidden),
            .o = try l.joined(i, &.{.{ .suffix = "self_attn.o_proj.weight", .rows = hidden }}, query),
            .gu = try l.joined(i, &.{
                .{ .suffix = "mlp.gate_proj.weight", .rows = inter },
                .{ .suffix = "mlp.up_proj.weight", .rows = inter },
            }, hidden),
            .down = try l.joined(i, &.{.{ .suffix = "mlp.down_proj.weight", .rows = hidden }}, inter),
        };
        w.norms[i] = .{
            .input = try l.raw(try n(&b, i, "input_layernorm.weight"), &.{hidden}),
            .post = try l.raw(try n(&b, i, "post_attention_layernorm.weight"), &.{hidden}),
            .q = try l.raw(try n(&b, i, "self_attn.q_norm.weight"), &.{128}),
            .k = try l.raw(try n(&b, i, "self_attn.k_norm.weight"), &.{128}),
            .k_ctx = try l.raw(try n(&b, i, "self_attn.k_norm_ctx.weight"), &.{128}),
        };
        @memcpy(std.mem.sliceAsBytes(dflare[i * hidden ..][0..hidden]), (try l.tensor(try n(&b, i, "dflare_norm.weight"), &.{hidden})).bytes);
    }
    @memcpy(std.mem.sliceAsBytes(dflare[3 * hidden ..]), (try l.tensor("markov_head.dflare_norm.weight", &.{hidden})).bytes);
    w.dflare = try l.upload(std.mem.sliceAsBytes(&dflare));
    const fuse = try l.tensor("fuse_weight", &.{ 4, 4 });
    const fuse_src: [*]align(1) const u16 = @ptrCast(fuse.bytes.ptr);
    var alpha: [16]f32 = undefined;
    for (0..4) |f| {
        var peak: f32 = -std.math.inf(f32);
        for (0..4) |j| {
            const v: f32 = @bitCast(@as(u32, fuse_src[f * 4 + j]) << 16);
            peak = @max(peak, v);
            alpha[f * 4 + j] = v;
        }
        var sum: f32 = 0;
        for (0..4) |j| {
            alpha[f * 4 + j] = @exp(alpha[f * 4 + j] - peak);
            sum += alpha[f * 4 + j];
        }
        for (0..4) |j| alpha[f * 4 + j] /= sum;
    }
    w.alpha = try l.upload(std.mem.sliceAsBytes(&alpha));
    w.mask = try l.raw("mask_embed", &.{hidden});
    w.norm = try l.raw("norm.weight", &.{hidden});
    w.hidden_w = try l.raw("markov_head.hidden_out.weight", &.{ hidden, rank });
    w.hidden_b = try l.raw("markov_head.hidden_out.bias", &.{hidden});
    w.markov = try l.raw("markov_head.markov_w1.weight", &.{ markov_vocab, rank });
    w.rnn = try l.raw("markov_head.rnn_proj.weight", &.{ 3 * rank, 2 * hidden + 2 * rank });
    w.rnn_b = try l.raw("markov_head.rnn_proj.bias", &.{3 * rank});
    var target = try core.Checkpoint.openModelPrefixes(gpa, io, target_dir, &.{"model.llm.unembed.weight"});
    defer target.close();
    const source = try target.expect("model.llm.unembed.weight", .bf16, &.{ vocab, hidden });
    if (valid == 0 or valid > vocab) return error.UnsupportedDSparkVocab;
    var aligned: ?[]u16 = null;
    defer if (aligned) |buf| gpa.free(buf);
    const src: []const u16 = if (@intFromPtr(source.bytes.ptr) % @alignOf(u16) == 0)
        @as([*]const u16, @ptrCast(@alignCast(source.bytes.ptr)))[0 .. valid * hidden]
    else blk: {
        const buf = try gpa.alloc(u16, valid * hidden);
        @memcpy(std.mem.sliceAsBytes(buf), source.bytes[0 .. valid * hidden * 2]);
        aligned = buf;
        break :blk buf;
    };
    const table = try gpa.alloc(u16, cuda.modelopt.padded(valid) * hidden);
    defer gpa.free(table);
    cuda.qmmf.packBf16(table, src, valid, hidden);
    w.unembed = cuda.modelopt.bf16Weight(try l.upload(std.mem.sliceAsBytes(table)), valid, hidden);
    return w;
}

pub const Kernels = @import("dspark_ops.zig").Kernels;

/// One stream's committed context: each layer's K/V ring, the newest fused Markov feature, and its span.
pub const Context = struct {
    k: [3]DeviceBuffer,
    v: [3]DeviceBuffer,
    rings: [3]u32,
    limit: u32,
    newest: DeviceBuffer,
    from: u64 = 0,
    end: u64 = 0,

    /// `capacity`: the global layer's longest committed context; `rings[1]` adds `rows` transient slots.
    pub fn init(d: *const cuda.Driver, capacity: u32) !Context {
        if (capacity == 0 or capacity > std.math.maxInt(u32) - rows) return error.DSparkContextFull;
        var c: Context = .{
            .k = undefined,
            .v = undefined,
            .rings = .{ local_ring, globalRing(capacity), local_ring },
            .limit = capacity,
            .newest = undefined,
        };
        var made: usize = 0;
        errdefer for (0..made) |i| {
            c.k[i].free();
            c.v[i].free();
        };
        for (c.rings, 0..) |ring, i| {
            c.k[i] = try DeviceBuffer.alloc(d, @as(usize, ring) * kv * 2);
            c.v[i] = DeviceBuffer.alloc(d, @as(usize, ring) * kv * 2) catch |e| {
                c.k[i].free();
                return e;
            };
            made += 1;
        }
        c.newest = try DeviceBuffer.alloc(d, hidden * 2);
        return c;
    }

    pub fn deinit(c: *Context) void {
        c.newest.free();
        for (0..3) |i| {
            c.k[i].free();
            c.v[i].free();
        }
    }

    pub const Span = struct { from: u64, end: u64 };

    /// Reject overlap and overflow before device work; a gap restarts the committed span.
    pub fn admit(c: *const Context, start: u64, n: usize) !Span {
        if (start < c.end) return error.DraftContextGap;
        const from = if (start > c.end) start else c.from;
        if (start + n - from > c.limit) return error.DSparkContextFull;
        return .{ .from = from, .end = start + n };
    }
};

test "a rejected absorb leaves the context's span whole and the next one lands" {
    var c: Context = .{
        .k = undefined,
        .v = undefined,
        .rings = .{ local_ring, globalRing(64), local_ring },
        .limit = 64,
        .newest = undefined,
        .from = 10,
        .end = 20,
    };
    try std.testing.expectError(error.DSparkContextFull, c.admit(100, 65));
    try std.testing.expectError(error.DraftContextGap, c.admit(19, 1));
    try std.testing.expectEqual(@as(u64, 10), c.from);
    try std.testing.expectEqual(@as(u64, 20), c.end);
    try std.testing.expectEqual(Context.Span{ .from = 10, .end = 74 }, try c.admit(20, 54));
    const s = try c.admit(100, 64);
    try std.testing.expectEqual(@as(u64, 100), s.from);
    try std.testing.expectEqual(@as(u64, 164), s.end);
    c.from, c.end = .{ s.from, s.end };
    try std.testing.expectError(error.DraftContextGap, c.admit(163, 1));
    try std.testing.expectEqual(@as(u64, 164), c.end);
    // Transient block rows fit past the committed context limit without aliasing committed slots.
    try std.testing.expectEqual(@as(u32, 64 + rows), c.rings[1]);
    try std.testing.expect(c.end + rows - c.from <= c.rings[1]);
}

/// The target's embedding and unembedding remain the plain server's exact weights.
pub const Target = struct { w: *const wts.Weights, lane: *const cuda.qmmf.Lane, unembed: Lane };
pub const Head = @import("dspark_head.zig").Head;
