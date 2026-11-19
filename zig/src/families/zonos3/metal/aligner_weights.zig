//! The forced aligner's weights on Metal from web/aligner (manifest, shards) in fp32: 8-bit matrices times scales.
const std = @import("std");
const Gpu = @import("gpu.zig").Gpu;

const A = std.mem.Allocator;

pub const dim = 1024;
pub const ffn = 4096;
pub const feat = 256;
pub const hop = 256;
pub const ups = 4;
pub const layers = 12;
const dia_width = 2048;

/// y [n] = W [n][k] x (+ b): fp32 on the device, b 0 without a bias.
pub const Linear = struct { w: u64, b: u64 = 0, n: usize, k: usize };
pub const DiaLayer = struct { qkv: Linear, o: Linear, gate_up: Linear, down: Linear };
pub const FaLayer = struct { qkv: Linear, out: Linear, w13: Linear, w2: Linear };
/// A transposed convolution's two phases: a linear over two neighbouring rows each.
pub const Up = struct { even: Linear, odd: Linear };
pub const Head = struct { in: Linear, up: [ups]Up, smooth: Linear, out: Linear };

pub const Weights = struct {
    gpa: A,
    bytes: usize = 0,
    embedding: []f32 = &.{}, // host [256][dim]
    dia_norm: []f32 = &.{},
    final_norm: []f32 = &.{},
    dia: [layers]DiaLayer = undefined,
    fa: [layers]FaLayer = undefined,
    patch: Linear = undefined,
    text_proj: Linear = undefined,
    start: Head = undefined,
    end: Head = undefined,
    mlp0: Linear = undefined,
    start2: Linear = undefined,
    end2: Linear = undefined,
    presence2: Linear = undefined,

    /// The host rows; the buffers go with their owner's release.
    pub fn deinit(w: *Weights) void {
        w.gpa.free(w.embedding);
        w.gpa.free(w.dia_norm);
        w.gpa.free(w.final_norm);
    }
};

const Loader = struct {
    gpa: A,
    io: std.Io,
    g: *Gpu,
    dir: []const u8,
    tensors: std.json.ObjectMap,
    shards: []const std.json.Value,
    held: ?usize = null,
    shard: []u8 = &.{},
    out: *Weights,

    fn bytesOf(l: *Loader, index: usize, offset: usize, n: usize) ![]const u8 {
        if (l.held != index) {
            l.gpa.free(l.shard);
            l.shard = &.{};
            l.held = null;
            const path = try std.fs.path.join(l.gpa, &.{ l.dir, l.shards[index].string });
            defer l.gpa.free(path);
            l.shard = try std.Io.Dir.cwd().readFileAlloc(l.io, path, l.gpa, .limited(1 << 30));
            l.held = index;
        }
        if (offset > l.shard.len or n > l.shard.len - offset) return error.BadAlignerShard;
        return l.shard[offset..][0..n];
    }

    fn int(o: std.json.ObjectMap, key: []const u8) !usize {
        const v = o.get(key) orelse return error.BadAlignerManifest;
        return if (v == .integer and v.integer >= 0) @intCast(v.integer) else error.BadAlignerManifest;
    }

    /// A tensor's values in fp32, row-major (caller frees): `count` of them, 8-bit ones dequantized.
    fn get(l: *Loader, comptime f: []const u8, args: anytype, count: usize) ![]f32 {
        var nb: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&nb, f, args) catch unreachable;
        const e = (l.tensors.get(name) orelse return error.MissingAlignerTensor).object;
        var n: usize = 1;
        for ((e.get("shape") orelse return error.BadAlignerManifest).array.items) |d| n *= @intCast(d.integer);
        if (n != count) return error.ShapeMismatch;
        const kind = (e.get("dtype") orelse return error.BadAlignerManifest).string;
        const index = try int(e, "shard");
        const out = try l.gpa.alloc(f32, n);
        errdefer l.gpa.free(out);
        if (std.mem.eql(u8, kind, "f32")) {
            const raw = try l.bytesOf(index, try int(e, "offset"), n * 4);
            @memcpy(std.mem.sliceAsBytes(out), raw);
        } else if (std.mem.eql(u8, kind, "q8")) {
            if (n % 32 != 0) return error.BadAlignerManifest;
            const q = try l.bytesOf(index, try int(e, "offset"), n);
            for (out, q) |*o, v| o.* = @floatFromInt(@as(i8, @bitCast(v)));
            const scales = try l.bytesOf(index, try int(e, "scales_offset"), n / 32 * 4);
            for (0..n / 32) |block| {
                const scale: f32 = @bitCast(std.mem.readInt(u32, scales[block * 4 ..][0..4], .little));
                for (out[block * 32 ..][0..32]) |*o| o.* *= scale;
            }
        } else return error.BadAlignerManifest;
        for (out) |v| if (!std.math.isFinite(v)) return error.BadAlignerWeights;
        return out;
    }

    fn device(l: *Loader, values: []const f32) !u64 {
        const at = try l.g.upload(std.mem.sliceAsBytes(values));
        l.out.bytes += values.len * 4;
        return at;
    }

    /// A matrix [n][k], its columns scaled by `fold` (the weight of the norm whose rows it reads), and its bias.
    fn linear(l: *Loader, comptime f: []const u8, args: anytype, n: usize, k: usize, fold: ?[]const f32, comptime bias: ?[]const u8) !Linear {
        const w = try l.get(f, args, n * k);
        defer l.gpa.free(w);
        if (fold) |g| for (0..n) |o| for (w[o * k ..][0..k], g) |*v, scale| {
            v.* *= scale;
        };
        var out: Linear = .{ .w = try l.device(w), .n = n, .k = k };
        if (bias) |b| {
            const values = try l.get(b, args, n);
            defer l.gpa.free(values);
            out.b = try l.device(values);
        }
        return out;
    }

    /// Taps [taps][n][k] as a linear over `taps` neighbouring rows: [n][taps * k].
    fn taps(l: *Loader, comptime f: []const u8, args: anytype, count: usize, n: usize, k: usize, comptime bias: []const u8) !Linear {
        const src = try l.get(f, args, count * n * k);
        defer l.gpa.free(src);
        const w = try l.gpa.alloc(f32, src.len);
        defer l.gpa.free(w);
        for (0..count) |t| for (0..n) |o| @memcpy(w[(o * count + t) * k ..][0..k], src[(t * n + o) * k ..][0..k]);
        const b = try l.get(bias, args, n);
        defer l.gpa.free(b);
        return .{ .w = try l.device(w), .b = try l.device(b), .n = n, .k = count * k };
    }

    fn head(l: *Loader, comptime name: []const u8, final: []const f32) !Head {
        var h: Head = undefined;
        h.in = try l.linear("fa." ++ name ++ ".in", .{}, feat, dim, final, "fa." ++ name ++ ".in_b");
        for (&h.up, 0..) |*u, s| {
            u.even = try l.taps("fa." ++ name ++ ".up{d}.even", .{s}, 2, feat, feat, "fa." ++ name ++ ".up{d}.b");
            u.odd = try l.taps("fa." ++ name ++ ".up{d}.odd", .{s}, 2, feat, feat, "fa." ++ name ++ ".up{d}.b");
        }
        h.smooth = try l.taps("fa." ++ name ++ ".smooth", .{}, 5, feat, feat, "fa." ++ name ++ ".smooth_b");
        h.out = try l.linear("fa." ++ name ++ ".out", .{}, feat, feat, null, "fa." ++ name ++ ".out_b");
        return h;
    }
};

/// The export at `dir`: norm weights folded into the matrices that read them, but the text encoder's last (host).
pub fn load(gpa: A, io: std.Io, g: *Gpu, dir: []const u8) !Weights {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "manifest.json" }), a, .limited(1 << 24));
    const root = (try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{})).object;
    const meta = (root.get("meta") orelse return error.BadAlignerManifest).object;
    const wants = .{ .{ "dim", dim }, .{ "n_layers", layers }, .{ "n_heads", 16 }, .{ "ffn_dim", ffn }, .{ "feat_dim", feat }, .{ "patch_window", 2 * hop }, .{ "patch_hop", hop }, .{ "n_upsample", ups }, .{ "dia_hidden", dim }, .{ "dia_layers", layers }, .{ "dia_heads", 16 }, .{ "dia_head_dim", 128 }, .{ "dia_intermediate", ffn } };
    inline for (wants) |want| if (try Loader.int(meta, want[0]) != want[1]) return error.UnsupportedAligner;
    if (!std.mem.eql(u8, (meta.get("format") orelse return error.BadAlignerManifest).string, "fusealign-gguf/1")) return error.UnsupportedAligner;
    var out: Weights = .{ .gpa = gpa };
    errdefer out.deinit();
    var l: Loader = .{ .gpa = gpa, .io = io, .g = g, .dir = dir, .tensors = (root.get("tensors") orelse return error.BadAlignerManifest).object, .shards = (root.get("shards") orelse return error.BadAlignerManifest).array.items, .out = &out };
    defer gpa.free(l.shard);
    out.embedding = try l.get("dia.embedding", .{}, 256 * dim);
    for (&out.dia, 0..) |*L, i| {
        const pre = try l.get("dia.{d}.pre_norm", .{i}, dim);
        defer gpa.free(pre);
        const post = try l.get("dia.{d}.post_norm", .{i}, dim);
        defer gpa.free(post);
        L.qkv = try l.linear("dia.{d}.qkv", .{i}, 3 * dia_width, dim, pre, null);
        L.o = try l.linear("dia.{d}.o", .{i}, dim, dia_width, null, null);
        L.gate_up = try l.linear("dia.{d}.gate_up", .{i}, 2 * ffn, dim, post, null);
        L.down = try l.linear("dia.{d}.down", .{i}, dim, ffn, null, null);
    }
    out.dia_norm = try l.get("dia.norm", .{}, dim);
    // Conv1d(1, dim, 512, stride 256) as a linear over two rows of 256 samples
    out.patch = try l.taps("fa.audio_patch", .{}, 2, dim, hop, "fa.audio_bias");
    out.text_proj = try l.linear("fa.text_proj", .{}, dim, dim, null, "fa.text_bias");
    for (&out.fa, 0..) |*L, i| {
        const attn = try l.get("fa.{d}.attn_norm", .{i}, dim);
        defer gpa.free(attn);
        const feed = try l.get("fa.{d}.ffn_norm", .{i}, dim);
        defer gpa.free(feed);
        L.qkv = try l.linear("fa.{d}.qkv", .{i}, 3 * dim, dim, attn, null);
        L.out = try l.linear("fa.{d}.out", .{i}, dim, dim, null, null);
        L.w13 = try l.linear("fa.{d}.w13", .{i}, 2 * ffn, dim, feed, null);
        L.w2 = try l.linear("fa.{d}.w2", .{i}, dim, ffn, null, null);
    }
    out.final_norm = try l.get("fa.final_norm", .{}, dim);
    out.start = try l.head("start", out.final_norm);
    out.end = try l.head("end", out.final_norm);
    out.mlp0 = try l.linear("fa.mlp0", .{}, 3 * dim, dim, out.final_norm, "fa.mlp0_b");
    out.start2 = try l.linear("fa.start_mlp2", .{}, feat, dim, null, "fa.start_mlp2_b");
    out.end2 = try l.linear("fa.end_mlp2", .{}, feat, dim, null, "fa.end_mlp2_b");
    out.presence2 = try l.linear("fa.presence_mlp2", .{}, 1, dim, null, "fa.presence_mlp2_b");
    return out;
}
