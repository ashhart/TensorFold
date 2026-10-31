//! Idol's image encoder weights: FLUX.2's VAE encoder, quant_conv and batch norm statistics, as the SDK holds them.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const ks = @import("vae_kernels.zig");
const Ref = ks.Ref;

/// The encoder's widths by stage, and the rest of the released geometry.
pub const widths = [4]usize{ 128, 256, 512, 512 };
pub const latent_channels = 32;
pub const resnets = 2;
pub const batch_norm_eps: f32 = 1e-4;

/// Accepts vae/config.json at FLUX.2's released encoder geometry, else an error.
pub fn check(gpa: std.mem.Allocator, text: []const u8) !void {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadConfig;
    const o = doc.value.object;
    const name = o.get("_class_name") orelse return error.BadConfig;
    if (name != .string or !std.mem.eql(u8, name.string, "AutoencoderKLFlux2")) return error.NotFlux2Vae;
    const want = .{ .{ "in_channels", 3 }, .{ "latent_channels", latent_channels }, .{ "layers_per_block", resnets }, .{ "norm_num_groups", ks.groups } };
    inline for (want) |kv| {
        const v = o.get(kv[0]) orelse return error.BadConfig;
        if (v != .integer or v.integer != kv[1]) {
            std.log.warn("idol vae: {s} is not {d}; the encoder runs the released geometry only", .{ kv[0], kv[1] });
            return error.UnsupportedGeometry;
        }
    }
    const lists = .{ .{ "block_out_channels", [_]i64{ 128, 256, 512, 512 } }, .{ "patch_size", [_]i64{ 2, 2 } } };
    inline for (lists) |kv| {
        const v = o.get(kv[0]) orelse return error.BadConfig;
        if (v != .array or v.array.items.len != kv[1].len) return error.UnsupportedGeometry;
        for (v.array.items, kv[1]) |item, n| if (item != .integer or item.integer != n) return error.UnsupportedGeometry;
    }
    inline for (.{ "mid_block_add_attention", "use_quant_conv" }) |f| {
        const v = o.get(f) orelse return error.BadConfig;
        if (v != .bool or !v.bool) return error.UnsupportedGeometry;
    }
    const blocks = o.get("down_block_types") orelse return error.BadConfig;
    if (blocks != .array or blocks.array.items.len != widths.len) return error.UnsupportedGeometry;
    for (blocks.array.items) |b| if (b != .string or !std.mem.eql(u8, b.string, "DownEncoderBlock2D")) return error.UnsupportedGeometry;
    const act = o.get("act_fn") orelse return error.BadConfig;
    if (act != .string or !std.mem.eql(u8, act.string, "silu")) return error.UnsupportedGeometry;
    const bn = o.get("batch_norm_eps") orelse return error.BadConfig;
    const bn_eps: f64 = switch (bn) {
        .float => |f| f,
        .integer => |n| @floatFromInt(n),
        else => return error.BadConfig,
    };
    if (@as(f32, @floatCast(bn_eps)) != batch_norm_eps) return error.UnsupportedGeometry;
}

/// A convolution's weights (1x1 [cin, ldw], stride-2 3x3 [9 * cin, ldw], 3x3 G g G^T [16, cin, ldw]) and bias.
pub const Conv = struct { w: Ref, b: Ref, cin: usize, cout: usize, ldw: usize, kind: Kind };
pub const Kind = enum { one, wino, strided, input };
pub const Norm = struct { g: Ref, b: Ref };
pub const Resnet = struct { norm1: Norm, conv1: Conv, norm2: Norm, conv2: Conv, shortcut: ?Conv };
pub const Stage = struct { resnets: [resnets]Resnet, down: ?Conv };

pub const Weights = struct {
    buf: mtl.Buffer,
    conv_in: Conv,
    stages: [4]Stage,
    mid: [2]Resnet,
    attn_norm: Norm,
    qkv: Conv,
    attn_out: Conv,
    norm_out: Norm,
    conv_out: Conv,
    quant: Conv,
    /// The batch norm's running mean and sqrt(running_var + eps), each of 128.
    bn_mean: Ref,
    bn_std: Ref,

    pub fn deinit(w: *Weights) void {
        w.buf.deinit();
    }
};

/// The bf16 value an fp32 weight rounds to (to nearest, ties to even), as torch's cast to bf16 makes it.
pub fn bf16(v: f32) f32 {
    if (std.math.isNan(v)) return v;
    const bits: u32 = @bitCast(v);
    const rounded = bits +% (0x7fff + ((bits >> 16) & 1));
    return @bitCast(rounded & 0xffff0000);
}

/// The folder `dir` (config.json, diffusion_pytorch_model.safetensors in bf16 or f32): every encoder tensor, packed.
pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, "config.json" }), arena, .limited(1 << 20));
    try check(gpa, config);
    var c = ck.Checkpoint.init(gpa);
    defer c.deinit();
    try c.addFile(device, try std.fmt.allocPrintSentinel(arena, "{s}/diffusion_pytorch_model.safetensors", .{dir}, 0), "");
    var w: Weights = undefined;
    var p: Packer = .{ .c = &c, .arena = arena };
    try layout(&w, &p);
    if (p.read != expected(&c)) {
        std.log.err("idol vae: the checkpoint has {d} encoder tensors; the encoder reads {d}", .{ expected(&c), p.read });
        return error.UnexpectedTensors;
    }
    w.buf = try device.buffer(p.at * 4, mtl.ResourceOptions.shared);
    errdefer w.buf.deinit();
    p.out = @as([*]f32, @ptrCast(@alignCast(w.buf.contents())))[0..p.at];
    p.at = 0;
    p.read = 0;
    p.buf = w.buf;
    try layout(&w, &p);
    return w;
}

/// Walks every tensor in the order it is packed: sizes it on the first pass, copies it on the second.
fn layout(w: *Weights, p: *Packer) !void {
    var name: [96]u8 = undefined;
    w.conv_in = try p.conv("encoder.conv_in", 3, widths[0], .input);
    var prev = widths[0];
    for (&w.stages, widths, 0..) |*s, out, i| {
        for (&s.resnets, 0..) |*r, j| {
            r.* = try p.resnet(try std.fmt.bufPrint(&name, "encoder.down_blocks.{d}.resnets.{d}", .{ i, j }), if (j == 0) prev else out, out);
        }
        s.down = if (i + 1 < widths.len) try p.conv(try std.fmt.bufPrint(&name, "encoder.down_blocks.{d}.downsamplers.0.conv", .{i}), out, out, .strided) else null;
        prev = out;
    }
    const W = widths[3];
    for (&w.mid, 0..) |*r, i| r.* = try p.resnet(try std.fmt.bufPrint(&name, "encoder.mid_block.resnets.{d}", .{i}), W, W);
    const at = "encoder.mid_block.attentions.0";
    w.attn_norm = try p.norm(at ++ ".group_norm", W);
    w.qkv = try p.linears(&.{ at ++ ".to_q", at ++ ".to_k", at ++ ".to_v" }, W);
    w.attn_out = try p.linears(&.{at ++ ".to_out.0"}, W);
    w.norm_out = try p.norm("encoder.conv_norm_out", W);
    w.conv_out = try p.conv("encoder.conv_out", W, 2 * latent_channels, .wino);
    w.quant = try p.conv("quant_conv", 2 * latent_channels, 2 * latent_channels, .one);
    const n = 4 * latent_channels;
    const mean = try p.vector("bn.running_mean", n);
    const var_ = try p.vector("bn.running_var", n);
    w.bn_mean = p.alloc(n);
    w.bn_std = p.alloc(n);
    if (p.out) |out| {
        // the SDK's sqrt(var + eps) in bf16, here in float64 rounded once
        for (out[w.bn_mean.off / 4 ..][0..n], out[w.bn_std.off / 4 ..][0..n], mean, var_) |*mo, *so, mv, v| {
            mo.* = mv;
            so.* = @floatCast(@sqrt(@as(f64, v) + 1e-4));
        }
    }
}

/// The checkpoint's encoder-side tensors: the encoder's, quant_conv's and the batch norm's two statistics.
fn expected(c: *const ck.Checkpoint) usize {
    var n: usize = 0;
    var it = c.tensors.keyIterator();
    while (it.next()) |k| {
        if (std.mem.startsWith(u8, k.*, "encoder.") or std.mem.startsWith(u8, k.*, "quant_conv.") or
            std.mem.eql(u8, k.*, "bn.running_mean") or std.mem.eql(u8, k.*, "bn.running_var")) n += 1;
    }
    return n;
}

/// Lays the encoder's tensors out in one fp32 buffer: sizes on the first pass (out null), copies on the second.
const Packer = struct {
    c: *const ck.Checkpoint,
    arena: std.mem.Allocator,
    buf: mtl.Buffer = undefined,
    out: ?[]f32 = null,
    at: usize = 0,
    read: usize = 0,

    fn alloc(p: *Packer, n: usize) Ref {
        const r: Ref = .{ .buf = p.buf, .off = p.at * 4 };
        p.at = std.mem.alignForward(usize, p.at + n, 64);
        return r;
    }

    /// Tensor `name` of `shape` as the bf16 values the SDK computes with (a bf16 file's own, an f32 file's rounded).
    fn tensor(p: *Packer, name: []const u8, shape: []const usize) ![]const f32 {
        const t = try p.c.get(name);
        if ((t.dtype != .f32 and t.dtype != .bf16) or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
            std.log.err("idol vae: {s} is {t} {any}; the encoder reads bf16 or f32 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
            return error.UnexpectedTensor;
        }
        p.read += 1;
        if (p.out == null) return &.{};
        const n = t.count();
        const v = try p.arena.alloc(f32, n);
        switch (t.dtype) {
            .bf16 => for (v, t.host(u16)) |*o, h| {
                o.* = @bitCast(@as(u32, h) << 16);
            },
            else => for (v, t.host(f32)) |*o, f| {
                o.* = bf16(f);
            },
        }
        return v;
    }

    fn vector(p: *Packer, name: []const u8, n: usize) ![]const f32 {
        return p.tensor(name, &.{n});
    }

    /// A copied vector `name` of n.
    fn copy(p: *Packer, name: []const u8, n: usize) !Ref {
        const v = try p.vector(name, n);
        const r = p.alloc(n);
        if (p.out) |out| @memcpy(out[r.off / 4 ..][0..n], v);
        return r;
    }

    /// Convolution `name` (.weight [cout, cin, k, k], .bias) in `kind`'s layout, its columns padded to ldw.
    fn conv(p: *Packer, name: []const u8, cin: usize, cout: usize, kind: Kind) !Conv {
        var buf: [96]u8 = undefined;
        const taps: usize = if (kind == .one) 1 else 9;
        const k: usize = if (kind == .one) 1 else 3;
        const ldw = if (kind == .input) cout else std.mem.alignForward(usize, cout, 64);
        const rows: usize = switch (kind) {
            .one => cin,
            .wino => 16 * cin,
            .strided, .input => 9 * cin,
        };
        const w = try p.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), &.{ cout, cin, k, k });
        const r = p.alloc(rows * ldw);
        if (p.out) |out| {
            const dst = out[r.off / 4 ..][0 .. rows * ldw];
            @memset(dst, 0);
            for (0..cout) |o| for (0..cin) |i| {
                const g = w[(o * cin + i) * taps ..][0..taps];
                switch (kind) {
                    .one => dst[i * ldw + o] = g[0],
                    .wino => for (ks.transform(g[0..9]), 0..) |v, pt| {
                        dst[(pt * cin + i) * ldw + o] = v;
                    },
                    .strided, .input => for (g, 0..) |v, tap| {
                        dst[(tap * cin + i) * ldw + o] = v;
                    },
                }
            };
        }
        const b = try p.copy(try std.fmt.bufPrint(&buf, "{s}.bias", .{name}), cout);
        return .{ .w = r, .b = b, .cin = cin, .cout = cout, .ldw = ldw, .kind = kind };
    }

    /// Linear layers `names` (.weight [width, width], .bias) stacked as one 1x1 convolution.
    fn linears(p: *Packer, names_: []const []const u8, width: usize) !Conv {
        var buf: [96]u8 = undefined;
        const n = names_.len * width;
        const r = p.alloc(width * n);
        for (names_, 0..) |name, part| {
            const w = try p.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), &.{ width, width });
            if (p.out) |out| {
                const dst = out[r.off / 4 ..][0 .. width * n];
                for (0..width) |o| for (0..width) |i| {
                    dst[i * n + part * width + o] = w[o * width + i];
                };
            }
        }
        const b = p.alloc(n);
        for (names_, 0..) |name, i| {
            const v = try p.vector(try std.fmt.bufPrint(&buf, "{s}.bias", .{name}), width);
            if (p.out) |out| @memcpy(out[b.off / 4 + i * width ..][0..width], v);
        }
        return .{ .w = r, .b = b, .cin = width, .cout = n, .ldw = n, .kind = .one };
    }

    fn norm(p: *Packer, name: []const u8, width: usize) !Norm {
        var buf: [96]u8 = undefined;
        return .{ .g = try p.copy(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), width), .b = try p.copy(try std.fmt.bufPrint(&buf, "{s}.bias", .{name}), width) };
    }

    fn resnet(p: *Packer, name: []const u8, cin: usize, cout: usize) !Resnet {
        var buf: [96]u8 = undefined;
        return .{
            .norm1 = try p.norm(try std.fmt.bufPrint(&buf, "{s}.norm1", .{name}), cin),
            .conv1 = try p.conv(try std.fmt.bufPrint(&buf, "{s}.conv1", .{name}), cin, cout, .wino),
            .norm2 = try p.norm(try std.fmt.bufPrint(&buf, "{s}.norm2", .{name}), cout),
            .conv2 = try p.conv(try std.fmt.bufPrint(&buf, "{s}.conv2", .{name}), cout, cout, .wino),
            .shortcut = if (cin != cout) try p.conv(try std.fmt.bufPrint(&buf, "{s}.conv_shortcut", .{name}), cin, cout, .one) else null,
        };
    }
};

test "only FLUX.2's released encoder geometry passes" {
    const a = std.testing.allocator;
    const sample =
        \\{"_class_name": "AutoencoderKLFlux2", "act_fn": "silu", "batch_norm_eps": 0.0001, "block_out_channels": [128, 256, 512, 512],
        \\ "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"], "in_channels": 3,
        \\ "latent_channels": 32, "layers_per_block": 2, "mid_block_add_attention": true, "norm_num_groups": 32,
        \\ "patch_size": [2, 2], "use_quant_conv": true}
    ;
    try check(a, sample);
    const narrow = try std.mem.replaceOwned(u8, a, sample, "[128, 256, 512, 512]", "[96, 192, 384, 384]");
    defer a.free(narrow);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, narrow));
    const no_quant = try std.mem.replaceOwned(u8, a, sample, "\"use_quant_conv\": true", "\"use_quant_conv\": false");
    defer a.free(no_quant);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, no_quant));
    const other = try std.mem.replaceOwned(u8, a, sample, "AutoencoderKLFlux2", "AutoencoderKL");
    defer a.free(other);
    try std.testing.expectError(error.NotFlux2Vae, check(a, other));
}

test "an f32 weight rounds to bf16 as torch's cast does: to nearest, ties to even" {
    try std.testing.expectEqual(@as(f32, 1.0), bf16(1.0 + 0x1p-9));
    try std.testing.expectEqual(@as(f32, 1.0 + 0x1p-7), bf16(1.0 + 0x1p-7 + 0x1p-9));
    try std.testing.expectEqual(@as(f32, 1.0 + 0x1p-6), bf16(1.0 + 0x1p-7 + 0x1p-8));
    try std.testing.expectEqual(@as(f32, 1.0 + 0x1p-7), bf16(1.0 + 0x1p-7 + 0x1p-8 - 0x1p-20));
}
