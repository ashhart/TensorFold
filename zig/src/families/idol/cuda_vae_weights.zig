//! Idol's image encoder weights on CUDA: FLUX.2's VAE encoder, quant_conv and batch norm, fp32 of the SDK's bf16.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("vae_config.zig");
const ks = @import("cuda_vae_kernels.zig");

/// A conv's weights [taps * cin, ldw] by tap, then input channel (ldw: cout in whole column tiles), and its bias.
pub const Conv = struct { w: u64, b: u64, cin: usize, cout: usize, ldw: usize, taps: usize };
pub const Norm = struct { g: u64, b: u64 };
pub const Resnet = struct { norm1: Norm, conv1: Conv, norm2: Norm, conv2: Conv, shortcut: ?Conv };
pub const Stage = struct { resnets: [cfg.resnets]Resnet, down: ?Conv };
/// One of the mid block's attention linears as torch holds it, w [512 out, 512 in] (cuBLASLt's X . W^T), and bias.
pub const Linear = struct { w: u64, b: u64 };

pub const Weights = struct {
    buf: cuda.DeviceBuffer,
    conv_in: Conv,
    stages: [4]Stage,
    mid: [2]Resnet,
    attn_norm: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm_out: Norm,
    conv_out: Conv,
    quant: Conv,
    /// The batch norm's running mean and sqrt(running_var + eps), each of 128.
    bn_mean: u64,
    bn_std: u64,

    pub fn deinit(w: *Weights) void {
        w.buf.free();
    }
};

/// The folder `dir` (config.json, diffusion_pytorch_model.safetensors in bf16 or f32): every encoder tensor, packed.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Weights {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, "config.json" }), arena, .limited(1 << 20));
    try cfg.check(gpa, config);
    var c: core.Checkpoint = .{ .gpa = gpa, .io = io };
    defer c.close();
    try c.add(dir, "diffusion_pytorch_model.safetensors");
    var w: Weights = undefined;
    var p: Packer = .{ .c = &c, .arena = arena };
    try layout(&w, &p);
    const left = unread(&c);
    if (left != 0) {
        std.log.err("idol vae: the checkpoint has {d} encoder tensors the encoder does not read", .{left});
        return error.UnexpectedTensors;
    }
    const host = try arena.alloc(f32, p.at);
    w.buf = try cuda.DeviceBuffer.alloc(d, p.at * 4);
    errdefer w.buf.free();
    p = .{ .c = &c, .arena = arena, .out = host, .base = w.buf.ptr };
    try layout(&w, &p);
    try w.buf.upload(0, std.mem.sliceAsBytes(host));
    return w;
}

/// Walks every tensor in the order it is packed: sizes it on the first pass, copies it on the second.
fn layout(w: *Weights, p: *Packer) !void {
    const widths = cfg.widths;
    var name: [96]u8 = undefined;
    w.conv_in = try p.conv("encoder.conv_in", 3, widths[0], 9, false);
    var prev = widths[0];
    for (&w.stages, widths, 0..) |*s, out, i| {
        for (&s.resnets, 0..) |*r, j| {
            r.* = try p.resnet(try std.fmt.bufPrint(&name, "encoder.down_blocks.{d}.resnets.{d}", .{ i, j }), if (j == 0) prev else out, out);
        }
        s.down = if (i + 1 < widths.len) try p.conv(try std.fmt.bufPrint(&name, "encoder.down_blocks.{d}.downsamplers.0.conv", .{i}), out, out, 9, true) else null;
        prev = out;
    }
    const W = widths[3];
    for (&w.mid, 0..) |*r, i| r.* = try p.resnet(try std.fmt.bufPrint(&name, "encoder.mid_block.resnets.{d}", .{i}), W, W);
    const at = "encoder.mid_block.attentions.0";
    w.attn_norm = try p.norm(at ++ ".group_norm", W);
    w.q = try p.linear(at ++ ".to_q", W);
    w.k = try p.linear(at ++ ".to_k", W);
    w.v = try p.linear(at ++ ".to_v", W);
    w.out = try p.linear(at ++ ".to_out.0", W);
    w.norm_out = try p.norm("encoder.conv_norm_out", W);
    w.conv_out = try p.conv("encoder.conv_out", W, 2 * cfg.latent_channels, 9, true);
    w.quant = try p.conv("quant_conv", 2 * cfg.latent_channels, 2 * cfg.latent_channels, 1, true);
    const n = 4 * cfg.latent_channels;
    const mean = try p.tensor("bn.running_mean", &.{n});
    const var_ = try p.tensor("bn.running_var", &.{n});
    const m_at = p.alloc(n);
    const s_at = p.alloc(n);
    w.bn_mean = p.addr(m_at);
    w.bn_std = p.addr(s_at);
    if (p.out) |out| for (out[m_at..][0..n], out[s_at..][0..n], mean, var_) |*mo, *so, mv, v| {
        mo.* = mv;
        so.* = cfg.bnStd(v);
    };
}

/// Encoder-side tensors (vae_config.encoderSide) no `get` took.
fn unread(c: *const core.Checkpoint) usize {
    var n: usize = 0;
    for (c.files.items) |*f| for (f.names.keys()) |k| {
        if (cfg.encoderSide(k) and !c.used.contains(k)) n += 1;
    };
    return n;
}

/// Lays the encoder's tensors out in one fp32 buffer: sizes on the first pass (out null), copies on the second.
const Packer = struct {
    c: *core.Checkpoint,
    arena: std.mem.Allocator,
    out: ?[]f32 = null,
    base: u64 = 0,
    at: usize = 0,

    /// n floats' offset, the next one 256 bytes on.
    fn alloc(p: *Packer, n: usize) usize {
        const r = p.at;
        p.at = std.mem.alignForward(usize, p.at + n, 64);
        return r;
    }

    fn addr(p: *const Packer, off: usize) u64 {
        return p.base + off * 4;
    }

    /// Tensor `name` of `shape` as the bf16 values the SDK computes with (a bf16 file's own, an f32 file's rounded).
    fn tensor(p: *Packer, name: []const u8, shape: []const usize) ![]const f32 {
        const t = try p.c.get(name);
        if ((t.dtype != .f32 and t.dtype != .bf16) or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
            std.log.err("idol vae: {s} is {t} {any}; the encoder reads bf16 or f32 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
            return error.UnexpectedTensor;
        }
        if (p.out == null) return &.{};
        const n = t.numel();
        const v = try p.arena.alloc(f32, n);
        switch (t.dtype) {
            .bf16 => for (v, 0..) |*o, i| {
                o.* = @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16);
            },
            else => for (v, 0..) |*o, i| {
                o.* = cfg.bf16(@bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little)));
            },
        }
        return v;
    }

    /// A copied vector `name` of n.
    fn copy(p: *Packer, name: []const u8, n: usize) !u64 {
        const v = try p.tensor(name, &.{n});
        const r = p.alloc(n);
        if (p.out) |out| @memcpy(out[r..][0..n], v);
        return p.addr(r);
    }

    /// Convolution `name` (.weight [cout, cin, k, k], .bias) in tf_iv_conv's layout (conv_in's [27, 128] unpadded).
    fn conv(p: *Packer, name: []const u8, cin: usize, cout: usize, taps: usize, padded: bool) !Conv {
        var buf: [96]u8 = undefined;
        const k: usize = if (taps == 1) 1 else 3;
        const ldw = if (padded) ks.ldwOf(cout) else cout;
        const w = try p.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), &.{ cout, cin, k, k });
        const r = p.alloc(taps * cin * ldw);
        if (p.out) |out| {
            const dst = out[r..][0 .. taps * cin * ldw];
            @memset(dst, 0);
            for (0..cout) |o| for (0..cin) |i| for (0..taps) |tap| {
                dst[(tap * cin + i) * ldw + o] = w[(o * cin + i) * taps + tap];
            };
        }
        const b = try p.copy(try std.fmt.bufPrint(&buf, "{s}.bias", .{name}), cout);
        return .{ .w = p.addr(r), .b = b, .cin = cin, .cout = cout, .ldw = ldw, .taps = taps };
    }

    /// Linear `name` (.weight [width, width], .bias) as stored.
    fn linear(p: *Packer, name: []const u8, width: usize) !Linear {
        var buf: [96]u8 = undefined;
        const w = try p.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), &.{ width, width });
        const r = p.alloc(width * width);
        if (p.out) |out| @memcpy(out[r..][0 .. width * width], w);
        return .{ .w = p.addr(r), .b = try p.copy(try std.fmt.bufPrint(&buf, "{s}.bias", .{name}), width) };
    }

    fn norm(p: *Packer, name: []const u8, width: usize) !Norm {
        var buf: [96]u8 = undefined;
        return .{ .g = try p.copy(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), width), .b = try p.copy(try std.fmt.bufPrint(&buf, "{s}.bias", .{name}), width) };
    }

    fn resnet(p: *Packer, name: []const u8, cin: usize, cout: usize) !Resnet {
        var buf: [96]u8 = undefined;
        return .{
            .norm1 = try p.norm(try std.fmt.bufPrint(&buf, "{s}.norm1", .{name}), cin),
            .conv1 = try p.conv(try std.fmt.bufPrint(&buf, "{s}.conv1", .{name}), cin, cout, 9, true),
            .norm2 = try p.norm(try std.fmt.bufPrint(&buf, "{s}.norm2", .{name}), cout),
            .conv2 = try p.conv(try std.fmt.bufPrint(&buf, "{s}.conv2", .{name}), cout, cout, 9, true),
            .shortcut = if (cin != cout) try p.conv(try std.fmt.bufPrint(&buf, "{s}.conv_shortcut", .{name}), cin, cout, 1, true) else null,
        };
    }
};

test "conv weights go tap-major, then input channel, into rows of whole column tiles" {
    try std.testing.expectEqual(@as(usize, 128), ks.ldwOf(64));
    try std.testing.expectEqual(@as(usize, 512), ks.ldwOf(512));
    try std.testing.expect(cfg.encoderSide("encoder.conv_in.weight") and cfg.encoderSide("bn.running_var"));
    try std.testing.expect(!cfg.encoderSide("decoder.conv_in.weight") and !cfg.encoderSide("post_quant_conv.bias"));
}
