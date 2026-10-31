//! MOSS-Image's image decoder on Metal: FLUX.2's VAE decoder (the checkpoint's vae/) in fp32, latents to 8-bit RGB.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const cfg = @import("config.zig");
const Ref = @import("../../core/qwen_vl/weights.zig").Ref;
const bind = @import("kernels.zig").Kernels.bind;

/// The decoder's widths from its last stage to its first (decoder_block_out_channels reversed), and the rest.
pub const widths = [4]usize{ 384, 384, 192, 96 };
pub const latent_channels = 32;
pub const groups = 32;
pub const resnets = 3;
pub const eps: f32 = 1e-6;
pub const batch_norm_eps: f32 = 1e-4;
/// Pixels a GroupNorm statistics part sums.
pub const chunk = 256;

/// Accepts vae/config.json at FLUX.2's released decoder geometry, else an error.
pub fn check(gpa: std.mem.Allocator, text: []const u8) !void {
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadConfig;
    const o = doc.value.object;
    const name = o.get("_class_name") orelse return error.BadConfig;
    if (name != .string or !std.mem.eql(u8, name.string, "AutoencoderKLFlux2")) return error.NotFlux2Vae;
    const want = .{ .{ "latent_channels", latent_channels }, .{ "layers_per_block", resnets - 1 }, .{ "norm_num_groups", groups }, .{ "out_channels", 3 } };
    inline for (want) |kv| {
        const v = o.get(kv[0]) orelse return error.BadConfig;
        if (v != .integer or v.integer != kv[1]) {
            std.log.warn("moss image decoder: {s} is not {d}; the decoder runs the released geometry only", .{ kv[0], kv[1] });
            return error.UnsupportedGeometry;
        }
    }
    const lists = .{ .{ "decoder_block_out_channels", [_]i64{ 96, 192, 384, 384 } }, .{ "patch_size", [_]i64{ 2, 2 } } };
    inline for (lists) |kv| {
        const v = o.get(kv[0]) orelse return error.BadConfig;
        if (v != .array or v.array.items.len != kv[1].len) return error.UnsupportedGeometry;
        for (v.array.items, kv[1]) |item, n| if (item != .integer or item.integer != n) return error.UnsupportedGeometry;
    }
    const flags = .{ "mid_block_add_attention", "use_post_quant_conv" };
    inline for (flags) |f| {
        const v = o.get(f) orelse return error.BadConfig;
        if (v != .bool or !v.bool) return error.UnsupportedGeometry;
    }
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

/// A convolution's weights (a 1x1's [cin, ldw], a 3x3's G g G^T [16, cin, ldw], ldw cout padded to 64) and bias.
pub const Conv = struct { w: Ref, b: Ref, cin: usize, cout: usize, ldw: usize, taps: usize };
pub const Norm = struct { g: Ref, b: Ref };
pub const Resnet = struct { norm1: Norm, conv1: Conv, norm2: Norm, conv2: Conv, shortcut: ?Conv };
pub const Stage = struct { resnets: [resnets]Resnet, upsample: ?Conv };

/// The decoder's pipelines, by function name.
pub const names = [_][]const u8{ "md_unpack", "md_conv", "md_wino", "md_wino_32", "md_norm", "md_group_part", "md_group_finish", "md_attention", "md_pixels" };

pub const Timing = struct { wall: f64, gpu: f64 };

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,
    p: [names.len]mtl.Pipeline,
    /// Every weight, reordered: 1x1 convolutions [cin, cout padded], 3x3s G g G^T, the attention's q | k | v as one.
    weights: mtl.Buffer,
    bn_scale: Ref,
    bn_mean: Ref,
    post_quant: Conv,
    conv_in: Conv,
    mid: [2]Resnet,
    attn_norm: Norm,
    qkv: Conv,
    attn_out: Conv,
    stages: [4]Stage,
    norm_out: Norm,
    conv_out: Conv,
    /// The picture this decoder's scratch is for.
    width: usize,
    height: usize,
    scratch: mtl.Buffer,
    lat: Ref,
    t: [5]Ref,
    qkv_rows: Ref,
    att: Ref,
    stats: Ref,
    part: Ref,
    out: Ref,
    rgb: Ref,

    fn get(d: *const Decoder, comptime name: []const u8) mtl.Pipeline {
        inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return d.p[i];
        @compileError("no pipeline " ++ name);
    }

    /// The vae/ folder `dir`, with scratch for one width x height picture.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, width: usize, height: usize) !*Decoder {
        const g = try cfg.grid(width, height);
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, "config.json" }), arena, .limited(1 << 20));
        try check(gpa, config);
        // qualified on the M5 alone, as the transformer is
        if (!device.tensorUnits()) {
            std.log.err("moss image decoder: {s} has no tensor units; this family runs on the M5's alone, the only chip it was checked on", .{device.name()});
            return error.TensorUnitsRequired;
        }
        const d = try gpa.create(Decoder);
        errdefer gpa.destroy(d);
        d.* = undefined;
        d.gpa = gpa;
        d.device = device;
        d.width = width;
        d.height = height;
        d.queue = try device.queue();
        errdefer d.queue.deinit();
        const src = try frags.source(device, gpa, sources.moss_image_decoder);
        defer gpa.free(src);
        d.lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer d.lib.deinit();
        var made: usize = 0;
        errdefer for (d.p[0..made]) |p| p.deinit();
        for (names, 0..) |name, i| {
            d.p[i] = try mtl.Pipeline.init(device, d.lib, name, false);
            made += 1;
        }
        var c = ck.Checkpoint.init(gpa);
        defer c.deinit();
        try c.addFile(device, try std.fmt.allocPrintSentinel(arena, "{s}/diffusion_pytorch_model.safetensors", .{dir}, 0), "");
        try d.pack(&c);
        errdefer d.weights.deinit();
        try d.plan(g.rows, g.columns);
        return d;
    }

    pub fn deinit(d: *Decoder) void {
        d.scratch.deinit();
        d.weights.deinit();
        for (d.p) |p| p.deinit();
        d.lib.deinit();
        d.queue.deinit();
        d.gpa.destroy(d);
    }

    /// The decoder's tensors of `c`, each checked for its shape, copied into one buffer in the kernels' layouts.
    fn pack(d: *Decoder, c: *const ck.Checkpoint) !void {
        var p: Packer = .{ .c = c };
        try d.layout(&p);
        if (p.read != p.expected(c)) {
            std.log.err("moss image decoder: the checkpoint has {d} decoder tensors; the decoder reads {d}", .{ p.expected(c), p.read });
            return error.UnexpectedTensors;
        }
        d.weights = try d.device.buffer(p.at * 4, mtl.ResourceOptions.shared);
        errdefer d.weights.deinit();
        p.out = @as([*]f32, @ptrCast(@alignCast(d.weights.contents())))[0..p.at];
        p.at = 0;
        p.read = 0;
        p.buf = d.weights;
        try d.layout(&p);
    }

    /// Walks every tensor in the order it is packed: sizes it on the first pass, copies it on the second.
    fn layout(d: *Decoder, p: *Packer) !void {
        const var_ = try p.tensor("bn.running_var", &.{128});
        const mean = try p.tensor("bn.running_mean", &.{128});
        d.bn_scale = p.alloc(128);
        d.bn_mean = p.alloc(128);
        if (p.out) |out| {
            const s = out[d.bn_scale.off / 4 ..][0..128];
            const m = out[d.bn_mean.off / 4 ..][0..128];
            for (s, m, var_, mean) |*so, *mo, v, mv| {
                so.* = @sqrt(v + batch_norm_eps);
                mo.* = mv;
            }
        }
        d.post_quant = try p.conv("post_quant_conv", latent_channels, latent_channels, 1);
        d.conv_in = try p.conv("decoder.conv_in", latent_channels, widths[0], 9);
        var name: [80]u8 = undefined;
        for (&d.mid, 0..) |*r, i| r.* = try p.resnet(try std.fmt.bufPrint(&name, "decoder.mid_block.resnets.{d}", .{i}), widths[0], widths[0]);
        d.attn_norm = try p.norm("decoder.mid_block.attentions.0.group_norm", widths[0]);
        d.qkv = try p.linears(&.{ "decoder.mid_block.attentions.0.to_q", "decoder.mid_block.attentions.0.to_k", "decoder.mid_block.attentions.0.to_v" }, widths[0]);
        d.attn_out = try p.linears(&.{"decoder.mid_block.attentions.0.to_out.0"}, widths[0]);
        var prev = widths[0];
        for (&d.stages, widths, 0..) |*s, out, i| {
            for (&s.resnets, 0..) |*r, j| {
                r.* = try p.resnet(try std.fmt.bufPrint(&name, "decoder.up_blocks.{d}.resnets.{d}", .{ i, j }), if (j == 0) prev else out, out);
            }
            s.upsample = if (i + 1 < widths.len) try p.conv(try std.fmt.bufPrint(&name, "decoder.up_blocks.{d}.upsamplers.0.conv", .{i}), out, out, 9) else null;
            prev = out;
        }
        d.norm_out = try p.norm("decoder.conv_norm_out", widths[3]);
        d.conv_out = try p.conv("decoder.conv_out", widths[3], 3, 9);
    }

    /// Scratch for a rows x columns latent grid: five picture tensors at the widest stage, the attention's rows.
    fn plan(d: *Decoder, rows: usize, columns: usize) !void {
        const R = rows * 2;
        const C = columns * 2;
        var most: usize = R * C * widths[0];
        var scale: usize = 1;
        for (widths, 0..) |w, i| {
            most = @max(most, R * C * scale * scale * w);
            if (i + 1 < widths.len) {
                scale *= 2;
                most = @max(most, R * C * scale * scale * w);
            }
        }
        const picture = R * C * 64;
        const attn_rows = std.mem.alignForward(usize, R * C, 32);
        const parts = [_]usize{ rows * columns * cfg.channels * 4, most * 4, most * 4, most * 4, most * 4, most * 4, attn_rows * 3 * widths[0] * 4, attn_rows * widths[0] * 4, 64 * 4, groups * chunks(picture) * 4, picture * 3 * 4, picture * 3 };
        var total: usize = 0;
        for (parts) |n| total += std.mem.alignForward(usize, n, 256);
        d.scratch = try d.device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        @memset(d.scratch.contents()[0..total], 0);
        var at: usize = 0;
        const views = [_]*Ref{ &d.lat, &d.t[0], &d.t[1], &d.t[2], &d.t[3], &d.t[4], &d.qkv_rows, &d.att, &d.stats, &d.part, &d.out, &d.rgb };
        for (views, parts) |v, n| {
            v.* = .{ .buf = d.scratch, .off = at };
            at += std.mem.alignForward(usize, n, 256);
        }
    }

    /// The 8-bit RGB picture [height, width, 3] of `latents` (rows x columns, 128), the transformer's output.
    pub fn decode(d: *Decoder, latents: []const f32, rgb: []u8) !Timing {
        const g = try cfg.grid(d.width, d.height);
        if (latents.len != g.rows * g.columns * cfg.channels) return error.LatentRows;
        if (rgb.len != d.width * d.height * 3) return error.PictureSize;
        @memcpy(floats(d.lat, latents.len), latents);
        const started = mtl.clock.seconds();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.queue.commandBuffer();
        const e = cb.compute(.serial);
        var R = g.rows * 2;
        var C = g.columns * 2;
        unpack(e, d.get("md_unpack"), d.lat, d.bn_scale, d.bn_mean, d.t[0], g.rows, g.columns);
        d.conv(e, d.post_quant, d.t[0], d.t[1], R, C, null);
        d.conv(e, d.conv_in, d.t[1], d.t[0], R, C, null);
        var x: usize = 0;
        x = d.resnet(e, d.mid[0], x, R, C);
        x = d.attention(e, x, R * C);
        x = d.resnet(e, d.mid[1], x, R, C);
        for (d.stages) |s| {
            for (s.resnets) |r| x = d.resnet(e, r, x, R, C);
            if (s.upsample) |u| {
                R *= 2;
                C *= 2;
                const y = (x + 1) % d.t.len;
                d.convUp(e, u, d.t[x], d.t[y], R, C);
                x = y;
            }
        }
        const n = (x + 1) % d.t.len;
        d.norm(e, d.t[x], d.t[n], R * C, widths[3], d.norm_out, true);
        d.conv(e, d.conv_out, d.t[n], d.out, R, C, null);
        pixels(e, d.get("md_pixels"), d.out, d.rgb, R * C * 3);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("moss image decoder: a pass failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
        @memcpy(rgb, d.rgb.addr()[0..rgb.len]);
        return .{ .wall = mtl.clock.seconds() - started, .gpu = cb.gpuSeconds() };
    }

    /// The last decode's output before the clamp [height, width, 3].
    pub fn output(d: *const Decoder) []const f32 {
        return floats(d.out, d.width * d.height * 3);
    }

    /// One residual block on t[x], its output in another of t; returns that one's index.
    fn resnet(d: *const Decoder, e: mtl.ComputeEncoder, r: Resnet, x: usize, R: usize, C: usize) usize {
        const n = (x + 1) % d.t.len;
        const h = (x + 2) % d.t.len;
        const o = (x + 3) % d.t.len;
        const s = (x + 4) % d.t.len;
        d.norm(e, d.t[x], d.t[n], R * C, r.conv1.cin, r.norm1, true);
        d.conv(e, r.conv1, d.t[n], d.t[h], R, C, null);
        d.norm(e, d.t[h], d.t[n], R * C, r.conv1.cout, r.norm2, true);
        const skip = if (r.shortcut) |sc| blk: {
            d.conv(e, sc, d.t[x], d.t[s], R, C, null);
            break :blk d.t[s];
        } else d.t[x];
        d.conv(e, r.conv2, d.t[n], d.t[o], R, C, skip);
        return o;
    }

    /// The middle block's self-attention on t[x] (`n` positions): GroupNorm, q | k | v, one head, out + t[x].
    fn attention(d: *const Decoder, e: mtl.ComputeEncoder, x: usize, n: usize) usize {
        const xn = (x + 1) % d.t.len;
        const o = (x + 2) % d.t.len;
        const W = widths[0];
        d.norm(e, d.t[x], d.t[xn], n, W, d.attn_norm, false);
        d.conv(e, d.qkv, d.t[xn], d.qkv_rows, n, 1, null);
        attend(e, d.get("md_attention"), d.qkv_rows, d.att, n);
        d.conv(e, d.attn_out, d.att, d.t[o], n, 1, d.t[x]);
        return o;
    }

    /// y = GroupNorm(x) with `nm` (then SiLU when `silu`), x and y [pixels, channels].
    fn norm(d: *const Decoder, e: mtl.ComputeEncoder, x: Ref, y: Ref, n: usize, channels: usize, nm: Norm, silu: bool) void {
        for ([_]bool{ false, true }) |centered| {
            groupPart(e, d.get("md_group_part"), x, d.stats, d.part, n, channels, centered);
            groupFinish(e, d.get("md_group_finish"), d.part, d.stats, n, channels, centered);
        }
        normed(e, d.get("md_norm"), x, d.stats, nm.g, nm.b, y, n, channels, silu);
    }

    /// y = conv(x) + b (+ r) at R x C.
    fn conv(d: *const Decoder, e: mtl.ComputeEncoder, c: Conv, x: Ref, y: Ref, R: usize, C: usize, r: ?Ref) void {
        const s: ConvShape = .{ .rows = R, .cols = C, .cin = c.cin, .cout = c.cout, .ldw = c.ldw };
        if (c.taps == 9) d.wino(e, c, x, r, y, s) else convolve(e, d.get("md_conv"), x, c.w, c.b, r, y, s);
    }

    /// md_wino_32 where md_wino's 64-channel groups would compute padding, else md_wino.
    fn wino(d: *const Decoder, e: mtl.ComputeEncoder, c: Conv, x: Ref, r: ?Ref, y: Ref, s: ConvShape) void {
        if (wide(c.cout)) winograd(e, d.get("md_wino"), 64, x, c.w, c.b, r, y, s) else winograd(e, d.get("md_wino_32"), 32, x, c.w, c.b, r, y, s);
    }

    /// y = conv(x read nearest from R/2 x C/2) + b at R x C.
    fn convUp(d: *const Decoder, e: mtl.ComputeEncoder, c: Conv, x: Ref, y: Ref, R: usize, C: usize) void {
        d.wino(e, c, x, null, y, .{ .rows = R, .cols = C, .cin = c.cin, .cout = c.cout, .ldw = c.ldw, .up = true });
    }
};

/// md_unpack: latents x [rows * columns, 128] to z [2 rows, 2 columns, 32], the batch norm undone (x * s + m).
pub fn unpack(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, s: Ref, m: Ref, z: Ref, rows: usize, columns: usize) void {
    bind(e, p, &.{ x, s, m, z }, [2]u32{ @intCast(rows), @intCast(columns) });
    e.dispatchThreads(mtl.Size.of(latent_channels, rows * columns * 4, 1), mtl.Size.of(32, 8, 1));
}

/// A convolution's size: the output picture, its channels, the weights' row length, and a 3x3's read from half size.
pub const ConvShape = struct { rows: usize, cols: usize, cin: usize, cout: usize, ldw: usize, up: bool = false };

fn convParams(s: ConvShape, add: bool) extern struct { rows: u32, cols: u32, cin: u32, cout: u32, ldw: u32, up: u32, add: u32 } {
    return .{ .rows = @intCast(s.rows), .cols = @intCast(s.cols), .cin = @intCast(s.cin), .cout = @intCast(s.cout), .ldw = @intCast(s.ldw), .up = @intFromBool(s.up), .add = @intFromBool(add) };
}

/// md_conv: y = x w + b (+ r when given), a 1x1 convolution of x [rows * cols, cin].
pub fn convolve(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, w: Ref, b: Ref, r: ?Ref, y: Ref, s: ConvShape) void {
    const params = convParams(s, r != null);
    bind(e, p, &.{ x, w, b, r orelse y, y }, params);
    e.dispatchGroups(mtl.Size.of(s.ldw / 64, (s.rows * s.cols + 63) / 64, 1), mtl.Size.of(128, 1, 1));
}

/// True when md_wino's 64 channels a threadgroup compute no padding past `cout`'s last 32.
pub fn wide(cout: usize) bool {
    return cout % 64 == 0 or cout % 64 > 32;
}

/// md_wino (`group` 64) or md_wino_32 (32): y = conv3x3(x) + b (+ r when given) by Winograd F(2x2, 3x3).
pub fn winograd(e: mtl.ComputeEncoder, p: mtl.Pipeline, group: usize, x: Ref, u: Ref, b: Ref, r: ?Ref, y: Ref, s: ConvShape) void {
    const params = convParams(s, r != null);
    bind(e, p, &.{ x, u, b, r orelse y, y }, params);
    e.dispatchGroups(mtl.Size.of((s.cout + group - 1) / group, (s.rows + 3) / 4, (s.cols + 15) / 16), mtl.Size.of(512, 1, 1));
}

/// The Winograd F(2x2, 3x3) form of one 3x3 kernel g: G g G^T in float64, each of its 16 values rounded once.
pub fn transform(g: *const [9]f32) [16]f32 {
    const G = [4][3]f64{ .{ 1, 0, 0 }, .{ 0.5, 0.5, 0.5 }, .{ 0.5, -0.5, 0.5 }, .{ 0, 0, 1 } };
    var out: [16]f32 = undefined;
    for (0..4) |i| for (0..4) |j| {
        var v: f64 = 0;
        for (0..3) |a| for (0..3) |b| {
            v += G[i][a] * @as(f64, g[a * 3 + b]) * G[j][b];
        };
        out[i * 4 + j] = @floatCast(v);
    };
    return out;
}

/// md_norm: y = GroupNorm(x) by stats (32 means, then 32 rstd) with g and beta, then SiLU when `silu`.
pub fn normed(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, stats: Ref, g: Ref, beta: Ref, y: Ref, n: usize, channels: usize, silu: bool) void {
    const params = extern struct { pixels: u32, channels: u32, silu: u32 }{ .pixels = @intCast(n), .channels = @intCast(channels), .silu = @intFromBool(silu) };
    bind(e, p, &.{ x, stats, g, beta, y }, params);
    e.dispatchThreads(mtl.Size.of(channels, n, 1), mtl.Size.of(32, 8, 1));
}

/// GroupNorm's parts over `n` pixels of x [n, channels]: one sum a chunk of `chunk` pixels and a group.
pub fn chunks(n: usize) usize {
    return (n + chunk - 1) / chunk;
}

fn groupParams(n: usize, channels: usize, centered: bool) extern struct { pixels: u32, channels: u32, chunk: u32, chunks: u32, centered: u32, eps: f32 } {
    return .{ .pixels = @intCast(n), .channels = @intCast(channels), .chunk = chunk, .chunks = @intCast(chunks(n)), .centered = @intFromBool(centered), .eps = eps };
}

/// md_group_part: part [32, chunks] = each chunk's sum of a group's values, or (centered) of (x - mean)^2.
pub fn groupPart(e: mtl.ComputeEncoder, p: mtl.Pipeline, x: Ref, stats: Ref, part: Ref, n: usize, channels: usize, centered: bool) void {
    bind(e, p, &.{ x, stats, part }, groupParams(n, channels, centered));
    e.dispatchGroups(mtl.Size.of(chunks(n), 1, 1), mtl.Size.of(channels, 1, 1));
}

/// md_group_finish: stats' 32 means from part, or (centered) its 32 rstd.
pub fn groupFinish(e: mtl.ComputeEncoder, p: mtl.Pipeline, part: Ref, stats: Ref, n: usize, channels: usize, centered: bool) void {
    bind(e, p, &.{ part, stats }, groupParams(n, channels, centered));
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32 * groups, 1, 1));
}

/// md_attention: o [rows, 384] = softmax(q k^T / sqrt(384)) v of qkv rows [rows, 1152], both padded to 32 rows.
pub fn attend(e: mtl.ComputeEncoder, p: mtl.Pipeline, qkv: Ref, o: Ref, rows: usize) void {
    const params = extern struct { rows: u32, scale: f32 }{ .rows = @intCast(rows), .scale = 1.0 / @sqrt(@as(f32, widths[0])) };
    bind(e, p, &.{ qkv, o }, params);
    e.dispatchGroups(mtl.Size.of((rows + 7) / 8, 1, 1), mtl.Size.of(128, 1, 1));
}

/// md_pixels: rgb = the n 8-bit values of y.
pub fn pixels(e: mtl.ComputeEncoder, p: mtl.Pipeline, y: Ref, rgb: Ref, n: usize) void {
    bind(e, p, &.{ y, rgb }, @as(u32, @intCast(n)));
    e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(256, 1, 1));
}

fn floats(r: Ref, n: usize) []f32 {
    return @as([*]f32, @ptrCast(@alignCast(r.addr())))[0..n];
}

/// Lays the decoder's tensors out in one fp32 buffer: sizes on the first pass (out null), copies on the second.
const Packer = struct {
    c: *const ck.Checkpoint,
    buf: mtl.Buffer = undefined,
    out: ?[]f32 = null,
    at: usize = 0,
    read: usize = 0,

    fn alloc(p: *Packer, n: usize) Ref {
        const r: Ref = .{ .buf = p.buf, .off = p.at * 4 };
        p.at = std.mem.alignForward(usize, p.at + n, 64);
        return r;
    }

    fn tensor(p: *Packer, name: []const u8, shape: []const usize) ![]const f32 {
        const t = try p.c.get(name);
        if (t.dtype != .f32 or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
            std.log.err("moss image decoder: {s} is {t} {any}; the decoder reads f32 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
            return error.UnexpectedTensor;
        }
        p.read += 1;
        return t.host(f32);
    }

    /// A copied vector `name` of n.
    fn copy(p: *Packer, name: []const u8, n: usize) !Ref {
        const v = try p.tensor(name, &.{n});
        const r = p.alloc(n);
        if (p.out) |out| @memcpy(out[r.off / 4 ..][0..n], v);
        return r;
    }

    /// Convolution `name` (.weight [cout, cin, k, k], .bias): a 1x1 as [cin, ldw], a 3x3 as G g G^T [16, cin, ldw].
    fn conv(p: *Packer, name: []const u8, cin: usize, cout: usize, taps: usize) !Conv {
        var buf: [96]u8 = undefined;
        const k: usize = if (taps == 9) 3 else 1;
        const ldw = std.mem.alignForward(usize, cout, 64);
        const points: usize = if (taps == 9) 16 else 1;
        const w = try p.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), &.{ cout, cin, k, k });
        const r = p.alloc(points * cin * ldw);
        if (p.out) |out| {
            const dst = out[r.off / 4 ..][0 .. points * cin * ldw];
            @memset(dst, 0);
            for (0..cout) |o| for (0..cin) |i| {
                const g = w[(o * cin + i) * taps ..][0..taps];
                if (taps == 9) {
                    for (transform(g[0..9]), 0..) |v, pt| dst[(pt * cin + i) * ldw + o] = v;
                } else dst[i * ldw + o] = g[0];
            };
        }
        const b = try p.copy(try std.fmt.bufPrint(&buf, "{s}.bias", .{name}), cout);
        return .{ .w = r, .b = b, .cin = cin, .cout = cout, .ldw = ldw, .taps = taps };
    }

    /// Linear layers `names` (.weight [width, width], .bias) stacked as one 1x1 convolution.
    fn linears(p: *Packer, names_: []const []const u8, width: usize) !Conv {
        var buf: [96]u8 = undefined;
        const n = names_.len * width;
        const ldw = std.mem.alignForward(usize, n, 64);
        const r = p.alloc(width * ldw);
        for (names_, 0..) |name, part| {
            const w = try p.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), &.{ width, width });
            if (p.out) |out| {
                const dst = out[r.off / 4 ..][0 .. width * ldw];
                for (0..width) |o| for (0..width) |i| {
                    dst[i * ldw + part * width + o] = w[o * width + i];
                };
            }
        }
        const b = p.alloc(n);
        for (names_, 0..) |name, i| {
            const v = try p.tensor(try std.fmt.bufPrint(&buf, "{s}.bias", .{name}), &.{width});
            if (p.out) |out| @memcpy(out[b.off / 4 + i * width ..][0..width], v);
        }
        return .{ .w = r, .b = b, .cin = width, .cout = n, .ldw = ldw, .taps = 1 };
    }

    fn norm(p: *Packer, name: []const u8, width: usize) !Norm {
        var buf: [96]u8 = undefined;
        return .{ .g = try p.copy(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), width), .b = try p.copy(try std.fmt.bufPrint(&buf, "{s}.bias", .{name}), width) };
    }

    fn resnet(p: *Packer, name: []const u8, cin: usize, cout: usize) !Resnet {
        var buf: [96]u8 = undefined;
        return .{
            .norm1 = try p.norm(try std.fmt.bufPrint(&buf, "{s}.norm1", .{name}), cin),
            .conv1 = try p.conv(try std.fmt.bufPrint(&buf, "{s}.conv1", .{name}), cin, cout, 9),
            .norm2 = try p.norm(try std.fmt.bufPrint(&buf, "{s}.norm2", .{name}), cout),
            .conv2 = try p.conv(try std.fmt.bufPrint(&buf, "{s}.conv2", .{name}), cout, cout, 9),
            .shortcut = if (cin != cout) try p.conv(try std.fmt.bufPrint(&buf, "{s}.conv_shortcut", .{name}), cin, cout, 1) else null,
        };
    }

    /// The checkpoint's decoder-side tensors: the decoder's, post_quant_conv's and the batch norm's two statistics.
    fn expected(_: *const Packer, c: *const ck.Checkpoint) usize {
        var n: usize = 0;
        var it = c.tensors.keyIterator();
        while (it.next()) |k| {
            if (std.mem.startsWith(u8, k.*, "decoder.") or std.mem.startsWith(u8, k.*, "post_quant_conv.") or
                std.mem.eql(u8, k.*, "bn.running_mean") or std.mem.eql(u8, k.*, "bn.running_var")) n += 1;
        }
        return n;
    }
};

test "only FLUX.2's released decoder geometry passes" {
    const a = std.testing.allocator;
    const sample =
        \\{"_class_name": "AutoencoderKLFlux2", "act_fn": "silu", "batch_norm_eps": 0.0001, "decoder_block_out_channels": [96, 192, 384, 384],
        \\ "latent_channels": 32, "layers_per_block": 2, "mid_block_add_attention": true, "norm_num_groups": 32, "out_channels": 3,
        \\ "patch_size": [2, 2], "use_post_quant_conv": true}
    ;
    try check(a, sample);
    const narrow = try std.mem.replaceOwned(u8, a, sample, "[96, 192, 384, 384]", "[128, 256, 512, 512]");
    defer a.free(narrow);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, narrow));
    const no_attn = try std.mem.replaceOwned(u8, a, sample, "\"mid_block_add_attention\": true", "\"mid_block_add_attention\": false");
    defer a.free(no_attn);
    try std.testing.expectError(error.UnsupportedGeometry, check(a, no_attn));
    const other = try std.mem.replaceOwned(u8, a, sample, "AutoencoderKLFlux2", "AutoencoderKL");
    defer a.free(other);
    try std.testing.expectError(error.NotFlux2Vae, check(a, other));
}
