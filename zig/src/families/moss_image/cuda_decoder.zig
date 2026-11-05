//! MOSS-Image's image decoder on CUDA: FLUX.2's VAE decoder (the checkpoint's vae/) in fp32, latents to a picture.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const ks = @import("cuda_decoder_kernels.zig");
const seconds = @import("cuda_model.zig").seconds;

/// The decoder's widths from its last stage to its first (decoder_block_out_channels reversed), and the rest.
pub const widths = [4]usize{ 384, 384, 192, 96 };
pub const latent_channels = 32;
pub const groups = 32;
pub const resnets = 3;
pub const eps: f32 = 1e-6;
pub const batch_norm_eps: f32 = 1e-4;
/// Pixels a GroupNorm statistics part sums.
pub const chunk = ks.chunk;

/// Accepts vae/config.json at FLUX.2's released decoder geometry, else an error (decoder.zig's check, without Metal).
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

/// A convolution: its weights' and bias's offset in the packed buffer, a 3x3's nine taps, or a 1x1's one.
pub const Conv = struct { w: usize, b: usize, cin: usize, cout: usize, taps: usize };
pub const Norm = struct { g: usize, b: usize };
pub const Resnet = struct { norm1: Norm, conv1: Conv, norm2: Norm, conv2: Conv, shortcut: ?Conv };
pub const Stage = struct { resnets: [resnets]Resnet, upsample: ?Conv };

pub const Timing = struct { wall: f64, gpu: f64 };

/// The parts of one decode's scratch, in `Views`' field order, in f32 values.
pub const Plan = struct {
    parts: [12]usize,

    fn mul(a: usize, b: usize) !usize {
        return std.math.mul(usize, a, b) catch error.ImageTooLarge;
    }

    /// A rows x columns latent grid's parts: five picture tensors at the widest stage, the attention's scores.
    pub fn of(rows: usize, columns: usize) !Plan {
        const grid = try mul(try mul(rows, 2), try mul(columns, 2));
        var most = try mul(grid, widths[0]);
        var scale: usize = 1;
        for (widths, 0..) |w, i| {
            most = @max(most, try mul(try mul(grid, try mul(scale, scale)), w));
            if (i + 1 < widths.len) {
                scale *= 2;
                most = @max(most, try mul(try mul(grid, try mul(scale, scale)), w));
            }
        }
        // the widest stage's pixels are the picture's: three doublings of 2 x 2 pixels
        const picture = try mul(grid, 64);
        return .{ .parts = .{
            try mul(try mul(rows, columns), cfg.channels), most,                     most, most,                                most,                most,
            try mul(grid, 3 * widths[0]),                  try mul(grid, widths[0]), 64,   try mul(groups, ks.chunks(picture)), try mul(grid, grid), try mul(picture, 3),
        } };
    }

    pub fn bytes(p: Plan) !usize {
        var total: usize = 0;
        for (p.parts) |n| {
            const bytes_ = std.math.mul(usize, n, 4) catch return error.ImageTooLarge;
            total = std.math.add(usize, total, std.mem.alignForward(usize, bytes_, 256)) catch return error.ImageTooLarge;
        }
        return total;
    }
};

/// One decode's scratch, seen as the buffers the decoder's steps take.
pub const Views = struct {
    lat: u64,
    t: [5]u64,
    qkv: u64,
    att: u64,
    stats: u64,
    part: u64,
    scores: u64,
    out: u64,

    pub fn of(base: u64, plan: Plan) Views {
        var at: [12]u64 = undefined;
        var off: u64 = 0;
        for (plan.parts, 0..) |n, i| {
            at[i] = base + off;
            off += std.mem.alignForward(u64, n * 4, 256);
        }
        return .{ .lat = at[0], .t = .{ at[1], at[2], at[3], at[4], at[5] }, .qkv = at[6], .att = at[7], .stats = at[8], .part = at[9], .scores = at[10], .out = at[11] };
    }
};

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    driver: cuda.Driver,
    ctx: cuda.Context,
    stream: cuda.Stream,
    k: ks.Kernels,
    /// Every weight, reordered: 1x1 convolutions [cin, cout], 3x3s their nine taps [tap, cin, cout].
    weights: cuda.DeviceBuffer,
    bn_scale: usize,
    bn_mean: usize,
    post_quant: Conv,
    conv_in: Conv,
    mid: [2]Resnet,
    attn_norm: Norm,
    qkv: Conv,
    attn_out: Conv,
    stages: [4]Stage,
    norm_out: Norm,
    conv_out: Conv,
    /// 1 / sqrt(384), attention's score scale.
    scale: f32,
    /// A decode's start and end on the stream, for its GPU time.
    marks: [2]cuda.Event,

    /// The vae/ folder `dir` on the first GPU: its config checked, its decoder tensors packed for the kernels.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Decoder {
        var arena_state = std.heap.ArenaAllocator.init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(arena, &.{ dir, "config.json" }), arena, .limited(1 << 20));
        try check(gpa, config);
        const d = try gpa.create(Decoder);
        errdefer gpa.destroy(d);
        d.* = undefined;
        d.gpa = gpa;
        d.driver = try cuda.Driver.open();
        errdefer d.driver.close();
        d.ctx = try cuda.Context.init(&d.driver, 0);
        errdefer d.ctx.deinit();
        const sm = try d.ctx.capability();
        if (sm != 120 and sm != 121) {
            std.log.err("moss image decoder: sm_{d} is not a GPU this family was checked on (sm_120, sm_121)", .{sm});
            return error.UncheckedGpu;
        }
        d.stream = try cuda.Stream.init(&d.driver, false);
        errdefer d.stream.deinit();
        d.k = try ks.Kernels.load(&d.driver);
        errdefer d.k.unload();
        var ck: core.Checkpoint = .{ .gpa = gpa, .io = io };
        defer ck.close();
        try ck.add(dir, "diffusion_pytorch_model.safetensors");
        try d.pack(&ck);
        errdefer d.weights.free();
        d.marks[0] = try cuda.Event.init(&d.driver, true);
        errdefer d.marks[0].deinit();
        d.marks[1] = try cuda.Event.init(&d.driver, true);
        errdefer d.marks[1].deinit();
        d.scale = 1.0 / @sqrt(@as(f32, widths[0]));
        return d;
    }

    pub fn deinit(d: *Decoder) void {
        d.marks[1].deinit();
        d.marks[0].deinit();
        d.weights.free();
        d.k.unload();
        d.stream.deinit();
        d.ctx.deinit();
        d.driver.close();
        d.gpa.destroy(d);
    }

    /// The GPU this decoder's context is on, for a tool's receipt.
    pub fn device(d: *const Decoder, buf: []u8) []const u8 {
        return d.ctx.name(buf) catch "a CUDA GPU";
    }

    /// A packed tensor's address on the device.
    fn at(d: *const Decoder, off: usize) u64 {
        return d.weights.ptr + off * 4;
    }

    /// The decoder's tensors of `ck`, each checked for its shape, copied into one buffer in the kernels' layouts.
    fn pack(d: *Decoder, ck: *core.Checkpoint) !void {
        var p: Packer = .{ .ck = ck };
        try d.layout(&p);
        const expected = expectedTensors(ck);
        if (p.read != expected) {
            std.log.err("moss image decoder: the checkpoint has {d} decoder tensors; the decoder reads {d}", .{ expected, p.read });
            return error.UnexpectedTensors;
        }
        const host = try d.gpa.alloc(f32, p.at);
        defer d.gpa.free(host);
        @memset(host, 0);
        p.out = host;
        p.at = 0;
        p.read = 0;
        try d.layout(&p);
        d.weights = try cuda.DeviceBuffer.fromHost(&d.driver, std.mem.sliceAsBytes(host));
    }

    /// Walks every tensor in the order it is packed: sizes it on the first pass, copies it on the second.
    fn layout(d: *Decoder, p: *Packer) !void {
        const var_ = try p.tensor("bn.running_var", &.{128});
        const mean = try p.tensor("bn.running_mean", &.{128});
        d.bn_scale = p.alloc(128);
        d.bn_mean = p.alloc(128);
        if (p.out) |out| {
            for (out[d.bn_scale..][0..128], out[d.bn_mean..][0..128], var_, mean) |*so, *mo, v, mv| {
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

    /// The raw picture [height, width, 3] of `latents` (rows x columns of 128), before the pipeline's clamp.
    pub fn decode(d: *Decoder, latents: []const f32, width: u32, height: u32, out: []f32) !Timing {
        const g = try cfg.grid(width, height);
        if (latents.len != g.rows * g.columns * cfg.channels) return error.LatentRows;
        if (out.len != @as(usize, width) * @as(usize, height) * 3) return error.PictureSize;
        const plan = try Plan.of(g.rows, g.columns);
        var buf = try cuda.DeviceBuffer.alloc(&d.driver, try plan.bytes());
        defer buf.free();
        const v = Views.of(buf.ptr, plan);
        const started = seconds();
        try buf.upload(@intCast(v.lat - buf.ptr), std.mem.sliceAsBytes(latents));
        try d.marks[0].record(d.stream);
        try d.run(&v, g.rows, g.columns);
        try d.marks[1].record(d.stream);
        try buf.download(@intCast(v.out - buf.ptr), std.mem.sliceAsBytes(out));
        return .{ .wall = seconds() - started, .gpu = @as(f64, try cuda.Event.elapsedMs(d.marks[0], d.marks[1])) / 1000 };
    }

    /// Every step of the decode in one stream: latents to picture, float32 throughout.
    fn run(d: *Decoder, v: *const Views, rows: usize, columns: usize) !void {
        var R = rows * 2;
        var C = columns * 2;
        try ks.unpack(&d.k, d.stream, v.lat, d.at(d.bn_scale), d.at(d.bn_mean), v.t[0], rows, columns);
        try d.conv(d.post_quant, v.t[0], v.t[1], R, C, 0, false);
        try d.conv(d.conv_in, v.t[1], v.t[0], R, C, 0, false);
        var x: usize = 0;
        x = try d.resnet(v, d.mid[0], x, R, C);
        x = try d.attention(v, x, R * C);
        x = try d.resnet(v, d.mid[1], x, R, C);
        for (d.stages) |stage| {
            for (stage.resnets) |r| x = try d.resnet(v, r, x, R, C);
            if (stage.upsample) |u| {
                R *= 2;
                C *= 2;
                const y = (x + 1) % v.t.len;
                try d.conv(u, v.t[x], v.t[y], R, C, 0, true);
                x = y;
            }
        }
        const n = (x + 1) % v.t.len;
        try d.norm(v, v.t[x], v.t[n], R * C, widths[3], d.norm_out, true);
        try d.conv(d.conv_out, v.t[n], v.out, R, C, 0, false);
    }

    /// One residual block on t[x], its output in another of t; returns that one's index.
    fn resnet(d: *Decoder, v: *const Views, r: Resnet, x: usize, R: usize, C: usize) !usize {
        const n = (x + 1) % v.t.len;
        const h = (x + 2) % v.t.len;
        const o = (x + 3) % v.t.len;
        const s = (x + 4) % v.t.len;
        try d.norm(v, v.t[x], v.t[n], R * C, r.conv1.cin, r.norm1, true);
        try d.conv(r.conv1, v.t[n], v.t[h], R, C, 0, false);
        try d.norm(v, v.t[h], v.t[n], R * C, r.conv1.cout, r.norm2, true);
        var skip = v.t[x];
        if (r.shortcut) |sc| {
            try d.conv(sc, v.t[x], v.t[s], R, C, 0, false);
            skip = v.t[s];
        }
        try d.conv(r.conv2, v.t[n], v.t[o], R, C, skip, false);
        return o;
    }

    /// The middle block's self-attention on t[x] (`n` positions, one head of 384): GroupNorm, q | k | v, out + t[x].
    fn attention(d: *Decoder, v: *const Views, x: usize, n: usize) !usize {
        const W = widths[0];
        const xn = (x + 1) % v.t.len;
        const o = (x + 2) % v.t.len;
        try d.norm(v, v.t[x], v.t[xn], n, W, d.attn_norm, false);
        try d.linear(v.t[xn], d.qkv, v.qkv, 0, n, false);
        // scores = q k^T: q is the packed rows' first third, k the second
        try ks.gemm(&d.k, d.stream, v.qkv, v.qkv + W * 4, 0, 0, v.scores, n, n, W, 3 * W, 3 * W, true, false);
        try ks.softmax(&d.k, d.stream, v.scores, n, d.scale);
        try ks.gemm(&d.k, d.stream, v.scores, v.qkv + 2 * W * 4, 0, 0, v.att, n, W, n, n, 3 * W, false, false);
        try d.linear(v.att, d.attn_out, v.t[o], v.t[x], n, true);
        return o;
    }

    /// y [pixels, cout] = x . weights + bias (+ r): a 1x1 convolution of x [pixels, cin].
    fn linear(d: *Decoder, x: u64, c: Conv, y: u64, r: u64, pixels: usize, add: bool) !void {
        try ks.gemm(&d.k, d.stream, x, d.at(c.w), d.at(c.b), r, y, pixels, c.cout, c.cin, c.cin, c.cout, false, add);
    }

    /// y = conv(x) + b (+ r) at R x C, x read nearest from half size when `up` (`r` 0 for none).
    fn conv(d: *Decoder, c: Conv, x: u64, y: u64, R: usize, C: usize, r: u64, up: bool) !void {
        if (c.taps == 9) {
            try ks.conv3x3(&d.k, d.stream, x, d.at(c.w), d.at(c.b), r, y, .{ .rows = R, .cols = C, .cin = c.cin, .cout = c.cout, .up = up }, r != 0);
        } else {
            try ks.gemm(&d.k, d.stream, x, d.at(c.w), d.at(c.b), r, y, R * C, c.cout, c.cin, c.cin, c.cout, false, r != 0);
        }
    }

    /// y = GroupNorm(x) (then SiLU when `silu`), x and y [pixels, channels]: the statistics in two fp32 passes.
    fn norm(d: *Decoder, v: *const Views, x: u64, y: u64, pixels: usize, channels: usize, nm: Norm, silu: bool) !void {
        for ([_]bool{ false, true }) |centered| {
            try ks.groupPart(&d.k, d.stream, x, v.stats, v.part, pixels, channels, centered);
            try ks.groupFinish(&d.k, d.stream, v.part, v.stats, pixels, channels, centered, eps);
        }
        try ks.norm(&d.k, d.stream, x, v.stats, d.at(nm.g), d.at(nm.b), y, pixels, channels, silu);
    }
};

/// The checkpoint's decoder-side tensors: the decoder's, post_quant_conv's and the batch norm's two statistics.
fn expectedTensors(ck: *const core.Checkpoint) usize {
    var n: usize = 0;
    for (ck.files.items) |*f| for (f.names.keys()) |k| {
        if (std.mem.startsWith(u8, k, "decoder.") or std.mem.startsWith(u8, k, "post_quant_conv.") or
            std.mem.eql(u8, k, "bn.running_mean") or std.mem.eql(u8, k, "bn.running_var")) n += 1;
    };
    return n;
}

/// Lays the decoder's tensors out in one fp32 buffer: sizes on the first pass (out null), copies on the second.
const Packer = struct {
    ck: *core.Checkpoint,
    out: ?[]f32 = null,
    at: usize = 0,
    read: usize = 0,

    fn alloc(p: *Packer, n: usize) usize {
        const at = p.at;
        p.at = std.mem.alignForward(usize, p.at + n, 64);
        return at;
    }

    fn tensor(p: *Packer, name: []const u8, shape: []const usize) ![]align(1) const f32 {
        const t = try p.ck.get(name);
        if (t.dtype != .f32 or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
            std.log.err("moss image decoder: {s} is {t} {any}; the decoder reads f32 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
            return error.UnexpectedTensor;
        }
        p.read += 1;
        return std.mem.bytesAsSlice(f32, t.bytes);
    }

    /// `dst` from the checkpoint's `src`, whose bytes need not be 4-aligned.
    fn take(dst: []f32, src: []align(1) const f32) void {
        for (dst, src) |*d, s| d.* = s;
    }

    /// A copied vector `name` of n.
    fn copy(p: *Packer, name: []const u8, n: usize) !usize {
        const v = try p.tensor(name, &.{n});
        const at = p.alloc(n);
        if (p.out) |out| take(out[at..][0..n], v);
        return at;
    }

    /// Convolution `name` (.weight [cout, cin, k, k], .bias): a 1x1 as [cin, cout], a 3x3 as its taps [tap, cin, cout].
    fn conv(p: *Packer, name: []const u8, cin: usize, cout: usize, taps: usize) !Conv {
        var buf: [96]u8 = undefined;
        const k: usize = if (taps == 9) 3 else 1;
        const w = try p.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), &.{ cout, cin, k, k });
        const at = p.alloc(taps * cin * cout);
        if (p.out) |out| {
            const dst = out[at..][0 .. taps * cin * cout];
            for (0..cout) |o| for (0..cin) |i| {
                const g = w[(o * cin + i) * taps ..][0..taps];
                for (0..taps) |tap| dst[(tap * cin + i) * cout + o] = g[tap];
            };
        }
        const b = try p.copy(try std.fmt.bufPrint(&buf, "{s}.bias", .{name}), cout);
        return .{ .w = at, .b = b, .cin = cin, .cout = cout, .taps = taps };
    }

    /// Linear layers `names` (.weight [width, width], .bias) stacked as one 1x1 convolution.
    fn linears(p: *Packer, names: []const []const u8, width: usize) !Conv {
        var buf: [96]u8 = undefined;
        const cout = names.len * width;
        const at = p.alloc(width * cout);
        for (names, 0..) |name, part| {
            const w = try p.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}), &.{ width, width });
            if (p.out) |out| {
                const dst = out[at..][0 .. width * cout];
                for (0..width) |o| {
                    for (0..width) |i| dst[i * cout + part * width + o] = w[o * width + i];
                }
            }
        }
        const b = p.alloc(cout);
        for (names, 0..) |name, i| {
            const v = try p.tensor(try std.fmt.bufPrint(&buf, "{s}.bias", .{name}), &.{width});
            if (p.out) |out| take(out[b..][i * width ..][0..width], v);
        }
        return .{ .w = at, .b = b, .cin = width, .cout = cout, .taps = 1 };
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

test "the plan sizes a 1024 x 1024 picture's parts, and every view lands inside the buffer" {
    const plan = try Plan.of(64, 64);
    try std.testing.expectEqual(@as(usize, 524288), plan.parts[0]);
    try std.testing.expectEqual(@as(usize, 201326592), plan.parts[1]);
    try std.testing.expectEqual(@as(usize, 18874368), plan.parts[6]);
    try std.testing.expectEqual(@as(usize, 6291456), plan.parts[7]);
    try std.testing.expectEqual(@as(usize, 131072), plan.parts[9]);
    try std.testing.expectEqual(@as(usize, 268435456), plan.parts[10]);
    try std.testing.expectEqual(@as(usize, 3145728), plan.parts[11]);
    const bytes = try plan.bytes();
    const v = Views.of(1 << 20, plan);
    var end: u64 = v.out + plan.parts[11] * 4;
    for (&v.t) |*t| end = @max(end, t.* + plan.parts[1] * 4);
    try std.testing.expect(end <= (1 << 20) + @as(u64, bytes));
    try std.testing.expectEqual(@as(usize, 16 * 16 * 3), (try Plan.of(1, 1)).parts[11]);
    try std.testing.expectError(error.BadImageSize, cfg.grid(0, 16));
}
