//! MiniMax-H3-Image's edit encoder on Metal in fp32: a reference's tiles through the VAE's CNN to blended moments.
const std = @import("std");
const mtl = @import("metal");
const st = @import("safetensors");
const kernels = @import("h3_image_encoder_kernels");
pub const reference = @import("picture_reference.zig");
pub const vae = @import("picture_vae.zig");
pub const picture = @import("picture.zig");

const Item = reference.Item;
const moments = reference.moments;
const cell = vae.cell_pixels;

/// The largest tile side the scratch is planned for (the release's is 256).
pub const max_tile = 512;
/// Pixels a GroupNorm part sums.
pub const chunk = 256;

/// A buffer and a byte offset into it.
pub const Ref = struct {
    buf: mtl.Buffer,
    off: usize = 0,

    pub fn floats(r: Ref, n: usize) []f32 {
        return @as([*]f32, @ptrCast(@alignCast(r.buf.contents() + r.off)))[0..n];
    }
};

pub const Timing = struct { wall: f64 = 0, gpu: f64 = 0 };

/// The four pipelines, compiled from source now with precise fp32 functions.
pub const Kernels = struct {
    lib: mtl.Library,
    p: [kernels.names.len]mtl.Pipeline,

    pub fn init(device: mtl.Device) !Kernels {
        var options = mtl.CompileOptions.mlx();
        options.math_functions = .precise;
        var k: Kernels = .{ .lib = try mtl.Library.fromSource(device, kernels.source, options), .p = undefined };
        errdefer k.lib.deinit();
        var made: usize = 0;
        errdefer for (k.p[0..made]) |p| p.deinit();
        for (kernels.names, 0..) |name, i| {
            k.p[i] = try mtl.Pipeline.init(device, k.lib, name, false);
            made += 1;
        }
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for (k.p) |p| p.deinit();
        k.lib.deinit();
    }

    fn get(k: *const Kernels, comptime name: []const u8) mtl.Pipeline {
        inline for (kernels.names, 0..) |n, i| if (comptime std.mem.eql(u8, n, name)) return k.p[i];
        @compileError("no pipeline " ++ name);
    }
};

fn bind(e: mtl.ComputeEncoder, p: mtl.Pipeline, refs: []const Ref, params: anytype) void {
    e.setPipeline(p);
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
    e.setValue(params, refs.len);
}

/// A layer's weights on the GPU: a convolution's [taps, cin, ldw] and bias, or a norm's gamma and beta.
pub const Layer = struct { item: Item, w: Ref, b: Ref };

/// y = conv(x) + b (+ r) at h x w input; `down`: stride 2, the padded row and column h, w read as h - 2, w - 2.
pub fn conv(e: mtl.ComputeEncoder, k: *const Kernels, l: Layer, x: Ref, y: Ref, r: ?Ref, h: usize, w: usize, down: bool) void {
    const side: u32 = if (l.item.kind == .conv3) 3 else 1;
    const oh = if (down) h / 2 else h;
    const ow = if (down) w / 2 else w;
    const shape = extern struct { h: u32, w: u32, oh: u32, ow: u32, cin: u32, cout: u32, ldw: u32, k: u32, stride: u32, pad: u32, add: u32 }{
        .h = @intCast(h),
        .w = @intCast(w),
        .oh = @intCast(oh),
        .ow = @intCast(ow),
        .cin = @intCast(l.item.cin),
        .cout = @intCast(l.item.cout),
        .ldw = @intCast(reference.rowLength(l.item.cout)),
        .k = side,
        .stride = if (down) 2 else 1,
        .pad = if (side == 3 and !down) 1 else 0,
        .add = @intFromBool(r != null),
    };
    bind(e, k.get("hie_conv"), &.{ x, l.w, l.b, r orelse y, y }, shape);
    e.dispatchGroups(mtl.Size.of((oh * ow + 63) / 64, shape.ldw / 64, 1), mtl.Size.of(256, 1, 1));
}

/// GroupNorm's parts for `pixels`.
pub fn chunks(pixels: usize) usize {
    return (pixels + chunk - 1) / chunk;
}

/// y = SiLU(GroupNorm(x)) over `pixels` of x [pixels, channels]: two passes of parts and their finish, then the map.
pub fn normSilu(e: mtl.ComputeEncoder, k: *const Kernels, l: Layer, x: Ref, y: Ref, stats: Ref, part: Ref, pixels: usize) void {
    const c = l.item.cin;
    for ([_]u32{ 0, 1 }) |centered| {
        const g = extern struct { pixels: u32, channels: u32, chunk: u32, chunks: u32, centered: u32, eps: f32 }{
            .pixels = @intCast(pixels),
            .channels = @intCast(c),
            .chunk = chunk,
            .chunks = @intCast(chunks(pixels)),
            .centered = centered,
            .eps = vae.group_eps,
        };
        bind(e, k.get("hie_group_part"), &.{ x, stats, part }, g);
        e.dispatchGroups(mtl.Size.of(chunks(pixels), 1, 1), mtl.Size.of(c, 1, 1));
        bind(e, k.get("hie_group_finish"), &.{ part, stats }, g);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32 * reference.groups, 1, 1));
    }
    const n = extern struct { pixels: u32, channels: u32 }{ .pixels = @intCast(pixels), .channels = @intCast(c) };
    bind(e, k.get("hie_norm_silu"), &.{ x, stats, l.w, l.b, y }, n);
    e.dispatchThreads(mtl.Size.of(c, pixels, 1), mtl.Size.of(32, 8, 1));
}

/// A block (norm1, conv1, norm2, conv2, any shortcut) on t[x] at h x w; returns the index of the t it wrote.
pub fn block(e: mtl.ComputeEncoder, k: *const Kernels, ls: []const Layer, t: [3]Ref, x: usize, stats: Ref, part: Ref, h: usize, w: usize) usize {
    const n = (x + 1) % 3;
    const o = (x + 2) % 3;
    normSilu(e, k, ls[0], t[x], t[n], stats, part, h * w);
    conv(e, k, ls[1], t[n], t[o], null, h, w, false);
    normSilu(e, k, ls[2], t[o], t[n], stats, part, h * w);
    if (ls.len == 5) {
        // the shortcut into o, then conv2 adds itself onto it in place
        conv(e, k, ls[4], t[x], t[o], null, h, w, false);
        conv(e, k, ls[3], t[n], t[o], t[o], h, w, false);
        return o;
    }
    conv(e, k, ls[3], t[n], t[x], t[x], h, w, false);
    return x;
}

/// Floats a layer takes in the weights buffer, each part aligned to 64.
fn footprint(it: Item) [2]usize {
    if (it.kind == .norm) return .{ std.mem.alignForward(usize, it.cin, 64), std.mem.alignForward(usize, it.cin, 64) };
    return .{ reference.taps(it) * it.cin * reference.rowLength(it.cout), std.mem.alignForward(usize, it.cout, 64) };
}

/// The layers of `items` packed into one shared buffer from `source.get(name)` (a []align(1) const f32 or null).
pub fn packLayers(gpa: std.mem.Allocator, device: mtl.Device, list: []const Item, source: anytype) !struct { mtl.Buffer, []Layer } {
    var total: usize = 0;
    for (list) |it| {
        const f = footprint(it);
        total += f[0] + f[1];
    }
    const buf = try device.buffer(total * 4, mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    const all = @as([*]f32, @ptrCast(@alignCast(buf.contents())))[0..total];
    const layers = try gpa.alloc(Layer, list.len);
    errdefer gpa.free(layers);
    var at: usize = 0;
    var name: [96]u8 = undefined;
    for (list, layers) |it, *l| {
        const f = footprint(it);
        l.* = .{ .item = it, .w = .{ .buf = buf, .off = at * 4 }, .b = .{ .buf = buf, .off = (at + f[0]) * 4 } };
        const w = source.get(try std.fmt.bufPrint(&name, "{s}.weight", .{it.name})) orelse return error.H3MissingEncoderTensor;
        const b = source.get(try std.fmt.bufPrint(&name, "{s}.bias", .{it.name})) orelse return error.H3MissingEncoderTensor;
        if (b.len != it.cout) return error.H3BadEncoderTensor;
        const wd = all[at..][0..f[0]];
        if (it.kind == .norm) {
            if (w.len != it.cin) return error.H3BadEncoderTensor;
            @memset(wd, 0);
            for (wd[0..it.cin], w) |*d, v| d.* = v;
        } else try reference.pack(it, w, wd);
        @memset(all[at + f[0] ..][0..f[1]], 0);
        for (all[at + f[0] ..][0..it.cout], b) |*d, v| d.* = v;
        at += f[0] + f[1];
    }
    return .{ buf, layers };
}

/// Tensors of a mapped checkpoint as float32 slices, for `packLayers`.
pub const FileSource = struct {
    file: *const st.File,

    pub fn get(s: FileSource, name: []const u8) ?[]align(1) const f32 {
        const t = s.file.get(name) orelse return null;
        if (t.dtype != .f32) return null;
        return std.mem.bytesAsSlice(f32, t.bytes);
    }
};

pub const Encoder = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    k: Kernels,
    weights: mtl.Buffer,
    layers: []Layer,
    /// The tile side the scratch holds.
    tile: usize,
    scratch: mtl.Buffer,
    input: Ref,
    t: [3]Ref,
    stats: Ref,
    part: Ref,
    out: Ref,

    /// Pipelines, `arch`'s layers packed from `source`, and scratch for tiles up to `tile` a side.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, arch: reference.Arch, tile: usize, source: anytype) !*Encoder {
        if (tile < 2 * cell or tile % cell != 0 or tile > max_tile) return error.H3BadTiles;
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const list = try reference.items(arena.allocator(), arch);
        const enc = try gpa.create(Encoder);
        errdefer gpa.destroy(enc);
        enc.* = undefined;
        enc.gpa = gpa;
        enc.device = device;
        enc.tile = tile;
        enc.queue = try device.queue();
        errdefer enc.queue.deinit();
        enc.k = try Kernels.init(device);
        errdefer enc.k.deinit();
        enc.weights, enc.layers = try packLayers(gpa, device, list, source);
        errdefer enc.weights.deinit();
        errdefer gpa.free(enc.layers);
        // the names live in the arena; a forward reads kinds and widths only
        for (enc.layers) |*l| l.item.name = "";
        try enc.plan(arch);
        return enc;
    }

    pub fn deinit(enc: *Encoder) void {
        enc.scratch.deinit();
        enc.weights.deinit();
        enc.gpa.free(enc.layers);
        enc.k.deinit();
        enc.queue.deinit();
        enc.gpa.destroy(enc);
    }

    /// The tensors a tile needs at most: input, three activations, GroupNorm statistics and parts, moments.
    fn plan(enc: *Encoder, arch: reference.Arch) !void {
        var most: usize = 0;
        var parts: usize = 0;
        var cin = arch.widths[0];
        for (arch.widths, 0..) |w, l| {
            const side = enc.tile >> @intCast(@min(l, reference.downsampled));
            const wide = @max(cin, w);
            most = @max(most, side * side * wide);
            parts = @max(parts, chunks(side * side) * wide);
            cin = w;
        }
        const last = enc.tile / cell;
        most = @max(most, last * last * moments);
        const sizes = [_]usize{ enc.tile * enc.tile * 3, most, most, most, 64, parts, last * last * moments };
        var total: usize = 0;
        for (sizes) |n| total += std.mem.alignForward(usize, n * 4, 256);
        enc.scratch = try enc.device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        var at: usize = 0;
        const views = [_]*Ref{ &enc.input, &enc.t[0], &enc.t[1], &enc.t[2], &enc.stats, &enc.part, &enc.out };
        for (views, sizes) |v, n| {
            v.* = .{ .buf = enc.scratch, .off = at };
            at += std.mem.alignForward(usize, n * 4, 256);
        }
    }

    /// One tile's moments [48, h / 16, w / 16] from its normalised pixels [h, w, 3].
    pub fn encodeTile(enc: *Encoder, pixels: []const f32, h: usize, w: usize, out: []f32) !Timing {
        if (h < 2 * cell or w < 2 * cell or h % cell != 0 or w % cell != 0 or h > enc.tile or w > enc.tile) return error.H3BadTiles;
        const lh = h / cell;
        const lw = w / cell;
        if (pixels.len != h * w * 3 or out.len != moments * lh * lw) return error.H3BadLatentLength;
        @memcpy(enc.input.floats(pixels.len), pixels);
        const started = mtl.clock.seconds();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = enc.queue.commandBuffer();
        const e = cb.compute(.serial);
        const ls = enc.layers;
        conv(e, &enc.k, ls[0], enc.input, enc.t[0], null, h, w, false);
        var at: usize = 1;
        var x: usize = 0;
        var rh = h;
        var rw = w;
        for (0..reference.levels) |l| {
            for (0..2) |_| {
                const n: usize = if (ls[at].item.cin != ls[at + 1].item.cout) 5 else 4;
                x = block(e, &enc.k, ls[at..][0..n], enc.t, x, enc.stats, enc.part, rh, rw);
                at += n;
            }
            if (l < reference.downsampled) {
                conv(e, &enc.k, ls[at], enc.t[x], enc.t[(x + 1) % 3], null, rh, rw, true);
                at += 1;
                x = (x + 1) % 3;
                rh /= 2;
                rw /= 2;
            }
        }
        const n = (x + 1) % 3;
        normSilu(e, &enc.k, ls[at], enc.t[x], enc.t[n], enc.stats, enc.part, rh * rw);
        conv(e, &enc.k, ls[at + 1], enc.t[n], enc.t[x], null, rh, rw, false);
        conv(e, &enc.k, ls[at + 2], enc.t[x], enc.out, null, rh, rw, false);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("h3 image encoder: a tile failed on the GPU: {s}", .{msg});
            return error.GpuFailure;
        }
        const got = enc.out.floats(lh * lw * moments);
        for (0..lh * lw) |p| for (0..moments) |c| {
            out[c * lh * lw + p] = got[p * moments + c];
        };
        return .{ .wall = mtl.clock.seconds() - started, .gpu = cb.gpuSeconds() };
    }

    /// A cropped picture's moments [48, height / 16, width / 16]: its tiles encoded, blended in latent cells.
    pub fn encode(enc: *Encoder, a: std.mem.Allocator, c: *const vae.Config, image: []const f32, s: reference.Size, out: []f32) !Timing {
        if (c.tile > enc.tile) return error.H3BadTiles;
        const g = reference.fullGrid(s);
        if (image.len != s.width * s.height * 3 or out.len != moments * g.rows * g.columns) return error.H3BadLatentLength;
        var arena = std.heap.ArenaAllocator.init(a);
        defer arena.deinit();
        const x = arena.allocator();
        const rows = try vae.split(x, s.height, c.tile, c.overlap);
        const cols = try vae.split(x, s.width, c.tile, c.overlap);
        var blend = try vae.Assembler.init(x, try cells(x, rows), try cells(x, cols), moments, out);
        const pixels = try x.alloc(f32, rows[0].len * cols[0].len * 3);
        const tile = try x.alloc(f32, moments * (rows[0].len / cell) * (cols[0].len / cell));
        var sum: Timing = .{};
        for (rows, 0..) |r, ri| for (cols, 0..) |col, ci| {
            for (0..r.len) |y| {
                @memcpy(pixels[y * col.len * 3 ..][0 .. col.len * 3], image[((r.start + y) * s.width + col.start) * 3 ..][0 .. col.len * 3]);
            }
            const t = try enc.encodeTile(pixels[0 .. r.len * col.len * 3], r.len, col.len, tile[0 .. moments * (r.len / cell) * (col.len / cell)]);
            sum.wall += t.wall;
            sum.gpu += t.gpu;
            try blend.place(ri, ci, tile[0 .. moments * (r.len / cell) * (col.len / cell)]);
        };
        return sum;
    }
};

/// Pixel spans in latent cells.
fn cells(a: std.mem.Allocator, spans: []const vae.Span) ![]vae.Span {
    const out = try a.alloc(vae.Span, spans.len);
    for (out, spans) |*o, s| o.* = .{ .start = s.start / cell, .len = s.len / cell, .overlap = s.overlap / cell };
    return out;
}

/// The released encoder from a `vae/` folder: its config (encoder tiling on) and checkpoint, header checked first.
pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8) !struct { *Encoder, vae.Config } {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const json = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 20));
    const c = try vae.config(a, json);
    try reference.checkTiling(a, json);
    var file = try st.File.openPrefixes(gpa, io, try std.fs.path.join(a, &.{ dir, "diffusion_pytorch_model.safetensors" }), &.{ "encoder.", "quant_conv." });
    defer file.close(io);
    var culprit: []const u8 = "";
    reference.checkHeader(a, &file.names, &culprit) catch |err| {
        std.log.err("h3 image encoder: {s} is missing or not at its released shape ({s})", .{ culprit, @errorName(err) });
        return err;
    };
    return .{ try Encoder.init(gpa, device, reference.released, c.tile, FileSource{ .file = &file }), c };
}
