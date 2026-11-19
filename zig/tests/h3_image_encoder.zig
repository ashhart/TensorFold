//! tf-h3-image-encoder: MiniMax-H3-Image's edit encoder on Metal, a reference picture to its clean rows, or a selftest.
const std = @import("std");
const mtl = @import("metal");
const he = @import("h3_image_encoder");
const codec = @import("picture_codec");
const noise = @import("torch_noise");
const selftest = @import("h3_image_encoder_selftest.zig");

const ref = he.reference;
const latent_channels = he.vae.latent_channels;

const usage =
    \\usage: tf-h3-image-encoder MODEL_DIR reference.png SEED OUT_PREFIX [CANVAS]
    \\       tf-h3-image-encoder selftest
    \\  MODEL_DIR  the MiniMax-H3-Image folder; vae/ holds config.json and diffusion_pytorch_model.safetensors
    \\  reference  a PNG, read upright in RGB (any alpha dropped), resized so its short edge lies in 1024..2048
    \\  SEED       the request's seed: the noise is torch.randn on one CPU generator seeded with it
    \\  CANVAS     WxH, the edit's canvas: the generator first draws the target's noise and this reference's anchor
    \\             noise, as a one-reference edit does, and OUT_PREFIX.anchors.f32 is written too; without it the
    \\             posterior's noise is the seed's first draw
    \\Writes OUT_PREFIX.rows.f32 (the clean rows [rows, 96]), .moments.f32 ([48, h, w], before sampling), .grid.json.
    \\selftest: each kernel on small synthetic weights against a float64 host reference; exit 1 on a miss.
    \\
;

/// A PNG upright in RGB, as Pillow's convert("RGB") gives it: alpha dropped, not composited.
fn load(io: std.Io, a: std.mem.Allocator, path: []const u8) !codec.Rgb {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(512 << 20));
    if (codec.sniff(bytes) != .png) return error.NotPng;
    var s = try codec.png.decodeWithLimits(a, bytes, .{ .side = 1 << 15, .pixels = 1 << 26 });
    if (s.channels == 4) {
        var i: usize = 3;
        while (i < s.data.len) : (i += 4) s.data[i] = 255;
    }
    return codec.finish(a, s);
}

/// CANVAS as width x height, both whole multiples of 32.
fn parseCanvas(text: []const u8) ?ref.Size {
    const x = std.mem.indexOfScalar(u8, text, 'x') orelse return null;
    const w = std.fmt.parseInt(usize, text[0..x], 10) catch return null;
    const h = std.fmt.parseInt(usize, text[x + 1 ..], 10) catch return null;
    if (w == 0 or h == 0 or w % 32 != 0 or h % 32 != 0 or w > 1 << 16 or h > 1 << 16) return null;
    return .{ .width = w, .height = h };
}

fn write(io: std.Io, a: std.mem.Allocator, prefix: []const u8, suffix: []const u8, data: []const u8) !void {
    const path = try std.fmt.allocPrint(a, "{s}{s}", .{ prefix, suffix });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and std.mem.eql(u8, args[1], "selftest")) return selftest.run(init.gpa, a);
    if (args.len != 5 and args.len != 6) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const seed = std.fmt.parseInt(u64, args[3], 10) catch {
        std.debug.print("SEED must be a whole number\n{s}", .{usage});
        return 2;
    };
    const canvas: ?ref.Size = if (args.len == 6) parseCanvas(args[5]) orelse {
        std.debug.print("CANVAS must be WxH, both multiples of 32\n", .{});
        return 2;
    } else null;

    const rgb = load(io, a, args[2]) catch |err| {
        std.debug.print("{s}: {s}\n", .{ args[2], if (err == error.NotPng) "not a PNG" else @errorName(err) });
        return 2;
    };
    const size = ref.bounded(rgb.width, rgb.height, ref.vae_bound) catch |err| {
        std.debug.print("{s}: {d}x{d} is refused: {s}\n", .{ args[2], rgb.width, rgb.height, @errorName(err) });
        return 2;
    };
    const same = size.width == rgb.width and size.height == rgb.height;
    const pixels = if (same) rgb.pixels else try codec.resample.resizeWith(a, .pillow, .lanczos, .of(rgb.pixels, rgb.width, rgb.height), size.width, size.height);
    const cropped = try ref.cropped(size);
    const image = try a.alloc(f32, cropped.width * cropped.height * 3);
    _ = try ref.input(pixels, size.width, size.height, image);
    const full = ref.fullGrid(cropped);
    const grid = try ref.evenGrid(full);

    // one generator, drawn in the edit's order: target, the reference's anchor, then the posterior on the full grid
    var gen = noise.Twister.init(seed);
    var anchor_noise: ?[]f32 = null;
    if (canvas) |c| {
        noise.draw(&gen, try a.alloc(f32, latent_channels * (c.height / 16) * (c.width / 16)));
        anchor_noise = try a.alloc(f32, latent_channels * grid.rows * grid.columns);
        noise.draw(&gen, anchor_noise.?);
    }
    const posterior = try a.alloc(f32, latent_channels * full.rows * full.columns);
    noise.draw(&gen, posterior);

    const device = try mtl.Device.init();
    defer device.deinit();
    const loading = mtl.clock.seconds();
    const enc, const config = try he.load(init.gpa, io, device, try std.fs.path.join(a, &.{ args[1], "vae" }));
    defer enc.deinit();
    const loaded = mtl.clock.seconds() - loading;
    const moments = try a.alloc(f32, ref.moments * full.rows * full.columns);
    const took = try enc.encode(a, &config, image, cropped, moments);
    const clean = try ref.clean(a, &config, moments, full, posterior);

    const prefix = args[4];
    try write(io, a, prefix, ".rows.f32", std.mem.sliceAsBytes(clean.rows));
    try write(io, a, prefix, ".moments.f32", std.mem.sliceAsBytes(moments));
    const json = try std.fmt.allocPrint(a, "{{\"rows\": {d}, \"columns\": {d}, \"moment_rows\": {d}, \"moment_columns\": {d}, \"width\": {d}, \"height\": {d}, \"seed\": {d}}}\n", .{ grid.rows, grid.columns, full.rows, full.columns, size.width, size.height, seed });
    try write(io, a, prefix, ".grid.json", json);
    if (anchor_noise) |z| {
        const noise_rows = try a.alloc(f32, z.len);
        try he.picture.patchify(z, grid, noise_rows);
        const out = try a.alloc(f32, z.len);
        try ref.anchors(clean.rows, noise_rows, out);
        try write(io, a, prefix, ".anchors.f32", std.mem.sliceAsBytes(out));
    }
    std.debug.print("{s}: {d}x{d} reference, VAE copy {d}x{d}, encoded {d}x{d}, grid {d} x {d} ({d} rows); loaded in {d:.1} s, encoded in {d:.2} s wall, {d:.2} s GPU\n", .{ device.name(), rgb.width, rgb.height, size.width, size.height, cropped.width, cropped.height, grid.rows, grid.columns, clean.rows.len / he.picture.row_width, loaded, took.wall, took.gpu });
    std.debug.print("wrote {s}.rows.f32, .moments.f32, .grid.json{s}\n", .{ prefix, if (anchor_noise != null) ", .anchors.f32" else "" });
    return 0;
}
