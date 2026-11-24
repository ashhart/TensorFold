//! tf-h3-image-decoder: MiniMax-H3-Image's image decoder on Metal, a latent to a PNG, or its self-test on the host.
const std = @import("std");
const mtl = @import("metal");
const dec = @import("h3_image_decoder");
const resample = @import("picture_writer").resample;
const png_file = @import("picture_writer").png;
const selftest = @import("h3_image_decoder_selftest.zig");

const vae = dec.vae;

const usage =
    \\usage: tf-h3-image-decoder MODEL_DIR latent.f32 ROWS COLUMNS out.png [WIDTHxHEIGHT]
    \\       tf-h3-image-decoder selftest
    \\  MODEL_DIR     the MiniMax-H3-Image folder: vae/config.json and vae/diffusion_pytorch_model.safetensors
    \\  latent.f32    the normalised latent [24, ROWS, COLUMNS] in float32, as tf-h3-image writes it
    \\  WIDTHxHEIGHT  the requested size: the 16 COLUMNS x 16 ROWS canvas is resized to it (Pillow's LANCZOS)
    \\selftest runs the kernels on synthetic weights against a float64 host reference; it exits 1 past a bound.
    \\
;

/// Tiles a pass decodes at most.
const batch = 16;

fn read(init: std.process.Init, a: std.mem.Allocator, path: []const u8, most: usize) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(most)) catch |err| {
        std.debug.print("cannot read {s}: {s}\n", .{ path, @errorName(err) });
        return err;
    };
}

/// The image decoder loaded once: its config and weights, its kernels built.
pub const Vae = struct {
    config: vae.Config,
    d: *dec.Decoder,

    pub fn load(init: std.process.Init, a: std.mem.Allocator, model_dir: []const u8) !Vae {
        const vae_dir = try std.fs.path.join(a, &.{ model_dir, "vae" });
        const config = try vae.config(a, try read(init, a, try std.fs.path.join(a, &.{ vae_dir, "config.json" }), 1 << 20));
        const device = try mtl.Device.init();
        const loading = mtl.clock.seconds();
        const d = try dec.Decoder.init(init.gpa, device, vae.layers, config.rope_theta);
        errdefer d.deinit();
        try d.load(try std.fmt.allocPrintSentinel(a, "{s}/diffusion_pytorch_model.safetensors", .{vae_dir}, 0));
        d.give_way = .{ .io = init.io };
        std.debug.print("{s}: decoder weights in bf16 in {d:.1} s\n", .{ device.name(), mtl.clock.seconds() - loading });
        return .{ .config = config, .d = d };
    }

    /// The normalised `latent` [24, rows, cols] decoded and written to `path` as a PNG of `want` (default the canvas).
    pub fn png(v: *const Vae, init: std.process.Init, a: std.mem.Allocator, latent: []f32, rows: usize, cols: usize, want: ?dec.host.Size, path: []const u8) !void {
        if (latent.len != try dec.host.latentLength(rows, cols)) return error.H3BadLatentLength;
        const tiles = try dec.host.tiles(a, &v.config, rows, cols);
        const width = cols * vae.cell_pixels;
        const height = rows * vae.cell_pixels;
        const size = want orelse dec.host.Size{ .width = width, .height = height };
        try vae.denormalize(&v.config, latent);
        const started = mtl.clock.seconds();
        const decoded = try a.alloc(f32, 3 * width * height);
        const took = try v.d.decode(a, &v.config, latent, rows, cols, vae.layers, batch, decoded);
        const rgb = try a.alloc(u8, 3 * width * height);
        try vae.pixels(decoded, height, width, rgb);
        const picture = if (size.width == width and size.height == height) rgb else try resample.resizeWith(a, .pillow, .lanczos, resample.View.of(rgb, width, height), size.width, size.height);
        const file = try png_file.encode(a, picture, size.width, size.height);
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = file });
        std.debug.print("{d}x{d} canvas: decoded {d} tiles in {d:.2} s ({d:.2} s on the GPU, {d:.2} s waited for speech), {d:.2} s in all; wrote {s} at {d}x{d}\n", .{ width, height, tiles.total(), took.wall, took.gpu, took.paused, mtl.clock.seconds() - started, path, size.width, size.height });
    }
};

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and std.mem.eql(u8, args[1], "selftest")) {
        if (!try selftest.run(init.gpa, a)) std.process.exit(1);
        return;
    }
    if (args.len != 6 and args.len != 7) {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    }
    const rows = std.fmt.parseInt(usize, args[3], 10) catch return error.H3BadPictureGrid;
    const cols = std.fmt.parseInt(usize, args[4], 10) catch return error.H3BadPictureGrid;
    const n = try dec.host.latentLength(rows, cols);
    const bytes = try read(init, a, args[2], n * 4 + 1);
    if (bytes.len != n * 4) {
        std.debug.print("{s} holds {d} bytes; a [24, {d}, {d}] float32 latent is {d}\n", .{ args[2], bytes.len, rows, cols, n * 4 });
        return error.H3BadLatentLength;
    }
    const latent = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(latent), bytes);
    const v = try Vae.load(init, a, args[1]);
    defer v.d.deinit();
    try v.png(init, a, latent, rows, cols, if (args.len == 7) try dec.host.size(args[6]) else null, args[5]);
}
