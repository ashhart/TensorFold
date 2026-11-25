//! tf-h3-image-decoder-host: MiniMax-H3-Image's image decoder on the CPU, a latent to a PNG, no GPU.
const std = @import("std");
const dec = @import("h3_image_decoder_host");

const host = dec.host;
const vae = dec.vae;

const usage =
    \\usage: tf-h3-image-decoder-host MODEL_DIR latent.f32 ROWS COLUMNS out.png [WIDTHxHEIGHT]
    \\  MODEL_DIR     the MiniMax-H3-Image folder: vae/config.json and vae/diffusion_pytorch_model.safetensors
    \\  latent.f32    the normalised latent [24, ROWS, COLUMNS] in float32, as tf-h3-image writes it
    \\  WIDTHxHEIGHT  the requested size: the 16 COLUMNS x 16 ROWS canvas is resized to it (Pillow's LANCZOS)
    \\
;

fn seconds(io: std.Io, since: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - since.toNanoseconds())) / 1e9;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 6 and args.len != 7) {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    }
    const rows = std.fmt.parseInt(usize, args[3], 10) catch return error.H3BadPictureGrid;
    const cols = std.fmt.parseInt(usize, args[4], 10) catch return error.H3BadPictureGrid;
    const n = try host.latentLength(rows, cols);
    const bytes = try dec.read(init.io, a, args[2], n * 4 + 1);
    if (bytes.len != n * 4) {
        std.debug.print("{s} holds {d} bytes; a [24, {d}, {d}] float32 latent is {d}\n", .{ args[2], bytes.len, rows, cols, n * 4 });
        return error.H3BadLatentLength;
    }
    const latent = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(latent), bytes);
    const c = try dec.config(init.io, a, args[1]);
    try vae.denormalize(&c, latent);
    var d = try dec.Decoder.init(init.gpa, vae.layers);
    defer d.deinit();
    const loading = std.Io.Clock.awake.now(init.io);
    try d.load(init.io, a, args[1]);
    std.debug.print("decoder weights in bf16 in {d:.1} s\n", .{seconds(init.io, loading)});
    const tiles = try host.tiles(a, &c, rows, cols);
    const height = rows * vae.cell_pixels;
    const width = cols * vae.cell_pixels;
    const decoded = try a.alloc(f32, 3 * width * height);
    const started = std.Io.Clock.awake.now(init.io);
    try d.decode(a, &c, latent, rows, cols, vae.layers, decoded);
    std.debug.print("decoder: {d:.2} s\n", .{seconds(init.io, started)});
    const size = if (args.len == 7) try host.size(args[6]) else host.Size{ .width = width, .height = height };
    try dec.write(init.io, a, decoded, height, width, size.width, size.height, args[5]);
    std.debug.print("{d}x{d} canvas: decoded {d} tiles; wrote {s} at {d}x{d}\n", .{ width, height, tiles.total(), args[5], size.width, size.height });
}
