//! MiniMax-H3-Image's image decoder on the host: the reference decoder over every tile, all cores, no GPU.
const std = @import("std");
const rounding = @import("bf16_round");
const writer = @import("picture_writer");
pub const host = @import("image_decoder.zig");
pub const vae = @import("picture_vae.zig");
pub const reference = @import("image_decoder_reference.zig");

const W = host.width;

/// A file's bytes, or why they cannot be read.
pub fn read(io: std.Io, a: std.mem.Allocator, path: []const u8, most: usize) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(most)) catch |err| {
        std.debug.print("cannot read {s}: {s}\n", .{ path, @errorName(err) });
        return err;
    };
}

/// `MODEL_DIR/vae/config.json` as a checked `Config`.
pub fn config(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !vae.Config {
    const json = try std.fs.path.join(a, &.{ model_dir, "vae", "config.json" });
    return vae.config(a, try read(io, a, json, 1 << 20));
}

/// The decoded picture to RGB bytes and a PNG at `path`, LANCZOS-resized to `want_width` x `want_height` when needed.
pub fn write(io: std.Io, a: std.mem.Allocator, decoded: []const f32, height: usize, width: usize, want_width: usize, want_height: usize, path: []const u8) !void {
    const rgb = try a.alloc(u8, 3 * width * height);
    try vae.pixels(decoded, height, width, rgb);
    const picture = if (want_width == width and want_height == height) rgb else try writer.resample.resizeWith(a, .pillow, .lanczos, writer.resample.View.of(rgb, width, height), want_width, want_height);
    const file = try writer.png.encode(a, picture, want_width, want_height);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = file });
}

/// The decoder on the host: every parameter in bf16 as `plan` says, tiles decoded across every core.
pub const Decoder = struct {
    gpa: std.mem.Allocator,
    plan: host.Plan,
    weights: []u16,

    /// `layers` blocks; fill `weights` with `load` or `synthetic` before `decode`.
    pub fn init(gpa: std.mem.Allocator, layers: usize) !Decoder {
        return .{ .gpa = gpa, .plan = try host.Plan.init(layers), .weights = try gpa.alloc(u16, 0) };
    }

    pub fn deinit(d: *Decoder) void {
        d.gpa.free(d.weights);
    }

    /// The decoder's tensors of `MODEL_DIR/vae/diffusion_pytorch_model.safetensors`, checked, rounded to bf16.
    pub fn load(d: *Decoder, io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !void {
        const path = try std.fs.path.join(a, &.{ model_dir, "vae", "diffusion_pytorch_model.safetensors" });
        const bytes = try read(io, a, path, 8 << 30);
        const w = try host.weights(a, bytes, d.plan);
        d.weights = try d.gpa.realloc(d.weights, d.plan.total());
        host.read(w, d.weights);
    }

    /// Synthetic weights from `seed` for a test, in the decoder's own buffer.
    pub fn synthetic(d: *Decoder, a: std.mem.Allocator, seed: u64) !void {
        d.weights = try d.gpa.realloc(d.weights, d.plan.total());
        try host.synthetic(a, d.plan, seed, d.weights);
    }

    /// The raw latent [24, rows, cols] decoded to `out` [3, 16 rows, 16 cols]: a tile a worker, blended in fp32.
    pub fn decode(d: *Decoder, a: std.mem.Allocator, c: *const vae.Config, raw: []const f32, rows: usize, cols: usize, layers: usize, out: []f32) !void {
        return d.decodeWith(a, c, raw, rows, cols, layers, out, std.Thread, std.Thread.getCpuCount() catch 1);
    }

    /// `decode` with up to `workers` workers (64 at most) from `Thread`, which a test swaps for one whose spawn fails.
    pub fn decodeWith(d: *Decoder, a: std.mem.Allocator, c: *const vae.Config, raw: []const f32, rows: usize, cols: usize, layers: usize, out: []f32, comptime Thread: type, workers: usize) !void {
        if (raw.len != try host.latentLength(rows, cols)) return error.H3BadLatentLength;
        const t = try host.tiles(a, c, rows, cols);
        if (layers > d.plan.layers) return error.H3BadTiles;
        const height = rows * vae.cell_pixels;
        const width = cols * vae.cell_pixels;
        if (out.len != 3 * height * width) return error.H3BadLatentLength;
        const model: reference.Model = .{ .w = d.weights, .plan = d.plan };
        const n = try host.tokens(t.h, t.w);
        const cells = t.h * t.w;
        const cos = try a.alloc(u16, n * vae.rotated);
        const sin = try a.alloc(u16, cos.len);
        try host.rotary(t.h, t.w, c.rope_theta, cos, sin);
        const pixels = 3 * t.rows[0].len * t.cols[0].len;
        const count = @min(t.total(), 64, @max(workers, 1));
        const stream = try a.alloc([]f32, count);
        const proj = try a.alloc([]f32, count);
        const bits = try a.alloc([]u16, count);
        const tile = try a.alloc([]f32, count);
        const picture = try a.alloc(f32, t.total() * pixels); // a slot a tile: a worker may not overwrite another's
        const scratches = try a.alloc(reference.Scratch, count);
        for (0..count) |me| {
            stream[me] = try a.alloc(f32, n * W);
            proj[me] = try a.alloc(f32, cells * host.patch);
            bits[me] = try a.alloc(u16, cells * host.patch);
            tile[me] = try a.alloc(f32, host.channels * cells);
            scratches[me] = try reference.Scratch.init(a, n);
            scratches[me].workers = 1; // the tiles already take every core
        }
        var w = Work{ .model = model, .raw = raw, .rows = rows, .cols = cols, .layers = layers, .tiles = t, .tokens = n, .cos = cos, .sin = sin, .stream = stream, .proj = proj, .bits = bits, .tile = tile, .picture = picture, .pixels = pixels, .scratches = scratches };
        const Job = struct {
            /// Worker `me`'s tiles: me, me + count, ...; worker 0 is this thread.
            fn run(work: *Work, me: usize, of: usize) void {
                var k = me;
                while (k < work.tiles.total()) : (k += of) work.one(me, k);
            }
        };
        {
            var threads: [64]Thread = undefined;
            var made: usize = 0;
            defer for (threads[0..made]) |th| th.join(); // every picture slot settled before the assembly blends them
            for (1..count) |me| {
                threads[made] = Thread.spawn(.{}, Job.run, .{ &w, me, count }) catch |err| {
                    w.stop();
                    return err;
                };
                made += 1;
            }
            Job.run(&w, 0, count);
        }
        if (w.failed.load(.monotonic)) return error.H3BadTiles;
        var assembly = try vae.Assembler.init(a, t.rows, t.cols, 3, out);
        for (0..t.total()) |k| {
            const at = t.at(k);
            try assembly.place(at.r, at.c, picture[k * pixels ..][0..pixels]);
        }
    }

    /// What a decoding worker needs: the model, the latent, one slot of every scratch buffer.
    const Work = struct {
        model: reference.Model,
        raw: []const f32,
        rows: usize,
        cols: usize,
        layers: usize,
        tiles: host.Tiles,
        tokens: usize,
        cos: []const u16,
        sin: []const u16,
        stream: [][]f32,
        proj: [][]f32,
        bits: [][]u16,
        tile: [][]f32,
        picture: []f32,
        pixels: usize,
        scratches: []reference.Scratch,
        failed: std.atomic.Value(bool) = .init(false),

        /// Tile `k`: crop, embed, the blocks, proj_out, then unpatchify into `k`'s picture slot.
        pub fn one(w: *Work, me: usize, k: usize) void {
            if (w.failed.load(.monotonic)) return;
            const at = w.tiles.at(k);
            const h = w.tiles.h;
            const ww = w.tiles.w;
            const cells = h * ww;
            reference.crop(w.raw, w.rows, w.cols, at.row, at.col, h, ww, w.tile[me]);
            reference.embed(w.model, w.tile[me], h, ww, w.stream[me]) catch return w.stop();
            for (0..w.layers) |i| reference.layer(w.model, w.scratches[me], i, w.stream[me], w.tokens, w.cos, w.sin) catch return w.stop();
            reference.final(w.model, w.scratches[me], w.stream[me], cells, w.proj[me]) catch return w.stop();
            for (w.bits[me], w.proj[me]) |*o, v| o.* = rounding.bits(v);
            host.unpatchify(w.bits[me], h, ww, w.picture[k * w.pixels ..][0..w.pixels]) catch return w.stop();
        }

        fn stop(w: *Work) void {
            w.failed.store(true, .monotonic);
        }
    };
};
