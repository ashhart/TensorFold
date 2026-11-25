//! tf-h3-image-decoder's self-test: the Metal decoder on synthetic weights against the float64 host reference.
const std = @import("std");
const mtl = @import("metal");
const dec = @import("h3_image_decoder");
const rounding = @import("bf16_round");

const host = dec.host;
const ref = dec.reference;
const vae = dec.vae;
const W = host.width;
const side = vae.cell_pixels;

/// Blocks checked one at a time against the reference, then chained; one tile also runs all 24 chained.
const checked = 2;

/// Errors count bf16 ULPs of max(|reference|, its row's RMS); the host's fp32 sums spread as far from its float64 ones.
pub const bounds = struct {
    /// fp32 sums against float64 move a bf16 output only where they straddle a rounding point: at most 1 ULP.
    pub const product_ulps: f64 = 1;
    pub const product_same: f64 = 0.99;
    /// One block on the GPU's own input (host fp32 sums: 95% identical, worst 4 ULPs, relative 7e-4).
    pub const block_ulps: f64 = 8;
    pub const block_same: f64 = 0.9;
    pub const block_relative: f64 = 2e-3;
    /// The output after two blocks chained (host fp32 sums: relative 2.9e-3, worst 3 ULPs) and after 24 (1.5e-2).
    pub const chain_ulps: f64 = 8;
    pub const chain_relative: f64 = 6e-3;
    pub const deep_relative: f64 = 3e-2;
    /// The assembled picture's bytes after two blocks: a value a ULP or two apart truncates at most two steps apart.
    pub const bytes_same: f64 = 0.8;
    pub const bytes_most: u8 = 2;
};

/// bf16's ULP at magnitude v: 2^(e - 8) for v in [2^(e - 1), 2^e).
fn ulp(v: f64) f64 {
    if (v == 0) return 0x1p-133;
    return std.math.ldexp(@as(f64, 1), std.math.frexp(v).exponent - 8);
}

const Stat = struct {
    count: usize = 0,
    same: usize = 0,
    worst: f64 = 0,
    diff: f64 = 0,
    norm: f64 = 0,

    /// GPU bf16 bits against bf16-exact reference values, in rows of `width`.
    fn add(s: *Stat, gpu: []const u16, want: []const f32, width: usize) void {
        var row: usize = 0;
        while (row < want.len) : (row += width) {
            const wr = want[row..][0..width];
            var ms: f64 = 0;
            for (wr) |v| ms += @as(f64, v) * v;
            const rms = @sqrt(ms / @as(f64, @floatFromInt(width)));
            for (gpu[row..][0..width], wr) |g, v| {
                const x = rounding.value(g);
                const d = @abs(@as(f64, x) - v);
                s.count += 1;
                s.same += @intFromBool(x == v);
                s.diff += d * d;
                s.norm += @as(f64, v) * v;
                s.worst = @max(s.worst, d / ulp(@max(@abs(v), rms)));
            }
        }
    }

    fn sameFraction(s: Stat) f64 {
        return @as(f64, @floatFromInt(s.same)) / @as(f64, @floatFromInt(@max(s.count, 1)));
    }

    fn relative(s: Stat) f64 {
        return @sqrt(s.diff / @max(s.norm, 1e-300));
    }

    fn report(s: Stat, name: []const u8, most_ulps: f64, least_same: f64, most_relative: f64) bool {
        const ok = s.worst <= most_ulps and s.sameFraction() >= least_same and s.relative() <= most_relative;
        std.debug.print("  {s}: {d} values, {d:.3}% identical, worst {d:.2} ULP, relative {e:.2} ({s})\n", .{ name, s.count, 100 * s.sameFraction(), s.worst, s.relative(), if (ok) "pass" else "FAIL" });
        return ok;
    }
};

/// A latent of rows x cols cells: its tiles in one pass for the stage checks, then decoded in passes of `batch`.
const Case = struct { rows: usize, cols: usize, batch: usize, deep: bool };

pub fn run(gpa: std.mem.Allocator, a: std.mem.Allocator) !bool {
    const device = try mtl.Device.init();
    const started = mtl.clock.seconds();
    const d = try dec.Decoder.init(gpa, device, vae.layers, 100);
    defer d.deinit();
    try d.synthetic(0x5eed);
    std.debug.print("{s}: kernels compiled, {d} synthetic blocks made in {d:.1} s\n", .{ device.name(), vae.layers, mtl.clock.seconds() - started });
    const model: ref.Model = .{ .w = d.params(), .plan = d.plan };
    const config: vae.Config = .{ .mean = @splat(0), .std = @splat(1), .tile = 256, .overlap = 64, .rope_theta = 100 };
    var ok = true;
    // one 6 x 10 tile (ids on axes of 6 and 10 cells), then 2 x 2 tiles of 16 x 16 decoded in passes of 3 and 1
    for ([_]Case{ .{ .rows = 6, .cols = 10, .batch = 1, .deep = true }, .{ .rows = 20, .cols = 28, .batch = 3, .deep = false } }) |c| {
        ok = try case(a, d, model, &config, c) and ok;
    }
    try timing(a, d, &config);
    std.debug.print("selftest: {s}\n", .{if (ok) "pass" else "FAIL"});
    return ok;
}

fn synthetic(a: std.mem.Allocator, rows: usize, cols: usize) ![]f32 {
    const latent = try a.alloc(f32, try host.latentLength(rows, cols));
    var prng = std.Random.DefaultPrng.init(rows * 1000 + cols);
    for (latent) |*v| v.* = (prng.random().float(f32) * 2 - 1) * 2;
    return latent;
}

/// The reference's sequences of every tile of `t`, from the latent through the embedding.
fn embedAll(a: std.mem.Allocator, model: ref.Model, latent: []const f32, c: Case, t: host.Tiles, out: []f32) !void {
    const cells = t.h * t.w;
    const per = out.len / t.total();
    const tile_latent = try a.alloc(f32, host.channels * cells);
    for (0..t.total()) |k| {
        const at = t.at(k);
        ref.crop(latent, c.rows, c.cols, at.row, at.col, t.h, t.w, tile_latent);
        try ref.embed(model, tile_latent, t.h, t.w, out[k * per ..][0..per]);
    }
}

fn case(a: std.mem.Allocator, d: *dec.Decoder, model: ref.Model, config: *const vae.Config, c: Case) !bool {
    const latent = try synthetic(a, c.rows, c.cols);
    const t = try host.tiles(a, config, c.rows, c.cols);
    const tiles = t.total();
    const cells = t.h * t.w;
    std.debug.print("{d} x {d} cells: {d} tiles of {d} x {d} cells, {d} tokens each\n", .{ c.rows, c.cols, tiles, t.h, t.w, cells + host.suffix });
    try d.prepare(latent, c.rows, c.cols, t.h, t.w, tiles);
    const n = d.tokens;
    const per = n * W;
    const pixels = 3 * cells * side * side;
    const origins = try a.alloc([2]u32, tiles);
    for (origins, 0..) |*o, k| {
        const at = t.at(k);
        o.* = .{ @intCast(at.row), @intCast(at.col) };
    }
    const cos = try a.alloc(u16, n * vae.rotated);
    const sin = try a.alloc(u16, n * vae.rotated);
    try host.rotary(t.h, t.w, config.rope_theta, cos, sin);
    const scratch = try ref.Scratch.init(a, n);
    var ok = true;

    // the embedding: post_quant_conv, x_embedder, the registers and the zero token
    _ = try d.pass(origins, .{ .to = 0, .final = false });
    const chained = try a.alloc(f32, tiles * per);
    try embedAll(a, model, latent, c, t, chained);
    var embedding: Stat = .{};
    embedding.add(d.stream(), chained, W);
    ok = embedding.report("embedding", bounds.product_ulps, bounds.product_same, 1) and ok;

    // each checked block on the GPU's own input, and the reference's chain beside it
    const input = try a.alloc(f32, tiles * per);
    for (0..checked) |i| {
        for (input, d.stream()) |*o, b| o.* = rounding.value(b);
        _ = try d.pass(origins, .{ .embed = false, .from = i, .to = i + 1, .final = false });
        for (0..tiles) |k| {
            try ref.layer(model, scratch, i, input[k * per ..][0..per], n, cos, sin);
            try ref.layer(model, scratch, i, chained[k * per ..][0..per], n, cos, sin);
        }
        var block: Stat = .{};
        block.add(d.stream(), input, W);
        var name: [32]u8 = undefined;
        ok = block.report(try std.fmt.bufPrint(&name, "block {d}", .{i}), bounds.block_ulps, bounds.block_same, bounds.block_relative) and ok;
    }

    // norm_out and proj_out on the GPU's stream, then the reference's chain against the GPU's
    for (input, d.stream()) |*o, b| o.* = rounding.value(b);
    _ = try d.pass(origins, .{ .embed = false, .from = checked, .to = checked });
    const want = try a.alloc(f32, cells * host.patch);
    const want_bits = try a.alloc(u16, want.len);
    const ref_tiles = try a.alloc(f32, tiles * pixels);
    const gpu_tiles = try a.alloc(f32, tiles * pixels);
    var final: Stat = .{};
    var chain: Stat = .{};
    for (0..tiles) |k| {
        const gpu = d.projected()[k * n * host.patch ..][0 .. cells * host.patch];
        try host.unpatchify(gpu, t.h, t.w, gpu_tiles[k * pixels ..][0..pixels]);
        try ref.final(model, scratch, input[k * per ..][0..per], cells, want);
        final.add(gpu, want, host.patch);
        try ref.final(model, scratch, chained[k * per ..][0..per], cells, want);
        chain.add(gpu, want, host.patch);
        for (want_bits, want) |*o, v| o.* = rounding.bits(v);
        try host.unpatchify(want_bits, t.h, t.w, ref_tiles[k * pixels ..][0..pixels]);
    }
    ok = final.report("norm_out and proj_out", bounds.product_ulps, bounds.product_same, 1) and ok;
    ok = chain.report("output after 2 blocks", bounds.chain_ulps, 0, bounds.chain_relative) and ok;

    // decode in passes of c.batch and of every tile: both equal the staged tiles assembled, bit for bit
    const height = c.rows * side;
    const width = c.cols * side;
    const staged = try assemble(a, t, gpu_tiles, 3 * height * width);
    const theirs_f = try assemble(a, t, ref_tiles, staged.len);
    const decoded = try a.alloc(f32, staged.len);
    for ([_]usize{ c.batch, tiles }, 0..) |b, i| {
        if (i == 1 and b == c.batch) continue;
        _ = try d.decode(a, config, latent, c.rows, c.cols, checked, b, decoded);
        const same = std.mem.eql(u32, @ptrCast(staged), @ptrCast(decoded));
        std.debug.print("  decode in passes of {d}: {s} the staged tiles assembled\n", .{ b, if (same) "bit-identical to" else "DIFFERENT from" });
        ok = ok and same;
    }
    const ours = try a.alloc(u8, 3 * height * width);
    const theirs = try a.alloc(u8, ours.len);
    try vae.pixels(decoded, height, width, ours);
    try vae.pixels(theirs_f, height, width, theirs);
    var same: usize = 0;
    var most: u8 = 0;
    for (ours, theirs) |x, y| {
        same += @intFromBool(x == y);
        most = @max(most, if (x > y) x - y else y - x);
    }
    const fraction = @as(f64, @floatFromInt(same)) / @as(f64, @floatFromInt(ours.len));
    const bytes_ok = fraction >= bounds.bytes_same and most <= bounds.bytes_most;
    std.debug.print("  picture bytes: {d:.3}% identical, at most {d} apart ({s})\n", .{ 100 * fraction, most, if (bytes_ok) "pass" else "FAIL" });
    ok = ok and bytes_ok;
    if (c.deep) ok = try deep(a, d, model, latent, c, t, origins, cos, sin, scratch) and ok;
    return ok;
}

fn assemble(a: std.mem.Allocator, t: host.Tiles, tiles: []const f32, len: usize) ![]f32 {
    const out = try a.alloc(f32, len);
    var assembly = try vae.Assembler.init(a, t.rows, t.cols, 3, out);
    const pixels = tiles.len / t.total();
    for (0..t.total()) |k| {
        const at = t.at(k);
        try assembly.place(at.r, at.c, tiles[k * pixels ..][0..pixels]);
    }
    return out;
}

/// All 24 blocks chained, the GPU from the latent in one pass and the reference beside it.
fn deep(a: std.mem.Allocator, d: *dec.Decoder, model: ref.Model, latent: []const f32, c: Case, t: host.Tiles, origins: []const [2]u32, cos: []const u16, sin: []const u16, scratch: ref.Scratch) !bool {
    try d.prepare(latent, c.rows, c.cols, t.h, t.w, t.total());
    _ = try d.pass(origins, .{ .to = vae.layers });
    const cells = t.h * t.w;
    const per = d.tokens * W;
    const x = try a.alloc(f32, t.total() * per);
    try embedAll(a, model, latent, c, t, x);
    const want = try a.alloc(f32, cells * host.patch);
    var s: Stat = .{};
    for (0..t.total()) |k| {
        for (0..vae.layers) |i| try ref.layer(model, scratch, i, x[k * per ..][0..per], d.tokens, cos, sin);
        try ref.final(model, scratch, x[k * per ..][0..per], cells, want);
        s.add(d.projected()[k * d.tokens * host.patch ..][0 .. cells * host.patch], want, host.patch);
    }
    return s.report("output after 24 blocks", std.math.inf(f64), 0, bounds.deep_relative);
}

/// All 24 blocks on a 1024 x 1024 canvas (64 x 64 cells, 5 x 5 tiles of 16 x 16), passes of 16 tiles.
fn timing(a: std.mem.Allocator, d: *dec.Decoder, config: *const vae.Config) !void {
    const cells = 64;
    const latent = try synthetic(a, cells, cells);
    const out = try a.alloc(f32, 3 * cells * cells * side * side);
    const tiles = (try host.tiles(a, config, cells, cells)).total();
    const took = try d.decode(a, config, latent, cells, cells, vae.layers, 16, out);
    std.debug.print("timing: {d} tiles of 16 x 16, 24 blocks: {d:.2} s in passes, {d:.2} s on the GPU, {d:.1} ms a tile\n", .{ tiles, took.wall, took.gpu, 1000 * took.gpu / @as(f64, @floatFromInt(tiles)) });
}
