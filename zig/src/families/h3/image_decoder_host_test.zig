//! The host image decoder (image_decoder_host.zig): tiles across cores, blended as the reference chain does.
const std = @import("std");
const rounding = @import("bf16_round");
const d = @import("image_decoder_host.zig");

const host = d.host;
const vae = d.vae;
const ref = d.reference;

test "the host image decoder: tiles across cores, the reference chain tile by tile, bit for bit" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // 7 x 7 cells as 2 x 2 tiles of 4 x 4 cells overlapping one cell: both blends exercised
    const c: vae.Config = .{ .mean = @splat(0), .std = @splat(1), .tile = 64, .overlap = 16, .rope_theta = 100 };
    var decoder = try d.Decoder.init(std.testing.allocator, 2);
    defer decoder.deinit();
    try decoder.synthetic(a, 0x5eed);
    var prng = std.Random.DefaultPrng.init(0x9);
    const latent = try a.alloc(f32, try host.latentLength(7, 7));
    for (latent) |*v| v.* = (prng.random().float(f32) * 2 - 1) * 2;
    const t = try host.tiles(a, &c, 7, 7);
    try std.testing.expectEqual(@as(usize, 4), t.total());
    // the serial chain the self-test runs beside the GPU: crop, embed, the blocks, proj_out, unpatchify
    const model: ref.Model = .{ .w = decoder.weights, .plan = decoder.plan };
    const n = try host.tokens(t.h, t.w);
    const cells = t.h * t.w;
    const cos = try a.alloc(u16, n * vae.rotated);
    const sin = try a.alloc(u16, cos.len);
    try host.rotary(t.h, t.w, c.rope_theta, cos, sin);
    const scratch = try ref.Scratch.init(a, n);
    const x = try a.alloc(f32, n * host.width);
    const proj = try a.alloc(f32, cells * host.patch);
    const bits = try a.alloc(u16, proj.len);
    const tile_latent = try a.alloc(f32, host.channels * cells);
    const tile = try a.alloc(f32, 3 * t.rows[0].len * t.cols[0].len);
    const want = try a.alloc(f32, 3 * 112 * 112);
    var assembly = try vae.Assembler.init(a, t.rows, t.cols, 3, want);
    for (0..t.total()) |k| {
        const at = t.at(k);
        ref.crop(latent, 7, 7, at.row, at.col, t.h, t.w, tile_latent);
        try ref.embed(model, tile_latent, t.h, t.w, x);
        for (0..2) |i| try ref.layer(model, scratch, i, x, n, cos, sin);
        try ref.final(model, scratch, x, cells, proj);
        for (bits, proj) |*o, v| o.* = rounding.bits(v);
        try host.unpatchify(bits, t.h, t.w, tile);
        try assembly.place(at.r, at.c, tile);
    }
    const got = try a.alloc(f32, want.len);
    try decoder.decode(a, &c, latent, 7, 7, 2, got);
    try std.testing.expectEqualSlices(f32, want, got);
}

/// std.Thread whose second spawn fails, as when the process runs out of threads.
const Exhausted = struct {
    inner: std.Thread,
    var spawned: usize = 0;

    pub fn spawn(config: std.Thread.SpawnConfig, comptime f: anytype, args: anytype) std.Thread.SpawnError!Exhausted {
        spawned += 1;
        if (spawned > 1) return error.ThreadQuotaExceeded;
        return .{ .inner = try std.Thread.spawn(config, f, args) };
    }

    pub fn join(self: Exhausted) void {
        self.inner.join();
    }
};

test "the host image decoder fails, not a partial picture, when a worker cannot be spawned" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c: vae.Config = .{ .mean = @splat(0), .std = @splat(1), .tile = 64, .overlap = 16, .rope_theta = 100 };
    var decoder = try d.Decoder.init(std.testing.allocator, 2);
    defer decoder.deinit();
    try decoder.synthetic(a, 0x5eed);
    const latent = try a.alloc(f32, try host.latentLength(7, 7));
    @memset(latent, 0.5);
    const got = try a.alloc(f32, 3 * 112 * 112);
    @memset(got, -7);
    try std.testing.expectError(error.ThreadQuotaExceeded, decoder.decodeWith(a, &c, latent, 7, 7, 2, got, Exhausted, 4));
    try std.testing.expectEqual(@as(usize, 2), Exhausted.spawned);
    for (got) |v| try std.testing.expectEqual(@as(f32, -7), v);
}
