//! Training passes holding several sequences against one a pass, and from kept prefixes against whole, on CUDA.
const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check_mod = @import("check.zig");

const train = nemotron.train;
const Trainer = train.Trainer;
const compare = @import("pack_compare.zig").For(train);

/// A pocket's neurons at a layer as the kernel holds them, the first `used` live, the last `open` of those scaled.
const neurons = 64;
const used = 6;
const open = 2;

/// MODEL's training passes over `count` of EXAMPLES (a pocket dump's), one a pass and side by side in each mode.
pub fn run(gpu: check_mod.Gpu, model: []const u8, path: []const u8, count: usize) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const kernels = std.c.getenv("TENSORFOLD_CUDA_KERNELS");
    const e = try nemotron.Engine.init(gpa, io, gpu.ctx, model, if (kernels) |k| std.mem.span(k) else null, .{ .context = 8192, .mtp = false, .graphs = false, .sampling = null, .segments = 1 });
    defer e.deinit();
    var living: ?nemotron.sites.Sites = null;
    const t0 = now(io);
    const t = try Trainer.init(gpa, io, e, model, &living);
    std.debug.print("RESULT trainer ready in {d:.1} s\n", .{now(io) - t0});
    defer t.deinit();
    const c = e.c;
    const d = c.hidden;
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    var layers: [4]usize = undefined;
    var found: usize = 0;
    for (0..c.layers) |l| if (found < layers.len and c.kinds[l] == .moe and l >= c.layers * (2 + 2 * found) / 10) {
        layers[found] = l;
        found += 1;
    };
    var held: [8]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (held[0..made]) |*x| x.free();
    for (layers[0..found]) |l| {
        const up = try gpa.alloc(f32, neurons * d);
        defer gpa.free(up);
        const down = try gpa.alloc(f32, neurons * d);
        defer gpa.free(down);
        @memset(up, 0);
        @memset(down, 0);
        for (up[0 .. used * d]) |*v| v.* = rng.floatNorm(f32) * 5e-4;
        for (down[0 .. used * d]) |*v| v.* = rng.floatNorm(f32) * 2e-2;
        held[made] = try cuda.DeviceBuffer.alloc(e.ctx.d, up.len * 4);
        held[made + 1] = try cuda.DeviceBuffer.alloc(e.ctx.d, down.len * 4);
        made += 2;
        try e.ops().upload(held[made - 2].ptr, std.mem.sliceAsBytes(up));
        try e.ops().upload(held[made - 1].ptr, std.mem.sliceAsBytes(down));
        e.w.blocks[l].moe.pocket = .{ .up = held[made - 2].ptr, .down = held[made - 1].ptr, .n = neurons };
        t.from[l] = used - open;
    }
    try e.stream.synchronize();
    // the checks' sequences and results live until the end
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const seqs = try compare.examples(a, io, path, count, layers[0..found], rng);
    try compare.pairs(a, t, seqs, layers[0..found], 85, false);
    try compare.check(a, io, t, seqs, layers[0..found]);
    try compare.prefixes(a, io, t, seqs, layers[0..found]);
}

fn now(io: std.Io) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
}
