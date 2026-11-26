//! RTX 5090 Mobile qualification of the default small-row FP8 prompt tile.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig_ = @import("flashnext_rig.zig");

const Rig = rig_.Rig;
const k_served = 5120;
const boundary_rows = [_]usize{ 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129 };
const boundary_columns = [_]usize{ 1, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025 };
const prompt_shapes = [_]struct { rows: usize, n: usize }{
    .{ .rows = 17, .n = 1024 },
    .{ .rows = 32, .n = 1024 },
    .{ .rows = 64, .n = 1024 },
    .{ .rows = 128, .n = 1024 },
};
const reps = 48;
const samples = 5;

pub fn run(gpu: check.Gpu) !void {
    const capability = try gpu.ctx.capability();
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    if (capability != 120 or sms != 82) return error.RequiresRtx5090MobileSm120;

    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x5090_4951);
    const random = prng.random();
    var r = try rig_.Rig.init(gpu, a, random);
    defer r.deinit();
    var prompt = try cuda.nvfp4.Prompt.load(gpu.d);
    defer prompt.unload();
    if (!prompt.smallm_enabled) return error.Prompt16SmallmDisabled;

    try boundaryBytes(&prompt, &r, a, random);
    try productionTiming(&prompt, &r, a, random);
}

fn boundaryBytes(prompt: *const cuda.nvfp4.Prompt, r: *Rig, a: std.mem.Allocator, random: std.Random) !void {
    const input = r.bfs(129 * 5248, 0.5);
    const x = try r.dev(std.mem.sliceAsBytes(input));
    const classic = try r.out(129 * 1025 * 2);
    const selected = try r.out(129 * 1025 * 2);
    var cases: usize = 0;

    for (boundary_columns) |n| {
        const w = try fp8Weight(r, a, random, n, k_served);
        for (boundary_rows) |rows| {
            const expected = rows > 16 and rows <= 128 and n <= 1024;
            if (prompt.useSmallm(rows, w) != expected) return error.Prompt16DispatchBoundaryMismatch;
            try prompt.matmulTile(r.s, x, k_served, rows, w, selected, .bf16, .lean);
            try prompt.matmulTile(r.s, x, k_served, rows, w, classic, .bf16, .classic);
            try r.s.synchronize();
            const bytes = rows * n * 2;
            try check.sameBytes("prompt16 default dispatch vs classic", try r.back(u8, selected, bytes), try r.back(u8, classic, bytes));
            cases += 1;
        }
    }

    for ([_]usize{ 4992, 5248 }) |k| {
        const w = try fp8Weight(r, a, random, 1024, k);
        if (prompt.useSmallm(64, w)) return error.Prompt16KBoundaryMismatch;
        try prompt.matmulTile(r.s, x, k, 64, w, selected, .bf16, .lean);
        try prompt.matmulTile(r.s, x, k, 64, w, classic, .bf16, .classic);
        try r.s.synchronize();
        try check.sameBytes("prompt16 K fallback vs classic", try r.back(u8, selected, 64 * 1024 * 2), try r.back(u8, classic, 64 * 1024 * 2));
        cases += 1;
    }

    std.debug.print("RESULT prompt16 boundary bytes: {d} shapes exact (143 K=5120 dispatch cases + 2 K fallbacks at 4992/5248); SM120/82, N edges 1..1025, M edges 16..129\n", .{cases});
    check.pass("prompt16 default dispatch matches classic bytes across M, N and K selection boundaries", .{});
}

fn productionTiming(prompt: *const cuda.nvfp4.Prompt, r: *Rig, a: std.mem.Allocator, random: std.Random) !void {
    const n: usize = 1024;
    const k: usize = k_served;
    const weights = try fp8Weight(r, a, random, n, k);
    const input = r.bfs(128 * k, 0.5);
    const x = try r.dev(std.mem.sliceAsBytes(input));
    const classic = try r.out(128 * n * 2);
    const selected = try r.out(128 * n * 2);
    var start = try cuda.Event.init(r.gpu.d, true);
    defer start.deinit();
    var stop = try cuda.Event.init(r.gpu.d, true);
    defer stop.deinit();

    for (prompt_shapes) |shape| {
        if (!prompt.useSmallm(shape.rows, weights)) return error.Prompt16ShapeNotSelected;
        try prompt.matmulTile(r.s, x, k, shape.rows, weights, selected, .bf16, .lean);
        try prompt.matmulTile(r.s, x, k, shape.rows, weights, classic, .bf16, .classic);
        try r.s.synchronize();
        try check.sameBytes("prompt16 served dispatch vs classic", try r.back(u8, selected, shape.rows * n * 2), try r.back(u8, classic, shape.rows * n * 2));

        var classic_us: [samples]f64 = undefined;
        var smallm_us: [samples]f64 = undefined;
        for (0..samples) |i| {
            if (i % 2 == 0) {
                classic_us[i] = try timePrompt(r.s, start, stop, prompt.*, x, k, shape.rows, weights, classic, .classic);
                smallm_us[i] = try timePrompt(r.s, start, stop, prompt.*, x, k, shape.rows, weights, selected, .lean);
            } else {
                smallm_us[i] = try timePrompt(r.s, start, stop, prompt.*, x, k, shape.rows, weights, selected, .lean);
                classic_us[i] = try timePrompt(r.s, start, stop, prompt.*, x, k, shape.rows, weights, classic, .classic);
            }
        }
        const columns = (n + 127) / 128;
        const classic_ctas = columns;
        const smallm_ctas = ((shape.rows + 15) / 16) * columns;
        std.debug.print("RESULT prompt16 rows {d}: classic {d:.2} us/{d} CTAs, default small-row {d:.2} us/{d} CTAs, exact bytes\n", .{
            shape.rows,
            check.median(&classic_us),
            classic_ctas,
            check.median(&smallm_us),
            smallm_ctas,
        });
    }
    check.pass("prompt16 small-row default path equals classic bytes for served rows 17, 32, 64 and 128", .{});
}

fn timePrompt(s: cuda.Stream, start: cuda.Event, stop: cuda.Event, prompt: cuda.nvfp4.Prompt, x: u64, k: usize, rows: usize, w: cuda.qmmf.Weight, out: u64, tile: cuda.nvfp4.Prompt.Tile) !f64 {
    try start.record(s);
    for (0..reps) |_| try prompt.matmulTile(s, x, k, rows, w, out, .bf16, tile);
    try stop.record(s);
    try stop.synchronize();
    return @as(f64, @floatCast(try start.elapsedMs(stop))) * 1000 / reps;
}

fn fp8Weight(r: *Rig, a: std.mem.Allocator, random: std.Random, n: usize, k: usize) !cuda.qmmf.Weight {
    const codes = try a.alloc(u8, n * k);
    for (codes) |*v| {
        v.* = 0x30 + random.uintLessThan(u8, 16);
        if (random.boolean()) v.* |= 0x80;
    }
    const tiled = try a.alloc(u8, cuda.fp8.padded(n) * k);
    cuda.fp8.packCodes(tiled, codes, n, k);
    return cuda.modelopt.fp8Weight(try r.dev(tiled), 0.125, n, k);
}
