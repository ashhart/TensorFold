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
const output_guard_bytes = 64;
const output_poison: u8 = 0xff;

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
    const output_capacity = 129 * 1025 * 2 + output_guard_bytes;
    const classic = try r.out(output_capacity);
    const selected = try r.out(output_capacity);
    const classic_buf: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = classic, .len = output_capacity };
    const selected_buf: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = selected, .len = output_capacity };
    var cases: usize = 0;

    for (boundary_columns) |n| {
        const w = try fp8Weight(r, a, random, n, k_served);
        for (boundary_rows) |rows| {
            const expected = rows > 16 and rows <= 128 and n == 1024;
            if (prompt.useSmallm(rows, w) != expected) return error.Prompt16DispatchBoundaryMismatch;
            try classic_buf.fill8(output_poison, r.s.handle);
            try selected_buf.fill8(output_poison, r.s.handle);
            try prompt.matmulTile(r.s, x, k_served, rows, w, selected, .bf16, .lean);
            try prompt.matmulTile(r.s, x, k_served, rows, w, classic, .bf16, .classic);
            try r.s.synchronize();
            const bytes = rows * n * 2;
            const selected_after = try r.back(u8, selected, output_capacity);
            const classic_after = try r.back(u8, classic, output_capacity);
            try check.sameBytes("prompt16 default dispatch vs classic", selected_after[0..bytes], classic_after[0..bytes]);
            try outputExtent(selected_after, bytes);
            try outputExtent(classic_after, bytes);
            cases += 1;
        }
    }

    for ([_]usize{ 4992, 5248 }) |k| {
        const w = try fp8Weight(r, a, random, 1024, k);
        if (prompt.useSmallm(64, w)) return error.Prompt16KBoundaryMismatch;
        try classic_buf.fill8(output_poison, r.s.handle);
        try selected_buf.fill8(output_poison, r.s.handle);
        try prompt.matmulTile(r.s, x, k, 64, w, selected, .bf16, .lean);
        try prompt.matmulTile(r.s, x, k, 64, w, classic, .bf16, .classic);
        try r.s.synchronize();
        const selected_after = try r.back(u8, selected, output_capacity);
        const classic_after = try r.back(u8, classic, output_capacity);
        const bytes = 64 * 1024 * 2;
        try check.sameBytes("prompt16 K fallback vs classic", selected_after[0..bytes], classic_after[0..bytes]);
        try outputExtent(selected_after, bytes);
        try outputExtent(classic_after, bytes);
        cases += 1;
    }

    std.debug.print("RESULT prompt16 boundary bytes: {d} shapes exact (143 K=5120 dispatch cases + 2 K fallbacks at 4992/5248); SM120/82, N edges 1..1025, M edges 16..129\n", .{cases});
    check.pass("prompt16 default dispatch matches classic bytes across M, N and K selection boundaries", .{});
}

fn outputExtent(bytes: []const u8, live_bytes: usize) !void {
    if (live_bytes > bytes.len or live_bytes % 2 != 0) return error.Prompt16OutputExtentChanged;
    var i: usize = 0;
    while (i < live_bytes) : (i += 2) {
        if (bytes[i] == output_poison and bytes[i + 1] == output_poison) return error.Prompt16OutputExtentChanged;
    }
    for (bytes[live_bytes..]) |byte| {
        if (byte != output_poison) return error.Prompt16OutputGuardChanged;
    }
}

fn productionTiming(prompt: *const cuda.nvfp4.Prompt, r: *Rig, a: std.mem.Allocator, random: std.Random) !void {
    const n: usize = 1024;
    const k: usize = k_served;
    const single_weight = try fp8Weight(r, a, random, n, k);
    const l2_bytes_i = try r.gpu.ctx.attribute(.l2_cache_size);
    const l2_bytes: usize = @intCast(@max(l2_bytes_i, 1));
    const weight_bytes = cuda.fp8.padded(n) * k;
    const rotating_count = (2 * l2_bytes) / weight_bytes + 1;
    if (rotating_count > 128) return error.RotatingWeightSetTooLarge;
    const rotating_weights = try a.alloc(cuda.qmmf.Weight, rotating_count);
    for (rotating_weights) |*w| w.* = try fp8Weight(r, a, random, n, k);
    const rotating_mib = rotating_count * weight_bytes / (1024 * 1024);
    const input = r.bfs(128 * k, 0.5);
    const x = try r.dev(std.mem.sliceAsBytes(input));
    const classic = try r.out(128 * n * 2);
    const selected = try r.out(128 * n * 2);
    var start = try cuda.Event.init(r.gpu.d, true);
    defer start.deinit();
    var stop = try cuda.Event.init(r.gpu.d, true);
    defer stop.deinit();

    for (prompt_shapes) |shape| {
        if (!prompt.useSmallm(shape.rows, single_weight)) return error.Prompt16ShapeNotSelected;
        try prompt.matmulTile(r.s, x, k, shape.rows, single_weight, selected, .bf16, .lean);
        try prompt.matmulTile(r.s, x, k, shape.rows, single_weight, classic, .bf16, .classic);
        try r.s.synchronize();
        try check.sameBytes("prompt16 served dispatch vs classic", try r.back(u8, selected, shape.rows * n * 2), try r.back(u8, classic, shape.rows * n * 2));

        var classic_us: [samples]f64 = undefined;
        var smallm_us: [samples]f64 = undefined;
        var classic_rotating_us: [samples]f64 = undefined;
        var smallm_rotating_us: [samples]f64 = undefined;
        for (0..samples) |i| {
            if (i % 2 == 0) {
                classic_us[i] = try timePrompt(r.s, start, stop, prompt.*, x, k, shape.rows, single_weight, classic, .classic);
                smallm_us[i] = try timePrompt(r.s, start, stop, prompt.*, x, k, shape.rows, single_weight, selected, .lean);
                classic_rotating_us[i] = try timePromptRotating(r.s, start, stop, prompt.*, x, k, shape.rows, rotating_weights, classic, .classic);
                smallm_rotating_us[i] = try timePromptRotating(r.s, start, stop, prompt.*, x, k, shape.rows, rotating_weights, selected, .lean);
            } else {
                smallm_us[i] = try timePrompt(r.s, start, stop, prompt.*, x, k, shape.rows, single_weight, selected, .lean);
                classic_us[i] = try timePrompt(r.s, start, stop, prompt.*, x, k, shape.rows, single_weight, classic, .classic);
                smallm_rotating_us[i] = try timePromptRotating(r.s, start, stop, prompt.*, x, k, shape.rows, rotating_weights, selected, .lean);
                classic_rotating_us[i] = try timePromptRotating(r.s, start, stop, prompt.*, x, k, shape.rows, rotating_weights, classic, .classic);
            }
        }
        const columns = (n + 127) / 128;
        const classic_ctas = columns;
        const smallm_ctas = ((shape.rows + 15) / 16) * columns;
        std.debug.print("RESULT prompt16 rows {d}: hot classic {d:.2} us/{d} CTAs, hot small-row {d:.2} us/{d} CTAs; rotating {d} MiB (L2 {d} MiB), classic {d:.2} us, small-row {d:.2} us; exact bytes\n", .{
            shape.rows,
            check.median(&classic_us),
            classic_ctas,
            check.median(&smallm_us),
            smallm_ctas,
            rotating_mib,
            l2_bytes / (1024 * 1024),
            check.median(&classic_rotating_us),
            check.median(&smallm_rotating_us),
        });
    }
    check.pass("prompt16 small-row default path equals classic bytes for served rows 17, 32, 64 and 128", .{});
}

fn timePromptRotating(s: cuda.Stream, start: cuda.Event, stop: cuda.Event, prompt: cuda.nvfp4.Prompt, x: u64, k: usize, rows: usize, weights: []const cuda.qmmf.Weight, out: u64, tile: cuda.nvfp4.Prompt.Tile) !f64 {
    if (weights.len == 0) return error.EmptyRotatingWeights;
    try start.record(s);
    for (0..reps) |i| try prompt.matmulTile(s, x, k, rows, weights[i % weights.len], out, .bf16, tile);
    try stop.record(s);
    try stop.synchronize();
    return @as(f64, @floatCast(try start.elapsedMs(stop))) * 1000 / reps;
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
