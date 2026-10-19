//! CSA2 sink comparisons against the FP64 ideal.
const sparse = @import("operator_csa2_fp64_sparse.zig");
const SparseSpec = sparse.SparseSpec;
const sparseCase = sparse.sparseCase;
const sparseCaseFrom = sparse.sparseCaseFrom;
const sparseCaseScaled = sparse.sparseCaseScaled;
const KeyPattern = sparse.KeyPattern;
const fixture = @import("csa2_fp64_fixture.zig");
const std = fixture.std;
const mtl = fixture.mtl;
const math = fixture.math;
const csa2 = fixture.csa2;
const width = fixture.width;
const ratio = fixture.ratio;
const Lcg = fixture.Lcg;

/// One live column isolates sink normalization; boundary rationale: docs/csa2-reference-cases.md.
fn sinkCase(
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    p: *const csa2.Pipelines,
    name: []const u8,
    rng: *Lcg,
    sink: f32,
) !void {
    const spec = SparseSpec{ .name = name, .heads = 1, .rows = 1, .position = 0, .ratio = 0, .compressed_rows = 0, .topk = 0 };
    const q = try gpa.alloc(u16, width);
    defer gpa.free(q);
    for (q) |*cell| cell.* = math.b(rng.scale(1.0));
    const window_rows = try gpa.alloc([]const f32, 1);
    defer gpa.free(window_rows);
    const cells = try gpa.alloc(f32, width);
    defer gpa.free(cells);
    for (cells) |*v| v.* = rng.scale(1.0);
    window_rows[0] = cells;
    const sinks = try gpa.alloc(f32, 1);
    defer gpa.free(sinks);
    sinks[0] = sink;
    const empty = try gpa.alloc([]const f32, 0);
    defer gpa.free(empty);
    const indices = try gpa.alloc(i32, 0);
    defer gpa.free(indices);
    try sparseCase(device, queue, p, spec, q, window_rows, empty, indices, sinks);
}

test "sink normalization sits at rounding level across the exp overflow boundaries" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var pipelines = try csa2.Pipelines.load(device);
    defer pipelines.deinit();
    var rng = Lcg{ .state = 0x51c0de51c0de51c0 };

    // Sweep underflow, balanced mass and overflow boundaries against the FP64 equation.
    const sweep = [_]struct { sink: f32, name: []const u8 }{
        .{ .sink = -104, .name = "sink_underflow" },
        .{ .sink = -90, .name = "sink_near_underflow" },
        .{ .sink = -60, .name = "sink_tiny" },
        .{ .sink = -12, .name = "sink_small" },
        .{ .sink = -1, .name = "sink_minus_one" },
        .{ .sink = 0, .name = "sink_zero" },
        .{ .sink = 1, .name = "sink_one" },
        .{ .sink = 12, .name = "sink_medium" },
        .{ .sink = 60, .name = "sink_large" },
        .{ .sink = 80, .name = "sink_edge80" },
        .{ .sink = 83, .name = "sink_edge83" },
        .{ .sink = 84, .name = "sink_edge84" },
        .{ .sink = 85, .name = "sink_edge85" },
        .{ .sink = 86, .name = "sink_over_edge" },
        .{ .sink = 90, .name = "sink_overflow" },
    };
    for (sweep) |entry| try sinkCase(gpa, device, queue, &pipelines, entry.name, &rng, entry.sink);

    // Distinct per-head sink masses check indexing and arithmetic.
    {
        const spec = SparseSpec{ .name = "sink_per_head", .heads = 4, .rows = 1, .position = 0, .ratio = 0, .compressed_rows = 0, .topk = 0 };
        var head_rng = Lcg{ .state = 0xabcdef0123456789 };
        const q = try gpa.alloc(u16, spec.heads * width);
        defer gpa.free(q);
        for (q) |*cell| cell.* = math.b(head_rng.scale(1.0));
        const window_rows = try gpa.alloc([]const f32, 1);
        defer gpa.free(window_rows);
        const cells = try gpa.alloc(f32, width);
        defer gpa.free(cells);
        for (cells) |*v| v.* = head_rng.scale(1.0);
        window_rows[0] = cells;
        const sinks = try gpa.alloc(f32, spec.heads);
        defer gpa.free(sinks);
        for (0..spec.heads) |h| sinks[h] = @as(f32, @floatFromInt(@as(i32, @intCast(h)) * 24 - 36));
        const empty = try gpa.alloc([]const f32, 0);
        defer gpa.free(empty);
        const indices = try gpa.alloc(i32, 0);
        defer gpa.free(indices);
        try sparseCase(device, queue, &pipelines, spec, q, window_rows, empty, indices, sinks);
    }
}

// Flat, ramped and descending scores exercise the online denominator; see docs/csa2-reference-cases.md.
test "tile denominators sit at rounding level for every tile shape" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var pipelines = try csa2.Pipelines.load(device);
    defer pipelines.deinit();
    var rng = Lcg{ .state = 0x1a2b3c4d5e6f7081 };

    // Ten tiles, 639 equal-score terms: the unsunk denominator is the term count.
    try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = "tiles_equal_scores", .heads = 1, .rows = 1, .position = 1022, .ratio = 2, .compressed_rows = 512, .topk = 512 }, &rng, 1.0, 0, 0, 0, KeyPattern.cancelling);
    // Ascending tile scores exercise lost correction mass at FP32 underflow.
    for ([_]i32{ -2, 0, 1, 2 }) |step| {
        var name_buf: [24]u8 = undefined;
        const label = std.fmt.bufPrint(&name_buf, "tiles_ramp{d}", .{step}) catch unreachable;
        try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = label, .heads = 1, .rows = 1, .position = 1000, .ratio = 2, .compressed_rows = 512, .topk = 512 }, &rng, 1.0, step, 0, 0, KeyPattern.near_tie);
    }
    // Descending scores leave the running maximum fixed, so the denominator is a plain sum.
    try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = "tiles_descending", .heads = 1, .rows = 1, .position = 1000, .ratio = 2, .compressed_rows = 512, .topk = 512 }, &rng, 1.0, 0, 1, 0, KeyPattern.near_tie);
    // One tile only, the boundary case of the online loop: the correction is applied zero times.
    try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = "tiles_single", .heads = 1, .rows = 1, .position = 20, .ratio = 0, .compressed_rows = 0, .topk = 0 }, &rng, 1.0, 0, 0, 0, KeyPattern.random);
}

// Different scale grids and cancelling mirror rows expose output-mix error; see docs/csa2-reference-cases.md.
test "the output mix sits at rounding level when window and compressed rows cancel" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var pipelines = try csa2.Pipelines.load(device);
    defer pipelines.deinit();
    var rng = Lcg{ .state = 0x77a1c0de_5b5e1a2b };

    // Reverse window/compressed dominance across six orders of magnitude.
    try sparseCaseScaled(gpa, device, queue, &pipelines, .{ .name = "mix_window_dominates", .heads = 1, .rows = 1, .position = 100, .ratio = 2, .compressed_rows = 100, .topk = 64 }, &rng, 1.0, 14, -12, 0, 0, KeyPattern.random);
    try sparseCaseScaled(gpa, device, queue, &pipelines, .{ .name = "mix_compressed_dominates", .heads = 1, .rows = 1, .position = 100, .ratio = 2, .compressed_rows = 100, .topk = 64 }, &rng, 1.0, -12, 14, 0, 0, KeyPattern.random);
    try sparseCaseScaled(gpa, device, queue, &pipelines, .{ .name = "mix_balanced_magnitudes", .heads = 1, .rows = 1, .position = 100, .ratio = 2, .compressed_rows = 100, .topk = 64 }, &rng, 1.0, 6, 6, 0, 0, KeyPattern.cancelling);

    // Alternating large shapes cancel in 128 rows, amplifying error relative to the small surviving sum.
    for ([_]i32{ 8, 16, 23 }) |shape_exponent| {
        var name_buf: [24]u8 = undefined;
        const label = std.fmt.bufPrint(&name_buf, "mirror_shape_e{d}", .{shape_exponent}) catch unreachable;
        const spec = SparseSpec{ .name = label, .heads = 1, .rows = 1, .position = 127, .ratio = 0, .compressed_rows = 0, .topk = 0 };
        const q = try gpa.alloc(u16, width);
        defer gpa.free(q);
        // The uniform query has zero dot product with the alternating shape.
        for (q) |*cell| cell.* = math.b(1.0);
        const window_rows = try gpa.alloc([]const f32, 128);
        defer gpa.free(window_rows);
        const shape = std.math.pow(f32, 2, @floatFromInt(shape_exponent));
        const shared = try gpa.alloc(f32, width);
        defer gpa.free(shared);
        // The shared perturbation is scaled by 2^-16; smaller terms can disappear in the FP8 grid.
        for (shared) |*v| v.* = rng.scale(shape * 0x1p-16);
        for (window_rows, 0..) |*row, i| {
            const cells = try gpa.alloc(f32, width);
            const side: f32 = if ((i & 1) == 0) 1 else -1;
            for (0..width) |d| {
                const flip: f32 = if ((d & 1) == 0) 1 else -1;
                cells[d] = side * flip * shape + shared[d];
            }
            row.* = cells;
        }
        const sinks = try gpa.alloc(f32, 1);
        defer gpa.free(sinks);
        sinks[0] = 0;
        const empty = try gpa.alloc([]const f32, 0);
        defer gpa.free(empty);
        const indices = try gpa.alloc(i32, 0);
        defer gpa.free(indices);
        try sparseCase(device, queue, &pipelines, spec, q, window_rows, empty, indices, sinks);
    }
}
