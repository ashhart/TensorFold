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

/// One live column pins the running maximum to a single known score, so `sink - max` is the whole
/// normalization. Sweeping it across the FP32 exp() overflow and underflow limits is the only path
/// by which the sink term can move the stage output, and it is also where an FP32 exp() would leave
/// the FP64 equation behind.
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

    // One case per boundary FP32 exp() has: underflow to zero, the overflow edge, and the
    // mass-balanced middle where the sink decides the output on its own. From about 83 upward the
    // sink exponent reaches the FP32 overflow boundary, so the cells that differ from the reference
    // sit near 1e-38 and a one-ULP difference in the running maximum decides whether either side
    // sees a finite denominator at all.
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

    // Per-head sinks: the kernel indexes the sink array by head, so heads whose sink masses differ
    // by orders of magnitude check that indexing as well as the arithmetic.
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

// Tile denominators: the online correction, the per-tile sum and the running maximum. A flat
// score set makes the denominator a pure sum of equal exponentials; a ramped score set drives the
// correction through the FP32 underflow limit; a descending set never applies a correction at all.
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

    // Ten tiles, so the denominator accumulates 639 exponentials and the accumulator takes 639
    // products. Identical key shapes make every score tie: the denominator is exactly the count.
    try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = "tiles_equal_scores", .heads = 1, .rows = 1, .position = 1022, .ratio = 2, .compressed_rows = 512, .topk = 512 }, &rng, 1.0, 0, 0, 0, KeyPattern.cancelling);
    // Ascending scores by tile: the correction falls one power of two per step until FP32 flushes
    // it, and the dropped mass is the whole question.
    for ([_]i32{ -2, 0, 1, 2 }) |step| {
        var name_buf: [24]u8 = undefined;
        const label = std.fmt.bufPrint(&name_buf, "tiles_ramp{d}", .{step}) catch unreachable;
        try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = label, .heads = 1, .rows = 1, .position = 1000, .ratio = 2, .compressed_rows = 512, .topk = 512 }, &rng, 1.0, step, 0, 0, KeyPattern.near_tie);
    }
    // Descending scores: the running maximum never moves, so every tile lands at full weight and
    // the denominator is a plain sum.
    try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = "tiles_descending", .heads = 1, .rows = 1, .position = 1000, .ratio = 2, .compressed_rows = 512, .topk = 512 }, &rng, 1.0, 0, 1, 0, KeyPattern.near_tie);
    // One tile only, the boundary case of the online loop: the correction is applied zero times.
    try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = "tiles_single", .heads = 1, .rows = 1, .position = 20, .ratio = 0, .compressed_rows = 0, .topk = 0 }, &rng, 1.0, 0, 0, 0, KeyPattern.random);
}

// The output mix: window rows and compressed rows meeting in one accumulator on different scale
// grids, and two mirror rows whose shared large shape cancels out of the weighted sum. The mirror
// case is the stage's real amplification probe - whatever the score arithmetic gets wrong shows up
// multiplied by the shape that the sum is cancelling.
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

    // Large window rows against tiny compressed rows, then the reverse: one accumulator holding
    // terms six orders of magnitude apart.
    try sparseCaseScaled(gpa, device, queue, &pipelines, .{ .name = "mix_window_dominates", .heads = 1, .rows = 1, .position = 100, .ratio = 2, .compressed_rows = 100, .topk = 64 }, &rng, 1.0, 14, -12, 0, 0, KeyPattern.random);
    try sparseCaseScaled(gpa, device, queue, &pipelines, .{ .name = "mix_compressed_dominates", .heads = 1, .rows = 1, .position = 100, .ratio = 2, .compressed_rows = 100, .topk = 64 }, &rng, 1.0, -12, 14, 0, 0, KeyPattern.random);
    try sparseCaseScaled(gpa, device, queue, &pipelines, .{ .name = "mix_balanced_magnitudes", .heads = 1, .rows = 1, .position = 100, .ratio = 2, .compressed_rows = 100, .topk = 64 }, &rng, 1.0, 6, 6, 0, 0, KeyPattern.cancelling);

    // The stage's real amplification probe: 128 window rows that carry a large alternating shape
    // which cancels out of both the score (a uniform query against a sign-alternating shape) and
    // the weighted sum, so every accumulator term is large and their total is small. Whatever the
    // arithmetic gets wrong arrives multiplied by that shape.
    for ([_]i32{ 8, 16, 23 }) |shape_exponent| {
        var name_buf: [24]u8 = undefined;
        const label = std.fmt.bufPrint(&name_buf, "mirror_shape_e{d}", .{shape_exponent}) catch unreachable;
        const spec = SparseSpec{ .name = label, .heads = 1, .rows = 1, .position = 127, .ratio = 0, .compressed_rows = 0, .topk = 0 };
        const q = try gpa.alloc(u16, width);
        defer gpa.free(q);
        // q dot shape == 0 exactly: a uniform query against a shape that alternates sign per
        // column, so the large part drops out of both scores and the accumulator alike.
        for (q) |*cell| cell.* = math.b(1.0);
        const window_rows = try gpa.alloc([]const f32, 128);
        defer gpa.free(window_rows);
        const shape = std.math.pow(f32, 2, @floatFromInt(shape_exponent));
        const shared = try gpa.alloc(f32, width);
        defer gpa.free(shared);
        // One 64th of the cancelling shape: near the FP8 window grid's floor, so the accumulator's
        // partial sums run 2^16 times the sum they end at. Below that the code grid cancels the
        // small part away and FP32 needs no help to be exact.
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
