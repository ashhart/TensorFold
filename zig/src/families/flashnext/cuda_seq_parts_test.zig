//! A lane stream's cache claims the carveout's planes, as newSeqOn does: the plan counts them into the room.
const std = @import("std");
const cuda = @import("cuda");
const prompt = @import("cuda_prompt.zig");
const carve_mod = @import("cuda_carve.zig");
const seq_parts = @import("cuda_seq_parts.zig");

// Virtual device addresses; the callbacks never open CUDA or touch device memory (restore_test's Device).
var next: u64 = 0x100000;
var fills: usize = 0;

fn alloc(out: *u64, n: usize) callconv(.c) c_int {
    out.* = next;
    next += n + 256;
    return 0;
}
fn free(_: u64) callconv(.c) c_int {
    return 0;
}
fn fillAsync(_: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) c_int {
    fills += 1;
    return 0;
}
fn fillSync(_: u64, _: u8, _: usize) callconv(.c) c_int {
    fills += 1;
    return 0;
}
fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
    return 0;
}

fn driver() cuda.Driver {
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = alloc;
    d.api.cuMemFree_v2 = free;
    d.api.cuMemsetD8Async = fillAsync;
    d.api.cuMemsetD8_v2 = fillSync;
    d.api.cuStreamSynchronize = sync;
    return d;
}

test "a lane stream's cache from a reused parts block takes the carveout's first planes" {
    const a = std.testing.allocator;
    var d = driver();
    var stream: cuda.Stream = .{ .d = &d, .handle = null };
    var engine: prompt.Engine = undefined;
    engine.gpa = a;
    engine.driver = &d;
    engine.stream = &stream;
    engine.eos = 7;
    engine.chunk = 16;
    engine.round_graph = .{};
    engine.heads = .full;
    engine.carve = null;

    // A whole free carveout of 256 MiB, the load plan fitted to this engine's window (cuda_lanes_load's shape).
    const host_buf = try a.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), 256 << 20);
    defer a.free(host_buf);
    var carve: cuda.Carveout = .{
        .d = &d,
        .fd = -1,
        .handle = 0,
        .host = host_buf,
        .dev = 0x7000000000,
        .flags = 0,
        .spans = .{ .size = 256 << 20 },
    };
    const window = 2048;
    const plan = carve_mod.Plan.of(&carve, window, engine.heads);
    try std.testing.expect(plan.n[0] > 0); // the plan has planes to hand out
    engine.carve = plan;

    // The lane opener's path: one reused parts block, then newSeqIn (cuda_lanes.zig open()'s branch).
    var block = try cuda.DeviceBuffer.alloc(&d, seq_parts.seqPartsBytes(engine.heads));
    defer block.free();
    var seq = try seq_parts.newSeqIn(&engine, 512, block);
    defer seq.deinit(a);

    try std.testing.expect(seq.carved.carve != null); // the cache holds carveout planes
    try std.testing.expect(seq.carved.of(.key).len == plan.n[0]);
    try std.testing.expect(seq.carved.of(.value).len == plan.n[1]);
    try std.testing.expect(seq.carved.of(.index).len == plan.n[2]);
    try std.testing.expect(seq.carved.of(.pooled).len == plan.n[3]);
    // the first of kc is the carveout's first key plane, borrowed (not a fresh allocation)
    try std.testing.expect(seq.kc[0].borrowed and seq.kc[0].ptr == carve.dev);
    try std.testing.expect(carve.freeBytes() < carve.bytes()); // the planes are out of the carveout

    // The second lane stream finds the carveout held: it takes nothing, its planes are its own.
    var seq2 = try seq_parts.newSeqIn(&engine, 512, block);
    defer seq2.deinit(a);
    try std.testing.expect(seq2.carved.carve == null);
    try std.testing.expect(!seq2.kc[0].borrowed);
}

test "newSeqIn with no carve plan: no carved planes, every buffer its own" {
    const a = std.testing.allocator;
    var d = driver();
    var stream: cuda.Stream = .{ .d = &d, .handle = null };
    var engine: prompt.Engine = undefined;
    engine.gpa = a;
    engine.driver = &d;
    engine.stream = &stream;
    engine.eos = 7;
    engine.chunk = 16;
    engine.round_graph = .{};
    engine.heads = .full;
    engine.carve = null;

    var block = try cuda.DeviceBuffer.alloc(&d, seq_parts.seqPartsBytes(engine.heads));
    defer block.free();
    var seq = try seq_parts.newSeqIn(&engine, 512, block);
    defer seq.deinit(a);
    try std.testing.expect(seq.carved.carve == null);
    try std.testing.expect(!seq.kc[0].borrowed);
}

test "carveout.open refuses tp 2 before it touches the context" {
    const env = @import("sys").env;
    _ = env.setenv("TF_CUDA_CARVEOUT", "1", 1);
    defer _ = env.unsetenv("TF_CUDA_CARVEOUT");
    _ = env.setenv("TF_CUDA_CARVEOUT_MIB", "64", 1);
    defer _ = env.unsetenv("TF_CUDA_CARVEOUT_MIB");
    // A context pointer the refusal path never dereferences: open() checks tp first.
    var ctx: cuda.Context = undefined;
    try std.testing.expectError(error.FlashNextCarveoutUnsupported, carve_mod.open(std.testing.allocator, &ctx, 2));
}
