//! One Qwen3 dense model on one HIP device for prompt-only heads: weights, a stream and scratch sized to a window.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");
const forward = @import("forward.zig");

const Allocator = std.mem.Allocator;

/// Inputs one pass pools at most (the embedding lane's default batch).
pub const max_inputs: usize = 128;
/// The least window the budget may shrink a load to.
pub const least_window: usize = 2048;

/// A pass's arguments before any upload: tokens = trunk ++ branches, each branch past the trunk and ending at `ends`.
pub fn checkBatch(tokens: []const u32, trunk: u32, ends: []const u32, out_len: usize, window: usize, vocab: usize, hidden: usize) error{InvalidBatch}!void {
    if (tokens.len == 0 or tokens.len > window or ends.len == 0 or ends.len > max_inputs or ends[0] <= trunk or
        ends[ends.len - 1] != tokens.len or out_len != ends.len * hidden) return error.InvalidBatch;
    for (tokens) |t| if (t >= vocab) return error.InvalidBatch;
}

pub const Engine = struct {
    gpa: Allocator,
    driver: hip.Runtime,
    ctx: hip.Context,
    lib: hip.Launcher,
    stream: hip.Stream,
    m: model.Model,
    /// Device bytes the weights took.
    weights: usize = 0,
    /// Tokens a pass takes at most, over all its inputs; set by `size`.
    window: usize = 0,
    arena: hip.Arena = undefined,
    ids_host: hip.HostBuffer = undefined,
    ids_dev: hip.DeviceBuffer = undefined,
    rows_dev: hip.DeviceBuffer = undefined,
    rows_host: hip.HostBuffer = undefined,
    sized: bool = false,

    /// The model on device `index`, its scratch not yet sized.
    pub fn load(gpa: Allocator, io: std.Io, dir: []const u8, index: c_int, choices: hip.Launcher.Choices) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = .{ .gpa = gpa, .driver = undefined, .ctx = undefined, .lib = undefined, .stream = undefined, .m = undefined };
        e.driver = try hip.Runtime.open();
        errdefer e.driver.close();
        e.ctx = try hip.Context.init(&e.driver, index);
        errdefer e.ctx.deinit();
        e.lib = try hip.Launcher.load(&e.driver, choices, hip.kernels.images);
        errdefer e.lib.unload();
        e.stream = try hip.Stream.init(&e.driver);
        errdefer e.stream.deinit();
        const before = hip.usage(false).device;
        // the WMMA tile fuses silu x up into the gate/up product, over the two weights interleaved in one
        e.m = try model.Model.load(gpa, io, &e.driver, dir, if (e.lib.dense.serve == .wmma12) .interleaved else .plain);
        e.weights = hip.usage(false).device - before;
        return e;
    }

    pub fn ops(e: *Engine, arena: *hip.Arena) hip.ops.Ops {
        return .{ .l = &e.lib, .bf16 = true, .stream = e.stream.handle, .arena = arena };
    }

    /// The pass's scratch for `window` tokens (one input of them: the longest cache) plus the ids and pooled rows.
    pub fn scratchBytes(e: *Engine, window: usize) !usize {
        var arena = hip.Arena.counting();
        var o = e.ops(&arena);
        o.stream = hip.counting;
        const out: hip.ops.Tensor = .{ .ptr = try arena.take(max_inputs * e.m.spec.hidden * 2), .kind = .bf16 };
        try forward.lastRows(o, &e.m, 0, 0, &[_]u32{@intCast(window)}, out);
        return arena.peak + Engine.fixedBytes(e, window);
    }

    /// The buffers beside the arena: ids for `window` tokens and `max_inputs` pooled BF16 rows.
    fn fixedBytes(e: *const Engine, window: usize) usize {
        return window * 4 + max_inputs * e.m.spec.hidden * 2;
    }

    /// Allocates the scratch for passes of `window` tokens; refuses when the bytes held differ from the count.
    pub fn size(e: *Engine, window: usize) !void {
        const want = try e.scratchBytes(window);
        const arena_bytes = want - e.fixedBytes(window);
        const before = hip.usage(false).device;
        e.arena = try hip.Arena.init(&e.driver, arena_bytes);
        errdefer e.arena.deinit();
        e.ids_dev = try hip.DeviceBuffer.alloc(&e.driver, window * 4);
        errdefer e.ids_dev.free();
        e.rows_dev = try hip.DeviceBuffer.alloc(&e.driver, max_inputs * e.m.spec.hidden * 2);
        errdefer e.rows_dev.free();
        e.ids_host = try hip.HostBuffer.alloc(&e.driver, window * 4);
        errdefer e.ids_host.free();
        e.rows_host = try hip.HostBuffer.alloc(&e.driver, max_inputs * e.m.spec.hidden * 2);
        errdefer e.rows_host.free();
        const held = hip.usage(false).device - before;
        if (held != want) {
            std.log.err("qwen3: sizing held {d} device bytes, the count says {d}", .{ held, want });
            return error.SizeMismatch;
        }
        e.window = window;
        e.sized = true;
    }

    /// Each branch's final-normed last row as fp32 into `out` (ends.len rows of hidden); see `checkBatch`.
    pub fn run(e: *Engine, tokens: []const u32, trunk: u32, ends: []const u32, out: []f32) !void {
        const hidden = e.m.spec.hidden;
        if (!e.sized) return error.InvalidBatch;
        try checkBatch(tokens, trunk, ends, out.len, e.window, e.m.spec.vocab, hidden);
        @memcpy(e.ids_host.slice(u32)[0..tokens.len], tokens);
        try e.ids_dev.uploadAsync(0, e.ids_host.view(0, tokens.len * 4), e.stream);
        const mark = e.arena.mark();
        defer e.arena.release(mark);
        const rows: hip.ops.Tensor = .{ .ptr = e.rows_dev.base(), .kind = .bf16 };
        try forward.lastRows(e.ops(&e.arena), &e.m, e.ids_dev.base(), trunk, ends, rows);
        try e.rows_dev.downloadAsync(0, e.rows_host.view(0, ends.len * hidden * 2), e.stream);
        try e.stream.synchronize();
        for (out, e.rows_host.slice(u16)[0..out.len]) |*o, b| o.* = @bitCast(@as(u32, b) << 16);
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        if (e.sized) {
            e.rows_host.free();
            e.ids_host.free();
            e.rows_dev.free();
            e.ids_dev.free();
            e.arena.deinit();
        }
        e.m.deinit();
        e.stream.deinit();
        e.lib.unload();
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }
};

test "a pass's arguments are checked before any upload: the trunk, the ends, the window, the vocabulary and the rows" {
    const tokens = [_]u32{ 1, 2, 3, 4 };
    try checkBatch(&tokens, 0, &.{ 2, 4 }, 6, 8, 10, 3);
    try checkBatch(&tokens, 1, &.{ 2, 4 }, 6, 4, 10, 3);
    const bad = error.InvalidBatch;
    try std.testing.expectError(bad, checkBatch(&tokens, 2, &.{ 2, 4 }, 6, 8, 10, 3)); // the first branch is empty
    try std.testing.expectError(bad, checkBatch(&tokens, 3, &.{ 2, 4 }, 6, 8, 10, 3)); // the trunk passes it
    try std.testing.expectError(bad, checkBatch(&tokens, 0, &.{}, 0, 8, 10, 3));
    try std.testing.expectError(bad, checkBatch(&.{}, 0, &.{0}, 3, 8, 10, 3));
    try std.testing.expectError(bad, checkBatch(&tokens, 0, &.{ 2, 3 }, 6, 8, 10, 3)); // the last end is not the count
    try std.testing.expectError(bad, checkBatch(&tokens, 0, &.{ 2, 4 }, 5, 8, 10, 3)); // the rows do not fit `out`
    try std.testing.expectError(bad, checkBatch(&tokens, 0, &.{ 2, 4 }, 6, 3, 10, 3)); // past the window
    try std.testing.expectError(bad, checkBatch(&tokens, 0, &.{ 2, 4 }, 6, 8, 4, 3)); // a token outside the vocabulary
    var many: [max_inputs + 1]u32 = undefined;
    for (&many, 1..) |*e, i| e.* = @intCast(i);
    const ids: [max_inputs + 1]u32 = @splat(0);
    try checkBatch(ids[0..max_inputs], 0, many[0..max_inputs], max_inputs, 1024, 10, 1);
    try std.testing.expectError(bad, checkBatch(&ids, 0, &many, max_inputs + 1, 1024, 10, 1));
}
