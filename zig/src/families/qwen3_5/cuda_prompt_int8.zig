//! --prompt-precision int8: the engine's switch, cuda.w4a8's kernels and a prompt chunk's quantized inputs.
const std = @import("std");
const cuda = @import("cuda");
const max_rows = @import("cuda_round.zig").max_rows;

pub const State = struct {
    on: bool = false,
    kernels: cuda.w4a8.Kernels = undefined,
    buf: ?cuda.DeviceBuffer = null, // the quantized inputs, taken only when int8 is on
    flag: ?cuda.DeviceBuffer = null, // the quantizer's mark for an exceptional group
    ws: u64 = 0,
    fell: u32 = 0, // this prompt chunk's projections that took bf16 (a non-finite input or a group's scale overflow)
    fallbacks: u64 = 0, // prompt chunks with any such projection

    pub fn resolve(s: *State, m: cuda.Module) !void {
        s.kernels = try cuda.w4a8.Kernels.resolve(m);
    }

    /// Takes the quantized inputs' memory once (a chunk's rows at the widest projection input) and turns int8 on.
    pub fn enable(s: *State, e: anytype) !void {
        if (s.buf == null) {
            const g = e.g;
            const k = @max(g.intermediate, g.valueWidth(), g.attnWidth(), g.hidden);
            s.buf = try cuda.DeviceBuffer.alloc(e.d, cuda.w4a8.Kernels.scratch(max_rows, k));
            s.ws = s.buf.?.ptr;
            s.flag = try cuda.DeviceBuffer.alloc(e.d, 4);
        }
        s.on = true;
    }

    pub fn deinit(s: *State) void {
        if (s.flag) |*f| f.free();
        s.flag = null;
        if (s.buf) |*b| b.free();
        s.buf = null;
        s.ws = 0;
        s.on = false;
    }
};

/// A prompt chunk's affine 4-bit projection (every prompt row) with int8 activations; false when it is exceptional.
pub fn project(e: anytype, x: u64, q: cuda.qlinear.Affine4, out: u64, rows: usize) !bool {
    var exceptional: u32 = 0;
    try e.int8.kernels.prompt(e.stream, x, q, out, rows, e.int8.ws, e.int8.flag.?.ptr, &exceptional);
    return exceptional == 0;
}

/// A prompt chunk's affine 4-bit projection in int8, with the group sums a short chunk needs.
pub fn projectRows(e: anytype, q: cuda.qlinear.Affine4, x: u64, xs: u64, out: u64, rows: usize, sums: bool, params: u64, k: usize) !bool {
    if (!try project(e, x, q, out, rows)) return false;
    if (rows <= 16 and sums) try e.run(e.fns.sums, .{ rows, (k / 64 + 63) / 64, 1 }, .{ 64, 1, 1 }, .{ x, xs, @as(c_int, @intCast(k)), params });
    return true;
}

/// The production allocation prefix, isolated so an inert driver can check ownership before startup.
pub fn Allocation(comptime Engine: type) type {
    return struct {
        pub fn create(gpa: std.mem.Allocator) !*Engine {
            const e = try gpa.create(Engine);
            e.* = undefined;
            e.int8 = .{};
            return e;
        }
    };
}

/// A prompt chunk starts: no projection of it has taken bf16 yet.
pub fn chunkStart(e: anytype) void {
    e.int8.fell = 0;
}

/// A projection of the current prompt chunk took bf16 (a non-finite input or a group's scale overflow).
pub fn fellBack(e: anytype) void {
    e.int8.fell += 1;
}

/// A prompt chunk of `rows` rows is through every layer: counted and logged once if any projection took bf16.
pub fn chunkEnd(e: anytype, rows: usize) void {
    if (e.int8.fell == 0) return;
    e.int8.fallbacks += 1;
    std.log.info("int8 prompt: a chunk of {d} rows took bf16 in {d} projections; {d} such chunks so far", .{ rows, e.int8.fell, e.int8.fallbacks });
    e.int8.fell = 0;
}

test "a chunk's returns to bf16 count once" {
    var e: struct { int8: State } = .{ .int8 = .{} };
    chunkStart(&e);
    chunkEnd(&e, max_rows);
    try std.testing.expectEqual(@as(u64, 0), e.int8.fallbacks);
    chunkStart(&e);
    for (0..64) |_| fellBack(&e);
    chunkEnd(&e, max_rows);
    chunkStart(&e);
    fellBack(&e);
    chunkEnd(&e, 16);
    try std.testing.expectEqual(@as(u64, 2), e.int8.fallbacks);
    try std.testing.expectEqual(@as(u32, 0), e.int8.fell);
}
