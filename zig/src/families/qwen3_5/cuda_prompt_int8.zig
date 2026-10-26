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
    fallbacks: u64 = 0, // prompt chunks that took bf16 instead: a non-finite input or a scale overflow in a group

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

/// An int8 prompt chunk of `rows` rows returned to bf16 (a non-finite input or a scale overflow): counted and logged.
pub fn fellBack(e: anytype, rows: usize) void {
    e.int8.fallbacks += 1;
    std.log.info("int8 prompt: a chunk of {d} rows took bf16 (a non-finite input or a group's scale overflow); {d} so far", .{ rows, e.int8.fallbacks });
}

test "a chunk's return to bf16 is counted" {
    var e: struct { int8: State } = .{ .int8 = .{} };
    fellBack(&e, max_rows);
    fellBack(&e, 16);
    try std.testing.expectEqual(@as(u64, 2), e.int8.fallbacks);
}
