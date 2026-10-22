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

/// A prompt chunk's affine 4-bit projection (every prompt row) with int8 activations.
/// False: the chunk holds an exceptional activation or an overflowed scale, and is recomputed in the default precision.
pub fn project(e: anytype, x: u64, q: cuda.qlinear.Affine4, out: u64, rows: usize) !bool {
    var exceptional: u32 = 0;
    try e.int8.kernels.prompt(e.stream, x, q, out, rows, e.int8.ws, e.int8.flag.?.ptr, &exceptional);
    return exceptional == 0;
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
