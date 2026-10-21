//! --prompt-precision int8: the engine's switch, cuda.w4a8's kernels and a prompt chunk's quantized inputs.
const cuda = @import("cuda");
const max_rows = @import("cuda_round.zig").max_rows;

pub const State = struct {
    on: bool = false,
    kernels: cuda.w4a8.Kernels = undefined,
    buf: ?cuda.DeviceBuffer = null, // the quantized inputs, taken only when int8 is on
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
        }
        s.on = true;
    }

    pub fn deinit(s: *State) void {
        if (s.buf) |*b| b.free();
    }
};

/// A prompt chunk's affine 4-bit projection (rows past 16) with int8 activations.
pub fn project(e: anytype, x: u64, q: cuda.qlinear.Affine4, out: u64, rows: usize) !void {
    return e.int8.kernels.prompt(e.stream, x, q, out, rows, e.int8.ws);
}
