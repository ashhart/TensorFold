//! FP64 references for the EXLR8 Metal projections, with first-order fp32 bounds that follow the kernels' stages.

const std = @import("std");
const ex = @import("exlr8");
const codec = ex.codec;

const u = 0x1p-24;
/// Rotation (7 adds), 128 fused terms, 48 block FMAs, the 4-lane sum, the output transform: 192 roundings, with room.
pub const gamma = 256 * u;

pub fn bf16Value(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
pub fn bf16Of(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b +% 0x7fff +% ((b >> 16) & 1)) >> 16);
}

/// One job's exact W (k_in, 128) and the |T| sums per input block and column, for the bound.
pub const Exact = struct {
    w: []f64,
    tabs: []f64,
    blocks: usize,

    pub fn init(gpa: std.mem.Allocator, codes: []const u8, blocks: usize, suh: []const f32, svh: []const f32) !Exact {
        const w = try gpa.alloc(f64, blocks * 128 * 128);
        errdefer gpa.free(w);
        try codec.reconstruct(codes, blocks * 128, 128, suh, svh, w);
        const tabs = try gpa.alloc(f64, blocks * 128);
        @memset(tabs, 0);
        for (0..blocks * 128) |k| for (0..128) |c| {
            tabs[(k / 128) * 128 + c] += @abs(codec.e4m3Value(codes[k * 128 + c]));
        };
        return .{ .w = w, .tabs = tabs, .blocks = blocks };
    }
    pub fn deinit(x: Exact, gpa: std.mem.Allocator) void {
        gpa.free(x.w);
        gpa.free(x.tabs);
    }

    /// y = x W for one row, and sum_m sum_b |s_b| A_b |T|_bm: the pipeline's magnitude before the output transform.
    pub fn row(x: Exact, in: []const f64, in_abs: []const f64, scales: []const f32, y: *[128]f64) f64 {
        @memset(y, 0);
        for (in, 0..) |v, k| for (0..128) |c| {
            y[c] += v * x.w[k * 128 + c];
        };
        var mag: f64 = 0;
        for (0..x.blocks) |b| {
            var a: f64 = 0;
            for (in_abs[b * 128 ..][0..128]) |v| a += v;
            var t: f64 = 0;
            for (x.tabs[b * 128 ..][0..128]) |v| t += v;
            mag += @abs(@as(f64, scales[b])) * a * t;
        }
        return mag;
    }
};

/// The codes of a run of tiles laid out as consecutive 64-tile input blocks (gate/up halves, down halves' block b).
pub fn blockCodes(book: *const codec.Codebook, k: u8, runs: []const []const u8, out: []u8) !void {
    for (runs, 0..) |run, b| try codec.decodeRun(book, k, .out_major, run, 128, 128, out[b * 128 * 128 ..][0 .. 128 * 128]);
}

/// Checks fp32 outputs against FP64 within gamma times the magnitude, plus a rounding of the output itself.
pub const Check = struct {
    values: usize = 0,
    max_error: f64 = 0,
    max_fraction: f64 = 0,

    pub fn add(c: *Check, got: f32, want: f64, mag: f64) !void {
        const err = @abs(@as(f64, got) - want);
        const bound = gamma * mag + u * @abs(want) + 0x1p-140;
        if (!std.math.isFinite(got) or !(err <= bound)) {
            std.debug.print("FP64 mismatch: gpu {d} want {d} error {e} bound {e}\n", .{ got, want, err, bound });
            return error.Exlr8Fp64Mismatch;
        }
        c.values += 1;
        c.max_error = @max(c.max_error, err);
        c.max_fraction = @max(c.max_fraction, err / bound);
    }
};
