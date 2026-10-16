//! EXLR8 dense cases for the Metal and CUDA checks: synthetic matrices in every row format, FP64 references, windows.

const std = @import("std");
const ex = @import("exlr8");
const st = ex.safetensors;
const dense = ex.dense;
const DenseJob = ex.job.DenseJob;
const a = std.heap.page_allocator;
pub const rows_max = ex.job.max_rows;
const u = 0x1p-24;

pub const Case = struct { name: []const u8, format: dense.Format, side: dense.Side, out: usize, in: usize, y_f32: bool };

pub const cases = [_]Case{
    .{ .name = "int4 g32 by output f32", .format = .int4, .side = .output, .out = 160, .in = 640, .y_f32 = true },
    .{ .name = "int4 g32 by input f32", .format = .int4, .side = .input, .out = 160, .in = 640, .y_f32 = true },
    .{ .name = "MXFP8 bf16", .format = .mxfp8, .side = .replicated, .out = 160, .in = 640, .y_f32 = false },
    .{ .name = "FP8 128x128 blocks f32", .format = .fp8_block, .side = .output, .out = 160, .in = 640, .y_f32 = true },
    .{ .name = "FP8 a scale a row f32", .format = .fp8_row, .side = .output, .out = 160, .in = 640, .y_f32 = true },
    .{ .name = "bf16 f32", .format = .bf16, .side = .output, .out = 96, .in = 640, .y_f32 = true },
    .{ .name = "fp32 bf16", .format = .fp32, .side = .replicated, .out = 96, .in = 384, .y_f32 = false },
};

/// A matrix's two data tensors packed for the GPU (the second 64-byte aligned) and the matrix over them.
pub const Packed = struct {
    bytes: []u8,
    second: usize,
    m: dense.Matrix,

    pub fn deinit(p: Packed) void {
        a.free(p.bytes);
    }

    /// The whole matrix as one job, outputs at y_col and inputs from x_col.
    pub fn job(p: Packed, out0: usize, outs: usize) !DenseJob {
        const r = p.m.record;
        return .{ .format = try r.kernelFormat(), .out0 = @intCast(out0), .outs = @intCast(outs), .in0 = 0, .ins = @intCast(r.in), .out_total = @intCast(r.out), .in_total = @intCast(r.in), .w = 0, .s = p.second };
    }
};

fn tensor(dtype: st.DType, shape: []const usize, bytes: []const u8) st.Tensor {
    var t: st.Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = bytes };
    @memcpy(t.shape[0..shape.len], shape);
    return t;
}

/// Packs a matrix's data tensors (the release's names and order) into one buffer.
pub fn pack(m: dense.Matrix) !Packed {
    const n0 = m.data[0].bytes.len;
    const second = std.mem.alignForward(usize, n0, 64);
    const n1 = if (dense.parts(m.record.format).len > 1) m.data[1].bytes.len else 0;
    const bytes = try a.alloc(u8, second + n1 + 64);
    @memset(bytes, 0);
    @memcpy(bytes[0..n0], m.data[0].bytes);
    if (n1 > 0) @memcpy(bytes[second..][0..n1], m.data[1].bytes);
    var out = m;
    out.data[0].bytes = bytes[0..n0];
    if (n1 > 0) out.data[1].bytes = bytes[second..][0..n1];
    return .{ .bytes = bytes, .second = second, .m = out };
}

/// A random matrix of the case's format in the release's tensor layout (codes 0x7F/0xFF, NaN, left out).
pub fn synthetic(c: Case, seed: u64) !Packed {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const rec: dense.Record = .{ .format = c.format, .out = c.out, .in = c.in, .k = 0, .side = c.side, .channels = if (c.side == .replicated) 0 else 32, .rotation = 0, .solve = 0 };
    var m: dense.Matrix = .{ .record = rec, .data = undefined, .checksum = null };
    var scratch: [2][]u8 = undefined;
    switch (c.format) {
        .int4 => {
            scratch[0] = try a.alloc(u8, c.out * c.in / 2);
            const s = try a.alloc(f16, c.out * c.in / 32);
            r.bytes(scratch[0]);
            for (s) |*v| v.* = @floatCast(@as(f32, @floatFromInt(r.intRangeLessThan(i32, 1, 30000))) * 0x1p-21 * (if (r.boolean()) @as(f32, 1) else -1));
            scratch[1] = std.mem.sliceAsBytes(s);
            m.data[0] = if (c.side == .input) tensor(.u8, &.{ c.in / 128, c.out, 64 }, scratch[0]) else tensor(.u8, &.{ c.out, c.in / 2 }, scratch[0]);
            m.data[1] = if (c.side == .input) tensor(.f16, &.{ c.in / 128, c.out, 4 }, scratch[1]) else tensor(.f16, &.{ c.out, c.in / 32 }, scratch[1]);
        },
        .mxfp8, .fp8_block, .fp8_row => {
            scratch[0] = try a.alloc(u8, c.out * c.in);
            for (scratch[0]) |*v| v.* = blk: {
                const b = r.int(u8);
                break :blk if (b & 0x7F == 0x7F) b ^ 1 else b;
            };
            m.data[0] = tensor(.u8, &.{ c.out, c.in }, scratch[0]);
            if (c.format == .mxfp8) {
                scratch[1] = try a.alloc(u8, c.out * c.in / 32);
                for (scratch[1]) |*v| v.* = r.intRangeAtMost(u8, 112, 130);
                m.data[1] = tensor(.u8, &.{ c.out, c.in / 32 }, scratch[1]);
            } else {
                const n = if (c.format == .fp8_row) c.out else ((c.out + 127) / 128) * ((c.in + 127) / 128);
                const s = try a.alloc(f32, n);
                for (s) |*v| v.* = @as(f32, @floatFromInt(r.intRangeLessThan(u32, 1 << 23, 1 << 24))) * 0x1p-33;
                scratch[1] = std.mem.sliceAsBytes(s);
                m.data[1] = if (c.format == .fp8_row) tensor(.f32, &.{c.out}, scratch[1]) else tensor(.f32, &.{ (c.out + 127) / 128, (c.in + 127) / 128 }, scratch[1]);
            }
        },
        .bf16 => {
            const v = try a.alloc(u16, c.out * c.in);
            for (v) |*x| x.* = @as(u16, r.intRangeAtMost(u16, 0x3800, 0x3F00)) | (if (r.boolean()) @as(u16, 0x8000) else 0);
            scratch[0] = std.mem.sliceAsBytes(v);
            m.data[0] = tensor(.bf16, &.{ c.out, c.in }, scratch[0]);
        },
        .fp32 => {
            const v = try a.alloc(f32, c.out * c.in);
            for (v) |*x| x.* = (r.float(f32) - 0.5) * 0.25;
            scratch[0] = std.mem.sliceAsBytes(v);
            m.data[0] = tensor(.f32, &.{ c.out, c.in }, scratch[0]);
        },
        .trellis => return error.TrellisHasItsOwnKernels,
    }
    const p = try pack(m);
    a.free(scratch[0]);
    if (dense.parts(c.format).len > 1) a.free(scratch[1]);
    return p;
}

/// Random fp32 input rows, about unit size.
pub fn inputs(n: usize, seed: u64) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const x = try a.alloc(f32, rows_max * n);
    for (x, 0..) |*v, i| v.* = if (i % 41 == 0) 0 else (prng.random().float(f32) - 0.5) * 3;
    return x;
}

/// Checks a projection's rows alone against FP64 of x W (W the reference's fp32 decode, exact in f64).
pub const Check = struct {
    values: usize = 0,
    max_error: f64 = 0,
    max_fraction: f64 = 0,

    /// Rows [0, rows) of `got` (rows x outs, f32 or bf16) against x W for matrix rows [out0, out0 + outs).
    pub fn rows(c: *Check, m: dense.Matrix, x: []const f32, out0: usize, outs: usize, got: []const u8, f32_out: bool) !void {
        const n = m.record.in;
        const w = try a.alloc(f32, n);
        defer a.free(w);
        for (0..outs) |o| {
            try m.rows(out0 + o, 1, w);
            for (0..rows_max) |r| {
                var want: f64 = 0;
                var mag: f64 = 0;
                for (w, x[r * n ..][0..n]) |wi, xi| {
                    want += @as(f64, wi) * xi;
                    mag += @abs(@as(f64, wi) * xi);
                }
                const i = r * outs + o;
                const g: f32 = if (f32_out) @bitCast(std.mem.readInt(u32, got[4 * i ..][0..4], .little)) else @bitCast(@as(u32, std.mem.readInt(u16, got[2 * i ..][0..2], .little)) << 16);
                // a lane's FMA chain of n / 32 terms, five butterfly adds, then the output's own rounding
                const out_u: f64 = if (f32_out) u else 0x1p-8;
                const bound = @as(f64, @floatFromInt(n / 32 + 6)) * u * mag + out_u * @abs(want) + 0x1p-140;
                const err = @abs(@as(f64, g) - want);
                if (!std.math.isFinite(g) or !(err <= bound)) {
                    std.debug.print("FP64 mismatch row {d} out {d}: gpu {d} want {d} error {e} bound {e}\n", .{ r, out0 + o, g, want, err, bound });
                    return error.Exlr8Fp64Mismatch;
                }
                c.values += 1;
                c.max_error = @max(c.max_error, err);
                c.max_fraction = @max(c.max_fraction, err / bound);
            }
        }
    }
};

/// Every row of each window (widths 1-16 in order, `row` bytes a row) equals the row alone; returns the pairs.
pub fn rowsEqual(windows: []const u8, alone: []const u8, row: usize) !usize {
    var at: usize = 0;
    var pairs: usize = 0;
    for (1..rows_max + 1) |m| {
        for (0..m) |r| {
            if (!std.mem.eql(u8, windows[at + r * row ..][0..row], alone[r * row ..][0..row])) {
                std.debug.print("row {d} of a {d}-row window differs from the row alone\n", .{ r, m });
                return error.RowInvariantMismatch;
            }
            pairs += 1;
        }
        at += m * row;
    }
    return pairs;
}

/// The fp32 weights a decode wrote equal the reference decode (the CPU's, equal to the loader's) bit for bit.
pub fn decodeEqual(m: dense.Matrix, out0: usize, outs: usize, got: []const f32) !void {
    const n = m.record.in;
    const w = try a.alloc(f32, n);
    defer a.free(w);
    for (0..outs) |o| {
        try m.rows(out0 + o, 1, w);
        for (w, got[o * n ..][0..n], 0..) |want, g, i| if (@as(u32, @bitCast(want)) != @as(u32, @bitCast(g))) {
            std.debug.print("decode mismatch row {d} input {d}: gpu {x} want {x}\n", .{ out0 + o, i, @as(u32, @bitCast(g)), @as(u32, @bitCast(want)) });
            return error.Exlr8DecodeMismatch;
        };
    }
}

pub fn digest(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

test "every synthetic case packs into the release's shapes" {
    for (cases, 0..) |c, i| {
        const p = try synthetic(c, i);
        defer p.deinit();
        const j = try p.job(0, c.out);
        try std.testing.expectEqual(@as(u32, @intCast(c.in)), j.ins);
        var w: [640]f32 = undefined;
        try p.m.rows(c.out - 1, 1, w[0..c.in]);
    }
}
