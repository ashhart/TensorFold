//! Borrowed NVFP4 table rows: decode requested mapped rows to BF16 without expanding the table.
const std = @import("std");
const Tensor = @import("safetensors.zig").Tensor;
const quant = @import("quant/quant.zig");

pub const Error = error{ BadNvfp4Table, BadNvfp4Rows, AliasedNvfp4Rows };

pub const Table = struct {
    codes: []const u8,
    scales: []const u8,
    rows: usize,
    width: usize,
    global: f32,

    /// Borrow exact row-major U8 codes and E4M3 scales per 16 values; the caller retains the maps.
    pub fn init(weight: Tensor, scales: Tensor, global: f32) Error!Table {
        if (weight.rank != 2 or weight.dtype != .u8 or weight.dim(0) == 0 or weight.dim(1) == 0)
            return error.BadNvfp4Table;
        const width = std.math.mul(usize, weight.dim(1), 2) catch return error.BadNvfp4Table;
        const size = std.math.mul(usize, weight.dim(0), weight.dim(1)) catch return error.BadNvfp4Table;
        if (width % 16 != 0 or weight.bytes.len != size or !std.math.isFinite(global)) return error.BadNvfp4Table;
        if (!scales.is(.f8_e4m3, &.{ weight.dim(0), width / 16 }) or scales.bytes.len != size / 8)
            return error.BadNvfp4Table;
        return .{ .codes = weight.bytes, .scales = scales.bytes, .rows = weight.dim(0), .width = width, .global = global };
    }

    /// Token-major requested rows, with FP32 products rounded once to BF16; malformed requests write nothing.
    pub fn gather(t: Table, ids: []const usize, out: []u16) Error!void {
        const count = std.math.mul(usize, ids.len, t.width) catch return error.BadNvfp4Rows;
        if (out.len != count) return error.BadNvfp4Rows;
        for (ids) |id| if (id >= t.rows) return error.BadNvfp4Rows;
        const bytes = std.mem.sliceAsBytes(out);
        if (overlaps(bytes, t.codes) or overlaps(bytes, t.scales) or overlaps(bytes, std.mem.sliceAsBytes(ids)))
            return error.AliasedNvfp4Rows;
        for (ids, 0..) |id, row| {
            for (0..t.width) |col| {
                const code = t.codes[id * (t.width / 2) + col / 2];
                const nib = if (col % 2 == 0) code & 15 else code >> 4;
                const a: f32 = @floatCast(quant.nvfp4.e2m1(nib));
                const b: f32 = @floatCast(quant.fp8.e4m3(t.scales[id * (t.width / 16) + col / 16]));
                const product: f32 = a * b;
                out[row * t.width + col] = quant.convert.bf16(product * t.global);
            }
        }
    }
};

fn overlaps(a: []const u8, b: []const u8) bool {
    if (a.len == 0 or b.len == 0) return false;
    const x = @intFromPtr(a.ptr);
    const y = @intFromPtr(b.ptr);
    return if (x <= y) y - x < a.len else x - y < b.len;
}

test {
    _ = @import("nvfp4_rows_test.zig");
}
