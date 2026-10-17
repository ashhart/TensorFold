//! Lossless byte-window4 scale planes, with sorted u24-position/u8-value exceptions.

const std = @import("std");

pub const codec = "byte-window4-fixed-stream-u24-exceptions/1";
pub const Error = error{ BadGeometry, BadFixedStream, BadExceptions, BadSlice, OverlappingOutput };

/// A validated, borrowed checkpoint plane; callers keep both streams immutable until decoding finishes.
pub const Plane = struct {
    rows: usize,
    columns: usize,
    fixed: []const u8,
    exceptions: []const u8,

    /// Slabs hold 16 row bases followed by those rows' low-even/high-odd packed offsets.
    pub fn init(rows: usize, columns: usize, fixed: []const u8, exceptions: []const u8) Error!Plane {
        if (rows == 0 or rows % 16 != 0 or columns == 0 or columns % 2 != 0) return error.BadGeometry;
        if (columns > 1 << 24 or rows > (1 << 24) / columns) return error.BadGeometry;
        const stride = 16 + 8 * columns;
        if (fixed.len != rows / 16 * stride) return error.BadFixedStream;
        for (0..rows / 16) |slab| {
            for (fixed[slab * stride ..][0..16]) |base| if (base > 240) return error.BadFixedStream;
        }
        if (exceptions.len % 4 != 0) return error.BadExceptions;
        var previous: ?u32 = null;
        var offset: usize = 0;
        while (offset < exceptions.len) : (offset += 4) {
            const position = word(exceptions, offset) & 0xffffff;
            if (position >= rows * columns) return error.BadExceptions;
            if (previous) |p| if (position <= p) return error.BadExceptions;
            previous = position;
        }
        return .{ .rows = rows, .columns = columns, .fixed = fixed, .exceptions = exceptions };
    }

    /// Decode a rectangular TP slice into exactly sized row-major bytes, without allocating or rounding.
    pub fn decode(self: Plane, out: []u8, row: usize, rows: usize, column: usize, columns: usize) Error!void {
        if (row > self.rows or rows > self.rows - row or column > self.columns or columns > self.columns - column)
            return error.BadSlice;
        if (out.len != rows * columns) return error.BadSlice;
        if (overlaps(out, self.fixed) or overlaps(out, self.exceptions)) return error.OverlappingOutput;
        if (rows == 0 or columns == 0) return;
        const stride = 16 + 8 * self.columns;
        for (0..rows) |r| {
            const source_row = row + r;
            const slab = source_row / 16 * stride;
            const base = self.fixed[slab + source_row % 16];
            const codes = slab + 16 + source_row % 16 * (self.columns / 2);
            for (0..columns) |c| {
                const source_column = column + c;
                const codes_byte = self.fixed[codes + source_column / 2];
                const shift: u3 = if (source_column % 2 == 0) 0 else 4;
                out[r * columns + c] = base + ((codes_byte >> shift) & 15);
            }
            var i = self.lowerBound(source_row * self.columns + column);
            const end = source_row * self.columns + column + columns;
            while (i < self.exceptions.len) : (i += 4) {
                const entry = word(self.exceptions, i);
                const position = entry & 0xffffff;
                if (position >= end) break;
                out[r * columns + position - (source_row * self.columns + column)] = @intCast(entry >> 24);
            }
        }
    }

    fn lowerBound(self: Plane, position: usize) usize {
        var lo: usize = 0;
        var hi = self.exceptions.len / 4;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (word(self.exceptions, mid * 4) & 0xffffff < position) lo = mid + 1 else hi = mid;
        }
        return lo * 4;
    }
};

fn word(bytes: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, bytes[offset..][0..4], .little);
}

fn overlaps(out: []u8, source: []const u8) bool {
    if (out.len == 0 or source.len == 0) return false;
    const a = @intFromPtr(out.ptr);
    const b = @intFromPtr(source.ptr);
    return if (a <= b) b - a < out.len else a - b < source.len;
}

test {
    _ = @import("nvfp4_csf_test.zig");
}
