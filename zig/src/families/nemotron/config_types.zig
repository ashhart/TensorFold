//! Config refusal text and affine matrix widths shared by the Nemotron backends.
const std = @import("std");

/// Why a check refused the checkpoint, kept for the caller's log line (tests read it instead).
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    /// Keep the reason, cut short if it is long.
    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

/// One matrix's MLX affine width: `bits` per weight, a scale and a bias for each `group` of them.
pub const Width = struct {
    bits: u8 = 4,
    group: u16 = 64,

    /// The width the 4-bit kernels read (lane tiles, the expert kernels, the embedding); others take the row kernels.
    pub fn four(w: Width) bool {
        return w.bits == 4 and w.group == 64;
    }

    /// A width the core's affine kernels unpack: 2 to 8 bits but 7, in groups of 32, 64 or 128.
    pub fn readable(w: Width) bool {
        return w.bits >= 2 and w.bits <= 8 and w.bits != 7 and (w.group == 32 or w.group == 64 or w.group == 128);
    }
};
