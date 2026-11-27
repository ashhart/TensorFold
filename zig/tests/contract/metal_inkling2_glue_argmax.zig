//! Argmax output checks for the Inkling 2 Mini glue contract.
const std = @import("std");
const p = @import("tensorfold").inkling2_metal_host.glue.plan;
const kv_conv = @import("metal_inkling2_glue_kv_conv.zig");

pub fn check(bytes: []const u8, values: []const f32, rows: usize, n: usize) ?[]const u8 {
    const got = std.mem.bytesAsSlice(u32, bytes);
    for (0..rows) |row| {
        if (got[row] != p.argmax(values[row * n ..][0..n])) return "argmax token differs";
    }
    return null;
}

/// Prepares argmax tie/nonfinite rows using the contract's existing float and upload helpers.
pub fn prepare(self: anytype, rows: usize, n: usize, nonfinite: bool, rng: std.Random) !void {
    self.f[0] = try kv_conv.fillFloats(self.arena.allocator(), rows * n, rng);
    if (n >= 4) {
        self.f[0][1] = 10.0;
        self.f[0][3] = 10.0;
    }
    if (nonfinite) {
        self.f[0][n - 1] = std.math.nan(f32);
        if (rows > 1) self.f[0][n] = std.math.nan(f32);
    }
    self.refs[0] = try self.rig.input(std.mem.sliceAsBytes(self.f[0]));
}
