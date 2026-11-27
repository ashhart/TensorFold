//! Argmax output checks for the Inkling 2 Mini glue contract.
const std = @import("std");
const p = @import("tensorfold").inkling2_metal_host.glue.plan;

pub fn check(bytes: []const u8, values: []const f32, rows: usize, n: usize) ?[]const u8 {
    const got = std.mem.bytesAsSlice(u32, bytes);
    for (0..rows) |row| {
        if (got[row] != p.argmax(values[row * n ..][0..n])) return "argmax token differs";
    }
    return null;
}
