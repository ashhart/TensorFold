//! Opt-in scratch receipts from the CUDA test runner; no inference module imports this owner.
const std = @import("std");
const options = @import("cuda_dump_options");

pub const DType = enum { bf16, f16, f32, u32, raw };
var records: usize = 0;

pub fn active() bool {
    if (comptime options.enabled) {
        const value = std.c.getenv("TF_CUDA_DUMP") orelse return false;
        return std.mem.eql(u8, std.mem.span(value), "1");
    }
    return false;
}

fn line(a: std.mem.Allocator, value: anytype) !void {
    const text = try std.json.Stringify.valueAlloc(a, value, .{});
    defer a.free(text);
    std.debug.print("DUMP {s}\n", .{text});
}

pub fn begin(a: std.mem.Allocator, suite: []const u8) !void {
    if (comptime options.enabled) {
        if (!active()) return;
        records = 0;
        try line(a, .{ .version = 1, .event = "begin", .suite = suite });
    }
}

pub fn end(a: std.mem.Allocator, suite: []const u8, ok: bool) !void {
    if (comptime options.enabled) {
        if (!active()) return;
        try line(a, .{ .version = 1, .event = "end", .suite = suite, .ok = ok, .records = records });
    }
}

/// Hash only the initialized, synchronized payload in its declared layout, without conversion or guards.
pub fn output(a: std.mem.Allocator, suite: []const u8, case: []const u8, tensor: []const u8, shape: []const usize, dtype: DType, meta: anytype, bytes: []const u8) !void {
    if (comptime options.enabled) {
        if (!active()) return;
        if (@import("builtin").cpu.arch.endian() != .little) return error.DumpEndian;
        const size: usize = switch (dtype) {
            .bf16, .f16 => 2,
            .f32, .u32 => 4,
            .raw => 1,
        };
        var expected: usize = size;
        for (shape) |dim| expected = try std.math.mul(usize, expected, dim);
        if (expected != bytes.len) return error.DumpSize;
        if (dtype != .raw and dtype != .u32) {
            var at: usize = 0;
            while (at < bytes.len) : (at += size) {
                const nonfinite = switch (dtype) {
                    .f32 => (std.mem.readInt(u32, bytes[at..][0..4], .little) & 0x7f800000) == 0x7f800000,
                    .bf16 => (std.mem.readInt(u16, bytes[at..][0..2], .little) & 0x7f80) == 0x7f80,
                    .f16 => (std.mem.readInt(u16, bytes[at..][0..2], .little) & 0x7c00) == 0x7c00,
                    .u32, .raw => false,
                };
                if (nonfinite) return error.DumpNonfinite;
            }
        }
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
        const hex = std.fmt.bytesToHex(hash, .lower);
        try line(a, .{ .version = 1, .event = "tensor", .suite = suite, .case = case, .tensor = tensor, .shape = shape, .dtype = @tagName(dtype), .endian = "little", .bytes = bytes.len, .sha256 = @as([]const u8, &hex), .meta = meta });
        records += 1;
    }
}
