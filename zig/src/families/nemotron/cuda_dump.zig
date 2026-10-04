//! Raw dumps of an eager forward (each block norm's h, y, xs, then logits and tokens) named as the oracle names them.

const std = @import("std");
const kern = @import("cuda_kernels.zig");

pub const Dump = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    i: usize = 0,

    fn write(d: *Dump, ops: kern.Ops, name: []const u8, ptr: u64, bytes: usize) !void {
        const host = try d.gpa.alloc(u8, bytes);
        defer d.gpa.free(host);
        try ops.download(host, ptr);
        try ops.s.synchronize();
        var buf: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/{s}.bin", .{ d.dir, name });
        try std.Io.Dir.cwd().writeFile(d.io, .{ .sub_path = path, .data = host });
    }

    pub fn norm(d: *Dump, ops: kern.Ops, h: u64, y: u64, xs: u64, rows: usize, hidden: usize) !void {
        var buf: [16]u8 = undefined;
        const n = rows * hidden;
        try d.write(ops, try std.fmt.bufPrint(&buf, "{d:0>3}_h", .{d.i}), h, n * 2);
        try d.write(ops, try std.fmt.bufPrint(&buf, "{d:0>3}_y", .{d.i}), y, n * 2);
        try d.write(ops, try std.fmt.bufPrint(&buf, "{d:0>3}_xs", .{d.i}), xs, n / 64 * 4);
        d.i += 1;
    }

    pub fn tail(d: *Dump, ops: kern.Ops, logits: u64, sampled: u64, rows: usize, vocab: usize) !void {
        try d.write(ops, "logits", logits, rows * vocab * 2);
        try d.write(ops, "sampled", sampled, rows * 4);
    }
};
