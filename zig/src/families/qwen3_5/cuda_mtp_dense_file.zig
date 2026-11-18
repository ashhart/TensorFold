//! Discovery and host-side norm conversion for the Loop's dense MTP checkpoint.
const std = @import("std");
const core = @import("core");

pub const Location = enum { root, child, direct };

fn hasShards(io: std.Io, path: []const u8) !bool {
    var d = std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true }) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => return false,
        else => return err,
    };
    defer d.close(io);
    var it = d.iterate();
    while (try it.next(io)) |entry| {
        if ((entry.kind == .file or entry.kind == .sym_link) and std.mem.endsWith(u8, entry.name, ".safetensors")) return true;
    }
    return false;
}

/// Root file wins; then the mtp child, or the mtp directory itself.
pub fn discover(io: std.Io, dir: []const u8) !Location {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = try std.fmt.bufPrint(&buf, "{s}/mtp.safetensors", .{dir});
    if (std.Io.Dir.cwd().access(io, root, .{})) |_| return .root else |err| switch (err) {
        error.FileNotFound => {},
        else => return err,
    }
    const child = try std.fmt.bufPrint(&buf, "{s}/mtp", .{dir});
    if (try hasShards(io, child)) return .child;
    if (std.mem.eql(u8, std.fs.path.basename(dir), "mtp") and try hasShards(io, dir)) return .direct;
    return error.NoLoopMtpFile;
}

/// Only the ordinary RMSNorm kernels can add one to a BF16 weight during execution.
pub fn rawNorm(t: core.checkpoint.Tensor, n: usize, supports_raw: bool) !bool {
    if (t.is(.bf16, &.{n})) return supports_raw;
    if (t.is(.f32, &.{n})) return false;
    return error.UnexpectedTensor;
}

/// Pre-shift an HF norm in FP32, rounding once to BF16 for kernels that need pre-shifted weights.
pub fn shift(t: core.checkpoint.Tensor, out: []u16) !void {
    if (!t.is(.bf16, &.{out.len}) and !t.is(.f32, &.{out.len})) return error.UnexpectedTensor;
    for (out, 0..) |*v, i| {
        const x: f32 = if (t.dtype == .bf16)
            @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16)
        else
            @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little));
        v.* = core.bf16_round.bits(1.0 + x);
    }
}

pub fn projection(t: core.checkpoint.Tensor, n: usize, k: usize) !void {
    if (!t.is(.bf16, &.{ n, k })) return error.UnexpectedTensor;
}
