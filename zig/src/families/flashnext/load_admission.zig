//! Host admission runs before the Flash Next loader allocates or initializes a device.
const std = @import("std");
const config = @import("config.zig");
const index = @import("index.zig");
const widths = @import("widths.zig");

pub const problem = "Flash Next runtime kernels read MLX affine 2- to 8-bit weights in groups of 32, 64 or 128 (n-gram tables in groups that divide a row); the log names the first other one";

pub const stored_problem = "the checkpoint's stored weights contradict the widths its config declares, or a layout one kernel reads at one width differs; the log names the first one";
pub const speed_up_problem = "speed-up mode (two Macs) splits the 6-bit group-32 kernels only; serve this checkpoint on one Mac";
pub const scale_problem = "an n-gram table scale must be one finite, nonzero value; the log names it";

pub fn problemFor(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.UnsupportedFlashAffineKernel, error.UnsupportedAffineBits, error.UnsupportedAffineGroup, error.UnsupportedAffineMode, error.UnsupportedQuantization => problem,
        error.InvalidAffineShape, error.InvalidAffineRank, error.InvalidAffineStorage, error.InvalidAffineWidth, error.InvalidAffineByteCount, error.UnquantizedAffineTensor, error.ExpertGateUpWidths, error.NgramTableWidths, error.LaneRowsNotFours, error.TooManyLaneMembers => stored_problem,
        error.SpeedUpNeeds6Bit32 => speed_up_problem,
        error.NgramTableScale => scale_problem,
        else => null,
    };
}

pub fn check(gpa: std.mem.Allocator, io: std.Io, directory: []const u8) !void {
    var refusal: index.Refusal = .{};
    checkWith(gpa, io, directory, &refusal) catch |err| {
        if (refusal.width) |w| std.log.warn("{s}: {d}-bit group {d} has no Flash Next device kernel", .{ refusal.name(), w.bits, w.group });
        if (refusal.len > 0 and refusal.width == null) std.log.warn("{s}: stored affine but declared unquantized", .{refusal.name()});
        return err;
    };
}

/// Every required matrix resolved at its own module width; the first one no device kernel reads lands in `refusal`.
pub fn checkWith(gpa: std.mem.Allocator, io: std.Io, directory: []const u8, refusal: *index.Refusal) !void {
    var c = try config.Config.read(gpa, io, directory);
    defer c.deinit();
    var w = try widths.Widths.init(gpa, &c);
    defer w.deinit();
    // refused before the index
    if (w.uniform() and !widths.launched(w.q.global)) return error.UnsupportedFlashAffineKernel;
    const path = try std.fs.path.join(gpa, &.{ directory, "model.safetensors.index.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(bytes);
    _ = try index.admitRuntime(gpa, bytes, &c, &w, refusal);
}

test {
    _ = @import("load_admission_test.zig");
    _ = @import("widths.zig");
}
