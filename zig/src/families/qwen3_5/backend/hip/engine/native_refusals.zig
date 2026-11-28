//! Why the native HIP engine cannot serve a checkpoint, decided from its config before any shard is read.

const std = @import("std");
const engine = @import("engine.zig");
const joint = @import("joint_readout.zig");
const config = @import("../../../weights/config.zig");
const serve = @import("hip_serve");

const Options = serve.Options;

/// A flag or a resource the engine cannot take: `problem` says why.
pub const Refused = error{Refused};

pub fn refuse(a: std.mem.Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Refused {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

pub fn gibs(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// Why `dir` cannot be served: BF16 Qwen3.5 serves only with its joint head, and the head only on the BF16 weights.
pub fn decisionOnly(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ?[]const u8 {
    // a config that does not read is the loader's to report
    var c = config.Config.read(gpa, io, dir) catch return null;
    defer c.deinit();
    const head = joint.present(io, a, dir);
    if (c.quant != .bf16 and head) {
        const what = "the joint schema head needs the BF16 checkpoint";
        return std.fmt.allocPrint(a, "{s}: {s} is quantized ({t})", .{ what, dir, c.quant }) catch what;
    }
    if (c.quant != .bf16 or c.modelopt or head) return null;
    if (engine.chatBf16(gpa, io, dir)) return null; // an image checkpoint: a tower the engine reads, no joint head
    const what = "BF16 Qwen3.5 serves as a decision backbone only";
    return std.fmt.allocPrint(a, "{s}: {s} has no joint_head_config.json", .{ what, dir }) catch what;
}

/// Why a Clef checkpoint cannot open under tensor parallelism: its head runs on one rank, so no rank opens it.
pub fn oneRank(a: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) ?[]const u8 {
    if (o.world <= 1 or !joint.present(io, a, dir)) return null;
    return "the joint schema head serves on one rank; drop --tp";
}

pub const bf16_drafter = "--drafter: the HIP engine runs its DFlash2 drafter over quantized checkpoints only; " ++
    "serve this BF16 checkpoint without --drafter";

/// Why --drafter cannot run over `dir`: the drafter's rounds are priced by timed windows, which BF16 targets skip.
pub fn bf16Drafter(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) ?[]const u8 {
    if (o.drafter == null) return null;
    // a config that does not read is the loader's to report
    var c = config.Config.read(gpa, io, dir) catch return null;
    defer c.deinit();
    return if (c.quant == .bf16) bf16_drafter else null;
}
