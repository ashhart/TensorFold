//! The CUDA engines whose own family module opens them, before the registration table is consulted.
const std = @import("std");
const api = @import("engine_api");
const refusals = @import("cuda_refusal.zig");
const Allocator = std.mem.Allocator;

/// `opened`: a family outside the table answered (null: refused, `problem` set); `not_mine`: the table decides.
pub const Answer = union(enum) { not_mine, opened: ?api.Opened };

/// Drama's, Qwen4's and LFM2's own engines, by the model type each serves.
pub fn special(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !Answer {
    if (std.mem.eql(u8, o.model_type, "fish_omni")) return .{ .opened = try @import("drama_open.zig").open(a, gpa, io, o, problem) };
    if (std.mem.eql(u8, o.model_type, "qwen4") or std.mem.eql(u8, o.model_type, "qwen4_text")) return .{ .opened = try @import("qwen4_cuda.zig").open(a, gpa, io, o, problem) };
    const lfm2 = @import("lfm2.zig");
    if (!std.mem.eql(u8, o.model_type, lfm2.family.model_type)) return .not_mine;
    if (refusals.lfm2(o)) |why| {
        problem.* = why;
        return .{ .opened = null };
    }
    return .{ .opened = try lfm2.open(a, gpa, io, o, problem) };
}
