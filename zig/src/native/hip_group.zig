//! HIP launch group checks, kept beside native engine construction.
const std = @import("std");
const api = @import("engine_api");
const hip_drafter = @import("hip_drafter.zig");
const Allocator = std.mem.Allocator;

const Refused = error{Refused};

pub fn refuse(a: Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Refused {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

/// The tensor-parallel flags as the Python ROCm server checks them, and the flags this engine cannot take.
pub fn checkGroup(a: Allocator, o: api.Open, problem: *[]const u8) Refused!void {
    if (hip_drafter.groupRefusal(a, o)) |why| return refuse(a, problem, "{s}", .{why});
    if (o.tp != 1 and o.tp != 2 and o.tp != 4 and o.tp != 8) return refuse(a, problem, "--tp {d} is not a supported ROCm world size; choose 1, 2, 4 or 8", .{o.tp});
    if (o.tp > 1 and o.master.len == 0) return refuse(a, problem, "--tp > 1 needs --master: rank 0's address on the link between the machines", .{});
    // the two-Mac engines take their ranks from --speed-up's settings file; this engine takes them from --tp
    if (o.speed_up != null) return refuse(a, problem, "--speed-up is the two-Mac engines' rank settings; the HIP engine takes its ranks from --tp", .{});
    if (o.tp == 1 and o.rank != 0) return refuse(a, problem, "--rank must be 0 when --tp 1", .{});
    if (o.rank >= o.tp) return refuse(a, problem, "--rank {d} not in [0, --tp {d})", .{ o.rank, o.tp });
    if (o.keep) |n| if (n < 0) return refuse(a, problem, "--checkpoint-slots must be 0 or more", .{});
    if (o.mtp_drafts) |n| if (n > 3) return refuse(a, problem, "--mtp-drafts {d}: the HIP engine drafts at most 3", .{n});
    if (o.mtp_confidence) |c| if (c < 0 or c > 1) return refuse(a, problem, "--mtp-confidence must be from 0 to 1", .{});
    if (o.prompt_cache_gib) |g| if (g < 0) return refuse(a, problem, "--prompt-cache-gib must be 0 or more", .{});
}
