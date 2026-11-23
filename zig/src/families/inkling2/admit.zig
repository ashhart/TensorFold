//! Inkling 2 on CUDA is admitted only as it was run: on GB10 (sm_121) at two ranks, placed by TF_INKLING2_RANK/WORLD.
const std = @import("std");
const cuda = @import("cuda");
const environment = @import("sys").environment;

pub const Refusal = error{ Inkling2RequiresSm121, Inkling2RequiresTwoRanks };

pub fn admit(sm: u32, world: usize) Refusal!void {
    if (sm != 121) return error.Inkling2RequiresSm121;
    if (world != 2) return error.Inkling2RequiresTwoRanks;
}

/// The first check of the family's open: the device's capability, then the rank count, before any image loads.
pub fn check(ctx: *const cuda.Context, world: usize) (Refusal || cuda.Error)!void {
    try admit(try ctx.capability(), world);
}

pub fn explain(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.Inkling2RequiresSm121 => "Inkling 2 on CUDA is validated on GB10 (sm_121) only",
        error.Inkling2RequiresTwoRanks => "Inkling 2 on CUDA runs on exactly two ranks: set TF_INKLING2_RANK (0 or 1) and TF_INKLING2_WORLD=2",
        error.Inkling2NoRank => "TF_INKLING2_RANK and TF_INKLING2_WORLD go together, with the rank below the world",
        else => null,
    };
}

/// This rank's place: TF_INKLING2_RANK of TF_INKLING2_WORLD (both, or neither for one rank).
pub const Place = struct { rank: usize, world: usize };

pub fn placeFrom(rank: ?[]const u8, world: ?[]const u8) !Place {
    return placeOf(try environment.value(?u32, "TF_INKLING2_RANK", rank, null), try environment.value(?u32, "TF_INKLING2_WORLD", world, null));
}

pub fn place() !Place {
    return placeOf(try environment.read(?u32, "TF_INKLING2_RANK", null), try environment.read(?u32, "TF_INKLING2_WORLD", null));
}

fn placeOf(r: ?u32, w: ?u32) error{Inkling2NoRank}!Place {
    if (r == null and w == null) return .{ .rank = 0, .world = 1 };
    if (r == null or w == null or w.? == 0 or r.? >= w.?) return error.Inkling2NoRank;
    return .{ .rank = r.?, .world = w.? };
}

test {
    _ = @import("admit_test.zig");
}
