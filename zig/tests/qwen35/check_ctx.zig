//! What the checks of one `check` run share: the engine, its lane session, the prompts and the report.

const std = @import("std");
const lanes = @import("lanes");
const qwen35 = @import("qwen3_5");
const group_mod = @import("group.zig");
const ids_file = @import("ids_file.zig");
const lanes_session = @import("lanes_session.zig");
const check_report = @import("check_report.zig");

pub const Ctx = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    /// Lives to the end of the run: replies, prompt sets.
    arena: std.mem.Allocator,
    e: *qwen35.engine.Engine,
    session: *lanes_session.Session,
    report: *check_report.Report,
    group: group_mod.Group,
    /// Ids for the long prompts, the row cases and the accuracy rows.
    ids: []const u32,
    short: []const ids_file.Prompt,
    long: []const ids_file.Prompt,
    /// Tokens each stream of an invariant run generates.
    tokens: u32,
    /// Pages the pool held before the first run: the scratch rows'.
    pages_at_start: usize = 0,

    /// Whether this run spans GPUs, where only what the lane core drives can run.
    pub fn tp(c: *const Ctx) bool {
        return c.group.world > 1;
    }
};

/// The ways a stream draws its tokens.
pub const Draw = struct { name: []const u8, sampling: ?lanes.Sampling };

pub const draws = [_]Draw{
    .{ .name = "greedy", .sampling = null },
    .{ .name = "seeded t=1", .sampling = .{ .seed = 7, .temperature = 1.0 } },
};
