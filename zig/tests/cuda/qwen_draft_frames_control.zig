//! Test-command-only drafter delegation: a fixed chain and the old shared-frame prompt arithmetic as a control.
const dr = @import("lanes").drafter;
const Shape = @import("lanes").shape.Shape;
const plan = @import("qwen_draft_frames_plan.zig");
pub const Stats = struct { frames: usize = 0, prompt_frames: usize = 0, max_rows: usize = 0 };

pub fn hold(d: anytype, items: []const dr.Hold, prompt_control: bool, stats: *Stats) !void {
    if (items.len == 0 or items.len > plan.streams or d.c.block != plan.block) return error.BadDraftFrame;
    const rows = items.len * plan.block;
    const old = d.projection_role;
    if (old != .proposal) return error.UnexpectedDraftRole;
    defer d.projection_role = old;
    const prompt = prompt_control and rows > 16;
    if (prompt) d.projection_role = .prompt;
    var parents: [plan.depth]i32 = undefined;
    var ranks: [plan.depth]u8 = @splat(0);
    var depths: [plan.depth]u8 = undefined;
    for (&parents, &depths, 0..) |*parent, *depth, i| {
        parent.* = @as(i32, @intCast(i)) - 1;
        depth.* = @intCast(i);
    }
    const shape: Shape = .{ .parents = &parents, .ranks = &ranks, .depths = &depths, .expected = 0 };
    var fixed: [plan.streams]dr.Hold = undefined;
    for (items, fixed[0..items.len]) |item, *copy| {
        if (item.depth != plan.depth or item.lanes != null) return error.BadDraftFrame;
        copy.* = item;
        copy.lanes = &shape;
    }
    try d.drafter().hold(fixed[0..items.len]);
    stats.frames += 1;
    stats.prompt_frames += @intFromBool(prompt);
    stats.max_rows = @max(stats.max_rows, rows);
}
