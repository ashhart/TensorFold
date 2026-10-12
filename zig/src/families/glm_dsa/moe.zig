//! A window's expert jobs from its picks: each picked expert's halves run only the rows routed to it.
const std = @import("std");
const ex = @import("../../core/exlr8.zig");
const cfg = @import("config.zig");
const wt = @import("weights.zig");
const gpu = @import("gpu.zig");
const Job = ex.job.Job;

/// The jobs (gate/up first, then down), member lists and per-pick tables one window's experts need.
pub const Plan = struct {
    jobs: std.ArrayList(Job) = .empty,
    members: std.ArrayList(u32) = .empty,
    gu_jobs: usize = 0,
    down_jobs: usize = 0,
    /// Per pick slot (row r, slot k; slot 8 the shared expert): routing weight, sv_block offset, down suh offset.
    wts: [gpu.max_rows * gpu.slots]f32 = undefined,
    sv: [gpu.max_rows * gpu.slots]u32 = undefined,
    suh: [gpu.max_rows * gpu.slots]u32 = undefined,

    pub fn deinit(p: *Plan, gpa: std.mem.Allocator) void {
        p.jobs.deinit(gpa);
        p.members.deinit(gpa);
    }

    /// Jobs for `rows` rows' picks (topk a row, as the route wrote them), experts ascending, the shared one last.
    pub fn build(gpa: std.mem.Allocator, c: *const cfg.Config, m: *const wt.Moe, rows: u32, pick: []const i32, wts: []const f32) !Plan {
        var p: Plan = .{};
        errdefer p.deinit(gpa);
        const k = c.topk;
        var users: [257]std.ArrayList(u32) = @splat(.empty);
        defer for (&users) |*u| u.deinit(gpa);
        for (0..rows) |r| for (0..k) |j| {
            const e: usize = @intCast(pick[r * k + j]);
            if (e >= c.experts) return error.BadPick;
            try users[e].append(gpa, @intCast(r * gpu.slots + j));
            p.wts[r * gpu.slots + j] = wts[r * k + j];
        };
        for (0..rows) |r| {
            try users[c.experts].append(gpa, @intCast(r * gpu.slots + gpu.slots - 1));
            p.wts[r * gpu.slots + gpu.slots - 1] = 1.0;
        }
        var down: std.ArrayList(Job) = .empty;
        defer down.deinit(gpa);
        const loc = c.localHalves();
        for (users, 0..) |u, e| {
            if (u.items.len == 0) continue;
            const x = &m.experts[e];
            for (u.items) |slot| {
                p.sv[slot] = x.down[0].block;
                p.suh[slot] = x.down_suh;
            }
            const gu_at: u32 = @intCast(p.members.items.len);
            for (u.items) |slot| try p.members.append(gpa, (slot / gpu.slots) << 16 | slot);
            const dn_at: u32 = @intCast(p.members.items.len);
            for (u.items) |slot| try p.members.append(gpa, slot << 16 | slot);
            const n: u32 = @intCast(u.items.len);
            for (0..loc) |hl| {
                const col: u32 = @intCast(hl * 128);
                try p.jobs.append(gpa, .{ .k = x.gate[hl].k, .blocks = 48, .base = x.gate_base[hl], .in_scale = x.gate[hl].block, .out_scale = x.gate[hl].chan, .y_col = col, .members = gu_at, .count = n });
                try p.jobs.append(gpa, .{ .k = x.up[hl].k, .blocks = 48, .base = x.up_base[hl], .in_scale = x.up[hl].block, .out_scale = x.up[hl].chan, .y_col = @intCast(loc * 128 + col), .members = gu_at, .count = n });
            }
            for (0..48) |ob| try down.append(gpa, .{ .k = x.down[0].k, .blocks = @intCast(loc), .base = x.down_base + @as(u32, @intCast(ob * loc)), .y_col = @intCast(ob * 128), .raw = 1, .members = dn_at, .count = n });
        }
        p.gu_jobs = p.jobs.items.len;
        p.down_jobs = down.items.len;
        try p.jobs.appendSlice(gpa, down.items);
        return p;
    }
};

test "a window's picks become member lists: every pick in its expert's list once, the shared expert every row" {
    const gpa = std.testing.allocator;
    var c: cfg.Config = .{};
    c.experts = 4;
    c.topk = 2;
    try cfg.split(&c, 0, 2);
    var experts: [5]wt.Expert = undefined;
    for (&experts, 0..) |*x, i| {
        x.down_suh = @intCast(1000 + i);
        x.down_base = @intCast(10 * i);
        for (0..wt.max_halves) |h| {
            x.gate[h] = .{ .at = 0, .k = 3, .chan = 0, .block = 0 };
            x.up[h] = x.gate[h];
            x.down[h] = .{ .at = 0, .k = 4, .chan = 0, .block = @intCast(i) };
            x.gate_base[h] = 0;
            x.up_base[h] = 0;
        }
    }
    const m: wt.Moe = .{ .router = .{ .job = 0, .in = 0, .out = 0 }, .bias = 0, .signs = 0, .experts = &experts };
    var p = try Plan.build(gpa, &c, &m, 2, &.{ 1, 3, 3, 0 }, &.{ 0.5, 0.25, 0.75, 0.125 });
    defer p.deinit(gpa);
    // experts 0, 1, 3 and the shared one: 16 gate/up jobs and 48 down jobs each
    try std.testing.expectEqual(@as(usize, 4 * 16), p.gu_jobs);
    try std.testing.expectEqual(@as(usize, 4 * 48), p.down_jobs);
    try std.testing.expectEqual(@as(u32, 1003), p.suh[1]);
    try std.testing.expectEqual(@as(f32, 0.75), p.wts[gpu.slots]);
    try std.testing.expectEqual(@as(f32, 1.0), p.wts[gpu.slots + gpu.slots - 1]);
    const shared_job = p.jobs.items[3 * 16];
    try std.testing.expectEqual(@as(u32, 2), shared_job.count);
    try std.testing.expectEqual(@as(u32, 1 << 16 | gpu.slots + 8), p.members.items[shared_job.members + 1]);
}
