//! Cached shared-expert plans against an on-demand plan, including each two-rank placement.
const std = @import("std");
const nemotron = @import("nemotron");
const check = @import("check.zig");

/// Compares only initialized members, items and counters, never unused item capacity.
pub fn compare(e: *nemotron.Engine) !void {
    const c = e.c;
    if (c.latent == 0 or c.bolt) return;
    const o = e.ops();
    for (1..nemotron.state.windowRows(c) + 1) |rows| {
        const cached = e.b.latentPlan(c, rows);
        const fresh = e.b.plan;
        try o.plan(e.b.sids, rows * c.ownShares(), c.ownShares(), 16, fresh);
        var want: [2]u32 = undefined;
        var got: [2]u32 = undefined;
        try o.download(std.mem.asBytes(&want), fresh.counts);
        try o.download(std.mem.asBytes(&got), cached.counts);
        try o.s.synchronize();
        try check.sameBytes("shared-plan counts", std.mem.asBytes(&got), std.mem.asBytes(&want));
        try same(e, cached.members, fresh.members, rows * c.ownShares() * 4);
        try same(e, cached.items, fresh.items, @as(usize, want[0]) * 12);
    }
    check.pass("cached shared plans: world {d}, rank {d}, every width equals an on-demand plan", .{ c.world, c.rank });
}

fn same(e: *nemotron.Engine, got_ptr: u64, want_ptr: u64, n: usize) !void {
    const got = try e.gpa.alloc(u8, n);
    defer e.gpa.free(got);
    const want = try e.gpa.alloc(u8, n);
    defer e.gpa.free(want);
    try e.ops().download(got, got_ptr);
    try e.ops().download(want, want_ptr);
    try e.stream.synchronize();
    try check.sameBytes("shared-plan payload", got, want);
}

/// Both ranks use the existing expert transport; rank zero runs the real snapshot and resume comparisons.
pub fn tp2(gpu: check.Gpu, model: []const u8, ids: []const u8) !void {
    var c = try nemotron.Config.read(gpu.gpa, gpu.io, model);
    try nemotron.latent.place(&c);
    if (c.world != 2 or c.latent == 0 or c.bolt) return error.ExpectedLatentTwoRanks;
    if (c.rank > 0) return nemotron.worker.serve(gpu.gpa, gpu.io, gpu.ctx, model, null);
    try @import("prompt_cuts.zig").run(gpu, model, ids);
}
