//! The production load-time measurement controller over recording timers, without devices or a checkpoint.
const std = @import("std");
const glm = @import("glm");
const costs = glm.lane_costs;
const cuda = @import("cuda");
const t = std.testing;
const m = costs.measurements;
const Recording = struct {
    windows: [costs.widths.len]usize = @splat(0),
    drafts: usize = 0,
    changes: usize = 0,
    fact: u64 = 0,
    ratio: [costs.widths.len]f64 = @splat(0.8), // a factored window's time over a classic one's, by width
    runs: usize = 0, // windows run so far
    slow: usize = 0, // the first this many windows run 15 ms over, as in a load's first seconds
    drift: f64 = 0, // a window's time moves by this fraction a window run before it
    switched: [2]f64 = .{ 0, 0 }, // classic's and factored's fraction over right after a window on the other form
    last: ?bool = null, // the last window's form, factored or not
    pub fn form(r: *@This(), fact: u64) !void {
        r.changes += 1;
        r.fact = fact;
    }
    fn classic(i: usize) f64 {
        return 20 + @as(f64, @floatFromInt(costs.widths[i]));
    }
    pub fn window(r: *@This(), i: usize) !f64 {
        r.windows[i] += 1;
        defer r.runs += 1;
        const f = (r.fact >> @intCast(i)) & 1 != 0;
        defer r.last = f;
        const after = if (r.last) |l| (if (l != f) r.switched[@intFromBool(f)] else 0) else 0;
        const ms = classic(i) * (if (f) r.ratio[i] else 1) * (1 + after);
        return ms * (1 + r.drift * @as(f64, @floatFromInt(r.runs))) + @as(f64, if (r.runs < r.slow) 15 else 0);
    }
    pub fn draft(r: *@This(), depth: usize) !f64 {
        r.drafts += 1;
        return 2 + 0.5 * @as(f64, @floatFromInt(depth));
    }
};

/// Each width's windows in a race: the untimed sweeps on both forms, then seven timed on each.
const raced = 2 * m.sweeps + 14;

test "GLM load prices: DFlash2 races the forms at every width without timing head steps or keeping prices" {
    var r: Recording = .{};
    const kept = try m.measure(&r, true, null);
    try t.expectEqual(m.all, r.fact);
    try t.expectEqual(m.all, kept.fact);
    try t.expectEqual(@as(usize, 0), r.drafts);
    for (r.windows) |count| try t.expectEqual(raced, count);
    try t.expect(m.reusable(kept, true));
    try t.expect(!m.reusable(kept, false));
    try t.expect(costs.tableFor(kept, true) == null);
}

test "GLM load prices: MTP keeps each width's raced time as its price, then the two-level head slope" {
    var r: Recording = .{};
    const kept = try m.measure(&r, false, null);
    try t.expectEqual(m.all, r.fact);
    try t.expectEqual(@as(usize, 16), r.drafts);
    for (r.windows, kept.ms, 0..) |count, ms, i| {
        try t.expectEqual(raced, count);
        try t.expectApproxEqAbs(Recording.classic(i) * 0.8, ms, 1e-12);
    }
    try t.expectApproxEqAbs(@as(f64, 0.5), kept.step_ms, 1e-12);
    try t.expect(m.reusable(kept, false));
    try t.expect(!m.reusable(kept, true));
    try t.expect(costs.tableFor(kept, false) != null);
    try t.expect(costs.tableFor(kept, true) == null);
}

test "GLM load prices: each width keeps the form its windows ran faster on, and classic inside the margin" {
    var r: Recording = .{};
    r.ratio = .{ 0.995, 0.97, 0.97, 1.02, 1.02, 0.97, 0.97, 0.97, 0.97, 0.97 };
    const kept = try m.measure(&r, false, null);
    const want = m.all & ~@as(u64, 0b1_1001); // classic at 1 row (inside the margin), 4 and 6
    try t.expectEqual(want, r.fact);
    try t.expectEqual(want, kept.fact);
    for (kept.ms, 0..) |ms, i| {
        const f = if ((want >> @intCast(i)) & 1 != 0) r.ratio[i] else 1;
        try t.expectApproxEqAbs(Recording.classic(i) * f, ms, 1e-12);
    }
}

test "GLM load prices: a dip is timed again at each width's own form, not the last raced one" {
    var r: Recording = .{};
    r.ratio = @splat(0.97);
    r.ratio[0] = 0.995; // inside the margin: 1 row stays classic
    r.ratio[1] = 0.9; // 2 rows then dip under 1 row's price
    const kept = try m.measure(&r, false, null);
    try t.expectEqual(@as(u64, 0), kept.fact & 1);
    try t.expect(r.windows[0] > raced);
    try t.expectApproxEqAbs(Recording.classic(0), kept.ms[0], 1e-12); // a factored retime would lower it
}

test "GLM load prices: a named expert form is set once, before the warm-up, and no forms race" {
    for ([_]bool{ false, true }) |pin| for ([_]bool{ false, true }) |block| {
        var r: Recording = .{};
        const kept = try m.measure(&r, block, pin);
        try t.expectEqual(@as(usize, 1), r.changes);
        try t.expectEqual(m.every(pin), r.fact);
        try t.expectEqual(m.every(pin), kept.fact);
        try t.expectEqual(@as(usize, if (block) 0 else 16), r.drafts);
        // as many untimed windows a width as a race's sweeps, then one fastest-of; DFlash2 times none
        for (r.windows) |count| try t.expectEqual(@as(usize, if (block) 0 else 2 * m.sweeps + 8), count);
        try t.expect(m.reusable(kept, block));
    };
}

test "GLM load prices: a slow start outlasting the old warm-up decides no width's form and sets no price" {
    var r: Recording = .{};
    r.ratio = @splat(1); // the two forms alike: classic at every width
    r.slow = 48; // longer than a warm-up at the widest alone, as the slowest observed start was
    const kept = try m.measure(&r, false, null);
    try t.expectEqual(@as(u64, 0), kept.fact);
    for (kept.ms, 0..) |ms, i| try t.expectApproxEqAbs(Recording.classic(i), ms, 1e-12);
}

test "GLM load prices: a drift through the race falls on both forms alike" {
    const Case = struct { drift: f64, ratio: f64, want: u64 };
    // faster as it runs with the forms alike, then slower with factored 2% faster at every width
    for ([_]Case{ .{ .drift = -0.0002, .ratio = 1, .want = 0 }, .{ .drift = 0.0002, .ratio = 0.98, .want = m.all } }) |c| {
        var r: Recording = .{};
        r.ratio = @splat(c.ratio);
        r.drift = c.drift;
        const kept = try m.measure(&r, true, null);
        try t.expectEqual(c.want, kept.fact);
    }
}

test "GLM load prices: a window run slow right after the other form decides no width's form" {
    var r: Recording = .{};
    r.ratio = @splat(0.97);
    r.switched = .{ 0.04, 0.13 }; // classic 4% and factored 13% over right after the other form
    const kept = try m.measure(&r, false, null);
    try t.expectEqual(m.all, kept.fact);
    for (kept.ms, 0..) |ms, i| try t.expectApproxEqAbs(Recording.classic(i) * 0.97, ms, 1e-12);
}

test "GLM load prices: a window takes the form of the narrowest timed width that holds it" {
    try t.expectEqual(@as(u64, 0), m.rows(0));
    try t.expectEqual(~@as(u64, 0), m.rows(m.all));
    try t.expectEqual(@as(u64, 1 << 3), m.rows(1 << 3)); // 4 rows alone
    try t.expectEqual(@as(u64, 0b11 << 4), m.rows(1 << 4)); // 6 rows: 5 and 6
    try t.expectEqual(~@as(u64, 0) << 24, m.rows(1 << 9)); // 32 rows: 25 and on
    var ks: glm.experts.Kernels = undefined;
    ks.fact = m.rows(m.all & ~@as(u64, 0b1_1000));
    for ([_]usize{ 0, 4, 5, 6 }) |rows| try t.expect(!ks.factored(rows));
    for ([_]usize{ 1, 3, 7, 32, 33, 64, 4096 }) |rows| try t.expect(ks.factored(rows));
    var k: m.Kept = .{ .ms = @splat(1), .step_ms = 0, .fact = m.all };
    try t.expect(m.reusable(k, false));
    k.fact = m.all + 1;
    try t.expect(!m.reusable(k, false));
}

const Device = struct {
    fn attribute(out: *c_int, attr: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = if (attr == .compute_capability_major) 12 else 1;
        return 0;
    }
    fn version(out: *c_int) callconv(.c) cuda.abi.Result {
        out.* = 13000;
        return 0;
    }
    fn name(out: [*]u8, n: c_int, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        if (n < 4) return 1;
        @memcpy(out[0..4], "GPU\x00");
        return 0;
    }
};

test "GLM load prices: an oversized NCCL setting disables the cache without rejecting the load identity" {
    const env = @import("sys").env;
    const variable = "NCCL_P2P_LEVEL";
    const a = t.allocator;
    const old = if (std.c.getenv(variable)) |v| try a.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer {
        if (old) |v| {
            _ = env.setenv(variable, v, 1);
            a.free(v);
        } else _ = env.unsetenv(variable);
    }
    var e: glm.engine.Engine = undefined;
    e.settings = .{};
    e.checkpoint_digest = @splat(1);
    e.share.world = 2;
    e.m.experts.nvfp4.warps = 4;
    e.drafter = null;
    var shape: [512]u8 = undefined;
    var name: [256]u8 = undefined;
    var driver: cuda.Driver = undefined;
    driver.api.cuDeviceGetAttribute = Device.attribute;
    driver.api.cuDriverGetVersion = Device.version;
    driver.api.cuDeviceGetName = Device.name;
    const ctx: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    const long = try a.allocSentinel(u8, 2048, 0);
    defer a.free(long);
    @memset(long, 'x');
    try t.expectEqual(@as(c_int, 0), env.setenv(variable, long, 1));
    try t.expect((try costs.keyParts(&e, &ctx, &name, &shape, null)) == null);
    try t.expectEqual(@as(c_int, 0), env.setenv(variable, "SYS", 1));
    const parts = (try costs.keyParts(&e, &ctx, &name, &shape, null)) orelse return error.ExpectedShortIdentity;
    try t.expectEqualStrings("glm-cuda", parts[0]);
    var dr: @TypeOf(e.drafter.?.*) = undefined;
    dr.bits = 4;
    dr.checkpoint_digest = @splat(2);
    e.drafter = &dr;
    const form = (try costs.keyParts(&e, &ctx, &name, &shape, null)) orelse return error.ExpectedFormIdentity;
    try t.expectEqualStrings("glm-cuda-form", form[0]);
}
