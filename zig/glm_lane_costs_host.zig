//! The production load-time measurement controller over recording timers, without devices or a checkpoint.
const std = @import("std");
const costs = @import("glm").lane_costs;
const cuda = @import("cuda");
const t = std.testing;
const Recording = struct {
    windows: [costs.widths.len]usize = @splat(0),
    drafts: usize = 0,
    forms: [3]bool = undefined,
    changes: usize = 0,
    fact: bool = false,
    pub fn form(r: *@This(), fact: bool) !void {
        if (r.changes >= r.forms.len) return error.TooManyFormChanges;
        r.forms[r.changes] = fact;
        r.changes += 1;
        r.fact = fact;
    }
    pub fn window(r: *@This(), i: usize) !f64 {
        r.windows[i] += 1;
        return (20 + @as(f64, @floatFromInt(costs.widths[i]))) * @as(f64, if (r.fact) 0.8 else 1);
    }
    pub fn draft(r: *@This(), depth: usize) !f64 {
        r.drafts += 1;
        return 2 + 0.5 * @as(f64, @floatFromInt(depth));
    }
};

test "GLM load prices: DFlash2 races expert forms without timing unused windows or head steps" {
    var r: Recording = .{};
    const kept = try costs.measurements.measure(&r, true, null);
    try t.expectEqual([3]bool{ false, true, true }, r.forms);
    try t.expectEqual(@as(usize, 0), r.drafts);
    for (r.windows, 0..) |count, i| {
        const expected: usize = if (i == 1 or i == 3 or i == 5) 16 else if (i == costs.widths.len - 1) 8 else 0;
        try t.expectEqual(expected, count);
    }
    try t.expect(costs.measurements.reusable(kept, true));
    try t.expect(!costs.measurements.reusable(kept, false));
    try t.expect(costs.tableFor(kept, true) == null);
}

test "GLM load prices: MTP retains every timed width and the two-level head slope" {
    var r: Recording = .{};
    const kept = try costs.measurements.measure(&r, false, null);
    try t.expectEqual([3]bool{ false, true, true }, r.forms);
    try t.expectEqual(@as(usize, 32), r.drafts);
    for (r.windows, 0..) |count, i| {
        const expected: usize = 8 + @as(usize, if (i == 1 or i == 3 or i == 5) 16 else if (i == costs.widths.len - 1) 8 else 0);
        try t.expectEqual(expected, count);
    }
    try t.expectApproxEqAbs(@as(f64, 0.5), kept.step_ms, 1e-12);
    try t.expect(costs.measurements.reusable(kept, false));
    try t.expect(!costs.measurements.reusable(kept, true));
    try t.expect(costs.tableFor(kept, false) != null);
    try t.expect(costs.tableFor(kept, true) == null);
}

test "GLM load prices: a named expert form is set once, before the warm-up, and no forms race" {
    for ([_]bool{ false, true }) |pin| for ([_]bool{ false, true }) |block| {
        var r: Recording = .{};
        const kept = try costs.measurements.measure(&r, block, pin);
        try t.expectEqual(@as(usize, 1), r.changes);
        try t.expectEqual(pin, r.forms[0]);
        try t.expectEqual(@as(u64, @intFromBool(pin)), kept.fact);
        try t.expectEqual(@as(usize, if (block) 0 else 16), r.drafts);
        for (r.windows, 0..) |count, i| {
            const warm: usize = if (i == costs.widths.len - 1) 8 else 0;
            try t.expectEqual(warm + @as(usize, if (block) 0 else 8), count);
        }
        try t.expect(costs.measurements.reusable(kept, block));
    };
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
    var e: @import("glm").engine.Engine = undefined;
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
