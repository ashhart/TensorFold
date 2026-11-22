//! Actual portable device-controller code versus DepthRule, and the production conditional/strict-setting boundaries.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const nemo = @import("nemotron");
const t = std.testing;
const policy = nemo.mtp_graph_policy;
extern fn tf_mtp_host_step(*const policy.Policy, *f64, c_uint, *c_uint, f32) c_uint;
extern fn tf_mtp_host_square(f64) f64;
extern fn tf_mtp_policy_bytes() c_uint;
extern fn tf_mtp_state_bytes() c_uint;

test "MTP graph: compiled portable controller agrees with actual DepthRule over costs and probabilities" {
    try t.expectEqual(@sizeOf(policy.Policy), tf_mtp_policy_bytes());
    try t.expectEqual(@sizeOf(policy.State), tf_mtp_state_bytes());
    const probabilities = [_]f32{ 1, 0.99, 0.8, 0.5, 0.1, 0, std.math.nan(f32), std.math.inf(f32) };
    for ([_]usize{ 1, 3, 15, 31 }) |most| for ([_]f64{ 0, 0.01, 0.5, 4 }) |slope| for ([_]?f64{ null, 0, 0.3, 1 }) |floor| {
        var costs: core.draft_depth.Costs = .{ .rows = 32, .level = 0.71 };
        for (costs.verify[1..], 1..) |*v, rows| v.* = 10 + slope * @as(f64, @floatFromInt(rows));
        var rule = core.draft_depth.DepthRule.init(costs, most, floor, 2);
        for (0..4) |round| {
            rule.done(round + 1, round + 2, round + 1);
            const inputs = policy.Policy.init(rule).?;
            for (probabilities) |p| {
                var raw_run: f64 = 1;
                var device_run: f64 = 1;
                var n: c_uint = 0;
                for (1..most + 1) |j| {
                    raw_run *= @as(f64, p);
                    const keep = rule.keep(j, raw_run);
                    const next = keep and rule.more(j, raw_run);
                    const before = n;
                    try t.expectEqual(next, tf_mtp_host_step(&inputs, &device_run, @intCast(j), &n, p) != 0);
                    try t.expectEqual(if (keep) @as(c_uint, @intCast(j)) else before, n);
                    if (!std.math.isNan(raw_run)) try t.expectEqual(@as(u64, @bitCast(raw_run)), @as(u64, @bitCast(device_run)));
                    if (!next) break;
                }
            }
        }
    };
}
test "MTP graph: controller square matches std pow including underflow and decision boundaries" {
    const probes = [_]f64{ 0, 1, 0.5, 0.999999999999, 1e-100, 1e-160, 1e-200, 1e-300, std.math.inf(f64) };
    for (probes) |v| try t.expectEqual(@as(u64, @bitCast(std.math.pow(f64, v, 2))), @as(u64, @bitCast(tf_mtp_host_square(v))));
    var prng = std.Random.DefaultPrng.init(42);
    for (0..10000) |_| {
        const v = prng.random().float(f64);
        try t.expectEqual(@as(u64, @bitCast(std.math.pow(f64, v, 2))), @as(u64, @bitCast(tf_mtp_host_square(v))));
    }
}
test "MTP graph: unsupported rule powers and missing conditional entry points retain host scheduling" {
    const Missing = struct {
        pub fn lookup(_: @This(), comptime T: type, _: [:0]const u8) ?T {
            return null;
        }
    };
    try t.expect(cuda.conditional.Api.resolve(Missing{}) == null);
    var e: nemo.Engine = undefined;
    e.k.triton = null;
    var h: nemo.Head = undefined;
    h.e = &e;
    h.auto_allowed = true;
    h.loop = null;
    try t.expect(!nemo.mtp_graph.available(&h));
    var costs: core.draft_depth.Costs = .{ .rows = 16, .level = 1 };
    costs.verify[2] = 10;
    const rule = core.draft_depth.DepthRule.init(costs, 15, null, 1);
    try t.expect(policy.Policy.init(rule) == null);
    try t.expect(try nemo.mtp_graph.depth(&h, &rule, .subset) == null);
}
test "MTP graph: strict opt-in is registered once, defaults off and actual reader refuses invalid text" {
    const sys = @import("sys");
    const name = "TF_NEMOTRON_MTP_GRAPH";
    var count: usize = 0;
    var scope_count: usize = 0;
    for (sys.environment.rules) |r| {
        scope_count += @intFromBool(r.scope == .nemotron);
    }
    try t.expectEqual(@as(usize, 3), scope_count);
    for (sys.environment.rules) |r| if (std.mem.eql(u8, r.name, name)) {
        count += 1;
    };
    try t.expectEqual(@as(usize, 1), count);
    _ = sys.env.unsetenv(name);
    defer _ = sys.env.unsetenv(name);
    try t.expect(!try nemo.mtp_graph.enabled());
    for ([_][:0]const u8{ "", "   ", "0", "1" }) |raw| {
        _ = sys.env.setenv(name, raw, 1);
        try t.expectEqual(std.mem.eql(u8, raw, "1"), try nemo.mtp_graph.enabled());
    }
    for ([_][:0]const u8{ "on", "yes", "2", "-1", "01", " 1" }) |raw| {
        _ = sys.env.setenv(name, raw, 1);
        try t.expectError(error.InvalidEnvironment, nemo.mtp_graph.enabled());
    }
}
test "MTP graph: actual graph recorder, depth and sample bodies compile without a driver" {
    std.mem.doNotOptimizeAway(&nemo.mtp_graph.record);
    std.mem.doNotOptimizeAway(&nemo.mtp_graph.depth);
    std.mem.doNotOptimizeAway(&nemo.Head.sample);
}

test "MTP graph: exact keep and more comparisons at one-ulp row-cost boundaries" {
    const p: f32 = 0.8;
    const score = std.math.pow(f64, @as(f64, p), 2);
    const bits: u64 = @bitCast(score);
    for ([_]u64{ bits - 1, bits, bits + 1 }) |at| {
        const edge: f64 = @bitCast(at);
        var costs: core.draft_depth.Costs = .{ .rows = 4, .level = 0 };
        costs.verify[2] = 0;
        costs.verify[3] = edge;
        costs.verify[4] = 2 * edge;
        var rule = core.draft_depth.DepthRule.init(costs, 3, null, 2);
        rule.tokens = 1;
        rule.ms = 1;
        const inputs = policy.Policy.init(rule).?;
        var run: f64 = 1;
        var n: c_uint = 1;
        const want = rule.keep(2, p) and rule.more(2, p);
        try t.expectEqual(want, tf_mtp_host_step(&inputs, &run, 2, &n, p) != 0);
        try t.expectEqual(if (rule.keep(2, p)) @as(c_uint, 2) else 1, n);
    }
}

test "MTP graph: native single-rank graphs only, with captured-glue and ranked callers using their host loop" {
    var e: nemo.Engine = undefined;
    e.k.triton = null;
    e.tp = null;
    var h: nemo.Head = undefined;
    h.e = &e;
    h.auto_allowed = true;
    h.loop = .{ .api = undefined, .device = undefined, .host = undefined, .ready = undefined };
    try t.expect(nemo.mtp_graph.available(&h));
    e.tp = @as(@typeInfo(@TypeOf(e.tp)).optional.child, undefined);
    try t.expect(!nemo.mtp_graph.available(&h));
    e.tp = null;
    e.k.triton = @as(@typeInfo(@TypeOf(e.k.triton)).optional.child, undefined);
    try t.expect(!nemo.mtp_graph.available(&h));
    e.k.triton = null;
    h.auto_allowed = false;
    try t.expect(!nemo.mtp_graph.available(&h));
}
