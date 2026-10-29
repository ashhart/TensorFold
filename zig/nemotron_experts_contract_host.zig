//! Host proofs for the Nemotron routed-expert contract: its census, route tables, float64 steps, stress and controls.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const cases = @import("tests/contract/nemotron_experts_cases.zig");
const fixture = @import("tests/contract/nemotron_experts_fixture.zig");
const Fixture = fixture.Fixture;
const a = std.testing.allocator;
const experts_source = @embedFile("kernels/metal/nemotron_experts.metal");

const Collector = struct {
    a: std.mem.Allocator,
    list: std.ArrayList(cases.Entry) = .empty,
    pub fn add(s: *@This(), v: cases.Entry, _: []const u8) !void {
        try s.list.append(s.a, v);
    }
};

test "Nemotron experts contract: all twenty pipelines and two controls reach the actual collector" {
    var found: Collector = .{ .a = a };
    defer found.list.deinit(a);
    try cases.visit(&found, null);
    try std.testing.expectEqual(@as(usize, 22), found.list.items.len);
    var plain: usize = 0;
    for (found.list.items, 0..) |v, i| {
        if (v.fault != .none) continue;
        plain += 1;
        for (found.list.items[i + 1 ..]) |w| if (w.fault == .none) try std.testing.expect(!std.mem.eql(u8, v.symbol, w.symbol));
    }
    try std.testing.expectEqual(@as(usize, 20), plain);
    for (found.list.items) |v| if (!v.geo) {
        const tag = try std.fmt.allocPrint(a, "[[host_name(\"{s}\")]]", .{v.symbol});
        defer a.free(tag);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, experts_source, tag));
    };
    const main = @embedFile("tests/contract/metal_main.zig");
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, main, "try nemotron_experts.collect(o.only);"));
    found.list.clearRetainingCapacity();
    try cases.visit(&found, "tf_xdown_rows4");
    try std.testing.expectEqual(@as(usize, 1), found.list.items.len);
}

test "Nemotron experts contract: replacements name w3, served windows reach state.max_rows and are admitted" {
    for (cases.entries) |v| {
        const served = cases.served(v);
        const s = cases.spec(v, "test", &served);
        try std.testing.expectEqual(v.base(), s.replaces == null);
        if (s.replaces) |r| try std.testing.expectEqualStrings(if (v.proj == .up) "tf_xup_w3" else "tf_xdown_w3", r);
        try std.testing.expectEqual(v.proj == .up and !v.base(), s.nan == .stated);
        try std.testing.expectEqual(v.proj == .up and v.base(), s.known != null);
        if (s.known) |k| try std.testing.expectEqual(c.Gate.nonfinite, k.gate);
        try std.testing.expectEqual(@as(usize, if (v.geo) 1 else 4), served[0].rows);
        var top: usize = 0;
        for (served) |shape| {
            try std.testing.expect(Fixture.takes(v.proj, shape));
            top = @max(top, shape.rows);
        }
        try std.testing.expectEqual(@as(usize, 128), top);
        const nk = cases.dims(v.proj);
        try std.testing.expect(!Fixture.takes(v.proj, .{ .rows = 1, .n = nk[1], .k = nk[0] }));
        try std.testing.expect(!Fixture.takes(v.proj, .{ .rows = 129, .n = nk[0], .k = nk[1] }));
    }
}

test "Nemotron experts contract: route tables keep six distinct experts a row and every pass and tail at four rows" {
    for ([_]usize{ 1, 2, 3, 4, 5, 8, 16, 17, 128 }) |rows| {
        const r = try fixture.Route.init(a, rows, 1494 + rows);
        defer r.deinit(a);
        var counts: [fixture.experts]usize = @splat(0);
        for (0..rows) |row| {
            const picks = r.ids[row * 6 ..][0..6];
            for (picks, 0..) |e, i| {
                counts[e] += 1;
                for (picks[i + 1 ..]) |o| try std.testing.expect(e != o);
            }
        }
        for (counts) |n| try std.testing.expect(n <= rows);
        if (rows == 1) {
            try std.testing.expectEqual(@as(i32, 6), r.ucount);
            try std.testing.expectEqual(@as(usize, 6), r.groups);
            for (0..6) |k| try std.testing.expectEqual(r.ids[k], r.uids[k]);
            continue;
        }
        try std.testing.expectEqual(@min(rows * 6, 128), r.groups);
        var seen: usize = 0;
        for (0..@intCast(r.ucount)) |u| {
            if (u > 0) try std.testing.expect(r.uids[u] > r.uids[u - 1]);
            try std.testing.expectEqual(@as(usize, @intCast(r.start[u])), seen);
            try std.testing.expectEqual(counts[r.uids[u]], @as(usize, @intCast(r.count[u])));
            for (r.members[seen..][0..@intCast(r.count[u])], 0..) |p, i| {
                try std.testing.expectEqual(r.uids[u], r.ids[@intCast(p)]);
                if (i > 0) try std.testing.expect(p > r.members[seen + i - 1]);
            }
            seen += @intCast(r.count[u]);
        }
        try std.testing.expectEqual(rows * 6, seen);
        if (rows == 4) try std.testing.expectEqualSlices(usize, &.{ 4, 4, 4, 4, 3, 2, 2, 1 }, &counts);
    }
}

test "Nemotron experts contract: the reference sums x as tf_load16 does, in bf16 steps" {
    const one = c.toBf16(1);
    const tiny = c.toBf16(0x1p-8);
    try std.testing.expectEqual(@as(f64, 1), fixture.chain(&.{ one, tiny, tiny, tiny }));
    try std.testing.expectEqual(@as(f64, 1 + 0x1p-7), fixture.chain(&.{ tiny, tiny, one, 0 }));
    try std.testing.expect(std.math.isNan(fixture.chain(&.{ c.toBf16(std.math.inf(f32)), c.toBf16(-std.math.inf(f32)), one, one })));
    const f = try Fixture.init(a, .down, .{ .rows = 2, .n = 4, .k = 1856 }, .random, 7);
    defer f.deinit();
    @memset(f.q.scales, c.toBf16(0));
    @memset(f.q.biases, c.toBf16(1));
    for (f.q.x, f.xf, 0..) |*x, *xf, i| {
        x.* = if (i % 4 == 0) one else tiny;
        xf.* = c.fromBf16(x.*);
    }
    for (f.chains, 0..) |*v, i| v.* = fixture.chain(f.q.x[4 * i ..][0..4]);
    const want = try a.alloc(f64, f.count());
    defer a.free(want);
    const mag = try a.alloc(f64, f.count());
    defer a.free(mag);
    f.reference(want, mag);
    for (want) |w| try std.testing.expectEqual(@as(f64, 1856 / 4), w);
}

test "Nemotron experts contract: the fp32 x-sum control leaves w3's float64 bound at a served window" {
    const v = for (cases.entries) |e| {
        if (e.fault == .fp32_sums) break e;
    } else unreachable;
    try std.testing.expectEqual(cases.Fault.fp32_sums, v.fault);
    const served = cases.served(v);
    const s = cases.spec(v, "control", &served);
    const shape: c.Shape = .{ .rows = 4, .n = 64, .k = 1856 };
    const f = try Fixture.init(a, .down, shape, .random, 0x6b63);
    defer f.deinit();
    const want = try a.alloc(f64, f.count());
    defer a.free(want);
    const mag = try a.alloc(f64, f.count());
    defer a.free(mag);
    f.reference(want, mag);
    for (f.chains, 0..) |*sum, i| {
        var plain: f32 = 0;
        for (f.q.x[4 * i ..][0..4]) |x| plain += c.fromBf16(x);
        sum.* = plain;
    }
    const plain = try a.alloc(f64, f.count());
    defer a.free(plain);
    f.reference(plain, mag);
    var out: usize = 0;
    for (want, plain, mag) |w, p, m| out += @intFromBool(@abs(c.fromBf16(c.toBf16(@floatCast(p))) - w) > c.bound(s, shape, w, m));
    try std.testing.expect(out > 0);
}

test "Nemotron experts contract: each control changes one witness, and a lost witness fails" {
    const coupled = try cases.source(a, experts_source, .coupled);
    defer a.free(coupled);
    const sums = try cases.source(a, experts_source, .fp32_sums);
    defer a.free(sums);
    const same = try cases.source(a, experts_source, .none);
    defer a.free(same);
    try std.testing.expectEqualStrings(experts_source, same);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, coupled, "x0 + xoff[0] + xl + k0"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, sums, "sum[b] += float(a) + float(bb) + float(c) + float(d);"));
    for ([_][]const u8{ coupled, sums }) |t| {
        try std.testing.expect(!std.mem.eql(u8, t, experts_source));
        try std.testing.expectEqual(std.mem.count(u8, experts_source, "[[host_name("), std.mem.count(u8, t, "[[host_name("));
    }
    try std.testing.expectError(error.SourceWitnessMissing, cases.source(a, "kernel void x() {}", .coupled));
}

test "Nemotron experts contract: stress sums are exact in fp32 in every order and far off in fp16 and bf16" {
    for ([_]cases.Proj{ .up, .down }) |proj| for ([_]u64{ 0x6b63, 0x1494 }) |seed| {
        const k = cases.dims(proj)[1];
        const shape: c.Shape = .{ .rows = 4, .n = 4, .k = k };
        const f = try Fixture.init(a, proj, shape, .stress, seed);
        defer f.deinit();
        const buf = try a.alloc(f64, 4 * f.count());
        defer a.free(buf);
        const want = buf[0..f.count()];
        const mag = buf[f.count()..][0..f.count()];
        const got = buf[2 * f.count() ..][0..f.count()];
        const allow = buf[3 * f.count() ..];
        f.reference(want, mag);
        for (f.dots) |d| try std.testing.expect(d >= 32 and d <= 224);
        const s = cases.spec(.{ .proj = proj, .symbol = "tf_x_w3", .rps = 4, .sg = 4, .mb = 3 }, "stress", &.{shape});
        if (proj == .up) {
            try f.bounds(want, allow);
        } else for (allow, want) |*l, w| {
            l.* = c.boundExact(s, w);
        }
        for ([_]contract.gates.Order{ .sequential, .pairwise, .strided }) |order| {
            try f.simulate(.f32, order, got);
            try std.testing.expectEqualSlices(f64, want, got);
            for ([_]c.Accum{ .f16, .bf16 }) |acc| {
                try f.simulate(acc, order, got);
                var most: f64 = 0;
                for (got, want, allow) |x, w, l| most = @max(most, if (std.math.isFinite(x)) @abs(x - w) / l else std.math.inf(f64));
                try std.testing.expect(most >= 100);
            }
        }
    };
}

test "Nemotron experts contract: relu2's interval holds every fp32 dot within the error and its roundings" {
    var prng = std.Random.DefaultPrng.init(1494);
    const rng = prng.random();
    for (0..2000) |_| {
        const d = (rng.float(f64) - 0.3) * 8;
        const e = rng.float(f64) * 0.01;
        const lo = @max(fixture.bfSide(d - e, false), 0);
        const hi = @max(fixture.bfSide(d + e, true), 0);
        const low = fixture.bfSide(lo * lo, false);
        const high = fixture.bfSide(hi * hi, true);
        for ([_]f64{ d - e, d, d + e, d - e / 3 }) |x| {
            const acc: f32 = @floatCast(x);
            const h = @max(c.fromBf16(c.toBf16(acc)), 0);
            const out = c.fromBf16(c.toBf16(h * h));
            try std.testing.expect(out >= low and out <= high);
        }
        try std.testing.expect(fixture.relu2(d) >= low and fixture.relu2(d) <= high);
    }
    try std.testing.expectEqual(@as(f64, 1), fixture.bfSide(1, false));
    try std.testing.expectEqual(@as(f64, 1 + 0x1p-7), fixture.bfSide(1 + 0x1p-20, true));
    try std.testing.expectEqual(@as(f64, -1 - 0x1p-7), fixture.bfSide(-1 - 0x1p-20, false));
    try std.testing.expectEqual(@as(f64, -1), fixture.bfSide(-1 - 0x1p-20, true));
    try std.testing.expect(std.math.isNan(fixture.relu2(std.math.nan(f64))));
}
