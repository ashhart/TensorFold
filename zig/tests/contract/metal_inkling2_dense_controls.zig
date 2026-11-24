//! Planted source controls and probe registration OOM tests for Inkling 2 Mini's multi-segment dense projections.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;

pub const Control = enum { none, order, dense4_seg, kv_input, ldy };

const Edit = struct { from: []const u8, to: []const u8 };

fn edit(ctl: Control) Edit {
    return switch (ctl) {
        .none => .{ .from = "", .to = "" },
        .order => .{
            .from = "acc[r] += dot(w0, float4(*(const device bfloat4*)x)) + dot(w1, float4(*(const device bfloat4*)(x + 4)));",
            .to = "acc[r] += dot(w0, float4(*(const device bfloat4*)x)); acc[r] += dot(w1, float4(*(const device bfloat4*)(x + 4)));",
        },
        .dense4_seg => .{
            .from = "for (uint s = 0; s < 3; ++s) if (g >= a.end[s]) seg = s + 1;",
            .to = "for (uint s = 0; s < 2; ++s) if (g >= a.end[s]) seg = s + 1;",
        },
        .kv_input => .{
            .from = "const device bfloat* X = seg == 4 ? Xk : seg == 5 ? Xv : Xq;",
            .to = "const device bfloat* X = Xq;",
        },
        .ldy => .{
            .from = "if (lane == 0 && r < rows) Y[(size_t)r * n_local + j] = bfloat(s);",
            .to = "if (lane == 0 && r < rows) Y[(size_t)r * (n_local + 1) + j] = bfloat(s);",
        },
    };
}

pub fn source(a: std.mem.Allocator, production: []const u8, ctl: Control) ![]u8 {
    if (ctl == .none) return a.dupe(u8, production);
    const e = edit(ctl);
    if (std.mem.count(u8, production, e.from) != 1) return error.ControlEditMissing;
    return std.mem.replaceOwned(u8, a, production, e.from, e.to);
}

pub fn label(ctl: Control, segs: u8) []const u8 {
    return switch (ctl) {
        .none => if (segs == 4) "metal inkling2 glue dense4 multi-segment tf_ink2_dense4" else "metal inkling2 glue dense6 multi-segment tf_ink2_dense6",
        .order => if (segs == 4) "control: metal inkling2 glue dense4, split dot accumulation order" else "control: metal inkling2 glue dense6, split dot accumulation order",
        .dense4_seg => "control: metal inkling2 glue dense4, last segment omitted in scan",
        .kv_input => "control: metal inkling2 glue dense6, k and v segments ignore xk and xv",
        .ldy => if (segs == 4) "control: metal inkling2 glue dense4, output row stride off by one" else "control: metal inkling2 glue dense6, output row stride off by one",
    };
}

test "every dense control applies once to glue sources and changes only its anchor" {
    const a = std.testing.allocator;
    const sources = @import("inkling2_metal");
    for ([_]Control{ .order, .dense4_seg, .kv_input, .ldy }) |ctl| {
        const e = edit(ctl);
        try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, sources.glue, e.from));
        const changed = try source(a, sources.glue, ctl);
        defer a.free(changed);
        const at = std.mem.indexOf(u8, sources.glue, e.from).?;
        try std.testing.expectEqualStrings(sources.glue[0..at], changed[0..at]);
        try std.testing.expectEqualStrings(e.to, changed[at..][0..e.to.len]);
        try std.testing.expectEqualStrings(sources.glue[at + e.from.len ..], changed[at + e.to.len ..]);
        try std.testing.expectError(error.ControlEditMissing, source(a, changed, ctl));
    }
}

const MockProbe = struct {
    a: std.mem.Allocator,
    destroyed: *usize,
    payload: []u8,

    fn init(a: std.mem.Allocator, destroyed: *usize) !MockProbe {
        return .{ .a = a, .destroyed = destroyed, .payload = try a.alloc(u8, 16) };
    }

    fn deinit(p: *MockProbe) void {
        p.a.free(p.payload);
        p.destroyed.* += 1;
    }

    fn prepare(_: *anyopaque, _: c.Shape, _: g.Fill, _: u64) !bool {
        return false;
    }
    fn run(_: *anyopaque) !void {}
    fn bytes(_: *anyopaque) ![]const u8 {
        return &.{};
    }
    fn intact(_: *anyopaque) !bool {
        return true;
    }
    fn overlap(_: *anyopaque) bool {
        return false;
    }
    fn count(_: *anyopaque) usize {
        return 0;
    }
    fn reference(_: *anyopaque, _: []f64, _: []f64) !void {}
    fn depth(_: c.Shape) usize {
        return 1;
    }

    const vt: g.VTable = .{
        .prepare = prepare,
        .run = run,
        .output = bytes,
        .inputs = bytes,
        .intact = intact,
        .overlap = overlap,
        .count = count,
        .reference = reference,
    };

    fn kernel(p: *MockProbe) g.Kernel {
        return .{ .spec = .{ .name = "mock", .accum = .none, .depth = depth, .out = .bf16, .served = &.{} }, .ctx = p, .vt = &vt };
    }
};

const MockSuite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*MockProbe) = .empty,
    kernels: *std.ArrayList(g.Kernel),

    fn add(s: *MockSuite, destroyed: *usize) !void {
        const probe = try s.a.create(MockProbe);
        errdefer s.a.destroy(probe);
        probe.* = try MockProbe.init(s.a, destroyed);
        errdefer probe.deinit();
        try contract.register(s.a, &s.probes, &s.kernels, probe, probe.kernel());
    }

    fn deinit(s: *MockSuite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};

test "dense probe registration cleans up candidate without publishing on kernel list OOM" {
    const base = std.testing.allocator;
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(base);
    var existing_destroyed: usize = 0;
    var candidate_destroyed: usize = 0;
    {
        var suite: MockSuite = .{ .a = base, .kernels = &kernels };
        defer suite.deinit();
        try suite.probes.ensureTotalCapacityPrecise(base, 1);
        try kernels.ensureTotalCapacityPrecise(base, 1);
        try suite.add(&existing_destroyed);
        const existing = suite.probes.items[0];
        // Fail remap so FailingAllocator hits kernels.ensureUnusedCapacity after probes grows.
        var failing = std.testing.FailingAllocator.init(base, .{ .fail_index = 3, .resize_fail_index = 0 });
        suite.a = failing.allocator();
        try std.testing.expectError(error.OutOfMemory, suite.add(&candidate_destroyed));
        suite.a = base;
        try std.testing.expect(failing.has_induced_failure);
        try std.testing.expect(suite.probes.capacity > 1);
        try std.testing.expectEqual(@as(usize, 1), kernels.capacity);
        try std.testing.expectEqual(@as(usize, 1), candidate_destroyed);
        try std.testing.expectEqual(@as(usize, 0), existing_destroyed);
        try std.testing.expectEqual(@as(usize, 1), suite.probes.items.len);
        try std.testing.expectEqual(existing, suite.probes.items[0]);
        try std.testing.expectEqual(@as(usize, 1), kernels.items.len);
        try std.testing.expectEqual(@as(*anyopaque, existing), kernels.items[0].ctx);
    }
    try std.testing.expectEqual(@as(usize, 1), existing_destroyed);
}
