//! Nemotron-H's 512-thread pick by rank against the wave a pick at a time: the same bytes, and their times.

const std = @import("std");
const rig = @import("../rig.zig");
const launch = @import("../../launches/nemotron_moe.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

/// One pick's inputs and outputs on the device: logits (rows, experts), the bias, and both kernels' picks and weights.
const Case = struct {
    rows: usize,
    experts: usize,
    top_k: usize,
    logits: DeviceBuffer,
    bias: DeviceBuffer,
    pick: [2]DeviceBuffer,
    wts: [2]DeviceBuffer,

    /// `levels` > 0 draws logits from that many values and a zero bias, so many experts tie.
    fn init(t: *Rig, rng: *rig.Rng, rows: usize, experts: usize, top_k: usize, levels: u32) !Case {
        const l = try gpa.alloc(f32, rows * experts);
        defer gpa.free(l);
        const b = try gpa.alloc(f32, experts);
        defer gpa.free(b);
        for (l) |*x| x.* = if (levels > 0) @floatFromInt(rng.next() % levels) else @floatCast(4 * rng.normal());
        for (b) |*x| x.* = if (levels > 0) 0 else rng.unit() / 16;
        var c: Case = .{
            .rows = rows,
            .experts = experts,
            .top_k = top_k,
            .logits = try t.upload(l),
            .bias = try t.upload(b),
            .pick = undefined,
            .wts = undefined,
        };
        for (0..2) |i| {
            c.pick[i] = try t.alloc(rows * top_k * 4);
            c.wts[i] = try t.alloc(rows * top_k * 4);
        }
        return c;
    }

    fn deinit(c: *Case) void {
        c.logits.free();
        c.bias.free();
        for (0..2) |i| {
            c.pick[i].free();
            c.wts[i].free();
        }
    }

    /// Kernel `which` (0 the rank pick, 1 the pass a pick at a time) over every row.
    fn run(c: *const Case, t: *Rig, which: usize, norm: bool) !void {
        const f = if (which == 0) &launch.select else &launch.selectPass;
        try f(
            &t.on,
            @ptrFromInt(at(c.logits)),
            @ptrFromInt(at(c.bias)),
            @ptrFromInt(at(c.pick[which])),
            @ptrFromInt(at(c.wts[which])),
            null,
            null,
            0,
            @intCast(c.rows),
            @intCast(c.experts),
            @intCast(c.top_k),
            2.5,
            norm,
            t.stream.handle,
        );
    }

    /// Whether both kernels wrote the same picks and weights, bit for bit.
    fn same(c: *const Case) !bool {
        var out: [2][]u32 = undefined;
        var w: [2][]u32 = undefined;
        for (0..2) |i| {
            out[i] = try rig.download(u32, c.pick[i]);
            w[i] = try rig.download(u32, c.wts[i]);
        }
        defer for (0..2) |i| {
            gpa.free(out[i]);
            gpa.free(w[i]);
        };
        return std.mem.eql(u32, out[0], out[1]) and std.mem.eql(u32, w[0], w[1]);
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

test "the 512-thread rank pick writes the bytes of the pass a pick at a time: random, tied and every row count" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9A1C };
    const shapes = [_][2]usize{ .{ 128, 6 }, .{ 128, 22 }, .{ 64, 6 }, .{ 256, 8 }, .{ 32, 8 }, .{ 1000, 32 } };
    var cases: usize = 0;
    for (shapes) |s| for ([_]usize{
        1,
        3,
        16,
        64,
    }) |rows| for ([_]u32{ 0, 3, 40 }) |levels| for ([_]bool{ true, false }) |norm| {
        var c = try Case.init(t, &rng, rows, s[0], s[1], levels);
        defer c.deinit();
        try c.run(t, 0, norm);
        try c.run(t, 1, norm);
        try t.stream.synchronize();
        try rig.expect(
            try c.same(),
            "E{d} top {d}, {d} rows, {d} levels, norm {}: the rank pick differs",
            .{ s[0], s[1], rows, levels, norm },
        );
        cases += 1;
    };
    std.debug.print("  nemotron rank pick: {d} cases, the same bytes as the pass a pick at a time\n", .{cases});
}

test "the rank pick's time against the pass a pick at a time, Nemotron's 128 experts and top 6" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x71AE };
    const launches = 2000;
    for ([_]usize{ 1, 4, 16, 64 }) |rows| {
        var c = try Case.init(t, &rng, rows, 128, 6, 0);
        defer c.deinit();
        var us: [2]f64 = undefined;
        for (0..2) |which| {
            for (0..20) |_| try c.run(t, which, true);
            try t.stream.synchronize();
            const t0 = nowNs();
            for (0..launches) |_| try c.run(t, which, true);
            try t.stream.synchronize();
            us[which] = @as(f64, @floatFromInt(nowNs() - t0)) / launches / 1e3;
        }
        std.debug.print(
            "  nemotron pick, {d} rows: rank {d:.2} us a launch, pass {d:.2} us (back to back)\n",
            .{ rows, us[0], us[1] },
        );
    }
}
