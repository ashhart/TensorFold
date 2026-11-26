//! Native MTP pick scheduling against the original host scan, without a device.
const std = @import("std");
const glm = @import("glm");
const loop = glm.mtp_loop;
const t = std.testing;
const max_depth = 15;
extern fn glm_mtp_pick_host([*]const u32, u32) u32;
extern fn glm_mtp_pick_control_host([*]const u32, u32, u32) u32;
extern fn glm_mtp_pick_idle_host([*]const u32, u32) u32;

const Ops = struct {
    enabled: bool = true,
    reads: usize = 0,
    uploads: usize = 0,
    forwards: usize = 0,
    queued: usize = 0,
    device_id: u32 = 0,
    previous: u32 = 17,
    mtp_len: usize = 101,
    in_head: usize = 0,
    fail: ?usize = null,
    finish_fail: bool = false,
    aborted: usize = 0,
    saved: [max_depth][2][4]u32 = undefined,

    fn words(o: *const Ops, j: usize) [2][4]u32 {
        const x: f32 = @floatFromInt((j * 7 + o.previous) % 31);
        return .{ .{ @bitCast(x), @intCast(j * 43 + 2), @bitCast(@as(f32, 1.25)), 0 }, .{ @bitCast(x + (if (j % 2 == 0) @as(f32, 1) else 0)), @intCast(j * 43 + 1), @bitCast(@as(f32, 2.5)), 0 } };
    }

    pub fn chained(o: *const Ops, depth: usize) bool {
        return o.enabled and depth > 1;
    }

    pub fn pick(o: *Ops, j: usize, sums: bool, _: *[1]u32) !glm.mtp_chance.Pick {
        if (o.fail == j) return error.QueueLost;
        o.reads += 1;
        const w = o.words(j);
        return glm.mtp_chance.pick(&w, sums);
    }

    pub fn queue(o: *Ops, j: usize, sums: bool, more: bool) !void {
        _ = sums;
        if (o.fail == j) return error.QueueLost;
        o.saved[j] = o.words(j);
        o.queued += 1;
        if (more) o.device_id = glm.mtp_chance.pick(&o.saved[j], false).token;
    }

    pub fn next(o: *Ops, token: u32, device: bool) !void {
        o.previous = if (device) o.device_id else token;
        o.uploads += @intFromBool(!device);
        o.forwards += 1;
        o.mtp_len += 1;
        o.in_head += 1;
    }

    pub fn finish(o: *Ops, depth: usize, out: []u32, chances: ?[]f64) !void {
        o.reads += 1;
        if (o.finish_fail) return error.GatherFailed;
        for (0..depth) |j| {
            const p = glm.mtp_chance.pick(&o.saved[j], chances != null);
            out[j] = p.token;
            if (chances) |c| c[j] = p.chance;
        }
    }
    pub fn abort(o: *Ops) void {
        o.aborted += 1;
    }
};

test "GLM MTP chain: one final read, no intermediate host ID, same IDs, chance bits and state" {
    for ([_]usize{ 2, 3, 7, max_depth }) |depth| for ([_]bool{ false, true }) |sums| {
        var old: Ops = .{ .enabled = false };
        var fast: Ops = .{};
        var want: [max_depth]u32 = @splat(0xdeadbeef);
        var got = want;
        var wc: [max_depth]f64 = @splat(-1);
        var gc = wc;
        try loop.run(&old, depth, &want, if (sums) &wc else null);
        try loop.run(&fast, depth, &got, if (sums) &gc else null);
        try t.expectEqualSlices(u32, &want, &got);
        try t.expectEqualSlices(u8, std.mem.asBytes(&wc), std.mem.asBytes(&gc));
        try t.expectEqual(old.previous, fast.previous);
        try t.expectEqual(old.mtp_len, fast.mtp_len);
        try t.expectEqual(old.in_head, fast.in_head);
        try t.expectEqual(depth - 1, fast.forwards);
        try t.expectEqual(depth, fast.queued);
        try t.expectEqual(@as(usize, 1), fast.reads);
        try t.expectEqual(@as(usize, 0), fast.uploads);
        try t.expectEqual(depth, old.reads);
        try t.expectEqual(depth - 1, old.uploads);
    };
}

test "GLM MTP chain: zero, one, disabled and back-to-back slots preserve the old path" {
    for ([_]usize{ 0, 1, 7 }) |depth| {
        var o: Ops = .{ .enabled = depth != 7 };
        var out: [max_depth]u32 = @splat(77);
        try loop.run(&o, depth, &out, null);
        try t.expectEqual(depth, o.reads);
        try t.expectEqual(@as(usize, 0), o.queued);
        try t.expectEqual(depth -| 1, o.uploads);
    }
    for (0..4) |slot| {
        var old: Ops = .{ .enabled = false, .previous = @intCast(17 + slot * 91), .mtp_len = 100 + slot };
        var fast = old;
        fast.enabled = true;
        var a: [max_depth]u32 = @splat(0);
        var b = a;
        try loop.run(&old, max_depth, &a, null);
        try loop.run(&fast, max_depth, &b, null);
        try t.expectEqualSlices(u32, &a, &b);
        try t.expectEqual(old.mtp_len, fast.mtp_len);
    }
}

test "GLM MTP chain: queue and gather failures stop before publishing draft outputs" {
    for (0..max_depth) |at| {
        var o: Ops = .{ .fail = at };
        var out: [max_depth]u32 = @splat(77);
        try t.expectError(error.QueueLost, loop.run(&o, max_depth, &out, null));
        try t.expectEqualSlices(u32, &@as([max_depth]u32, @splat(77)), &out);
        try t.expectEqual(at, o.queued);
        try t.expectEqual(at, o.forwards);
        try t.expectEqual(@as(usize, 0), o.reads);
        try t.expectEqual(@as(usize, 1), o.aborted);
    }
    var o: Ops = .{ .finish_fail = true };
    var out: [max_depth]u32 = @splat(77);
    try t.expectError(error.GatherFailed, loop.run(&o, max_depth, &out, null));
    try t.expectEqual(@as(usize, 1), o.aborted);
    try t.expectEqualSlices(u32, &@as([max_depth]u32, @splat(77)), &out);
}

test "GLM MTP chain: actual kernel body equals the host scan at exceptional values and random rank words" {
    const special = [_]u32{ 0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x3f800000, 0xbf800000, 0x7f800000, 0xff800000, 0x7fc12345, 0xffcabcde, 0x7f800001 };
    for (special) |a| for (special) |b| {
        const words = [_][4]u32{ .{ a, 17, 0, 0xabcdef01 }, .{ b, 3, 0, 0xabcdef02 } };
        const copy = words;
        try t.expectEqual(glm.mtp_chance.pick(&words, false).token, glm_mtp_pick_host(@ptrCast(&words), 2));
        try t.expectEqual(@as(u32, 0xfeedface), glm_mtp_pick_idle_host(@ptrCast(&words), 2));
        try t.expectEqualSlices(u8, std.mem.asBytes(&copy), std.mem.asBytes(&words));
    };
    var prng = std.Random.DefaultPrng.init(0xb63a24);
    const rng = prng.random();
    var words: [16][4]u32 = undefined;
    for (1..17) |world| for (0..4096) |_| {
        for (&words) |*w| {
            for (w) |*v| v.* = rng.int(u32);
        }
        try t.expectEqual(glm.mtp_chance.pick(words[0..world], false).token, glm_mtp_pick_host(@ptrCast(&words), @intCast(world)));
    };
    const tie = [_][4]u32{ .{ 0x3f800000, 17, 0, 0 }, .{ 0x3f800000, 3, 0, 0 } };
    const nan = [_][4]u32{ .{ 0x7fc12345, 17, 0, 0 }, .{ 0x7f800000, 3, 0, 0 } };
    try t.expect(glm_mtp_pick_host(@ptrCast(&tie), 2) != glm_mtp_pick_control_host(@ptrCast(&tie), 2, 1));
    try t.expect(glm_mtp_pick_host(@ptrCast(&nan), 2) != glm_mtp_pick_control_host(@ptrCast(&nan), 2, 2));
}

test "GLM MTP chain: production adapter records one final read at both ranks with exact IDs and chance bits" {
    const Fixture = @import("test_support/glm_mtp_chain_fixture.zig").Fixture;
    for (0..2) |rank| for ([_]usize{ 2, 7, 15 }) |depth| for ([_]bool{ false, true }) |sums| {
        var old: Fixture = .{};
        old.init(rank, false);
        defer old.deinit();
        var fast: Fixture = .{};
        fast.init(rank, true);
        defer fast.deinit();
        var want: [max_depth]u32 = @splat(77);
        var got = want;
        var wc: [max_depth]f64 = @splat(-1);
        var gc = wc;
        try old.run(depth, &want, if (sums) &wc else null);
        try fast.run(depth, &got, if (sums) &gc else null);
        try t.expectEqualSlices(u32, &want, &got);
        try t.expectEqualSlices(u8, std.mem.asBytes(&wc), std.mem.asBytes(&gc));
        try t.expectEqual(old.cur.in_head, fast.cur.in_head);
        try t.expectEqual(old.state.mtp_len, fast.state.mtp_len);
        try t.expectEqual(@as(usize, 1), fast.syncs);
        try t.expectEqual(@as(usize, 1), fast.reads);
        try t.expectEqual(@as(usize, 1), fast.checks);
        try t.expectEqual(depth, fast.gathers);
        try t.expectEqual(depth - 1, fast.picks);
        try t.expectEqual(@as(usize, 0), fast.uploads);
        try t.expectEqual(depth, old.reads);
    };
}

test "GLM MTP chain: admission and recording adapter fallback leave other modes on their old path" {
    const Fixture = @import("test_support/glm_mtp_chain_fixture.zig").Fixture;
    for (0..8) |mode| {
        var f: Fixture = .{};
        f.init(0, true);
        defer f.deinit();
        switch (mode) {
            0 => f.e.settings.mtp_chain = false,
            1 => f.e.cur.sampling = 9,
            2 => f.e.structured = true,
            3 => f.e.mtp_pick = null,
            4 => f.e.c.dsa = true,
            5 => f.e.local = &.{1},
            6 => f.e.share.world = 1,
            7 => f.e.pairs.len = 32,
            else => unreachable,
        }
        var out: [max_depth]u32 = @splat(77);
        try f.run(7, &out, null);
        try t.expectEqual(@as(usize, 7), f.reads);
        try t.expectEqual(@as(usize, 6), f.uploads);
        try t.expectEqual(@as(usize, 0), f.gathers);
    }
    try t.expect(!loop.eligible(true, 3, 0, false, false, false, 7));
    try t.expect(!loop.eligible(true, 2, 0, false, false, false, max_depth + 1));
}

test "GLM MTP chain: gather or download refusal publishes no output and exact scratch extent fits" {
    const Fixture = @import("test_support/glm_mtp_chain_fixture.zig").Fixture;
    for ([_]bool{ false, true }) |read| {
        var f: Fixture = .{};
        f.init(0, true);
        defer f.deinit();
        f.fail_check = !read;
        f.fail_read = read;
        var out: [max_depth]u32 = @splat(77);
        if (f.run(max_depth, &out, null)) |_| return error.TestUnexpectedResult else |_| {}
        try t.expectEqualSlices(u32, &@as([max_depth]u32, @splat(77)), &out);
        try t.expectEqual(@as(usize, @intFromBool(read)), f.reads);
    }
    var f: Fixture = .{};
    f.init(1, true);
    defer f.deinit();
    f.e.pairs.len = max_depth * 2 * 16;
    var out: [max_depth]u32 = @splat(77);
    try f.run(max_depth, &out, null);
    try t.expectEqual(@as(usize, 1), f.reads);
}

test "GLM MTP chain: the same final read also uses the discrete-device download path" {
    const Fixture = @import("test_support/glm_mtp_chain_fixture.zig").Fixture;
    var f: Fixture = .{ .integrated = false };
    f.init(0, true);
    defer f.deinit();
    var out: [max_depth]u32 = @splat(77);
    try f.run(max_depth, &out, null);
    try t.expectEqual(@as(usize, 1), f.reads);
}

test "GLM MTP chain: registered switch defaults off, refuses bad forms and disagrees across ranks" {
    const Lookup = struct {
        var text: ?[]const u8 = null;
        fn get(name: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_GLM_MTP_CHAIN")) text else null;
        }
    };
    var problem: []const u8 = "";
    Lookup.text = null;
    try t.expect(!(try glm.ranks.Settings.read(Lookup.get, &problem)).mtp_chain);
    for ([_][]const u8{ "0", "1", "  " }) |text| {
        Lookup.text = text;
        try t.expectEqual(std.mem.eql(u8, text, "1"), (try glm.ranks.Settings.read(Lookup.get, &problem)).mtp_chain);
    }
    for ([_][]const u8{ "true", "2", "-1" }) |text| {
        Lookup.text = text;
        try t.expectError(error.BadSetting, glm.ranks.Settings.read(Lookup.get, &problem));
        try t.expect(std.mem.indexOf(u8, problem, "TF_GLM_MTP_CHAIN") != null);
    }
    var all = [_]glm.ranks.Fingerprint{
        glm.ranks.Fingerprint.of(2, 0, .{}, 4096, true, @splat(0), @splat(1), @splat(2)),
        glm.ranks.Fingerprint.of(2, 1, .{}, 4096, true, @splat(0), @splat(1), @splat(2)),
    };
    try t.expectEqual(@as(?glm.ranks.Mismatch, null), glm.ranks.mismatch(&all));
    all[1].mtp_chain = 1;
    try t.expectEqual(glm.ranks.Mismatch{ .field = .mtp_chain, .rank = 1 }, glm.ranks.mismatch(&all).?);
}
