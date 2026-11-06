//! Arithmetic identity regressions over the real CUDA identity path and a recording driver, with no GPU calls.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const disk = @import("cuda_disk.zig");
const identity = @import("disk_identity.zig");
const t = std.testing;

const Recording = struct {
    var calls: [16]enum { name, major, minor, version } = undefined;
    var count: usize = 0;
    var value: ?[:0]const u8 = null;

    fn name(out: [*]u8, size: c_int, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        std.debug.assert(size >= 5);
        @memcpy(out[0..5], "fake\x00");
        calls[count] = .name;
        count += 1;
        return 0;
    }
    fn attribute(out: *c_int, which: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        calls[count] = if (which == .compute_capability_major) .major else .minor;
        count += 1;
        out.* = if (which == .compute_capability_major) 12 else 1;
        return 0;
    }
    fn version(out: *c_int) callconv(.c) cuda.abi.Result {
        calls[count] = .version;
        count += 1;
        out.* = 13000;
        return 0;
    }
    fn environment(key: [:0]const u8) ?[:0]const u8 {
        std.debug.assert(std.mem.eql(u8, key, "TF_QWEN_KV_PAGES_ALL"));
        return value;
    }
};

// Main has no arithmetic row yet; this reclassified fake row stands in for a future registered arithmetic switch.
const arithmetic: api.profile.Entry = .{ .id = .TF_QWEN_KV_PAGES_ALL, .families = &.{"nemotron_h"}, .class = .arithmetic, .syntax = .one, .effect = "test arithmetic", .evidence = "test fixture" };

test "disk identity: an arithmetic value changes actual CUDA identity and zero equals unset" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{}" });
    var driver: cuda.Driver = undefined;
    driver.api.cuDeviceGetName = Recording.name;
    driver.api.cuDeviceGetAttribute = Recording.attribute;
    driver.api.cuDriverGetVersion = Recording.version;
    const gpu: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    const profile: identity.Profile = .{ .family = "nemotron_h", .backend = "cuda", .entries = &.{arithmetic}, .environment = Recording.environment };
    Recording.count = 0;
    Recording.value = null;
    const unset = try disk.identityOf(a, t.io, &gpu, dir, null, &.{1}, 42, profile);
    Recording.value = "1";
    const enabled = try disk.identityOf(a, t.io, &gpu, dir, null, &.{1}, 42, profile);
    Recording.value = "0";
    const disabled = try disk.identityOf(a, t.io, &gpu, dir, null, &.{1}, 42, profile);
    try t.expectEqual(@as(usize, 12), Recording.count);
    for (0..3) |i| try t.expectEqualSlices(@TypeOf(Recording.calls[0]), &.{ .name, .major, .minor, .version }, Recording.calls[i * 4 ..][0..4]);
    try t.expectEqual(unset, disabled);
    try t.expect(unset != enabled);
    const root = try std.fs.path.join(a, &.{ dir, "states" });
    const Imprint = api.prompt_imprint.Imprint;
    const key = Imprint.keyOf(&.{ 1, 2, 3 });
    {
        var saved = try Imprint.open(a, root, unset, 1 << 20);
        defer saved.deinit();
        try saved.add(key, 2, &.{ 1, 2, 3 }, &.{}, 8);
    }
    var same = try Imprint.open(a, root, disabled, 1 << 20);
    defer same.deinit();
    try t.expect(same.has(key));
    var stale = try Imprint.open(a, root, enabled, 1 << 20);
    defer stale.deinit();
    try t.expect(!stale.has(key));
    try t.expect(stale.best(&.{ 1, 2, 3 }, &.{}, false, 0) == null);
    try t.expect(same.has(key));
}

const Values = struct {
    const Pair = struct { api.profile.Switch, ?[:0]const u8 };
    var values: []const Pair = &.{};
    var lookups: usize = 0;
    fn environment(name: [:0]const u8) ?[:0]const u8 {
        lookups += 1;
        for (values) |pair| if (std.mem.eql(u8, @tagName(pair[0]), name)) return pair[1];
        return null;
    }
};

fn profileHash(entries: []const api.profile.Entry, family: []const u8, backend: []const u8, values: []const Values.Pair) u64 {
    Values.values = values;
    Values.lookups = 0;
    var h: std.hash.Wyhash = .init(7);
    identity.profile(&h, .{ .family = family, .backend = backend, .entries = entries, .environment = Values.environment });
    return h.final();
}

test "disk identity: every non-identical class is scoped by family and backend" {
    for ([_]api.profile.Class{ .arithmetic, .experimental, .debug, .identical }) |class| {
        var entry = arithmetic;
        entry.class = class;
        const entries = [_]api.profile.Entry{entry};
        const unset = profileHash(&entries, "nemotron_h", "cuda", &.{});
        const zero = profileHash(&entries, "nemotron_h", "cuda", &.{.{ entry.id, "0" }});
        const one = profileHash(&entries, "nemotron_h", "cuda", &.{.{ entry.id, "1" }});
        try t.expectEqual(unset, zero);
        try t.expectEqual(class != .identical, unset != one);
        if (class == .identical) try t.expectEqual(@as(usize, 0), Values.lookups);
        for ([_][2][]const u8{ .{ "qwen3_5", "cuda" }, .{ "nemotron_h", "metal" } }) |scope| {
            const off_scope = profileHash(&entries, scope[0], scope[1], &.{});
            try t.expectEqual(off_scope, profileHash(&entries, scope[0], scope[1], &.{.{ entry.id, "1" }}));
            try t.expectEqual(@as(usize, 0), Values.lookups);
            try t.expectEqual(off_scope, profileHash(&.{}, scope[0], scope[1], &.{}));
        }
    }
}

test "disk identity: every applicable row contributes, including later registry rows" {
    var entries = [_]api.profile.Entry{ arithmetic, arithmetic, arithmetic };
    entries[1].id = .TF_QWEN_KV_SHARE;
    entries[2].id = .TF_KV_SHARE;
    const unset = profileHash(&entries, "nemotron_h", "cuda", &.{});
    for (entries) |entry| {
        try t.expect(unset != profileHash(&entries, "nemotron_h", "cuda", &.{.{ entry.id, "1" }}));
        try t.expectEqual(@as(usize, 3), Values.lookups);
    }
}

test "disk identity: defaults resolve before counts, booleans and auto values are hashed" {
    var entry = arithmetic;
    entry.syntax = .count;
    entry.stable = "4";
    try t.expectEqual(profileHash(&.{entry}, "nemotron_h", "cuda", &.{}), profileHash(&.{entry}, "nemotron_h", "cuda", &.{.{ entry.id, "0004" }}));
    try t.expect(profileHash(&.{entry}, "nemotron_h", "cuda", &.{}) != profileHash(&.{entry}, "nemotron_h", "cuda", &.{.{ entry.id, "0" }}));
    try t.expect(profileHash(&.{entry}, "nemotron_h", "cuda", &.{.{ entry.id, "broken" }}) != profileHash(&.{entry}, "nemotron_h", "cuda", &.{.{ entry.id, "0" }}));
    entry.syntax = .one;
    entry.stable = "1";
    try t.expectEqual(profileHash(&.{entry}, "nemotron_h", "cuda", &.{}), profileHash(&.{entry}, "nemotron_h", "cuda", &.{.{ entry.id, "1" }}));
    entry.syntax = .auto;
    entry.stable = null;
    try t.expectEqual(profileHash(&.{entry}, "nemotron_h", "cuda", &.{}), profileHash(&.{entry}, "nemotron_h", "cuda", &.{.{ entry.id, "auto" }}));
    entry.report_values = &.{ "0", "all" };
    try t.expect(profileHash(&.{entry}, "nemotron_h", "cuda", &.{}) != profileHash(&.{entry}, "nemotron_h", "cuda", &.{.{ entry.id, "0" }}));
}

test "disk identity: the shared helper also binds applicable Metal profile values" {
    var entry = arithmetic;
    entry.backend = "metal";
    const unset = profileHash(&.{entry}, "nemotron_h", "metal", &.{});
    try t.expect(unset != profileHash(&.{entry}, "nemotron_h", "metal", &.{.{ entry.id, "1" }}));
    try t.expectEqual(@as(usize, 1), Values.lookups);
}

test "disk identity: explicit values hash their exact bytes for DeepSeek alone" {
    const entries = api.profile.entries;
    var legacy: std.ArrayList(api.profile.Entry) = .empty;
    defer legacy.deinit(t.allocator);
    for (entries) |e| if (e.syntax != .explicit) try legacy.append(t.allocator, e);
    for (entries) |e| {
        if (e.syntax != .explicit) continue;
        const empty = profileHash(&entries, "deepseek_v41", "cuda", &.{});
        const zero = profileHash(&entries, "deepseek_v41", "cuda", &.{.{ e.id, "0" }});
        const one = profileHash(&entries, "deepseek_v41", "cuda", &.{.{ e.id, "1" }});
        try t.expect(empty != zero and zero != one);
        try t.expect(one != profileHash(&entries, "deepseek_v41", "cuda", &.{.{ e.id, "01" }}));
        for (entries) |other| for (other.families) |family| {
            if (std.mem.eql(u8, family, "deepseek_v41")) continue;
            for ([_][]const u8{ "cuda", "metal", "hip" }) |backend| {
                const before = profileHash(legacy.items, family, backend, &.{});
                try t.expectEqual(before, profileHash(&entries, family, backend, &.{}));
                try t.expectEqual(before, profileHash(&entries, family, backend, &.{.{ e.id, "1" }}));
            }
        };
    }
}
