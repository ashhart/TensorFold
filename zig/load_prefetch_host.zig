//! Real family reader adapters over small owned files; no CUDA driver or GPU calls.
const std = @import("std");
const core = @import("core");
const ahead = core.checkpoint_read_ahead;
const t = std.testing;
const Family = enum { nemotron, bolt, glm, minimax, kolibri, spark, deepseek };
const families = std.enums.values(Family);
const ds = @import("src/families/deepseek_v41/cuda_load_prefetch.zig");
const Pack = @import("src/families/deepseek_v41/pack.zig").Pack;
const Config = @import("src/families/deepseek_v41/config.zig").Config;
const cfg: Config = .{ .layers = 2 };
const Fixture = struct {
    tmp: t.TmpDir,
    dir: [:0]u8,
    ck: core.Checkpoint,
    pack: Pack,
    fn deinit(f: *Fixture) void {
        f.pack.deinit();
        f.ck.close();
        t.allocator.free(f.dir);
        f.tmp.cleanup();
    }
};
fn fixture(family: Family) !Fixture {
    var tmp = t.tmpDir(.{});
    errdefer tmp.cleanup();
    const prefix = switch (family) {
        .nemotron => "backbone.layers.",
        .bolt => "language_model.backbone.layers.",
        .glm => "model.language_model.layers.",
        .minimax => "language_model.model.layers.",
        .kolibri, .spark => "model.layers.",
        .deepseek => "layers.",
    };
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(t.allocator);
    try header.append(t.allocator, '{');
    for ([_]u32{ 0, 1, 2, 10, 1 }, 0..) |layer, i| {
        const pre = if (family == .deepseek and layer == 2) "mtp." else prefix;
        const index = if (family == .deepseek and layer == 2) 0 else layer;
        try header.print(t.allocator, "{s}\"{s}{d}.{s}\":{{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[{d},{d}]}}", .{ if (i == 0) "" else ",", pre, index, if (i == 4) "b" else "a", i * 4, i * 4 + 4 });
    }
    try header.append(t.allocator, '}');
    const bytes = try t.allocator.alloc(u8, 8 + header.items.len + 20);
    defer t.allocator.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header.items.len, .little);
    @memcpy(bytes[8..][0..header.items.len], header.items);
    @memset(bytes[8 + header.items.len ..], 7);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = bytes });
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    errdefer t.allocator.free(dir);
    var ck = try core.Checkpoint.openModel(t.allocator, t.io, dir);
    errdefer ck.close();
    const pack = try Pack.open(t.allocator, t.io, dir);
    return .{ .tmp = tmp, .dir = dir, .ck = ck, .pack = pack };
}
fn begin(family: Family, f: *Fixture, layer: usize, count: usize) !ahead.Scope {
    return switch (family) {
        .nemotron, .bolt => @import("src/families/nemotron/cuda_load_prefetch.zig").begin(&f.ck, if (family == .bolt) "language_model." else "", layer, count),
        .glm => @import("src/families/glm/cuda_load_prefetch.zig").begin(&f.ck, layer, count),
        .minimax => @import("src/families/minimax_m3/cuda_load_prefetch.zig").begin(&f.ck, layer, count),
        .kolibri => @import("src/families/kolibri1/cuda_load_prefetch.zig").begin(&f.ck, layer, count),
        .spark => @import("src/families/spark25/cuda_load_prefetch.zig").begin(&f.ck, layer, count),
        .deepseek => ds.begin(f.ck.gpa, t.io, &f.pack, &cfg, @intCast(layer), count),
    };
}

test "load prefetch: every family's disabled adapter dereferences no source and starts no reader" {
    for (families) |family| {
        var f: Fixture = undefined;
        var reading = try begin(family, &f, 0, 0);
        defer reading.deinit();
        try t.expect(reading.reader.work == null);
        try t.expectEqual(@as(u64, 0), try reading.check());
    }
}

test "load prefetch: actual family adapters isolate layer prefixes and join before drop" {
    for (families) |family| {
        var f = try fixture(family);
        defer f.deinit();
        const before = try f.tmp.dir.readFileAlloc(t.io, "model.safetensors", t.allocator, .limited(1 << 20));
        defer t.allocator.free(before);
        for (0..3) |layer| {
            var reading = try begin(family, &f, layer, 3);
            defer reading.deinit();
            try t.expectEqual(@as(u64, if (layer == 1) 8 else 4), try reading.check());
            try t.expectEqual(@as(usize, if (layer == 1) 2 else 1), reading.reader.work.?.completed.load(.monotonic));
            for (reading.reader.threads) |thread| try t.expect(thread == null);
        }
        const after = try f.tmp.dir.readFileAlloc(t.io, "model.safetensors", t.allocator, .limited(1 << 20));
        defer t.allocator.free(after);
        try t.expectEqualSlices(u8, before, after);
    }
}

test "load prefetch: each family's failed reader names its source and frees the scope" {
    for (families) |family| {
        var f = try fixture(family);
        defer f.deinit();
        const original = f.ck.files.items[0].path;
        f.ck.files.items[0].path = "missing-load-prefetch-fixture.safetensors";
        defer f.ck.files.items[0].path = original;
        const dir = f.pack.dir;
        f.pack.dir = "missing-load-prefetch-fixture";
        defer f.pack.dir = dir;
        var reading = try begin(family, &f, 1, 3);
        defer reading.deinit();
        try t.expectError(error.LoadPrefetchRead, reading.check());
        const problem = reading.reader.problem().?;
        try t.expect(std.mem.indexOf(u8, std.mem.span(problem.path), "missing-load-prefetch-fixture") != null);
        try t.expectEqual(error.FileNotFound, problem.reason);
        for (reading.reader.threads) |thread| try t.expect(thread == null);
        reading.deinit();
        try t.expect(reading.reader.work == null and reading.arena == null);
    }
}

test "load prefetch: every family balances allocation failures while readers may be live" {
    const Control = struct {
        fn run(a: std.mem.Allocator, family: Family, input: *Fixture) !void {
            var f = input.*;
            f.ck.gpa = a;
            var reading = try begin(family, &f, 1, 3);
            defer reading.deinit();
            _ = try reading.check();
        }
    };
    for (families) |family| {
        var f = try fixture(family);
        defer f.deinit();
        try t.checkAllAllocationFailures(t.allocator, Control.run, .{ family, &f });
    }
}

test "load prefetch: malformed options and truncated data refuse without changing the source" {
    try t.expectEqual(@as(usize, 16), try ahead.readerCount(null));
    try t.expectEqual(@as(usize, 0), try ahead.readerCount("0"));
    for ([_][]const u8{ "17", "word", "-1" }) |bad| try t.expectError(error.BadLoadPrefetch, ahead.readerCount(bad));
    var f = try fixture(.spark);
    defer f.deinit();
    const spans = try t.allocator.alloc(ahead.Span, 1);
    spans[0] = .{ .path = f.ck.files.items[0].path.ptr, .lo = 1 << 20, .hi = (1 << 20) + 4 };
    var reader = try ahead.start(t.allocator, t.io, spans, 1);
    defer reader.deinit();
    try t.expectError(error.LoadPrefetchRead, reader.check());
    try t.expectEqual(error.EndOfStream, reader.problem().?.reason);
}

test "load prefetch: actual GLM loop follows a rank's layer order and stops on failed reads" {
    const Fake = struct {
        ck: *core.Checkpoint,
        w: struct { config: struct { layers: u32 = 3 } = .{} } = .{},
        calls: [3]u32 = @splat(99),
        n: usize = 0,
        pub fn layer(f: *@This(), i: u32) !u32 {
            f.calls[f.n] = i;
            f.n += 1;
            return i;
        }
    };
    const production = @import("src/families/glm/cuda_load_ahead.zig");
    for ([_]usize{ 0, 3 }) |count| {
        var f = try fixture(.glm);
        defer f.deinit();
        var loader: Fake = .{ .ck = &f.ck };
        var out: [3]u32 = @splat(99);
        try production.layersWithCount(&loader, &.{ 2, 0, 1 }, &out, count);
        try t.expectEqualSlices(u32, &.{ 2, 0, 1 }, &loader.calls);
        try t.expectEqualSlices(u32, &loader.calls, &out);
        try t.expectEqual(@as(usize, 3), loader.n);
        if (count > 0) {
            const old = f.ck.files.items[0].path;
            f.ck.files.items[0].path = "missing-load-prefetch-fixture.safetensors";
            defer f.ck.files.items[0].path = old;
            loader.n = 0;
            try t.expectError(error.LoadPrefetchRead, production.layersWithCount(&loader, &.{ 2, 0, 1 }, &out, count));
            try t.expectEqual(@as(usize, 1), loader.n);
        }
    }
}

test "load prefetch: chunk edges read every byte once with several readers" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const size = (8 << 20) + 3;
    const bytes = try t.allocator.alloc(u8, size);
    defer t.allocator.free(bytes);
    @memset(bytes, 17);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "source", .data = bytes });
    const path = try tmp.dir.realPathFileAlloc(t.io, "source", t.allocator);
    defer t.allocator.free(path);
    const spans = try t.allocator.alloc(ahead.Span, 2);
    spans[0] = .{ .path = path.ptr, .lo = 0, .hi = 8 << 20 };
    spans[1] = .{ .path = path.ptr, .lo = 8 << 20, .hi = size };
    var reader = try ahead.start(t.allocator, t.io, spans, 16);
    defer reader.deinit();
    try t.expectEqual(@as(u64, size), try reader.check());
    try t.expectEqual(@as(usize, 2), reader.work.?.completed.load(.monotonic));
    const file = try tmp.dir.openFile(t.io, "source", .{});
    defer file.close(t.io);
    try t.expectEqual(@as(u64, size), try file.length(t.io));
    const after = try t.allocator.alloc(u8, size);
    defer t.allocator.free(after);
    try t.expectEqual(@as(usize, size), try file.readPositionalAll(t.io, after, 0));
    try t.expectEqualSlices(u8, bytes, after);
}
