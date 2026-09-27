//! Full-checkpoint cache property checks, independent of proposal acceptance rates.
const std = @import("std");
const mx = @import("mlx.zig");
const dense = @import("model.zig");
const nemotron = @import("nemotron.zig");
const flash = @import("flash.zig");
const equal = @import("sampling_checks.zig").equal;
fn Pass(comptime M: type) type {
    return if (M == dense.Model) dense.Pass else if (M == nemotron.Model) nemotron.Pass else flash.Pass;
}
fn forward(comptime M: type, m: *M, tokens: []const i32, parents: []const i32) !Pass(M) {
    return if (M == dense.Model) m.forward(tokens, parents) else m.forward(tokens);
}
fn commit(comptime M: type, m: *M, p: *Pass(M), rows: []const i32) !void {
    if (M == dense.Model) try m.commit(p, rows) else try m.commit(p, rows.len);
}
fn Snapshot(comptime M: type) type {
    return struct {
        const Self = @This();
        cache: @TypeOf(@as(M, undefined).cache) = @splat(.{}),
        position: i32 = 0,
        fn capture(m: *M) !Self {
            var out = Self{ .position = m.position };
            errdefer out.deinit();
            for (m.cache, &out.cache) |c, *saved| saved.* = try c.clone();
            return out;
        }
        fn deinit(s: *Self) void {
            for (&s.cache) |*c| c.deinit();
        }
        fn restore(s: *const Self, m: *M) !void {
            var next: @TypeOf(m.cache) = @splat(.{});
            errdefer for (&next) |*c| c.deinit();
            for (s.cache, &next) |c, *saved| saved.* = try c.clone();
            m.reset();
            m.cache = next;
            m.position = s.position;
        }
        fn compare(s: *const Self, m: *M, scope: *mx.Scope) !void {
            if (s.position != m.position) return error.CachePositionMismatch;
            for (s.cache, m.cache) |expected, actual| {
                inline for (.{ "a", "b", "raw", "pooled", "ple" }) |field| {
                    if (@hasField(@TypeOf(actual), field)) {
                        const a = @field(actual, field);
                        const b = @field(expected, field);
                        if ((a.ctx == null) != (b.ctx == null)) return error.CachePresenceMismatch;
                        if (a.ctx != null) try equal(scope, a, b);
                    }
                }
                if (@hasField(@TypeOf(actual), "offset")) {
                    if (actual.offset != expected.offset or !std.mem.eql(i32, &actual.history, &expected.history)) return error.CacheMetadataMismatch;
                }
            }
        }
    };
}
fn prefill(comptime M: type, m: *M, count: usize, random: std.Random) !void {
    var tokens: [128]i32 = undefined;
    var parents: [128]i32 = undefined;
    var rows: [128]i32 = undefined;
    var offset: usize = 0;
    while (offset < count) {
        const n = @min(if (M == dense.Model) @as(usize, 128) else 16, count - offset);
        for (0..n) |i| {
            tokens[i] = random.intRangeAtMost(i32, 100, 10000);
            parents[i] = @as(i32, @intCast(i)) - 1;
            rows[i] = @intCast(i);
        }
        var p = try forward(M, m, tokens[0..n], parents[0..n]);
        defer p.deinit();
        try commit(M, m, &p, rows[0..n]);
        offset += n;
    }
}
fn neverEos(_: i32) bool {
    return false;
}
pub fn checkSerial(comptime M: type, m: *M, long: bool) !void {
    const a = mx.allocator;
    var rng = std.Random.DefaultPrng.init(0x50495045);
    const prefixes: []const usize = if (long) &.{ 9999, 10007 } else &.{ 0, 31 };
    var checks: usize = 0;
    for (prefixes) |prefix| {
        m.reset();
        try prefill(M, m, prefix, rng.random());
        var base = try Snapshot(M).capture(m);
        defer base.deinit();
        for ([_]f64{ 0, 0.7 }) |temperature| for ([_]usize{ 1, 2, 17 }) |limit| {
            var scope = mx.Scope{};
            defer scope.deinit();
            const settings = @import("sampling.zig").Sampling{ .metal = true, .seed = 5678, .temperature = temperature, .top_k = 12, .top_p = 0.8 };
            try base.restore(m);
            var expected: std.ArrayList(u32) = .empty;
            defer expected.deinit(a);
            try expected.append(a, 42);
            while (expected.items.len < limit) {
                var p = try forward(M, m, &.{@intCast(expected.items[expected.items.len - 1])}, &.{-1});
                defer p.deinit();
                const ids = try @import("sampling.zig").rows(&m.kernels, &p.scope, p.logits, &.{m.position + 1}, settings);
                defer a.free(ids);
                try expected.append(a, @intCast(ids[0]));
                try commit(M, m, &p, &.{0});
            }
            var cache = try Snapshot(M).capture(m);
            defer cache.deinit();
            var reference = try forward(M, m, &.{97}, &.{-1});
            defer reference.deinit();
            try base.restore(m);
            var actual: std.ArrayList(u32) = .empty;
            defer actual.deinit(a);
            try actual.append(a, 42);
            const result = try @import("serial_pipeline.zig").generate(M, m, a, &actual, limit, settings, neverEos, null);
            try std.testing.expectEqualSlices(u32, expected.items, actual.items);
            try std.testing.expectEqual(limit - 1, result.rounds);
            try std.testing.expectEqual(limit -| 2, result.queued_ahead);
            try cache.compare(m, &scope);
            var continued = try forward(M, m, &.{97}, &.{-1});
            defer continued.deinit();
            try equal(&scope, reference.logits, continued.logits);
            checks += 1;
        };
        std.debug.print("PASS: serial pipeline prefix {d}, greedy/sampled, budgets 1/2/17; every token, cache and continuation exact\n", .{prefix});
    }
    m.reset();
    if (!long) {
        var baseline: usize = 0;
        var maximum: usize = 0;
        for (0..72) |cycle| {
            {
                var generated: std.ArrayList(u32) = .empty;
                defer generated.deinit(a);
                try generated.append(a, 42);
                _ = try @import("serial_pipeline.zig").generate(M, m, a, &generated, 5, .{ .metal = true, .temperature = 0 }, neverEos, null);
            }
            m.reset();
            try mx.check(mx.c.mlx_synchronize(mx.stream));
            var active: usize = 0;
            try mx.check(mx.c.mlx_get_active_memory(&active));
            if (cycle == 7) baseline = active;
            if (cycle >= 8) {
                maximum = @max(maximum, active);
                if (active > baseline + 8 * 1024 * 1024) return error.ActiveMemoryGrowth;
            }
        }
        std.debug.print("PASS: 64 serial pipeline/reset cycles; active MLX memory baseline={d}, max={d}\n", .{ baseline, maximum });
    }
    std.debug.print("PASS: {d} full-model serial pipeline/cache/continuation comparisons\n", .{checks});
}
pub fn check(comptime M: type, m: *M) !void {
    return checkPrefixes(M, m, if (M == dense.Model) &.{ 0, 1, 15, 16, 17, 63, 64, 127, 128, 511, 512, 513 } else &.{ 0, 1, 15, 16, 17, 33 }, true);
}
pub fn checkLong(comptime M: type, m: *M) !void {
    return checkPrefixes(M, m, if (M == flash.Model) &.{ 2044, 2051, 2063 } else &.{ 9999, 10007 }, false);
}
fn checkPrefixes(comptime M: type, m: *M, prefixes: []const usize, short: bool) !void {
    if (@hasDecl(M, "forwardArray")) {
        var scope = mx.Scope{};
        defer scope.deinit();
        try std.testing.expectError(error.InvalidToken, m.forwardArray(mx.empty));
        try std.testing.expectError(error.InvalidToken, m.forwardArray(try scope.scalar(42)));
        try std.testing.expectError(error.InvalidLaneWidth, m.forwardArray(try scope.zeros(&.{ 1, 1 }, mx.i32t)));
        try std.testing.expectError(error.InvalidLaneWidth, m.forwardArray(try scope.zeros(&.{0}, mx.i32t)));
        try std.testing.expectError(error.InvalidLaneWidth, m.forwardArray(try scope.zeros(&.{17}, mx.i32t)));
    }
    var rng = std.Random.DefaultPrng.init(0x4341434845);
    const random = rng.random();
    var tokens: [128]i32 = undefined;
    var parents: [128]i32 = undefined;
    var rows: [128]i32 = undefined;
    var checks: usize = 0;
    for (prefixes, 0..) |prefix, cycle| {
        var checked_row: usize = 0;
        var checked_stage: []const u8 = "prefill";
        errdefer std.debug.print("FAIL: prefix {d}, row {d}, stage {s}\n", .{ prefix, checked_row, checked_stage });
        m.reset();
        try prefill(M, m, prefix, random);
        var base = try Snapshot(M).capture(m);
        defer base.deinit();
        const width: usize = if (M == dense.Model and cycle == prefixes.len - 1) 128 else if (M == dense.Model and cycle % 2 == 1) 32 else 16;
        for (0..width) |i| {
            tokens[i] = random.intRangeAtMost(i32, 100, 10000);
            parents[i] = @as(i32, @intCast(i)) - 1;
            rows[i] = @intCast(i);
        }
        tokens[4] = if (M == nemotron.Model) 2 else 248044;
        tokens[11] = if (M == nemotron.Model) 11 else 248046;
        var input_scope = mx.Scope{};
        defer input_scope.deinit();
        var batch = if (@hasDecl(M, "forwardArray")) blk: {
            // Keep the IDs on the GPU, including rows beyond both EOS IDs. The
            // serial host-token path below checks every possible retained prefix.
            const ids = try input_scope.cast(try input_scope.ints(tokens[0..width]), mx.c.MLX_UINT32);
            const zero = try input_scope.cast(try input_scope.ints(&.{0}), mx.c.MLX_UINT32);
            break :blk try m.forwardArray(try input_scope.binary(mx.c.mlx_add, ids, zero));
        } else try forward(M, m, tokens[0..width], parents[0..width]);
        defer batch.deinit();
        if (M == dense.Model) {
            try std.testing.expectError(error.EmptyCommit, m.commit(&batch, &.{}));
            try std.testing.expectError(error.InvalidCommit, m.commit(&batch, &.{1}));
            try std.testing.expectError(error.InvalidCommit, m.commit(&batch, &.{ 0, @intCast(width) }));
            var scope = mx.Scope{};
            defer scope.deinit();
            try base.compare(m, &scope);
        } else {
            try std.testing.expectError(error.InvalidCommit, m.commit(&batch, 0));
            try std.testing.expectError(error.InvalidCommit, m.commit(&batch, width + 1));
        }
        for (0..width) |j| {
            checked_row = j;
            checked_stage = "verified logits";
            var scope = mx.Scope{};
            defer scope.deinit();
            var serial = try forward(M, m, tokens[j..][0..1], &.{-1});
            defer serial.deinit();
            const axis: usize = if (M == dense.Model) 1 else 0;
            try equal(&scope, serial.logits, try scope.slice(batch.logits, axis, @intCast(j), @intCast(j + 1)));
            try commit(M, m, &serial, &.{0});
            var expected = try Snapshot(M).capture(m);
            defer expected.deinit();
            var serial_next = try forward(M, m, &.{97}, &.{-1});
            defer serial_next.deinit();
            try base.restore(m);
            try commit(M, m, &batch, rows[0 .. j + 1]);
            checked_stage = "committed cache";
            try expected.compare(m, &scope);
            checked_stage = "continuation logits";
            var continued = try forward(M, m, &.{97}, &.{-1});
            defer continued.deinit();
            try equal(&scope, serial_next.logits, continued.logits);
            try expected.restore(m);
            checks += 1;
        }
        // Dropping an entire speculative pass must leave the committed cache intact.
        var before = try Snapshot(M).capture(m);
        defer before.deinit();
        for (0..3) |_| {
            var rejected = try forward(M, m, tokens[0..16], parents[0..16]);
            rejected.deinit();
        }
        var scope = mx.Scope{};
        defer scope.deinit();
        try before.compare(m, &scope);
        std.debug.print("PASS: cache cycle {d}, prefix {d}, every acceptance length 1..{d}, EOS history, rejection and continuation\n", .{ cycle, prefix, width });
    }
    if (M == dense.Model and short) try trees(m, random);
    m.reset();
    if (m.position != 0) return error.CacheResetMismatch;
    for (m.cache) |c| if (c.a.ctx != null or c.b.ctx != null) return error.CacheResetMismatch;
    std.debug.print("PASS: {d} accepted-prefix cache/continuation checks across {d} resets\n", .{ checks, prefixes.len });
    if (short) try memory(M, m);
}
fn memory(comptime M: type, m: *M) !void {
    var baseline: usize = 0;
    var maximum: usize = 0;
    for (0..136) |cycle| {
        {
            var p = try forward(M, m, &.{ 42, 97, 100 }, &.{ -1, 0, 1 });
            defer p.deinit();
            try commit(M, m, &p, &.{ 0, 1 });
        }
        m.reset();
        try mx.check(mx.c.mlx_synchronize(mx.stream));
        var active: usize = 0;
        try mx.check(mx.c.mlx_get_active_memory(&active));
        if (cycle == 7) baseline = active;
        if (cycle >= 8) {
            maximum = @max(maximum, active);
            if (active > baseline + 8 * 1024 * 1024) return error.ActiveMemoryGrowth;
        }
    }
    std.debug.print("PASS: 128 post-warmup forward/partial-commit/reset cycles; active MLX memory baseline={d}, max={d} bytes\n", .{ baseline, maximum });
}
fn trees(m: *dense.Model, random: std.Random) !void {
    var tokens: [32]i32 = undefined;
    var parents: [32]i32 = undefined;
    for ([_]usize{ 2, 7, 15, 16, 17, 31, 32 }) |width| {
        m.reset();
        try prefill(dense.Model, m, 65, random);
        var base = try Snapshot(dense.Model).capture(m);
        defer base.deinit();
        for (0..width) |i| {
            tokens[i] = random.intRangeAtMost(i32, 100, 10000);
            parents[i] = if (i == 0) -1 else @intCast(random.uintLessThan(usize, i));
        }
        var batch = try m.forward(tokens[0..width], parents[0..width]);
        defer batch.deinit();
        const tree = try @import("lanes.zig").Tree.init(parents[0..width]);
        for (0..width) |leaf| {
            const path = tree.paths[leaf * 128 ..][0..@intCast(tree.depths[leaf] + 1)];
            try base.restore(m);
            for (path) |row| {
                var serial = try m.forward(tokens[@intCast(row)..][0..1], &.{-1});
                defer serial.deinit();
                try equal(&serial.scope, serial.logits, try serial.scope.slice(batch.logits, 1, row, row + 1));
                try m.commit(&serial, &.{0});
            }
            var expected = try Snapshot(dense.Model).capture(m);
            defer expected.deinit();
            try base.restore(m);
            try m.commit(&batch, path);
            var scope = mx.Scope{};
            defer scope.deinit();
            try expected.compare(m, &scope);
        }
        std.debug.print("PASS: random {d}-row tree, every leaf path and all cache arrays\n", .{width});
    }
}
