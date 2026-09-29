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
        rope_delta: i32 = 0,
        fn capture(m: *M) !Self {
            var out = Self{ .position = m.position };
            if (@hasField(M, "rope_delta")) out.rope_delta = m.rope_delta;
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
            if (@hasField(M, "rope_delta")) m.rope_delta = s.rope_delta;
        }
        fn compare(s: *const Self, m: *M, scope: *mx.Scope) !void {
            if (s.position != m.position) return error.CachePositionMismatch;
            if (@hasField(M, "rope_delta")) if (s.rope_delta != m.rope_delta) return error.CachePositionMismatch;
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
                    if (actual.offset != expected.offset) return error.CacheMetadataMismatch;
                    // Resident PLE keeps these same two token IDs on the GPU.
                    const ah = if (actual.token_history.ctx != null) actual.token_history else try scope.ints(&actual.history);
                    const eh = if (expected.token_history.ctx != null) expected.token_history else try scope.ints(&expected.history);
                    try equal(scope, ah, eh);
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
// Single-threaded diagnostic: treat a known first draw as the terminal marker.
// This forces a real Flash pass to be queued and discarded, independent of how
// likely the checkpoint is to emit its ordinary EOS IDs in a short fixture.
var terminal_marker: i32 = -1;
fn terminalEos(token: i32) bool {
    return token == terminal_marker;
}
fn gpuTokens(comptime M: type, m: *const M) bool {
    return if (@hasDecl(M, "gpuTokensEnabled")) m.gpuTokensEnabled() else true;
}
pub fn checkResident(m: *flash.Model, long: bool) !void {
    if (!m.gpuTokensEnabled()) return error.RequiresResidentPLE;
    const original = m.resident_ple;
    defer m.resident_ple = original;
    const prefixes: []const usize = if (long) &.{ 2044, 2051, 2063 } else &.{ 0, 31 };
    var rng = std.Random.DefaultPrng.init(0x504c455354415445);
    const ids = [_]i32{ 42, 97, 100, 103, 248044, 109, 112, 115, 118, 121, 124, 248046, 130, 133, 136, 139 };
    var checks: usize = 0;
    for (prefixes) |prefix| {
        var stage: []const u8 = "bounded prefill";
        var retained: usize = 0;
        errdefer {
            var peak: usize = 0;
            _ = mx.c.mlx_get_peak_memory(&peak);
            std.debug.print("Resident comparison failed: prefix {d}, position {d}, keep {d}, stage {s}, peak MLX bytes {d}\n", .{ prefix, m.position, retained, stage, peak });
        }
        const random_state = rng;
        m.reset();
        m.resident_ple = false;
        try prefill(flash.Model, m, prefix, rng.random());
        var bounded = try Snapshot(flash.Model).capture(m);
        defer bounded.deinit();
        var expected = try m.forward(&ids);
        defer expected.deinit();
        m.reset();
        m.resident_ple = true;
        rng = random_state;
        stage = "resident prefill";
        try prefill(flash.Model, m, prefix, rng.random());
        var resident = try Snapshot(flash.Model).capture(m);
        defer resident.deinit();
        var s = mx.Scope{};
        defer s.deinit();
        try bounded.compare(m, &s);
        stage = "resident verification";
        var actual = try m.forwardArray(try s.cast(try s.ints(&ids), mx.c.MLX_UINT32));
        defer actual.deinit();
        try equal(&s, expected.logits, actual.logits);
        for (0..17) |keep| {
            retained = keep;
            stage = "bounded continuation";
            var step = mx.Scope{};
            defer step.deinit();
            m.resident_ple = false;
            try bounded.restore(m);
            if (keep > 0) try m.commit(&expected, keep);
            var cache = try Snapshot(flash.Model).capture(m);
            defer cache.deinit();
            var reference = try m.forward(&.{97});
            defer reference.deinit();
            m.resident_ple = true;
            stage = "resident continuation";
            try resident.restore(m);
            if (keep > 0) try m.commit(&actual, keep);
            try cache.compare(m, &step);
            var continued = try m.forwardArray(try step.ints(&.{97}));
            defer continued.deinit();
            try equal(&step, reference.logits, continued.logits);
            checks += 1;
        }
        std.debug.print("PASS: resident/bounded Flash at {d} past: prefill cache, all logits, all 17 retained prefixes, EOS history and continuation exact\n", .{prefix});
    }
    m.reset();
    std.debug.print("PASS: {d} full-model resident/bounded PLE state comparisons\n", .{checks});
}
pub fn checkBufferReuse(comptime M: type, m: *M) !void {
    const kv = @import("kv_buffer.zig");
    if (!kv.enabled) return error.BuffersDisabled;
    kv.track_reuse = true;
    defer kv.track_reuse = false;
    var rng = std.Random.DefaultPrng.init(0x444f4e415445);
    m.reset();
    try prefill(M, m, 33, rng.random());
    kv.attempted = 0;
    kv.reused = 0;
    const pipeline = @hasDecl(M, "SerialPass") and gpuTokens(M, m);
    if (@hasDecl(M, "SerialPass")) {
        if (pipeline) {
            var generated: std.ArrayList(u32) = .empty;
            defer generated.deinit(mx.allocator);
            try generated.append(mx.allocator, 42);
            _ = try @import("serial_pipeline.zig").generate(M, m, mx.allocator, &generated, 33, .{ .metal = true, .temperature = 0 }, neverEos, null);
        }
    }
    if (!pipeline) {
        for (0..32) |_| {
            var p = try forward(M, m, &.{42}, &.{-1});
            defer p.deinit();
            try commit(M, m, &p, &.{0});
        }
    }
    if (kv.attempted == 0 or kv.reused != kv.attempted) {
        std.debug.print("Buffer donation: {d}/{d} exact allocations reused\n", .{ kv.reused, kv.attempted });
        return error.BufferWasNotReused;
    }
    std.debug.print("PASS: {d}/{d} real-model attention writes reused their exact donor allocations ({s})\n", .{ kv.reused, kv.attempted, if (pipeline) "pipelined" else "synchronous" });
    m.reset();
}
pub fn checkBuffered(comptime M: type, m: *M, long: bool) !void {
    const kv = @import("kv_buffer.zig");
    const original = kv.enabled;
    defer kv.enabled = original;
    var rng = std.Random.DefaultPrng.init(0x425546464552);
    const prefixes: []const usize = if (long) (if (M == flash.Model) &.{ 2044, 2051, 2063 } else &.{ 9999, 10007 }) else &.{ 0, 31, 2044 };
    var checks: usize = 0;
    for (prefixes) |prefix| {
        const state = rng;
        m.reset();
        kv.enabled = false;
        try prefill(M, m, prefix, rng.random());
        var expected_cache = try Snapshot(M).capture(m);
        defer expected_cache.deinit();
        var expected = try forward(M, m, &.{ 42, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139 }, &.{ -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 });
        defer expected.deinit();
        // Compare separately-prefilled buffer layouts, not only two schedules
        // that both use the new buffers and might share a rounding difference.
        m.reset();
        kv.enabled = true;
        rng = state;
        try prefill(M, m, prefix, rng.random());
        var scope = mx.Scope{};
        defer scope.deinit();
        try expected_cache.compare(m, &scope);
        var actual = try forward(M, m, &.{ 42, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127, 130, 133, 136, 139 }, &.{ -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 });
        defer actual.deinit();
        try equal(&scope, expected.logits, actual.logits);
        checks += 1;
        std.debug.print("PASS: buffered/unbuffered prefill caches and every verification logit at prefix {d}\n", .{prefix});
    }
    m.reset();
    std.debug.print("PASS: {d} independently-prefilled buffered/unbuffered model comparisons\n", .{checks});
}
pub fn checkSerial(comptime M: type, m: *M, long: bool) !void {
    const a = mx.allocator;
    var rng = std.Random.DefaultPrng.init(0x50495045);
    const prefixes: []const usize = if (long) (if (M == flash.Model) &.{ 2044, 2051, 2063 } else &.{ 9999, 10007 }) else &.{ 0, 31 };
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
            var first_cache: ?Snapshot(M) = null;
            defer if (first_cache) |*cache| cache.deinit();
            try expected.append(a, 42);
            while (expected.items.len < limit) {
                var p = try forward(M, m, &.{@intCast(expected.items[expected.items.len - 1])}, &.{-1});
                defer p.deinit();
                const ids = try @import("sampling.zig").rows(&m.kernels, &p.scope, p.logits, &.{m.position + 1}, settings);
                defer a.free(ids);
                try expected.append(a, @intCast(ids[0]));
                try commit(M, m, &p, &.{0});
                if (M == flash.Model and limit == 17 and expected.items.len == 2) first_cache = try Snapshot(M).capture(m);
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
            if (M == flash.Model and limit == 17) {
                terminal_marker = @intCast(expected.items[1]);
                try std.testing.expect(terminal_marker != 42);
                try first_cache.?.restore(m);
                var terminal_reference = try forward(M, m, &.{97}, &.{-1});
                defer terminal_reference.deinit();
                try base.restore(m);
                actual.clearRetainingCapacity();
                try actual.append(a, 42);
                const terminal = try @import("serial_pipeline.zig").generate(M, m, a, &actual, limit, settings, terminalEos, null);
                try std.testing.expectEqualSlices(u32, expected.items[0..2], actual.items);
                try std.testing.expectEqual(@as(usize, 1), terminal.rounds);
                try std.testing.expectEqual(@as(usize, 1), terminal.queued_ahead);
                try first_cache.?.compare(m, &scope);
                var terminal_next = try forward(M, m, &.{97}, &.{-1});
                defer terminal_next.deinit();
                try equal(&scope, terminal_reference.logits, terminal_next.logits);
                checks += 1;
            }
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
        if (!gpuTokens(M, m)) try std.testing.expectError(error.RequiresResidentPLE, m.forwardArray(try scope.ints(&.{42})));
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
            if (!gpuTokens(M, m)) break :blk try forward(M, m, tokens[0..width], parents[0..width]);
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
