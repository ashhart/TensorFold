//! Compare batched early MTP with independent one-row replay, every retained
//! prefix and its continuation, including MTP's own sparse/10K attention branches.
const std = @import("std");
const mx = @import("mlx.zig");
const equal = @import("sampling_checks.zig").equal;

fn caches(comptime C: type, s: *mx.Scope, a: C, b: C) !void {
    inline for (.{ "a", "b", "raw", "pooled", "ple" }) |field| if (@hasField(C, field)) {
        const x = @field(a, field);
        const y = @field(b, field);
        if ((x.ctx == null) != (y.ctx == null)) return error.CachePresenceMismatch;
        if (x.ctx != null) try equal(s, x, y);
    };
    if (@hasField(C, "offset")) {
        if (a.offset != b.offset or !std.mem.eql(i32, &a.history, &b.history)) return error.CacheMetadataMismatch;
    }
}

pub fn check(comptime M: type, m: *M) !void {
    var ids: [16]i32 = undefined;
    for (&ids, 0..) |*id, j| id.* = @intCast(1000 + j * 37);
    var target = try m.forward(&ids);
    defer target.deinit();
    // Fixed real target states isolate the MTP cache from unrelated target prefill
    // rounding while exercising its actual checkpoint and complete attention path.
    var base = M.DraftCache{};
    defer base.deinit();
    var length: usize = 0;
    var checks: usize = 0;
    for ([_]usize{ 0, 31, 2044, 2051, 9999, 10007 }) |past| {
        while (length < past) {
            const count = @min(16, past - length);
            var s = mx.Scope{};
            defer s.deinit();
            _ = try m.draftStepArray(&s, try s.slice(target.hidden, 0, 0, @intCast(count)), try s.ints(ids[0..count]), &base, false);
            length += count;
        }
        for ([_]usize{ 1, 3, 16 }) |rows| {
            var s = mx.Scope{};
            defer s.deinit();
            var serial = try base.clone();
            defer serial.deinit();
            var snapshots: [16]M.DraftCache = @splat(.{});
            defer for (&snapshots) |*cache| cache.deinit();
            var outputs: [16]mx.Array = undefined;
            for (0..rows) |j| {
                outputs[j] = try m.draftStep(&s, try s.slice(target.hidden, 0, @intCast(j), @intCast(j + 1)), ids[j], &serial);
                snapshots[j] = try serial.clone();
            }
            var batched = try base.clone();
            defer batched.deinit();
            const hidden = try m.draftStepArray(&s, try s.slice(target.hidden, 0, 0, @intCast(rows)), try s.ints(ids[0..rows]), &batched, true);
            const logits = try m.draftHead(&s, hidden);
            for (0..rows) |j| {
                try equal(&s, outputs[j], try s.slice(hidden, 0, @intCast(j), @intCast(j + 1)));
                try equal(&s, try m.draftHead(&s, outputs[j]), try s.slice(logits, 0, @intCast(j), @intCast(j + 1)));
            }
            for (0..rows + 1) |keep| {
                var step = mx.Scope{};
                defer step.deinit();
                var prefix = try M.draftPrefix(&step, batched, rows, keep);
                defer prefix.deinit();
                var expected = try (if (keep == 0) base else snapshots[keep - 1]).clone();
                defer expected.deinit();
                try caches(M.DraftCache, &step, prefix, expected);
                const next = try step.slice(target.hidden, 0, 0, 1);
                const actual = try m.draftStep(&step, next, 77, &prefix);
                const reference = try m.draftStep(&step, next, 77, &expected);
                try equal(&step, actual, reference);
                try equal(&step, try m.draftHead(&step, actual), try m.draftHead(&step, reference));
                try caches(M.DraftCache, &step, prefix, expected);
                checks += 1;
            }
            std.debug.print("PASS: MTP {d} rows at {d} past, all hidden/logit rows, {d} prefixes and continuations\n", .{ rows, past, rows + 1 });
        }
    }
    std.debug.print("PASS: {d} MTP accepted-prefix/cache/continuation checks\n", .{checks});
}
