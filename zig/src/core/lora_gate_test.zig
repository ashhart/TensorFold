//! The gates of `merge --quantize` end to end on synthetic folders: identity, zero effect, ratios and the point bound.
const std = @import("std");
const st = @import("safetensors.zig");
const se = @import("shard_edit.zig");
const fx = @import("lora_fixture.zig");
const dir = @import("lora_merge_dir.zig");
const lm = dir.math;
const lp = dir.plan;
const Io = std.Io;

const io = fx.io;
const module = fx.module;

fn contains(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}

/// Merges `vals` (n x k) with a zero adapter of `alpha`, then runs the gates: true when they all pass.
fn zeroCase(vals: []const f32, n: usize, k: usize, alpha: []const u8, neg: bool) !bool {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeSingle(&s, n, k, vals);
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .n = n, .k = k, .r = 1, .zero = !neg, .neg_zero = neg }}, .alpha = alpha, .rank = "1" });
    var buf: [4096]u8 = undefined;
    try std.testing.expectEqual(@as(u8, 0), (try fx.merge(&s, base, adapter, &buf)).code);
    var big: [16384]u8 = undefined;
    const g = try fx.gates(&s, base, adapter, &big);
    try std.testing.expect(contains(g.out, "zero effect: true"));
    return g.ok;
}

test "a zero adapter on exactly representable tensors passes: zeros, halves and an integer grid" {
    const a = std.testing.allocator;
    const zeros = try a.alloc(f32, 2 * 64);
    defer a.free(zeros);
    @memset(zeros, 0);
    try std.testing.expect(try zeroCase(zeros, 2, 64, "2", false));
    @memset(zeros, 0.5);
    try std.testing.expect(try zeroCase(zeros, 2, 64, "2", false));
    for (zeros, 0..) |*v, i| v.* = @floatFromInt(i % 16);
    try std.testing.expect(try zeroCase(zeros, 2, 64, "2", false));
}

/// The reviewer's group [0, 15, 2.5, 0, ...]: shipped scale -1, bias 15, 2.5 coded 13 (value 2).
fn tiedGroup() [64]f32 {
    var v: [64]f32 = @splat(0);
    v[1] = 15;
    v[2] = 2.5;
    return v;
}

/// OUT's words with code `at` of the single row replaced by `code`, written over the merged shard.
fn tamperCode(s: *fx.Scratch, at: usize, code: u32) !void {
    var buf: [256]u8 = undefined;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var err = Io.Writer.fixed(&buf);
    var shards = try lp.Shards.init(std.testing.allocator, arena.allocator(), io, &err, try s.path(&.{"merged"}));
    defer shards.deinit();
    const t = (try shards.get(module ++ ".weight")).?;
    var words: [8]u32 = undefined;
    @memcpy(std.mem.sliceAsBytes(&words), t.bytes);
    words[at / 8] = (words[at / 8] & ~(@as(u32, 0xf) << @intCast(4 * (at % 8)))) | (code << @intCast(4 * (at % 8)));
    try se.overwrite(std.testing.allocator, io, try s.path(&.{"merged"}), &.{.{ .name = module ++ ".weight", .dtype = .u32, .shape = &.{ 1, 8 }, .bytes = std.mem.sliceAsBytes(&words) }});
}

test "a zero-effect adapter (alpha 0, or negative-zero factors) must leave every byte: a changed code fails" {
    const v = tiedGroup();
    for ([_]struct { alpha: []const u8, neg: bool }{ .{ .alpha = "0", .neg = false }, .{ .alpha = "2", .neg = true } }) |c| {
        var s = try fx.Scratch.init();
        defer s.deinit();
        const base = try fx.makeSingle(&s, 1, 64, &v);
        const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .n = 1, .k = 64, .r = 1, .value = if (c.neg) null else 1, .neg_zero = c.neg }}, .alpha = c.alpha, .rank = "1" });
        var buf: [4096]u8 = undefined;
        try std.testing.expectEqual(@as(u8, 0), (try fx.merge(&s, base, adapter, &buf)).code);
        var big: [16384]u8 = undefined;
        try std.testing.expect((try fx.gates(&s, base, adapter, &big)).ok); // untouched: passes
        try tamperCode(&s, 2, 12); // 2.5 decodes as 3 instead of 2: the same error as before, a different byte
        const g = try fx.gates(&s, base, adapter, &big);
        try std.testing.expect(!g.ok);
        try std.testing.expect(contains(g.out, "zero effect but the shard differs"));
    }
}

test "a collapsed grid in one group fails the recomputation although the global ratios pass" {
    const a = std.testing.allocator;
    var vals = try a.alloc(f32, 2048);
    defer a.free(vals);
    @memset(vals[0..64], 1);
    for (1..32) |g| {
        @memset(vals[g * 64 ..][0..64], 0);
        vals[g * 64 + 1] = 150;
        vals[g * 64 + 2] = 25;
    }
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeSingle(&s, 1, 2048, vals);
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .n = 1, .k = 2048, .r = 1, .value = 1 }}, .alpha = "0.000001", .rank = "1" });
    var buf: [4096]u8 = undefined;
    try std.testing.expectEqual(@as(u8, 0), (try fx.merge(&s, base, adapter, &buf)).code);
    var big: [16384]u8 = undefined;
    try std.testing.expect((try fx.gates(&s, base, adapter, &big)).ok);
    // group 0 of OUT: scale = bias = 0 and every code zero
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var err_buf: [256]u8 = undefined;
    var err = Io.Writer.fixed(&err_buf);
    var shards = try lp.Shards.init(a, arena.allocator(), io, &err, try s.path(&.{"merged"}));
    defer shards.deinit();
    const j: lp.Job = .{ .module = module, .a = "", .b = "", .n = 1, .k = 2048, .group = 64, .dtype = .u32 };
    const q = try lp.quantOf(a, &shards, j);
    defer lp.free(a, q);
    q.scales[0] = 0;
    q.biases[0] = 0;
    @memset(q.words[0..8], 0);
    const shapes = [3][2]usize{ .{ 1, 256 }, .{ 1, 32 }, .{ 1, 32 } };
    try se.overwrite(a, io, try s.path(&.{"merged"}), &.{
        .{ .name = module ++ ".weight", .dtype = .u32, .shape = &shapes[0], .bytes = std.mem.sliceAsBytes(q.words) },
        .{ .name = module ++ ".scales", .dtype = .bf16, .shape = &shapes[1], .bytes = std.mem.sliceAsBytes(q.scales) },
        .{ .name = module ++ ".biases", .dtype = .bf16, .shape = &shapes[2], .bytes = std.mem.sliceAsBytes(q.biases) },
    });
    const g = try fx.gates(&s, base, adapter, &big);
    try std.testing.expect(!g.ok);
    try std.testing.expect(contains(g.out, "within sanity bound: yes")); // the global ratios are fine; the group is not
    try std.testing.expect(contains(g.out, "1 of 32 groups differ from OUT"));
}

test "identity: appended bytes, a changed byte outside the payload, extra files and a differing file all fail" {
    const Tamper = enum { append, other_tensor, extra_file, config };
    for ([_]Tamper{ .append, .other_tensor, .extra_file, .config }) |t| {
        var s = try fx.Scratch.init();
        defer s.deinit();
        const base = try fx.makeBase(&s, fx.base_config);
        const adapter = try fx.makeAdapter(&s, .{});
        var buf: [4096]u8 = undefined;
        try std.testing.expectEqual(@as(u8, 0), (try fx.merge(&s, base, adapter, &buf)).code);
        var big: [16384]u8 = undefined;
        try std.testing.expect((try fx.gates(&s, base, adapter, &big)).ok);
        const shard = try s.read("merged", fx.files[2]);
        switch (t) {
            .append => try s.write("merged", fx.files[2], try std.mem.concat(s.arena.allocator(), u8, &.{ shard, "extra" })),
            .other_tensor => {
                const copy = try s.arena.allocator().dupe(u8, shard);
                var f = try st.File.open(std.testing.allocator, io, try s.path(&.{ "merged", fx.files[2] }));
                const at = f.data + f.names.get("backbone.embeddings.weight").?.begin;
                f.close(io);
                copy[at] ^= 1;
                try s.write("merged", fx.files[2], copy);
            },
            .extra_file => try s.write("merged", "extra.txt", "x"),
            .config => try s.write("merged", "config.json", "{}"),
        }
        const g = try fx.gates(&s, base, adapter, &big);
        try std.testing.expect(!g.ok);
        try std.testing.expect(contains(g.out, "identity: ") and contains(g.out, "FAIL"));
    }
}

test "a zero adapter whose shard differs fails the identity gate" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .zero = true }} });
    var buf: [4096]u8 = undefined;
    try std.testing.expectEqual(@as(u8, 0), (try fx.merge(&s, base, adapter, &buf)).code);
    const shard = try s.arena.allocator().dupe(u8, try s.read("merged", fx.files[2]));
    var f = try st.File.open(std.testing.allocator, io, try s.path(&.{ "merged", fx.files[2] }));
    shard[f.data + f.names.get(module ++ ".weight").?.begin] ^= 1;
    f.close(io);
    try s.write("merged", fx.files[2], shard);
    var big: [16384]u8 = undefined;
    const g = try fx.gates(&s, base, adapter, &big);
    try std.testing.expect(!g.ok and contains(g.out, "zero effect but the shard differs"));
}

test "a wrong group in OUT fails the recompute gate even where the precision gates would pass" {
    const v = tiedGroup();
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeSingle(&s, 1, 64, &v);
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .n = 1, .k = 64, .r = 1, .value = 0.1 }}, .alpha = "2", .rank = "1" });
    var buf: [4096]u8 = undefined;
    try std.testing.expectEqual(@as(u8, 0), (try fx.merge(&s, base, adapter, &buf)).code);
    var big: [16384]u8 = undefined;
    const good = try fx.gates(&s, base, adapter, &big);
    try std.testing.expect(good.ok and contains(good.out, "0 of 1 groups differ from OUT"));
    try tamperCode(&s, 2, 5);
    const g = try fx.gates(&s, base, adapter, &big);
    try std.testing.expect(!g.ok);
    try std.testing.expect(contains(g.out, "1 of 1 groups differ from OUT"));
}

test "identity: a dangling symlink in OUT, or in SHIPPED, fails the gate" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    const adapter = try fx.makeAdapter(&s, .{});
    var buf: [4096]u8 = undefined;
    try std.testing.expectEqual(@as(u8, 0), (try fx.merge(&s, base, adapter, &buf)).code);
    var big: [16384]u8 = undefined;
    try std.testing.expect((try fx.gates(&s, base, adapter, &big)).ok);
    try Io.Dir.cwd().symLink(io, "no-such-file", try s.path(&.{ "merged", "extra.bin" }), .{});
    var g = try fx.gates(&s, base, adapter, &big);
    try std.testing.expect(!g.ok and contains(g.out, "extra.bin"));
    try Io.Dir.cwd().deleteFile(io, try s.path(&.{ "merged", "extra.bin" }));
    try Io.Dir.cwd().symLink(io, "no-such-file", try s.path(&.{ "base", "extra.bin" }), .{});
    g = try fx.gates(&s, base, adapter, &big);
    try std.testing.expect(!g.ok and contains(g.out, "extra.bin"));
}
