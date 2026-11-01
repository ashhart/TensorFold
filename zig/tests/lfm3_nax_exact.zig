//! Real-weight LFM3 tensor-unit prompt route: chunk seams, resumed == fresh, and windows against one-row decode.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").lfm3;
const st = q.state;
const fwd = q.forward;
const c = q.config;

fn equalBytes(expected: []const u8, actual: []const u8) !void {
    if (expected.len != actual.len) return error.ExactBytesMismatch;
    if (std.mem.eql(u8, expected, actual)) return;
    for (expected, actual, 0..) |want, got, i| if (want != got) {
        std.debug.print("  first byte difference at {d}: expected {x:0>2}, actual {x:0>2}\n", .{ i, want, got });
        break;
    };
    return error.ExactBytesMismatch;
}

fn equalState(a: *const st.Cache, b: *const st.Cache) !void {
    if (a.len != b.len) return error.CacheLengthMismatch;
    for (a.blocks, b.blocks) |x, y| switch (x) {
        .conv => |s| try equalBytes(s.contents()[0..st.conv_bytes], y.conv.contents()[0..st.conv_bytes]),
        .attention => |k| {
            for ([_]mtl.Buffer{ k.keys, k.values }, [_]mtl.Buffer{ y.attention.keys, y.attention.values }) |a_buf, b_buf| {
                for (0..c.kv_heads) |h| {
                    const ao = h * a.capacity * c.head_dim * 2;
                    const bo = h * b.capacity * c.head_dim * 2;
                    const bytes = a.len * c.head_dim * 2;
                    try equalBytes((a_buf.contents() + ao)[0..bytes], (b_buf.contents() + bo)[0..bytes]);
                }
            }
        },
    };
}

fn equal(a: *const st.Cache, b: *const st.Cache) !void {
    try equalState(a, b);
    try equalBytes(a.logits.contents()[0 .. c.vocab * 2], b.logits.contents()[0 .. c.vocab * 2]);
}

fn tokens(gpa: std.mem.Allocator, length: usize) ![]u32 {
    const ids = try gpa.alloc(u32, length);
    const public = [_]u32{ 825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373 };
    for (ids, 0..) |*id, i| id.* = public[(i * 7 + i / 11) % public.len];
    return ids;
}

/// The served prompt route over `ids[from..to]` in chunks of `rows` (the last one shorter).
fn prompt(m: *q.Model, scratch: *st.Scratch, cache: *st.Cache, ids: []const u32, from: usize, to: usize, rows: usize) !void {
    var at = from;
    while (at < to) {
        const n = @min(rows, to - at);
        try fwd.runPromptWithImages(m, scratch, &.{.{ .cache = cache, .rows = n }}, ids[at..][0..n], .last, null, .nax);
        at += n;
    }
}

/// Prompts cut at every offset of `cuts` (each piece one chunk) equal the same prompt in 2048-row chunks.
fn cutsEqual(m: *q.Model, gpa: std.mem.Allocator, scratch: *st.Scratch, length: usize, cuts: []const usize) !void {
    const ids = try tokens(gpa, length);
    defer gpa.free(ids);
    var a = try st.Cache.init(gpa, m.device, &m.config.layer_types, length + 64);
    defer a.deinit();
    var b = try st.Cache.init(gpa, m.device, &m.config.layer_types, length + 64);
    defer b.deinit();
    try prompt(m, scratch, &a, ids, 0, length, 2048);
    var at: usize = 0;
    for (cuts) |cut| {
        if (cut > at and cut <= length) {
            try prompt(m, scratch, &b, ids, at, cut, 2048);
            at = cut;
        }
    }
    try prompt(m, scratch, &b, ids, at, length, 2048);
    try equal(&a, &b);
}

fn uniform(m: *q.Model, gpa: std.mem.Allocator, scratch: *st.Scratch, length: usize, rows: usize) !void {
    const ids = try tokens(gpa, length);
    defer gpa.free(ids);
    var a = try st.Cache.init(gpa, m.device, &m.config.layer_types, length + 64);
    defer a.deinit();
    var b = try st.Cache.init(gpa, m.device, &m.config.layer_types, length + 64);
    defer b.deinit();
    try prompt(m, scratch, &a, ids, 0, length, 2048);
    try prompt(m, scratch, &b, ids, 0, length, rows);
    try equal(&a, &b);
}

/// A saved prefix resumes into the state of the same prompt run fresh, for prefixes that are not chunk multiples.
fn resumed(m: *q.Model, gpa: std.mem.Allocator, scratch: *st.Scratch, length: usize, prefix: usize) !void {
    const ids = try tokens(gpa, length);
    defer gpa.free(ids);
    var fresh = try st.Cache.init(gpa, m.device, &m.config.layer_types, length + 64);
    defer fresh.deinit();
    try prompt(m, scratch, &fresh, ids, 0, length, 2048);
    var source = try st.Cache.init(gpa, m.device, &m.config.layer_types, length + 64);
    defer source.deinit();
    try prompt(m, scratch, &source, ids, 0, prefix, 2048);
    const snap = try q.snapshot.save(gpa, &m.config.layer_types, &source, @intCast(prefix));
    defer q.snapshot.drop(gpa, snap);
    var restored = try st.Cache.init(gpa, m.device, &m.config.layer_types, length + 64);
    defer restored.deinit();
    try q.snapshot.restore(&restored, snap);
    try equalState(&source, &restored);
    try prompt(m, scratch, &restored, ids, prefix, length, 2048);
    try equal(&fresh, &restored);
}

/// A window of `rows` verified in one pass equals the same rows decoded one at a time, after a served-route prompt.
fn window(m: *q.Model, gpa: std.mem.Allocator, scratch: *st.Scratch, wide: *st.Scratch, solo: *st.Scratch, prefix: usize, rows: usize) !void {
    const ids = try tokens(gpa, prefix + rows + 1);
    defer gpa.free(ids);
    var a = try st.Cache.init(gpa, m.device, &m.config.layer_types, prefix + 64);
    defer a.deinit();
    var b = try st.Cache.init(gpa, m.device, &m.config.layer_types, prefix + 64);
    defer b.deinit();
    try prompt(m, scratch, &a, ids, 0, prefix, 2048);
    try prompt(m, scratch, &b, ids, 0, prefix, 2048);
    try fwd.run(m, wide, &.{.{ .cache = &a, .rows = rows }}, ids[prefix..][0..rows], true, .all);
    for (0..rows) |r| {
        try fwd.run(m, solo, &.{.{ .cache = &b, .rows = 1 }}, ids[prefix + r ..][0..1], false, .all);
        try equalBytes(solo.logits.contents()[0 .. c.vocab * 2], (wide.logits.contents() + r * c.vocab * 2)[0 .. c.vocab * 2]);
    }
    try equal(&a, &b);
}

const Result = struct {
    passed: usize = 0,
    failed: usize = 0,

    fn record(self: *Result, name: []const u8, result: anyerror!void) void {
        if (result) |_| {
            self.passed += 1;
            std.debug.print("ok {s}\n", .{name});
        } else |err| {
            self.failed += 1;
            std.debug.print("FAIL {s}: {s}\n", .{ name, @errorName(err) });
        }
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) {
        std.debug.print("usage: tf-lfm3-nax-exact MODEL_DIR\n", .{});
        return error.ExpectedModel;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.Model.load(init.gpa, init.io, args[1]);
    defer m.deinit();
    try m.enableNaxPrompt();
    try m.enableTiledAttention();
    if (m.prompt_nax == null or m.tiled_attention == null or m.split_attention == null) return error.NaxPromptUnavailable;
    var scratch = try st.Scratch.init(init.gpa, m.device, &m.config.layer_types, 2048);
    defer scratch.deinit();
    var wide = try st.Scratch.init(init.gpa, m.device, &m.config.layer_types, st.batch_rows);
    defer wide.deinit();
    var solo = try st.Scratch.init(init.gpa, m.device, &m.config.layer_types, 1);
    defer solo.deinit();
    var result: Result = .{};
    var name: [128]u8 = undefined;
    for ([_]usize{ 1, 8, 9, 63, 64, 65, 127, 128, 129, 1000, 2047 }) |rows| {
        result.record(try std.fmt.bufPrint(&name, "prompt 4500: chunks of {d} equal chunks of 2048", .{rows}), uniform(m, init.gpa, &scratch, 4500, rows));
    }
    const cut_sets = [_][]const usize{ &.{ 1, 2, 4500 - 1 }, &.{ 2047, 2049 }, &.{ 2048, 2051 }, &.{ 4090, 4095, 4098 }, &.{ 3, 11, 300, 2053, 4000 } };
    for (cut_sets, 0..) |cuts, i| {
        result.record(try std.fmt.bufPrint(&name, "prompt 4500: cut set {d} equals chunks of 2048", .{i}), cutsEqual(m, init.gpa, &scratch, 4500, cuts));
    }
    for ([_]usize{ 2047, 2048, 2049, 2051, 4097, 8193 }) |length| {
        result.record(try std.fmt.bufPrint(&name, "prompt {d}: cut at 9 and 2048 equals chunks of 2048", .{length}), cutsEqual(m, init.gpa, &scratch, length, &.{ 9, 2048 }));
    }
    for ([_]usize{ 1, 9, 128, 1000, 2048, 2051 }) |prefix| {
        result.record(try std.fmt.bufPrint(&name, "prompt 4500: resumed from {d} equals fresh", .{prefix}), resumed(m, init.gpa, &scratch, 4500, prefix));
    }
    for ([_]usize{ 127, 1000, 4096 }) |prefix| {
        for ([_]usize{ 2, 8, 9, 16, 32 }) |rows| {
            result.record(try std.fmt.bufPrint(&name, "prefix {d}: window of {d} equals one-row decode", .{ prefix, rows }), window(m, init.gpa, &scratch, &wide, &solo, prefix, rows));
        }
    }
    std.debug.print("LFM3_NAX_EXACT passed={d} failed={d}\n", .{ result.passed, result.failed });
    if (result.failed != 0) return error.ExactnessFailed;
}
