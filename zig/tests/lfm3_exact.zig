//! Real-weight LFM3 prompt chunks, windows, partial keeps and unequal streams against native one-row execution.
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

fn equal(a: *const st.Cache, b: *const st.Cache) !void {
    if (a.len != b.len) return error.CacheLengthMismatch;
    try equalBytes(a.logits.contents()[0 .. c.vocab * 2], b.logits.contents()[0 .. c.vocab * 2]);
    try equalState(a, b);
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

fn prompt(m: *q.Model, scratch: *st.Scratch, cache: *st.Cache, ids: []const u32) !void {
    var at: usize = 0;
    while (at < ids.len) {
        const rows: usize = @min(scratch.rows, ids.len - at);
        try fwd.run(m, scratch, &.{.{ .cache = cache, .rows = rows }}, ids[at..][0..rows], false, .last);
        at += rows;
    }
}

/// A prompt of `prefix` rows, then plain decode to `len`: what a window's kept rows equal.
fn decoded(m: *q.Model, solo: *st.Scratch, cache: *st.Cache, ids: []const u32, prefix: usize, len: usize) !void {
    try prompt(m, solo, cache, ids[0..prefix]);
    for (prefix..len) |r| try fwd.run(m, solo, &.{.{ .cache = cache, .rows = 1 }}, ids[r..][0..1], false, .all);
}

fn window(m: *q.Model, wide: *st.Scratch, solo: *st.Scratch, ids: []const u32, prefix: usize, rows: usize, keep: usize) !void {
    var a = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 512);
    defer a.deinit();
    var b = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 529);
    defer b.deinit();
    try prompt(m, wide, &a, ids[0..prefix]);
    try prompt(m, solo, &b, ids[0..prefix]);
    if (prefix == 0) try equalState(&a, &b) else try equal(&a, &b);
    try fwd.run(m, wide, &.{.{ .cache = &a, .rows = rows }}, ids[prefix..][0..rows], true, .all);
    for (0..rows) |r| {
        try fwd.run(m, solo, &.{.{ .cache = &b, .rows = 1 }}, ids[prefix + r ..][0..1], false, .all);
        const want = solo.logits.contents()[0 .. c.vocab * 2];
        const actual = (wide.logits.contents() + r * c.vocab * 2)[0 .. c.vocab * 2];
        try equalBytes(want, actual);
    }
    try equal(&a, &b);
    var path: [32]u32 = undefined;
    for (&path, 0..) |*r, i| r.* = @intCast(i);
    try a.keep(wide, path[0..keep]);
    var accepted = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 512);
    defer accepted.deinit();
    try prompt(m, solo, &accepted, ids[0..prefix]);
    for (0..keep) |r| try fwd.run(m, solo, &.{.{ .cache = &accepted, .rows = 1 }}, ids[prefix + r ..][0..1], false, .all);
    try equal(&a, &accepted);
    try fwd.run(m, wide, &.{.{ .cache = &a, .rows = 1 }}, ids[prefix + keep ..][0..1], false, .all);
    try fwd.run(m, solo, &.{.{ .cache = &accepted, .rows = 1 }}, ids[prefix + keep ..][0..1], false, .all);
    try equal(&a, &accepted);
    try equalBytes(a.logits.contents()[0 .. c.vocab * 2], accepted.logits.contents()[0 .. c.vocab * 2]);
}

fn shared(m: *q.Model, wide: *st.Scratch, solo: *st.Scratch, ids: []const u32) !void {
    var caches: [4]st.Cache = undefined;
    var n: usize = 0;
    defer for (caches[0..n]) |*cache| cache.deinit();
    for (&caches) |*cache| {
        cache.* = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 512);
        n += 1;
    }
    for (&caches, 0..) |*cache, i| try prompt(m, solo, cache, ids[0..if (i % 2 == 0) @as(usize, 127) else 256]);
    var tokens: [20]u32 = undefined;
    @memcpy(tokens[0..7], ids[127..][0..7]);
    @memcpy(tokens[7..20], ids[256..][0..13]);
    try fwd.run(m, wide, &.{ .{ .cache = &caches[0], .rows = 7 }, .{ .cache = &caches[1], .rows = 13 } }, &tokens, true, .all);
    for ([_]usize{ 7, 13 }, 0..) |rows, stream| {
        const prefix: usize = if (stream == 0) 127 else 256;
        const base: usize = if (stream == 0) 0 else 7;
        for (0..rows) |r| {
            try fwd.run(m, solo, &.{.{ .cache = &caches[stream + 2], .rows = 1 }}, ids[prefix + r ..][0..1], false, .all);
            try equalBytes(solo.logits.contents()[0 .. c.vocab * 2], (wide.logits.contents() + (base + r) * c.vocab * 2)[0 .. c.vocab * 2]);
        }
        try equal(&caches[stream], &caches[stream + 2]);
    }
    try caches[0].keep(wide, &.{ 0, 1, 2 });
    try caches[1].keep(wide, &.{0});
    for ([_]usize{ 127, 256 }, [_]usize{ 130, 257 }, 0..) |prefix, length, stream| {
        var fresh = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 512);
        defer fresh.deinit();
        try decoded(m, solo, &fresh, ids, prefix, length);
        try equal(&caches[stream], &fresh);
    }
}

/// One window shared by streams at `lens`, prompted to `prompts` (0 rows skip one): it equals one-row decode.
fn round(
    m: *q.Model,
    wide: *st.Scratch,
    solo: *st.Scratch,
    ids: []const u32,
    caches: []st.Cache,
    prompts: []const usize,
    lens: []const usize,
    rows: []const usize,
) !void {
    var segs: [4]fwd.Segment = undefined;
    var tokens: [32]u32 = undefined;
    var n: usize = 0;
    var total: usize = 0;
    for (caches, lens, rows) |*cache, len, r| {
        if (r == 0) continue;
        segs[n] = .{ .cache = cache, .rows = r };
        @memcpy(tokens[total..][0..r], ids[len..][0..r]);
        n += 1;
        total += r;
    }
    try fwd.run(m, wide, segs[0..n], tokens[0..total], true, .all);
    var base: usize = 0;
    for (caches, prompts, lens, rows) |*cache, start, len, r| {
        if (r == 0) continue;
        var ref = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 512);
        defer ref.deinit();
        try decoded(m, solo, &ref, ids, start, len);
        for (0..r) |i| {
            try fwd.run(m, solo, &.{.{ .cache = &ref, .rows = 1 }}, ids[len + i ..][0..1], false, .all);
            const actual = (wide.logits.contents() + (base + i) * c.vocab * 2)[0 .. c.vocab * 2];
            try equalBytes(solo.logits.contents()[0 .. c.vocab * 2], actual);
        }
        try equal(cache, &ref);
        base += r;
    }
}

/// Four unequal streams share 32 rows, keep unequal paths, then share 16 rows with one stream idle.
fn shared4(m: *q.Model, wide: *st.Scratch, solo: *st.Scratch, ids: []const u32) !void {
    var caches: [4]st.Cache = undefined;
    var n: usize = 0;
    defer for (caches[0..n]) |*cache| cache.deinit();
    const prompts = [_]usize{ 1, 127, 128, 256 };
    var lens = prompts;
    for (&caches, lens) |*cache, len| {
        cache.* = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 512);
        n += 1;
        try prompt(m, solo, cache, ids[0..len]);
    }
    try round(m, wide, solo, ids, &caches, &prompts, &lens, &.{ 1, 5, 10, 16 });
    const keep = [_]usize{ 1, 3, 10, 7 };
    var path: [16]u32 = undefined;
    for (&path, 0..) |*r, i| r.* = @intCast(i);
    for (&caches, prompts, &lens, keep) |*cache, start, *len, kept| {
        try cache.keep(wide, path[0..kept]);
        len.* += kept;
        var fresh = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 512);
        defer fresh.deinit();
        try decoded(m, solo, &fresh, ids, start, len.*);
        try equal(cache, &fresh);
    }
    try round(m, wide, solo, ids, &caches, &prompts, &lens, &.{ 3, 4, 9, 0 });
}

/// A long prompt in 4096-row chunks equals it in 2048-row chunks across fixed attention partitions.
fn chunks(m: *q.Model, gpa: std.mem.Allocator) !void {
    const length = 4500;
    const ids = try gpa.alloc(u32, length);
    defer gpa.free(ids);
    const public = [_]u32{ 825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373 };
    for (ids, 0..) |*id, i| id.* = public[(i * 7 + i / 11) % public.len];
    var a = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, length + 64);
    defer a.deinit();
    var b = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, length + 64);
    defer b.deinit();
    for ([_]usize{ 4096, 2048 }, 0..) |rows, i| {
        var scratch = try st.Scratch.init(gpa, m.device, &m.config.layer_types, rows);
        defer scratch.deinit();
        try prompt(m, &scratch, if (i == 0) &a else &b, ids);
    }
    try equal(&a, &b);
}

/// Long prompt chunks equal one-row execution across attention slices and 4096-key partitions.
fn seam(m: *q.Model, gpa: std.mem.Allocator, length: usize) !void {
    const ids = try gpa.alloc(u32, length);
    defer gpa.free(ids);
    const public = [_]u32{ 825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373 };
    for (ids, 0..) |*id, i| id.* = public[(i * 7 + i / 11) % public.len];
    var a = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, length + 64);
    defer a.deinit();
    var b = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, length + 64);
    defer b.deinit();
    var served = try st.Scratch.init(gpa, m.device, &m.config.layer_types, 2048);
    defer served.deinit();
    var one = try st.Scratch.init(gpa, m.device, &m.config.layer_types, 1);
    defer one.deinit();
    try prompt(m, &served, &a, ids);
    try prompt(m, &one, &b, ids);
    try equal(&a, &b);
}

/// A saved prefix resumes exactly; dropping a speculative request leaves a later fresh cache unchanged.
fn snapshotsAndReset(m: *q.Model, chunk: *st.Scratch, solo: *st.Scratch, ids: []const u32) !void {
    var reference = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 512);
    defer reference.deinit();
    try prompt(m, solo, &reference, ids);
    var source = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 520);
    defer source.deinit();
    try prompt(m, chunk, &source, ids[0..128]);
    const snap = try q.snapshot.save(m.gpa, &m.config.layer_types, &source, 128);
    defer q.snapshot.drop(m.gpa, snap);
    var restored = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 529);
    defer restored.deinit();
    try q.snapshot.restore(&restored, snap);
    try equalState(&source, &restored);
    try prompt(m, chunk, &restored, ids[128..]);
    try equal(&reference, &restored);
    {
        var cancelled = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 537);
        defer cancelled.deinit();
        try prompt(m, chunk, &cancelled, ids[0..129]);
        try fwd.run(m, chunk, &.{.{ .cache = &cancelled, .rows = 16 }}, ids[129..145], true, .all);
    }
    var reset = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 541);
    defer reset.deinit();
    try prompt(m, chunk, &reset, ids);
    try equal(&reference, &reset);
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

fn chunkCase(m: *q.Model, chunk: *st.Scratch, solo: *st.Scratch, ids: []const u32) !void {
    var a = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 512);
    defer a.deinit();
    var b = try st.Cache.init(m.gpa, m.device, &m.config.layer_types, 529);
    defer b.deinit();
    try prompt(m, chunk, &a, ids);
    try prompt(m, solo, &b, ids);
    try equal(&a, &b);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const long = args.len == 3 and std.mem.eql(u8, args[2], "--long");
    if (args.len != 2 and !long) {
        std.debug.print("usage: tf-lfm3-exact MODEL_DIR [--long]\n", .{});
        return error.ExpectedModel;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.Model.load(init.gpa, init.io, args[1]);
    defer m.deinit();
    var wide = try st.Scratch.init(init.gpa, m.device, &m.config.layer_types, st.batch_rows);
    defer wide.deinit();
    var solo = try st.Scratch.init(init.gpa, m.device, &m.config.layer_types, 1);
    defer solo.deinit();
    var chunk = try st.Scratch.init(init.gpa, m.device, &m.config.layer_types, 128);
    defer chunk.deinit();
    var ids: [300]u32 = undefined;
    const public = [_]u32{ 825, 264, 15352, 4829, 9944, 13, 271, 279, 854, 22373 };
    for (&ids, 0..) |*id, i| id.* = public[(i * 7 + i / 11) % public.len];
    var result: Result = .{};
    for ([_]usize{ 1, 127, 128, 129, 300 }) |length| {
        var name: [128]u8 = undefined;
        result.record(try std.fmt.bufPrint(&name, "prompt {d}: chunk128 versus one row", .{length}), chunkCase(m, &chunk, &solo, ids[0..length]));
    }
    for ([_]usize{ 0, 127, 128, 129, 255, 256 }) |prefix| {
        for ([_]usize{ 1, 2, 7, 8, 9, 15, 16, 17, 24, 32 }) |rows| {
            const keep = @max(1, rows / 2);
            var name: [128]u8 = undefined;
            result.record(try std.fmt.bufPrint(&name, "prefix {d}, window {d}, keep {d}", .{ prefix, rows, keep }), window(m, &wide, &solo, &ids, prefix, rows, keep));
        }
    }
    result.record("two unequal streams, partial keeps and fresh continuation", shared(m, &wide, &solo, &ids));
    result.record("four unequal streams, 32/16 aggregate rows and idle stream", shared4(m, &wide, &solo, &ids));
    result.record("prefix snapshot restore and fresh cache after cancellation surrogate", snapshotsAndReset(m, &chunk, &solo, &ids));
    if (long) {
        result.record("prompt4500: chunk4096 versus chunk2048", chunks(m, init.gpa));
        for ([_]usize{ 2047, 2049, 4097, 8193 }) |length| {
            var name: [128]u8 = undefined;
            result.record(try std.fmt.bufPrint(&name, "long seam {d}: chunk2048 versus one row", .{length}), seam(m, init.gpa, length));
        }
    }
    std.debug.print("LFM3_EXACT mode={s} passed={d} failed={d} skipped=0 prompt=5 windows=60 shared=2 snapshots=1 long={d} gpu_seconds={d:.3}\n", .{
        if (long) "long" else "default", result.passed, result.failed, @as(usize, if (long) 5 else 0), fwd.gpu_seconds,
    });
    if (result.failed != 0) return error.ExactnessFailed;
}
