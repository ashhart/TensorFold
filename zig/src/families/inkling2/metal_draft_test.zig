//! DSpark head on Metal: blocks against an f64 reference, chunking, rollback, stream isolation, refusals, tap contract.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const config = @import("config.zig");
const draft = @import("metal_draft.zig");
const fxm = @import("metal_draft_fixture.zig");

const t = std.testing;
const dr = lanes.drafter;
const depth = 7;

pub const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,

    pub fn init() !Gpu {
        const device = mtl.Device.init() catch return error.SkipZigTest;
        errdefer device.deinit();
        return .{ .device = device, .queue = device.queue() catch return error.SkipZigTest };
    }
    pub fn deinit(g: Gpu) void {
        g.queue.deinit();
        g.device.deinit();
    }
};

pub fn features(rows: []const u16, n: usize, start: u64, row_bytes: u32) dr.Absorb {
    return .{ .stream = undefined, .features = .{ .buffer = @intFromPtr(rows.ptr), .rows = @intCast(n), .row_bytes = row_bytes, .space = .host, .dtype = .bf16 }, .start = start, .follow = &.{} };
}

pub fn absorb(d: dr.Drafter, s: *lanes.Stream, rows: []const u16, n: usize, start: u64, row_bytes: u32) !void {
    var item = features(rows, n, start, row_bytes);
    item.stream = s;
    try d.absorb(&.{item});
}

/// The drafts of one hold at `p` committed rows after `anchor`.
pub fn drafts(d: dr.Drafter, s: *lanes.Stream, anchor: u32, p: u64) ![depth]u32 {
    try d.hold(&.{.{ .stream = s, .pending = anchor, .position = p + 1, .depth = depth }});
    var out: [depth]u32 = undefined;
    try d.held(&.{s}, &.{out[0..]});
    return out;
}

pub fn xBits(a: *draft.Adapter, rows: usize, D: usize) []const u16 {
    return a.core.buf("x").slice(u16, rows * D)[0 .. rows * D];
}

pub fn relError(got: []const u16, ref: []const f64) f64 {
    var err: f64 = 0;
    var mag: f64 = 0;
    for (got, ref) |g, r| {
        err += (fxm.unbf(g) - r) * (fxm.unbf(g) - r);
        mag += r * r;
    }
    return @sqrt(err / mag);
}

test "the config narrows to Mini's head structure: logical taps, 3 layers sliding 0 and 2, 3 heads a key/value head" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const c = try fxm.tinyConfig(arena.allocator());
    const m = c.mtp.?;
    try t.expectEqual(@as(u32, 3), m.layers);
    try t.expectEqual([3]bool{ true, false, true }, m.local[0..3].*);
    try t.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, m.taps[0..m.taps_n]);
    try t.expectEqual(@as(u32, 3), m.q_heads / m.kv_heads);
    try t.expectEqual(@as(u32, 8), m.block);
    // the taps are visits of the looped backbone, not layers of the 24 physical ones: 48 is its final norm
    try t.expectEqual(@as(u32, 48), c.text.executionLayers());
    try t.expect(m.taps[m.taps_n - 1] == c.text.executionLayers());
    try t.expect(c.text.layers < m.taps[m.taps_n - 1]);
}

test "a block over committed rows matches the f64 reference: rows, Markov steps, chunked and rolled-back contexts, streams" {
    const g = try Gpu.init();
    defer g.deinit();
    const fx = try fxm.Fixture.init(t.allocator, t.io, g.device);
    defer fx.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const D = fx.d.D;
    const row_bytes: u32 = @intCast(fx.d.taps * D * 2);
    const row_words = fx.d.taps * D;
    const ad = try draft.Adapter.init(t.allocator, t.io, fx.dir, g.device, g.queue, &fx.cfg, fx.tensors(), .{ .streams = 2, .capacity = 64 });
    defer ad.deinit();
    const d = ad.drafter();
    try t.expectEqual(@as(u32, depth), d.facts().depth);
    try t.expect(d.facts().fixed_block and d.facts().step_ms > 0);
    try t.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, d.taps());
    var s1: lanes.Stream = undefined;
    var s2: lanes.Stream = undefined;
    try d.open(&s1);
    try d.open(&s2);
    const rows = try fxm.features(a, fx.d, 12, 7);
    const anchor: u32 = 1234;
    // the reference over 6 committed rows
    try absorb(d, &s1, rows, 6, 0, row_bytes);
    const first = try drafts(d, &s1, anchor, 6);
    const ref = try fx.ref.blockRows(a, rows, 6, anchor);
    try t.expect(relError(xBits(ad, fx.d.block, D), ref) < 0.02);
    var best: [depth]f64 = undefined;
    var got: [depth]f64 = undefined;
    try fx.ref.steps(a, ref, rows[5 * row_words ..][0..row_words], anchor, &first, null, &best, &got);
    for (first, best, got) |tok, top, at| {
        try t.expect(tok < fx.d.unpadded);
        try t.expect(at >= top - 0.5);
    }
    // the same rows in two absorbs, and rolled back and absorbed again, draft the same bits
    try absorb(d, &s2, rows, 2, 0, row_bytes);
    try absorb(d, &s2, rows[2 * row_words ..], 4, 2, row_bytes);
    try t.expectEqual(first, try drafts(d, &s2, anchor, 6));
    const bits = try a.dupe(u16, xBits(ad, fx.d.block, D));
    try absorb(d, &s2, rows[4 * row_words ..], 8, 4, row_bytes);
    _ = try drafts(d, &s2, anchor, 12);
    try absorb(d, &s2, rows[4 * row_words ..], 2, 4, row_bytes);
    try t.expectEqual(first, try drafts(d, &s2, anchor, 6));
    try t.expectEqualSlices(u16, bits, xBits(ad, fx.d.block, D));
    // another stream's rows in between change nothing
    try absorb(d, &s2, rows[6 * row_words ..], 6, 6, row_bytes);
    try t.expectEqual(first, try drafts(d, &s1, anchor, 6));
    const other = try drafts(d, &s2, anchor + 1, 12);
    try d.hold(&.{
        .{ .stream = &s1, .pending = anchor, .position = 7, .depth = depth },
        .{ .stream = &s2, .pending = anchor + 1, .position = 13, .depth = depth },
    });
    var held: [2][depth]u32 = undefined;
    try d.held(&.{ &s2, &s1 }, &.{ &held[0], &held[1] });
    try t.expectEqual(other, held[0]);
    try t.expectEqual(first, held[1]);
    // refusals
    try t.expectError(error.DraftContextGap, d.hold(&.{.{ .stream = &s1, .pending = 1, .position = 9, .depth = 3 }}));
    try t.expectError(error.DraftDepth, d.hold(&.{.{ .stream = &s1, .pending = 1, .position = 7, .depth = 8 }}));
    try t.expectError(error.DraftDepth, d.hold(&.{.{ .stream = &s1, .pending = 1, .position = 7, .depth = 0 }}));
    var late = features(rows, 1, 0, row_bytes);
    late.stream = &s1;
    late.features.ready = .{ .metal_event = 1 };
    try t.expectError(error.BadDraftFeatures, d.absorb(&.{late}));
    var wide = features(rows, 1, 0, row_bytes + 2);
    wide.stream = &s1;
    try t.expectError(error.BadDraftFeatures, d.absorb(&.{wide}));
    // a feature row of the 8 taps Small stores is not Mini's 4
    var small = features(rows, 1, 0, row_bytes * 2);
    small.stream = &s1;
    try t.expectError(error.BadDraftFeatures, d.absorb(&.{small}));
    var over = features(rows, 4, 62, row_bytes);
    over.stream = &s1;
    try t.expectError(error.DraftContextFull, d.absorb(&.{over}));
    var third: lanes.Stream = undefined;
    try t.expectError(error.DraftStreamsFull, d.open(&third));
    try t.expectError(error.DraftStreamOpen, d.open(&s1));
    var first_scratch: [depth]u32 = undefined;
    d.release(&s1);
    try d.open(&s1);
    try t.expectError(error.ShortDraftChain, d.held(&.{&s1}, &.{&first_scratch}));
    try t.expectError(error.DraftContextGap, d.hold(&.{.{ .stream = &s1, .pending = anchor, .position = 7, .depth = depth }}));
    try absorb(d, &s1, rows, 6, 0, row_bytes);
    try t.expectEqual(first, try drafts(d, &s1, anchor, 6));
    d.release(&s1);
    d.release(&s2);
    try d.open(&third);
    d.release(&third);
}

test "a context past the sliding window and the ring matches the reference: sliding layers read the last 1024 rows" {
    const g = try Gpu.init();
    defer g.deinit();
    const fx = try fxm.Fixture.init(t.allocator, t.io, g.device);
    defer fx.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = 1170; // past the ring's 1152 slots: the oldest positions' slots were rewritten
    const ad = try draft.Adapter.init(t.allocator, t.io, fx.dir, g.device, g.queue, &fx.cfg, fx.tensors(), .{ .capacity = 1200 });
    defer ad.deinit();
    const d = ad.drafter();
    var s: lanes.Stream = undefined;
    try d.open(&s);
    defer d.release(&s);
    const rows = try fxm.features(a, fx.d, p, 11);
    try absorb(d, &s, rows, p, 0, @intCast(fx.d.taps * fx.d.D * 2)); // two passes: 1024 rows and 146
    _ = try drafts(d, &s, 77, p);
    const ref = try fx.ref.blockRows(a, rows, p, 77);
    try t.expect(relError(xBits(ad, fx.d.block, fx.d.D), ref) < 0.02);
}

test "the draft head's host-side numbers: lane bytes, refusals before any allocation" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const c = try fxm.tinyConfig(arena.allocator());
    const D = c.text.hidden;
    // two sliding layers of 1152 slots and one full layer of `capacity`, K and V, bf16 [kv_heads * 128] a slot
    const slot: u64 = 2 * 128 * 2;
    try t.expectEqual(2 * slot * (2 * 1152 + 2048) + D * 2 + c.mtp.?.block * @sizeOf(u32), draft.laneBytes(&c, 2048));
    var no_head = c;
    no_head.mtp = null;
    try t.expectEqual(@as(u64, 0), draft.laneBytes(&no_head, 2048));
    const g = try Gpu.init();
    defer g.deinit();
    var empty = std.testing.tmpDir(.{});
    defer empty.cleanup();
    const path = try empty.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(path);
    const none: mtl.Buffer = undefined;
    const tensor = fxm.tensorOf(none, &.{0});
    const tgt: draft.Target = .{ .embed = tensor, .embed_norm = tensor, .unembed = tensor };
    try t.expectError(error.NoDraftHead, draft.Adapter.init(t.allocator, t.io, path, g.device, g.queue, &no_head, tgt, .{ .capacity = 8 }));
    try t.expectError(error.UnexpectedDraftTarget, draft.Adapter.init(t.allocator, t.io, path, g.device, g.queue, &c, tgt, .{ .capacity = 8 }));
}

test "a tap past the backbone's 48 logical visits is refused, 48 (the final norm) is not" {
    const g = try Gpu.init();
    defer g.deinit();
    const fx = try fxm.Fixture.init(t.allocator, t.io, g.device);
    defer fx.deinit();
    var bad = fx.cfg;
    bad.mtp.?.taps[3] = bad.text.executionLayers() + 1;
    try t.expectError(error.UnsupportedDraftHead, draft.Adapter.init(t.allocator, t.io, fx.dir, g.device, g.queue, &bad, fx.tensors(), .{ .capacity = 16 }));
    const ok = try draft.Adapter.init(t.allocator, t.io, fx.dir, g.device, g.queue, &fx.cfg, fx.tensors(), .{ .capacity = 16 });
    ok.deinit();
}
