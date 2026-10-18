//! Actual GLM prompt workflow with recording device primitives and sequence storage.
const std = @import("std");
const glm = @import("glm");
const t = std.testing;
const fw = struct {
    pub const Taps = struct { out: u64 };
    pub const prefill_rows = 2048;
    pub const window_rows = 8;
    pub const State = struct {
        pos: usize = 0,
        mtp_len: usize = 0,
        prompt_hidden: u64 = 0,
        pub fn setMtpLen(s: *State, _: u8, n: usize) !void {
            s.mtp_len = n;
        }
    };
    pub const Seg = struct { st: *State, row0: usize, rows: usize };
    pub const Buffers = struct { fnormed: u64, logits: u64 };
    pub const Rank = struct { w: *u8, st: *State, b: *Buffers, segs: []const Seg = &.{} };
    pub var observed_ring = false;
    pub var observed_taps = false;
    pub const Forward = struct {
        d: u8,
        m: *u8,
        stream: u8,
        ranks: []const Rank,
        exchange: u8,
        pieces: ?*u8,
        split: bool,
        kda_wide: bool,
        ring: bool = false,
        taps: ?Taps = null,
        pub fn run(f: *Forward, ids: []const i32) !void {
            observed_ring = f.ring;
            observed_taps = f.taps != null;
            if (f.taps) |taps| if (taps.out != 0) {
                for (ids, 0..) |id, i| {
                    const ptr: *u32 = @ptrFromInt(taps.out + i * glm.dflash.TAP_ROW);
                    ptr.* = @as(u32, @intCast(id)) + @as(u32, f.d) * 1000;
                }
            };
            const b = f.ranks[0].b;
            const hidden: [*]u16 = @ptrFromInt(b.fnormed);
            const logits: [*]u16 = @ptrFromInt(b.logits);
            for (ids, 0..) |id, i| hidden[i] = @intCast(id);
            for (f.ranks[0].segs, 0..) |g, i| logits[i] = @intCast(ids[g.row0 + g.rows - 1]);
        }
        pub fn mtp(f: *Forward, _: []const i32, _: []const u64) !void {
            const first: *u16 = @ptrFromInt(f.ranks[0].b.fnormed);
            first.* = 777;
        }
        pub fn commit(f: *Forward, n: usize) !void {
            f.ranks[0].st.pos += n;
        }
    };
};
const engine = struct {
    pub const max_together = 16;
    pub const prompt_rows = 2048;
    pub const Prompt = glm.engine.Prompt;
    pub const Seq = struct { st: fw.State = .{}, held_at: u64 = 0 };
    pub const Engine = struct {
        pub const holdPrompt = glm.prompt_hold.save;
        capacity: usize = 4096,
        c: struct { hidden: usize = 1 } = .{},
        hold: struct { ptr: u64 },
        mtp: bool = true,
        cur: *Seq = undefined,
        seqs: [32]Seq = @splat(.{}),
        bp: fw.Buffers,
        share: struct { vocab: Vocab = .{} } = .{},
        settings: struct { split: bool = true, ring: bool = true, kda_wide: bool = true } = .{},
        d: u8 = 0,
        m: u8 = 0,
        stream: u8 = 0,
        w: u8 = 0,
        pieces: u8 = 0,
        drafter: ?*Drafter = null,
        const Vocab = struct {
            pub fn len(_: Vocab) usize {
                return 1;
            }
        };
        pub fn bind(e: *Engine, slot: u32) !void {
            if (slot >= 32) return error.BadSlot;
            e.cur = &e.seqs[slot];
        }
        pub fn gate(_: *Engine, _: ?u8) !void {}
        pub fn exchange(_: *Engine) u8 {
            return 0;
        }
        pub fn draw(_: *Engine, logits: u64, _: usize, _: usize, out: []u32) !void {
            const p: *const u16 = @ptrFromInt(logits);
            out[0] = p.*;
        }
        pub fn copy(_: *Engine, dst: u64, src: u64, n: usize) !void {
            const d: [*]u8 = @ptrFromInt(dst);
            const s: [*]const u8 = @ptrFromInt(src);
            @memcpy(d[0..n], s[0..n]);
        }
    };
    pub const Drafter = struct {
        taps_prefill: u64 = 0,
        states: [32]Slot = @splat(.{}),
        const Slot = struct { ring: ?struct { ptr: u64 } = null };
        pub fn taps(dr: *Drafter, _: bool) fw.Taps {
            return .{ .out = dr.taps_prefill };
        }
        pub const promptSegment = glm.dflash.ingestPrompt;
        pub fn slot(dr: *Drafter, i: u32) !*Slot {
            return &dr.states[i];
        }
        pub fn ingest(_: *Drafter, _: *Engine, runs: []const glm.dflash.Run) !void {
            for (runs) |run| for (0..run.n) |i| {
                const src: *const u32 = @ptrFromInt(run.taps + i * glm.dflash.TAP_ROW);
                const dst: [*]u32 = @ptrFromInt(run.ring);
                dst[@as(usize, @intCast(run.pos0)) + i] = src.*;
            };
        }
    };
};
const FakeEngine = engine.Engine;
const Box = struct {
    hidden: [2048]u16 = @splat(0),
    logits: [16]u16 = @splat(0),
    hold: [32]u16 = @splat(0),
    fn make(b: *Box) FakeEngine {
        var e: FakeEngine = .{ .hold = .{ .ptr = @intFromPtr(&b.hold) }, .bp = .{ .fnormed = @intFromPtr(&b.hidden), .logits = @intFromPtr(&b.logits) } };
        for (&e.seqs, 0..) |*q, i| q.st.prompt_hidden = @intFromPtr(&b.hold[i]);
        return e;
    }
};
fn pass(e: *FakeEngine, parts: []const engine.Prompt) !void {
    var first: [16]u32 = undefined;
    try glm.prompts.pass(e, parts, &first, fw.Forward, fw.Rank);
}
fn held(e: *FakeEngine, slot: usize) u16 {
    e.bind(@intCast(slot)) catch unreachable;
    const p: *const u16 = @ptrFromInt(glm.prompt_hold.input(e));
    return p.*;
}
test "GLM prompt workflow: multiple passes, nonzero and reused slots retain the first-draft rows" {
    var b: Box = .{};
    var e = b.make();
    const a: [1025]u32 = @splat(41);
    const z: [1025]u32 = @splat(93);
    try pass(&e, &.{.{ .slot = 7, .ids = &a }});
    try pass(&e, &.{.{ .slot = 11, .ids = &z }});
    try t.expectEqual(@as(u16, 41), held(&e, 7));
    try t.expectEqual(@as(u16, 93), held(&e, 11));
    e.seqs[11].st.pos = 0;
    e.seqs[11].st.mtp_len = 0;
    try pass(&e, &.{.{ .slot = 11, .ids = &.{ 12, 13, 14 } }});
    try t.expectEqual(@as(u16, 41), held(&e, 7));
    try t.expectEqual(@as(u16, 14), held(&e, 11));
}
test "GLM prompt workflow: normal path and batched passes retain the incumbent" {
    var b: Box = .{};
    var e = b.make();
    var prior: u16 = 71;
    try e.bind(23);
    try e.holdPrompt(@intFromPtr(&prior));
    try pass(&e, &.{ .{ .slot = 4, .ids = &.{ 31, 32 } }, .{ .slot = 9, .ids = &.{ 61, 62, 63 } } });
    try t.expectEqual(@as(u16, 71), held(&e, 23));
    try t.expectEqual(@as(u16, 32), held(&e, 4));
    try t.expectEqual(@as(u16, 63), held(&e, 9));
    prior = 81;
    try e.bind(9);
    try e.holdPrompt(@intFromPtr(&prior));
    try t.expectEqual(@as(u16, 71), held(&e, 23));
    try t.expectEqual(@as(u16, 32), held(&e, 4));
    try t.expectEqual(@as(u16, 81), held(&e, 9));
}
test "GLM prompt workflow: ring and enabled drafter taps reach the actual pass" {
    var b: Box = .{};
    var e = b.make();
    var dr: engine.Drafter = .{};
    e.drafter = &dr;
    var tap_rows: [2 * glm.dflash.TAP_ROW]u8 align(8) = @splat(0);
    var contexts: [32][16]u32 = @splat(@splat(0));
    dr.taps_prefill = @intFromPtr(&tap_rows);
    for (&dr.states, 0..) |*v, i| v.ring = .{ .ptr = @intFromPtr(&contexts[i]) };
    try pass(&e, &.{ .{ .slot = 3, .ids = &.{41} }, .{ .slot = 8, .ids = &.{93} } });
    try t.expect(fw.observed_ring);
    try t.expect(fw.observed_taps);
}

test "GLM prompt workflow: DFlash context equals ordinary ingest at own offsets and positions on both ranks" {
    const aa = [_]u32{ 11, 12, 13 };
    const bb = [_]u32{ 51, 52, 53, 54, 55 };
    for (0..2) |rank| {
        var b: Box = .{};
        var e = b.make();
        e.d = @intCast(rank);
        var tap_rows: [8 * glm.dflash.TAP_ROW]u8 align(8) = @splat(0xa5);
        var actual: [32][16]u32 = @splat(@splat(0xcccccccc));
        var ordinary: [32][16]u32 = actual;
        var dr: engine.Drafter = .{ .taps_prefill = @intFromPtr(&tap_rows) };
        for (&dr.states, 0..) |*v, i| v.ring = .{ .ptr = @intFromPtr(&actual[i]) };
        e.drafter = &dr;
        try pass(&e, &.{ .{ .slot = 4, .ids = &aa }, .{ .slot = 9, .ids = &bb } });
        var single: [5 * glm.dflash.TAP_ROW]u8 align(8) = @splat(0xa5);
        dr.taps_prefill = @intFromPtr(&single);
        for (&dr.states, 0..) |*v, i| v.ring = .{ .ptr = @intFromPtr(&ordinary[i]) };
        for ([_][]const u32{ &aa, &bb }, [_]u32{ 4, 9 }) |ids, slot| {
            for (ids, 0..) |id, i| {
                const ptr: *u32 = @ptrFromInt(dr.taps_prefill + i * glm.dflash.TAP_ROW);
                ptr.* = id + @as(u32, @intCast(rank)) * 1000;
            }
            try glm.dflash.Drafter.prompt(&dr, &e, slot, 0, ids.len, ids.len);
            for (ids, 0..) |id, i| try t.expectEqual(id + @as(u32, @intCast(rank)) * 1000, actual[slot][i]);
        }
        try t.expectEqualSlices(u8, std.mem.asBytes(&ordinary), std.mem.asBytes(&actual));
    }
}
test "GLM prompt workflow: context window cut preserves sequence positions independently of tap offsets" {
    const dr = glm.dflash;
    const run = (try dr.promptRun(1000, 2000, 2048, 2048, 5000, 0)).?;
    try t.expectEqual(@as(i64, 2952), run.pos0);
    try t.expectEqual(@as(usize, 1144), run.n);
    try t.expectEqual(@as(u64, 2000 + 904 * dr.TAP_ROW), run.taps);
    const shifted = (try dr.promptRun(1000, 2000, 0, 5, 5, 11)).?;
    try t.expectEqual(@as(i64, 0), shifted.pos0);
    try t.expectEqual(@as(u64, 2000 + 11 * dr.TAP_ROW), shifted.taps);
    try t.expectEqual(@as(?dr.Run, null), try dr.promptRun(1000, 2000, 0, 2048, 5000, 0));
    try t.expectError(error.BadPromptContext, dr.promptRun(0, 0, 9, 2, 10, 0));
    try t.expectError(error.DrafterContextTooWide, dr.promptRun(0, 0, 0, 5, 5, 2044));
}

test "GLM prompt workflow: both ranks share exact packets and reject malformed batches" {
    const wire = glm.prompt_wire;
    var buf: [wire.words]u32 = undefined;
    const parts = [_]engine.Prompt{ .{ .slot = 7, .ids = &.{ 11, 12 } }, .{ .slot = 13, .ids = &.{ 21, 22, 23 } } };
    const ids = try wire.encode(&parts, 4096, &buf);
    var decoded: [16]engine.Prompt = undefined;
    const back = try wire.decode(ids, 2, 4096, &decoded);
    for (parts, back) |a, b| {
        try t.expectEqual(a.slot, b.slot);
        try t.expectEqualSlices(u32, a.ids, b.ids);
    }
    try t.expectEqual(@as(u32, 12), @backingInt(glm.follow.Kind.block));
    try t.expectEqual(@as(u32, 13), @backingInt(glm.follow.Kind.many));
    try t.expectError(error.BadStep, wire.decode(ids, 0, 4096, &decoded));
    try t.expectError(error.BadStep, wire.decode(ids, 17, 4096, &decoded));
    for (0..ids.len) |cut| try t.expectError(error.BadStep, wire.decode(ids[0..cut], 2, 4096, &decoded));
    buf[ids.len] = 99;
    try t.expectError(error.BadStep, wire.decode(buf[0 .. ids.len + 1], 2, 4096, &decoded));
    buf[2] = buf[0];
    try t.expectError(error.BadStep, wire.decode(ids, 2, 4096, &decoded));
    buf[2] = 13;
    buf[1] = 0;
    try t.expectError(error.BadStep, wire.decode(ids, 2, 4096, &decoded));
    try t.expectError(error.BadPrompts, wire.encode(&.{.{ .slot = 1, .ids = &.{} }}, 4096, &buf));
}
