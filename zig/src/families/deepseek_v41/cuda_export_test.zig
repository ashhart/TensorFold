//! Export glue over synthetic snapshots: parts equal the library's export, import accepts them, bad snapshots refuse.
const std = @import("std");
const fabric = @import("fabric");
const g = @import("handoff/geometry.zig");
const plan = @import("handoff/plan.zig");
const ex = @import("handoff/export.zig");
const metal = @import("handoff/metal.zig");
const im = @import("handoff/import.zig");
const p2 = @import("handoff/p2.zig");
const s = @import("handoff/synth.zig");
const cx = @import("cuda_export.zig");

const geo = g.release;
const gpa = std.testing.allocator;

/// What the engine's snapshot and page reads hold for a synthetic world: bounded roles by name, pages as read.
pub const Snap = struct { pieces: []cx.Piece, pages: []cx.Pages };

pub fn snapshotOf(arena: std.mem.Allocator, w: s.World) !Snap {
    var list: std.ArrayList(cx.Piece) = .empty;
    for (w.cuda.swa, 0..) |ring, l| {
        try list.append(arena, .{ .role = try std.fmt.allocPrint(arena, "s.L{d}.swa.v", .{l}), .bytes = try arena.dupe(u8, ring.?.values) });
        try list.append(arena, .{ .role = try std.fmt.allocPrint(arena, "s.L{d}.swa.s", .{l}), .bytes = try arena.dupe(u8, ring.?.scales) });
        if (geo.sourceIndex(@intCast(l))) |si| if (geo.sources[si].ratio == 2)
            try list.append(arena, .{ .role = try std.fmt.allocPrint(arena, "s.L{d}.carry", .{l}), .bytes = try arena.dupe(u8, w.cuda.carries[si]) });
    }
    for (w.cuda.drafter, 0..) |ring, i| {
        try list.append(arena, .{ .role = try std.fmt.allocPrint(arena, "s.L{d}.ring.v", .{geo.layers + i}), .bytes = try arena.dupe(u8, ring.values) });
        try list.append(arena, .{ .role = try std.fmt.allocPrint(arena, "s.L{d}.ring.s", .{geo.layers + i}), .bytes = try arena.dupe(u8, ring.scales) });
    }
    const pages = try arena.alloc(cx.Pages, geo.sources.len);
    for (geo.sources, w.cuda.sources, pages) |src, st, *p| p.* = .{ .layer = src.layer, .comp = st.comp.ranks[0], .index = st.index.ranks[0] };
    return .{ .pieces = list.items, .pages = pages };
}

fn tokensOf(arena: std.mem.Allocator, n: u64) ![]u32 {
    const t = try arena.alloc(u32, n);
    for (t, 0..) |*x, i| x.* = @intCast(i * 5 + 3);
    return t;
}

fn target(arena: std.mem.Allocator, capacity: u32) !metal.Packed {
    const probe: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = 43, .capacity = capacity };
    return .{ .live = try arena.alloc(u8, probe.liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(capacity)), .layers = 43, .capacity = capacity };
}

/// Every part of the export's sources in manifest order.
fn partsOf(exp: fabric.producer.Export, out: *std.ArrayList([]const u8)) !void {
    for (exp.sources) |src| for (src.parts) |p| try out.append(std.testing.allocator, p);
}

fn exportCase(upto: u64, index: ex.IndexFormat, stale_drafter: ?u64) !void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const want_rows: u32 = @intCast(@min(upto, geo.window));
    const w = try s.make(arena, .{ .upto = upto, .index = index, .ds_rows = want_rows, .seed = upto * 7 + 1 });
    const snap = try snapshotOf(arena, w);
    const tokens = try tokensOf(arena, upto);
    const id: [16]u8 = @splat(0x5c);
    const built = try cx.build(gpa, id, tokens, .{ .pos = upto, .ds_rows = stale_drafter orelse want_rows, .index_format = index, .pieces = snap.pieces, .pages = snap.pages });
    defer built.deinit(gpa);

    // the library's own export of the same state, parts and layers
    const shape: plan.Shape = .{ .upto = upto, .ds_rows = want_rows };
    const want = try ex.build(arena, shape, w.cuda);
    const laid = try p2.layers(arena, want);
    try std.testing.expectEqual(laid.layers.len, built.exp.layers.len);
    var got_parts: std.ArrayList([]const u8) = .empty;
    defer got_parts.deinit(gpa);
    try partsOf(built.exp, &got_parts);
    var want_parts: std.ArrayList([]const u8) = .empty;
    defer want_parts.deinit(gpa);
    for (laid.sources) |src| for (src.parts) |p| try want_parts.append(gpa, p);
    try std.testing.expectEqual(want_parts.items.len, got_parts.items.len);
    var total: u64 = 0;
    for (want_parts.items, got_parts.items, 0..) |a, b, i| {
        total += b.len;
        if (stale_drafter) |valid| if (isRing(built.exp, i)) {
            // the rows below the valid ones are zero, the valid ones are the state's
            const cut = (want_rows - @min(valid, want_rows)) * 1024;
            for (b[0..cut]) |byte| try std.testing.expectEqual(@as(u8, 0), byte);
            try std.testing.expectEqualSlices(u8, a[cut..], b[cut..]);
            continue;
        };
        try std.testing.expectEqualSlices(u8, a, b);
    }
    try std.testing.expectEqual(total, built.bytes);
    try std.testing.expectEqual(total, cx.estimate(upto));
    try std.testing.expectEqualSlices(u32, tokens, built.exp.tokens);
    try std.testing.expectEqual(@as(u16, 2), built.exp.protocol());

    // the manifest is exactly the plan, and the library's importer takes every part
    const manifest = built.exp.toManifest("dsv41-test", 0, 1);
    try p2.check(arena, shape, manifest);
    const via = try target(arena, 2048);
    var imp = try im.Import.init(arena, shape, .{ .@"packed" = via }, .{});
    var next: usize = 0;
    for (built.exp.layers, built.exp.sources) |layer, src| {
        for (layer.parts, src.parts) |part, bytes| {
            const dtype: plan.Dtype = if (part.dtype == .float32) .f32 else .bf16;
            try imp.put(0, layer.index, part.name, dtype, part.shape[0], part.shape[1], bytes);
            next += 1;
        }
    }
    try std.testing.expectEqual(got_parts.items.len, next);
    const res = try imp.complete();
    try std.testing.expectEqual(upto, res.upto);
    try std.testing.expectEqual(upto - want_rows, res.ds_first);
}

/// Whether part `i` (in manifest order) is a drafter ring (layers past the backbone).
fn isRing(exp: fabric.producer.Export, i: usize) bool {
    var at: usize = 0;
    for (exp.layers) |l| {
        if (i < at + l.parts.len) return l.index >= geo.layers;
        at += l.parts.len;
    }
    return false;
}

test "an engine snapshot exports as the library's own parts, and the importer takes them: short, odd and even prompts" {
    inline for (.{ 2, 3, 130, 301, 517 }) |upto| try exportCase(upto, .fp8, null);
}

test "a bf16 index family exports as the library does" {
    try exportCase(300, .bf16, null);
}

test "drafter rows the snapshot never held are zero in the export, the valid ones are the state's" {
    try exportCase(300, .fp8, 40);
    try exportCase(300, .fp8, 0);
    try exportCase(20, .fp8, 5);
}

test "a snapshot without the drafter's rings exports without them" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = 200, .seed = 9 });
    const snap = try snapshotOf(arena, w);
    const tokens = try tokensOf(arena, 200);
    const built = try cx.build(gpa, @splat(1), tokens, .{ .pos = 200, .ds_rows = 0, .pieces = snap.pieces, .pages = snap.pages });
    defer built.deinit(gpa);
    for (built.exp.layers) |l| try std.testing.expect(l.index < geo.layers);
    try p2.check(arena, .{ .upto = 200 }, built.exp.toManifest("dsv41-test", 0, 1));
}

test "a snapshot without the decoder layers' rings is refused, never zero-filled" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = 200, .seed = 9 });
    const snap = try snapshotOf(arena, w);
    var kept: std.ArrayList(cx.Piece) = .empty;
    for (snap.pieces) |p| {
        // the decoder starts at layer 20: its rings and the drafter's are not in this snapshot
        const l = std.fmt.parseInt(u32, p.role[3..std.mem.indexOfScalarPos(u8, p.role, 3, '.').?], 10) catch unreachable;
        if (l < 20) try kept.append(arena, p);
    }
    const tokens = try tokensOf(arena, 200);
    try std.testing.expectError(error.MissingRing, cx.build(gpa, @splat(1), tokens, .{ .pos = 200, .ds_rows = 0, .pieces = kept.items, .pages = snap.pages }));
}

test "a missing carry, a missing source, a short pool and a mismatched prompt are refused by name" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = 200, .seed = 11 });
    const snap = try snapshotOf(arena, w);
    const tokens = try tokensOf(arena, 200);
    const st: cx.State = .{ .pos = 200, .ds_rows = 0, .pieces = snap.pieces, .pages = snap.pages };

    var no_carry: std.ArrayList(cx.Piece) = .empty;
    for (snap.pieces) |p| if (!std.mem.endsWith(u8, p.role, ".carry")) try no_carry.append(arena, p);
    var bad = st;
    bad.pieces = no_carry.items;
    try std.testing.expectError(error.MissingCarry, cx.build(gpa, @splat(1), tokens, bad));

    bad = st;
    bad.pages = snap.pages[1..];
    try std.testing.expectError(error.MissingSource, cx.build(gpa, @splat(1), tokens, bad));

    const short = try arena.dupe(cx.Pages, snap.pages);
    short[3].comp = short[3].comp[0 .. 100 * g.cuda_record_bytes]; // rows 0..99 of the 200 it needs
    bad = st;
    bad.pages = short;
    try std.testing.expectError(error.ShortPool, cx.build(gpa, @splat(1), tokens, bad));

    try std.testing.expectError(error.NotAHandoffPrompt, cx.build(gpa, @splat(1), tokens[0..199], st));
    try std.testing.expectError(error.NotAHandoffPrompt, cx.build(gpa, @splat(1), &.{}, .{ .pos = 0, .ds_rows = 0, .pieces = &.{}, .pages = &.{} }));

    // rings of different row counts are not one engine's
    const odd = try arena.dupe(cx.Piece, snap.pieces);
    for (odd) |*p| if (std.mem.eql(u8, p.role, "s.L5.swa.v")) {
        p.bytes = p.bytes[0 .. p.bytes.len / 2];
    };
    for (odd) |*p| if (std.mem.eql(u8, p.role, "s.L5.swa.s")) {
        p.bytes = p.bytes[0 .. p.bytes.len / 2];
    };
    bad = st;
    bad.pieces = odd;
    try std.testing.expectError(error.BadSnapshot, cx.build(gpa, @splat(1), tokens, bad));
}

/// A replay export of `upto` tokens: the decoder layers' rings hold another prompt's row where the replay never wrote.
fn replayCase(upto: u64) !void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = upto, .seed = upto * 3 + 5 });
    const full = try snapshotOf(arena, w);
    const tokens = try tokensOf(arena, upto);
    const keep: u64 = 127;
    const d: u32 = 20;
    // the replay's live slot: every row the replay wrote is the full forward's, the one before them is stale
    const live = try arena.dupe(cx.Piece, full.pieces);
    const stale = (upto -| 128) % 256;
    for (live) |*p| {
        const l = std.fmt.parseInt(u32, p.role[3..std.mem.indexOfScalarPos(u8, p.role, 3, '.').?], 10) catch unreachable;
        if (l < d or !std.mem.startsWith(u8, p.role[std.mem.indexOfScalarPos(u8, p.role, 3, '.').? + 1 ..], "swa")) continue;
        const row: usize = if (std.mem.endsWith(u8, p.role, ".v")) g.cuda_values_bytes else g.cuda_scale_bytes;
        p.bytes = try arena.dupe(u8, p.bytes);
        if (upto >= 128) @memset(p.bytes[stale * row ..][0..row], 0xa7);
    }
    const base: cx.State = .{ .pos = upto, .ds_rows = 0, .pieces = full.pieces, .pages = full.pages };
    var rep = base;
    rep.pieces = live;
    rep.decoder_from = d;
    rep.replay_rows = keep;
    const a = try cx.build(gpa, @splat(1), tokens, base);
    defer a.deinit(gpa);
    const b = try cx.build(gpa, @splat(1), tokens, rep);
    defer b.deinit(gpa);
    try std.testing.expectEqual(a.bytes, b.bytes);
    var zeroed: usize = 0;
    for (a.exp.layers, b.exp.layers, a.exp.sources, b.exp.sources) |la, lb, sa, sb| {
        try std.testing.expectEqual(la.index, lb.index);
        for (la.parts, sa.parts, sb.parts) |part, x, y| {
            const swa = std.mem.eql(u8, part.name, "swa");
            if (!(swa and la.index >= d and upto >= 128)) {
                try std.testing.expectEqualSlices(u8, x, y);
                continue;
            }
            // the window's oldest position: zero, the replay's rows are the full forward's
            for (y[0..1024]) |byte| try std.testing.expectEqual(@as(u8, 0), byte);
            try std.testing.expectEqualSlices(u8, x[1024..], y[1024..]);
            zeroed += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, if (upto >= 128) geo.layers - d else 0), zeroed);
    // the library's importer takes the replay export
    const shape: plan.Shape = .{ .upto = upto };
    try p2.check(arena, shape, b.exp.toManifest("dsv41-test", 0, 1));
}

test "a replay-mode export equals the full export but for the window's oldest row of the decoder layers, which is zero" {
    inline for (.{ 2, 100, 127, 128, 129, 300, 520 }) |upto| try replayCase(upto);
}
