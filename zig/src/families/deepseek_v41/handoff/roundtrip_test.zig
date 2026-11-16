//! CUDA state -> neutral parts -> Metal buffers, checked against the synthetic state's independent expectation.
const std = @import("std");
const g = @import("geometry.zig");
const c = @import("codec.zig");
const fp4 = @import("fp4.zig");
const plan = @import("plan.zig");
const ex = @import("export.zig");
const metal = @import("metal.zig");
const im = @import("import.zig");
const s = @import("synth.zig");

const geo = g.release;
const chunk: u64 = 70 << 10; // small parts, so a few pages cut several windows

const Case = struct { upto: u64, world: u32 = 1, split: bool = false, index: ex.IndexFormat = .fp8, ds_rows: u32 = 0 };

fn shapeFor(cs: Case, shard: u32) plan.Shape {
    return .{ .upto = cs.upto, .shard = shard, .shards = cs.world, .split = cs.split, .ds_rows = cs.ds_rows, .chunk = chunk };
}

/// Every shard's parts into the import, in a shuffled order.
fn land(arena: std.mem.Allocator, cs: Case, w: s.World, imp: *im.Import, rng: *s.Rng) !void {
    var all: std.ArrayList(struct { shard: u32, part: ex.Part }) = .empty;
    for (0..cs.world) |sh| {
        const parts = try ex.build(arena, shapeFor(cs, @intCast(sh)), w.cuda);
        for (parts) |p| try all.append(arena, .{ .shard = @intCast(sh), .part = p });
    }
    rng.p.random().shuffle(@TypeOf(all.items[0]), all.items);
    var buf: [40]u8 = undefined;
    for (all.items) |e| try imp.put(e.shard, e.part.spec.layer, e.part.spec.name(&buf), e.part.spec.dtype, e.part.spec.rows, e.part.spec.cols, e.part.bytes);
}

fn bytesOf(row: []const u16, out: []u8) void {
    for (row, 0..) |v, i| std.mem.writeInt(u16, out[2 * i ..][0..2], v, .little);
}

test "release layout: every landed byte is the independent expectation, everything else untouched" {
    inline for (.{
        Case{ .upto = 1 },
        Case{ .upto = 2 },
        Case{ .upto = 3, .ds_rows = 3 },
        Case{ .upto = 130, .ds_rows = 128 },
        Case{ .upto = 517, .world = 2, .split = true, .index = .bf16 },
        Case{ .upto = 1200, .world = 3, .split = true, .ds_rows = 77 },
        Case{ .upto = 1200, .index = .bf16, .ds_rows = 5 },
    }) |cs| try releaseCase(cs);
}

fn releaseCase(cs: Case) !void {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = cs.upto, .world = cs.world, .split = cs.split, .index = cs.index, .ds_rows = cs.ds_rows, .seed = cs.upto + cs.world });
    const slots = 144;
    const layers = try arena.alloc(metal.Release.Layer, geo.layers);
    for (layers, 0..) |*l, i| {
        l.* = .{ .window = try arena.alloc(u8, 2 * slots * 1024) };
        @memset(l.window, 0xEE);
        if (geo.sourceIndex(@intCast(i))) |si| {
            const rows: usize = @intCast(geo.rowsOf(si, cs.upto) + 1); // poolRows: one spare
            l.pool = try arena.alloc(u8, rows * 1024);
            l.keys = try arena.alloc(u8, rows * 68);
            @memset(l.pool, 0x77);
            @memset(l.keys, 0x77);
            l.carry = try arena.alloc(u8, 4 * 2048);
            @memset(l.carry, 0xEE);
        }
    }
    const drafter = try arena.alloc([]u8, 3);
    for (drafter) |*d| {
        d.* = try arena.alloc(u8, 133 * 1024);
        @memset(d.*, 0x33);
    }
    var rng = s.Rng.init(cs.upto);
    var imp = try im.Import.init(arena, shapeFor(cs, 0), .{ .release = .{ .layers = layers, .drafter = drafter } }, .{});
    try std.testing.expectError(error.Incomplete, imp.complete());
    try land(arena, cs, w, &imp, &rng);
    const res = try imp.complete();
    try std.testing.expectEqual(cs.upto, res.upto);
    try std.testing.expectEqual(cs.upto - cs.ds_rows, res.ds_first);

    const wrows = @min(cs.upto, geo.window);
    var want: [1024]u8 = undefined;
    for (layers, 0..) |l, li| {
        // the window: both mirrors of each live position; every other slot is zero (begin cleared it)
        var live: [2 * slots]bool = @splat(false);
        for (0..wrows) |i| {
            const pos = cs.upto - wrows + i;
            bytesOf(w.truth.swa[li][i * 512 ..][0..512], &want);
            for ([_]usize{ pos % slots, pos % slots + slots }) |slot| {
                try std.testing.expectEqualSlices(u8, &want, l.window[slot * 1024 ..][0..1024]);
                live[slot] = true;
            }
        }
        for (live, 0..) |on, slot| if (!on) try std.testing.expect(std.mem.allEqual(u8, l.window[slot * 1024 ..][0..1024], 0));
        const si = geo.sourceIndex(@intCast(li)) orelse continue;
        const rows = geo.rowsOf(si, cs.upto);
        for (0..@intCast(rows)) |r| {
            bytesOf(w.truth.comp[si][r * 512 ..][0..512], &want);
            try std.testing.expectEqualSlices(u8, &want, l.pool[r * 1024 ..][0..1024]);
            var key: [68]u8 = undefined;
            fp4.releaseIndexKey(w.truth.idx[si][r * 128 ..][0..128], &key);
            try std.testing.expectEqualSlices(u8, &key, l.keys[r * 68 ..][0..68]);
        }
        try std.testing.expect(std.mem.allEqual(u8, l.pool[@intCast(rows * 1024)..], 0x77)); // rows past the prefix untouched
        if (geo.sources[si].ratio == 2) {
            const slot: usize = @intCast((cs.upto - 1) % 2);
            const other = 1 - slot;
            const cw = l.carry;
            // kv slot / gate slot hold CUDA's carry row; the other slot is a new slot's: kv 0, score -inf
            try std.testing.expect(std.mem.eql(u8, cw[slot * 2048 ..][0..2048], w.cuda.carries[si][0..2048]));
            try std.testing.expect(std.mem.eql(u8, cw[(2 + slot) * 2048 ..][0..2048], w.cuda.carries[si][2048..4096]));
            try std.testing.expect(std.mem.allEqual(u8, cw[other * 2048 ..][0..2048], 0));
            for (0..512) |d| try std.testing.expectEqual(-std.math.inf(f32), @as(f32, @bitCast(std.mem.readInt(u32, cw[(2 + other) * 2048 + 4 * d ..][0..4], .little))));
        }
    }
    for (drafter, 0..) |d, stage| {
        for (0..cs.ds_rows) |i| {
            bytesOf(w.truth.ds[stage][i * 512 ..][0..512], &want);
            const pos = cs.upto - cs.ds_rows + i;
            try std.testing.expectEqualSlices(u8, &want, d[(pos % 128) * 1024 ..][0..1024]);
        }
        if (cs.ds_rows == 0) try std.testing.expect(std.mem.allEqual(u8, d, 0x33));
    }
}

test "served-lane layout: fp8 window rows, fp4 main and index rows, partials and capture where the engine reads them" {
    inline for (.{
        Case{ .upto = 1 },
        Case{ .upto = 3, .ds_rows = 3 },
        Case{ .upto = 300, .ds_rows = 128 },
        Case{ .upto = 517, .world = 2, .split = true },
        Case{ .upto = 1200, .world = 2, .split = true, .index = .bf16, .ds_rows = 9 },
    }) |cs| try packedCase(cs);
}

fn packedCase(cs: Case) !void {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = cs.upto, .world = cs.world, .split = cs.split, .index = cs.index, .ds_rows = cs.ds_rows, .seed = cs.upto * 7 + cs.world });
    const capacity: u32 = 2048;
    const pk: metal.Packed = .{ .live = try arena.alloc(u8, (metal.Packed{ .live = &.{}, .storage = &.{}, .layers = 43, .capacity = capacity }).liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(capacity)), .layers = 43, .capacity = capacity };
    @memset(pk.live, 0xEE);
    @memset(pk.storage, 0x77);
    var rng = s.Rng.init(cs.upto);
    var imp = try im.Import.init(arena, shapeFor(cs, 0), .{ .@"packed" = pk }, .{});
    try land(arena, cs, w, &imp, &rng);
    _ = try imp.complete();

    const P = metal.Packed;
    const wrows = @min(cs.upto, geo.window);
    var ringrow: [528]u8 = undefined;
    for (0..40) |li| {
        var live: [128]bool = @splat(false);
        for (0..wrows) |i| {
            const pos = cs.upto - wrows + i;
            fp4.window(w.truth.swa[li][i * 512 ..][0..512], &ringrow);
            const at = (li * 128 + pos % 128) * 528;
            try std.testing.expectEqualSlices(u8, &ringrow, pk.live[at..][0..528]);
            live[pos % 128] = true;
        }
        for (live, 0..) |on, slot| if (!on) try std.testing.expect(std.mem.allEqual(u8, pk.live[(li * 128 + slot) * 528 ..][0..528], 0));
    }
    for (0..3) |stage| for (0..cs.ds_rows) |i| {
        const pos = cs.upto - cs.ds_rows + i;
        fp4.window(w.truth.ds[stage][i * 512 ..][0..512], &ringrow);
        try std.testing.expectEqualSlices(u8, &ringrow, pk.live[((40 + stage) * 128 + pos % 128) * 528 ..][0..528]);
    };
    var main_off: usize = 0;
    for (geo.sources, 0..) |src, si| {
        const cap_rows: usize = capacity / src.ratio;
        const rows: usize = @intCast(geo.rowsOf(@intCast(si), cs.upto));
        var m: [288]u8 = undefined;
        var key: [68]u8 = undefined;
        for (0..rows) |r| {
            fp4.mainRow(w.truth.comp[si][r * 512 ..][0..512], &m);
            try std.testing.expectEqualSlices(u8, &m, pk.storage[main_off + r * 288 ..][0..288]);
            fp4.index(w.truth.idx[si][r * 128 ..][0..128], &key);
            try std.testing.expectEqualSlices(u8, &key, pk.storage[main_off + cap_rows * 288 + r * 68 ..][0..68]);
        }
        try std.testing.expect(std.mem.allEqual(u8, pk.storage[main_off + rows * 288 ..][0 .. (cap_rows - rows) * 288], 0x77));
        main_off += cap_rows * (288 + 68);
        if (src.ratio == 2) {
            const slot: usize = @intCast((cs.upto - 1) % 2);
            var part_off: usize = 43 * 128 * 528;
            for (P.ratios[0..si]) |r| part_off += 2 * r * 2048;
            try std.testing.expectEqualSlices(u8, w.cuda.carries[si][0..2048], pk.live[part_off + slot * 2048 ..][0..2048]);
            try std.testing.expectEqualSlices(u8, w.cuda.carries[si][2048..4096], pk.live[part_off + (2 + slot) * 2048 ..][0..2048]);
            const cap = 43 * 128 * 528 + P.partial_bytes + si * P.capture_bytes + @as(usize, @intCast((cs.upto - 1) % 8)) * 4096;
            try std.testing.expectEqualSlices(u8, w.cuda.carries[si][0..2048], pk.live[cap..][0..2048]);
            try std.testing.expectEqualSlices(u8, w.cuda.carries[si][2048..4096], pk.live[cap + 2048 ..][0..2048]);
        }
    }
}

test "the importer refuses a wrong, repeated, out-of-shape or non-finite part, and a layout that cannot hold the state" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cs: Case = .{ .upto = 300, .ds_rows = 4 };
    const w = try s.make(arena, .{ .upto = 300, .ds_rows = 4 });
    const layers = try arena.alloc(metal.Release.Layer, geo.layers);
    for (layers, 0..) |*l, i| {
        l.* = .{ .window = try arena.alloc(u8, 2 * 144 * 1024) };
        if (geo.sourceIndex(@intCast(i)) != null) l.* = .{ .window = l.window, .pool = try arena.alloc(u8, 301 * 1024), .keys = try arena.alloc(u8, 301 * 68), .carry = try arena.alloc(u8, 4 * 2048) };
    }
    const drafter = try arena.alloc([]u8, 3);
    for (drafter) |*d| d.* = try arena.alloc(u8, 128 * 1024);
    var imp = try im.Import.init(arena, shapeFor(cs, 0), .{ .release = .{ .layers = layers, .drafter = drafter } }, .{});
    const parts = try ex.build(arena, shapeFor(cs, 0), w.cuda);
    var buf: [40]u8 = undefined;
    const p0 = parts[0];
    const n0 = p0.spec.name(&buf);
    try std.testing.expectError(error.BadPart, imp.put(1, p0.spec.layer, n0, .bf16, p0.spec.rows, 512, p0.bytes)); // no shard 1
    try std.testing.expectError(error.BadPart, imp.put(0, p0.spec.layer, n0, .f32, p0.spec.rows, 512, p0.bytes));
    try std.testing.expectError(error.BadPart, imp.put(0, p0.spec.layer, n0, .bf16, p0.spec.rows - 1, 512, p0.bytes));
    try std.testing.expectError(error.BadPart, imp.put(0, p0.spec.layer, n0, .bf16, p0.spec.rows, 512, p0.bytes[1..]));
    try std.testing.expectError(error.BadPart, imp.put(0, p0.spec.layer, "comp@0", .bf16, p0.spec.rows, 512, p0.bytes));
    const bad = try arena.dupe(u8, p0.bytes);
    std.mem.writeInt(u16, bad[10..][0..2], 0x7f80, .little);
    try std.testing.expectError(error.NonFinite, imp.put(0, p0.spec.layer, n0, .bf16, p0.spec.rows, 512, bad));
    try imp.put(0, p0.spec.layer, n0, .bf16, p0.spec.rows, 512, p0.bytes);
    try std.testing.expectError(error.Duplicate, imp.put(0, p0.spec.layer, n0, .bf16, p0.spec.rows, 512, p0.bytes));
    try std.testing.expectError(error.Incomplete, imp.complete());

    // layouts that cannot hold the state
    layers[2].pool = layers[2].pool[0..1024];
    try std.testing.expectError(error.TargetTooSmall, im.Import.init(arena, shapeFor(cs, 0), .{ .release = .{ .layers = layers, .drafter = drafter } }, .{}));
    try std.testing.expectError(error.LayoutMismatch, im.Import.init(arena, shapeFor(cs, 0), .{ .release = .{ .layers = layers[0..39] } }, .{}));
    var pk: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = 40, .capacity = 299 };
    try std.testing.expectError(error.TargetTooSmall, im.Import.init(arena, shapeFor(cs, 0), .{ .@"packed" = pk }, .{}));
    pk.layers = 41;
    try std.testing.expectError(error.LayoutMismatch, im.Import.init(arena, shapeFor(cs, 0), .{ .@"packed" = pk }, .{}));
}

test "a layout without drafter rings, or a caller who declines them, drops them and says so" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cs: Case = .{ .upto = 140, .ds_rows = 6 };
    const w = try s.make(arena, .{ .upto = 140, .ds_rows = 6 });
    const capacity: u32 = 256;
    const pk: metal.Packed = .{ .live = try arena.alloc(u8, (metal.Packed{ .live = &.{}, .storage = &.{}, .layers = 40, .capacity = capacity }).liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(capacity)), .layers = 40, .capacity = capacity };
    var rng = s.Rng.init(1);
    var imp = try im.Import.init(arena, shapeFor(cs, 0), .{ .@"packed" = pk }, .{});
    try land(arena, cs, w, &imp, &rng);
    const res = try imp.complete();
    try std.testing.expect(res.drafter_dropped);
    try std.testing.expectEqual(res.ds_end, res.ds_first);
    var declined = try im.Import.init(arena, shapeFor(cs, 0), .{ .@"packed" = pk }, .{ .drafter = false });
    try land(arena, cs, w, &declined, &rng);
    try std.testing.expect((try declined.complete()).drafter_dropped);
}

test "Engram's lookback ids are the last ngram - 1 prompt tokens before upto, fewer for a short prompt" {
    const toks = [_]u32{ 10, 11, 12, 13, 14, 15 };
    try std.testing.expectEqualSlices(u32, &.{ 13, 14, 15 }, im.lookbackIds(&toks, 6, 3));
    try std.testing.expectEqualSlices(u32, &.{ 11, 12, 13 }, im.lookbackIds(&toks, 4, 3));
    try std.testing.expectEqualSlices(u32, &.{ 10, 11 }, im.lookbackIds(&toks, 2, 3));
    try std.testing.expectEqualSlices(u32, &.{}, im.lookbackIds(&toks, 0, 3));
    try std.testing.expectEqualSlices(u32, &.{ 13, 14, 15 }, im.lookbackIds(&toks, 99, 3));
}

test "the exporter refuses state the CUDA snapshot does not hold: a missing ring, shard, carry or a short pool" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const w = try s.make(arena, .{ .upto = 200, .world = 2, .split = true });
    var shape = shapeFor(.{ .upto = 200, .world = 2, .split = true }, 0);
    _ = try ex.build(arena, shape, w.cuda);
    // a CED replay snapshot: rings below the decoder only
    const swa = try arena.dupe(?ex.Ring, w.cuda.swa);
    swa[25] = null;
    var cuda = w.cuda;
    cuda.swa = swa;
    try std.testing.expectError(error.MissingRing, ex.build(arena, shape, cuda));
    cuda = w.cuda;
    cuda.carries = w.cuda.carries[0..1];
    try std.testing.expectError(error.MissingCarry, ex.build(arena, shape, cuda));
    const srcs = try arena.dupe(ex.SourceState, w.cuda.sources);
    const ranks = try arena.dupe([]const u8, srcs[0].comp.ranks);
    ranks[0] = &.{}; // this shard's own rank holds nothing
    srcs[0].comp.ranks = ranks;
    cuda = w.cuda;
    cuda.sources = srcs;
    try std.testing.expectError(error.MissingShard, ex.build(arena, shape, cuda));
    ranks[0] = srcs[1].comp.ranks[0][0..100];
    try std.testing.expectError(error.ShortPool, ex.build(arena, shape, cuda));
    cuda = w.cuda;
    cuda.sources = srcs[1..];
    try std.testing.expectError(error.MissingSource, ex.build(arena, shape, cuda));
    cuda = w.cuda;
    cuda.rings.rows = 100;
    try std.testing.expectError(error.BadRingRows, ex.build(arena, shape, cuda));
    shape.ds_rows = 4;
    try std.testing.expectError(error.MissingRing, ex.build(arena, shape, w.cuda)); // drafter rows asked, none held
}
