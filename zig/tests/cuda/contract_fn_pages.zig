//! flashnext_attn.cu's caches through a page table: a slot's prep, pooling, scores and attention, paged == contiguous.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const media = @import("contract_fn_pages_media.zig");

const q_heads = 24;
const kv_heads = 2;
const head_dim = 256;
const index_heads = 4;
const index_dim = 128;
const half = 32;
const proj_n = 13952;
const out_k = q_heads * head_dim;
const ratio = 4;
const top = 512;
const idw = top * ratio + ratio;
const dense_keys = (top + 1) * ratio - 1;
const att_rows = 256;
const fused_rows = 1024;
const nch = 5;
const eps: f32 = 1e-6;
/// A key's or value's row: two heads of 256.
const row_kv = kv_heads * head_dim;
/// Elements a position adds to an attention slot of a page: its key and value, indexer key and a quarter pooled row.
const slot_elems = 2 * row_kv + index_dim + index_dim / ratio;

/// Two streams' cached positions, the rows each adds, positions a page, the slot the layer takes and slots a page.
pub const Case = struct { lens: [2]usize, rows: usize, ptok: usize, slot: usize, slots: usize, image: ?media.Spec = null };
pub const cases = [_]Case{
    .{ .lens = .{ 0, 3 }, .rows = 1, .ptok = 16, .slot = 1, .slots = 12 }, // a stream's first key
    .{ .lens = .{ 127, 128 }, .rows = 16, .ptok = 32, .slot = 1, .slots = 12 }, // a verify window over page edges
    .{ .lens = .{ 700, 33 }, .rows = 40, .ptok = 48, .slot = 0, .slots = 12 }, // a page no 512-key chunk divides
    .{ .lens = .{ 2040, 2100 }, .rows = 16, .ptok = 64, .slot = 5, .slots = 12 }, // past 2,048 keys: selected blocks
    .{ .lens = .{ 4095, 5000 }, .rows = 16, .ptok = 256, .slot = 12, .slots = 13 }, // the MTP head's slot
    .{ .lens = .{ 9000, 300 }, .rows = 1, .ptok = 512, .slot = 11, .slots = 12 }, // a long sparse decode row
    .{ .lens = .{ 2600, 0 }, .rows = 300, .ptok = 128, .slot = 2, .slots = 12 }, // prompt rows on the fused launch
    .{ .lens = .{ 0, 1 }, .rows = 64, .ptok = 32, .slot = 0, .slots = 12, .image = media.specs[0] },
    .{ .lens = .{ 63, 126 }, .rows = 16, .ptok = 64, .slot = 1, .slots = 12, .image = media.specs[1] },
    .{ .lens = .{ 2045, 2101 }, .rows = 16, .ptok = 96, .slot = 5, .slots = 12, .image = media.specs[2] },
    .{ .lens = .{ 126, 257 }, .rows = 300, .ptok = 128, .slot = 12, .slots = 13, .image = media.specs[3] },
    .{ .lens = .{ 4095, 5000 }, .rows = 1, .ptok = 256, .slot = 12, .slots = 13, .image = media.specs[2] },
    .{ .lens = .{ 9000, 3000 }, .rows = 1, .ptok = 512, .slot = 11, .slots = 12, .image = media.specs[2] },
};

/// A cache of the slot: K, V, indexer keys, pooled keys.
const Cache = enum { k, v, ik, pooled };

/// Elements a row of `c` holds.
fn rowOf(c: Cache) usize {
    return switch (c) {
        .k, .v => row_kv,
        .ik, .pooled => index_dim,
    };
}

/// One case's host data: each stream's logical caches (rows as the contiguous layout keeps them), inputs, page ids.
pub const Fixture = struct {
    c: Case,
    cap: [2]usize, // positions a stream's pages cover (and its contiguous buffers hold)
    caches: [2][4][]u16, // [stream][cache] rows [0, cap): written rows random, the rest NaN
    proj: [2][]u16, // [rows][proj_n] a stream
    qw: []f32,
    kw: []f32,
    iw: []f32,
    ikw: []f32,
    inv: []f32,
    image: ?[2]media.Plan,
    ids: [2][]u32, // each stream's pages in the arena, shuffled across both
    pages: usize, // the arena's pages: both streams' and two spare

    /// A page holds every slot's caches, addressed in elements without narrowing a byte stride to row indices.
    fn pageElems(f: Fixture) usize {
        std.debug.assert(f.c.ptok % ratio == 0);
        return f.c.slots * f.c.ptok * slot_elems;
    }

    /// Where cache `c` of the slot starts in a page, in elements.
    fn slotAt(f: Fixture, c: Cache) usize {
        const p = f.c.ptok;
        const base = f.c.slot * p * slot_elems;
        return base + switch (c) {
            .k => 0,
            .v => p * row_kv,
            .ik => 2 * p * row_kv,
            .pooled => 2 * p * row_kv + p * index_dim,
        };
    }

    /// Rows of `c` a page holds.
    fn perPage(f: Fixture, c: Cache) usize {
        return if (c == .pooled) f.c.ptok / ratio else f.c.ptok;
    }

    fn rows(f: Fixture, st: usize, c: Cache) usize {
        return if (c == .pooled) f.cap[st] / ratio else f.cap[st];
    }

    /// The arena element where row `p` of stream `st`'s cache `c` sits under page ids `ids`.
    fn placed(f: Fixture, ids: []const u32, c: Cache, p: usize) usize {
        const per = f.perPage(c);
        return ids[p / per] * f.pageElems() + f.slotAt(c) + (p % per) * rowOf(c);
    }

    pub fn make(a: std.mem.Allocator, c: Case, seed: u64) !Fixture {
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var f: Fixture = undefined;
        f.c = c;
        f.image = if (c.image) |im| try im.plans(a) else null;
        var total: usize = 2;
        for (0..2) |st| {
            f.cap[st] = (c.lens[st] + c.rows + c.ptok - 1) / c.ptok * c.ptok;
            total += f.cap[st] / c.ptok;
            for (std.enums.values(Cache), 0..) |cache, i| {
                const n = f.rows(st, cache);
                const written = if (cache == .pooled) c.lens[st] / ratio else c.lens[st];
                const x = try a.alloc(u16, n * rowOf(cache));
                for (x[0 .. written * rowOf(cache)]) |*v| v.* = bf16(rng.float(f32) * 2 - 1);
                @memset(x[written * rowOf(cache) ..], nan); // rows no pass wrote: a read of one shows
                f.caches[st][i] = x;
            }
            f.proj[st] = try a.alloc(u16, c.rows * proj_n);
            for (f.proj[st]) |*v| v.* = bf16((rng.float(f32) * 2 - 1) * 2);
        }
        f.pages = total;
        f.qw = try weights(a, rng, head_dim);
        f.kw = try weights(a, rng, head_dim);
        f.iw = try weights(a, rng, index_dim);
        f.ikw = try weights(a, rng, index_dim);
        f.inv = try a.alloc(f32, half);
        for (f.inv, 0..) |*o, i| o.* = @floatCast(std.math.pow(f64, 10_000_000.0, -@as(f64, @floatFromInt(i)) / half));
        f.ids = try shuffled(a, f, rng);
        return f;
    }

    /// Both streams' page ids from one shuffle of the arena's pages (two left over).
    fn shuffled(a: std.mem.Allocator, f: Fixture, rng: std.Random) ![2][]u32 {
        const order = try a.alloc(u32, f.pages);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        rng.shuffle(u32, order);
        const n0 = f.cap[0] / f.c.ptok;
        return .{ order[0..n0], order[n0..][0 .. f.cap[1] / f.c.ptok] };
    }

    /// The arena: both streams' caches at their pages under `ids`, every other element NaN (free pages and slots).
    fn arena(f: Fixture, a: std.mem.Allocator, ids: [2][]const u32) ![]u16 {
        const out = try a.alloc(u16, f.pages * f.pageElems());
        @memset(out, nan);
        for (0..2) |st| for (std.enums.values(Cache), 0..) |c, i| for (0..f.rows(st, c)) |p| {
            @memcpy(out[f.placed(ids[st], c, p)..][0..rowOf(c)], f.caches[st][i][p * rowOf(c) ..][0..rowOf(c)]);
        };
        return out;
    }

    /// The arena's rows of stream `st` read back as its contiguous caches.
    fn unpage(f: Fixture, a: std.mem.Allocator, all: []const u16, ids: []const u32, st: usize) ![4][]u16 {
        var out: [4][]u16 = undefined;
        for (std.enums.values(Cache), 0..) |c, i| {
            out[i] = try a.alloc(u16, f.rows(st, c) * rowOf(c));
            for (0..f.rows(st, c)) |p| @memcpy(out[i][p * rowOf(c) ..][0..rowOf(c)], all[f.placed(ids, c, p)..][0..rowOf(c)]);
        }
        return out;
    }

    /// Whether arena element `e` belongs to a row of either stream's caches (else it must never change).
    fn owned(f: Fixture, a: std.mem.Allocator, ids: [2][]const u32) ![]bool {
        const out = try a.alloc(bool, f.pages * f.pageElems());
        @memset(out, false);
        for (0..2) |st| for (std.enums.values(Cache)) |c| for (0..f.rows(st, c)) |p| {
            @memset(out[f.placed(ids[st], c, p)..][0..rowOf(c)], true);
        };
        return out;
    }
};

const nan: u16 = 0x7fc1; // a quiet NaN no kernel's arithmetic makes

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn weights(a: std.mem.Allocator, rng: std.Random, n: usize) ![]f32 {
    const out = try a.alloc(f32, n);
    for (out) |*x| x.* = 0.5 + rng.float(f32);
    return out;
}

fn cdiv(x: usize, y: usize) usize {
    return (x + y - 1) / y;
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

/// A cached value row for dense attention, or a pooled row scored by sparse attention, replaced with NaNs.
fn plant(f: Fixture, all: []u16, ids: [2][]const u32, row: usize) usize {
    const st: usize = if (f.c.lens[0] > 0) 0 else 1;
    const cache: Cache = if ((f.c.lens[st] + 1) / ratio <= top) .v else .pooled;
    @memset(all[f.placed(ids[st], cache, row)..][0..rowOf(cache)], nan);
    return st;
}

/// A dense value row is read; a sparse pooled row must have a positive score so replacing it with NaNs changes it.
fn controlRow(f: Fixture, scores: []const u8) !usize {
    const st: usize = if (f.c.lens[0] > 0) 0 else 1;
    if ((f.c.lens[st] + 1) / ratio <= top) return 0;
    for (0..f.c.lens[st] / ratio) |row| {
        const score: f32 = @bitCast(std.mem.readInt(u32, scores[row * 4 ..][0..4], .little));
        if (std.math.isFinite(score) and score > 0) return row;
    }
    return error.ControlHasNoPositiveScore;
}

/// One stream's pass outputs: queries, indexer queries, attention rows, the last block's scores and selection.
const Out = struct { q: []u8, iq: []u8, ao: []u8, sc: []u8, ids: []u8, nk: []u8, spr: []u8 };

/// Both streams' outputs and caches after a launch; on the arena, its bytes before and after.
pub const Run = struct { out: [2]Out, caches: [2][4][]u16, arena: []u16 = &.{}, before: []u16 = &.{} };

/// Device buffers of one stream's pass.
const Bufs = struct { proj: u64, pos: u64, qw: u64, kw: u64, iw: u64, ikw: u64, inv: u64, q: u64, iq: u64, kc: u64, vc: u64, ikc: u64, pooled: u64, sc: u64, ids: u64, nk: u64, spr: u64, po: u64, pm: u64, pl: u64, ao: u64 };

/// cuda_batch.zig's attnRows over `n` rows from `pos0`: prep, pooling, each block's scores and selection, attention.
fn pass(g: *const flash.Glue, s: cuda.Stream, select: cuda.Function, pos0: usize, n: usize, nb: usize, b: Bufs, pg: flash.Pg, image: ?media.View) !void {
    if (image) |im| {
        try im.attn(s, b.proj, b.pos, b.qw, b.kw, b.iw, b.inv, b.q, b.kc, b.vc, b.iq, b.ikc, eps, n, pg);
        try im.pooled(s, b.ikc, b.pooled, b.pos, b.ikw, b.inv, eps, n, pg);
    } else {
        try g.attnPrep(s, b.proj, b.pos, b.qw, b.kw, b.iw, b.inv, b.q, b.kc, b.vc, b.iq, b.ikc, eps, n, pg);
        try g.qsaPool(s, b.ikc, b.pooled, b.pos, b.ikw, b.inv, eps, n, pg);
    }
    const fused = n > att_rows;
    const group: usize = if (fused) fused_rows else att_rows;
    var g0: usize = 0;
    while (g0 < n) : (g0 += group) {
        const gm = @min(group, n - g0);
        var r0 = g0;
        while (r0 < g0 + gm) : (r0 += att_rows) {
            const m = @min(att_rows, g0 + gm - r0);
            const ends = pos0 + r0 + m;
            const pos = b.pos + (r0 / att_rows) * 16;
            const sel = r0 - g0;
            if (ends / ratio > top) try g.qsaScores(s, b.iq + r0 * index_heads * index_dim * 2, b.pooled, pos, b.sc, nb, m, @min(nb, cdiv(ends, ratio)), pg);
            var a: cuda.Args = .{};
            inline for (.{ b.sc, pos, b.ids + sel * idw * 4, b.nk + sel * 4, b.spr + sel * 4, cint(nb), cint(idw), cint(top), cint(ratio) }) |x| a.add(x);
            try cuda.launch.launch(select, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 1024 } }, s, &a);
            if (!fused) {
                const chunks = cdiv(@min(ends, dense_keys), 512);
                try g.attnChunks(s, b.q + r0 * q_heads * head_dim * 2, b.kc, b.vc, pos, b.po, b.pm, b.pl, b.ids, b.nk, b.spr, m, kv_heads, chunks, true, pg);
                try g.attnMerge(s, b.po, b.pm, b.pl, pos, b.ao + r0 * out_k * 2, b.nk, b.spr, m, kv_heads, true);
            }
        }
        if (fused) try g.attnFused(s, b.q + g0 * q_heads * head_dim * 2, b.kc, b.vc, b.pos + (g0 / att_rows) * 16, b.ao + g0 * out_k * 2, b.ids, b.nk, b.spr, gm, kv_heads, pg);
    }
}

/// Both streams' passes: contiguous (`placed` null), or on the arena at the pages `placed` names, read through `given`.
pub fn launch(gpu: check.Gpu, s: cuda.Stream, g: *const flash.Glue, select: cuda.Function, f: Fixture, placed: ?[2][]const u32, given: ?[2][]const u32, planted: ?usize, a: std.mem.Allocator) !Run {
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer {
        cuda.completion.drain(gpu.d, s, "the page contract did not drain");
        rig.deinit();
    }
    var r: Run = .{ .out = undefined, .caches = undefined };
    var kv: [2][4]u64 = undefined; // each stream's cache bases: its own buffers, or the arena's slot in page 0
    var arena_ptr: u64 = 0;
    if (placed) |ids| {
        r.before = try f.arena(a, ids);
        if (planted) |row| _ = plant(f, r.before, ids, row);
        arena_ptr = try rig.scratch(r.before.len * 2);
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = arena_ptr, .len = r.before.len * 2 }).uploadAsync(0, std.mem.sliceAsBytes(r.before), s.handle);
        // the bases as the engine lays a page out (prompt.Paged.at); the rows were placed by this file's own map
        for (0..2) |st| for (std.enums.values(Cache), 0..) |c, i| {
            kv[st][i] = arena_ptr + 2 * f.slotAt(c);
        };
    } else for (0..2) |st| for (0..4) |i| {
        const bytes = std.mem.sliceAsBytes(f.caches[st][i]);
        kv[st][i] = try rig.scratch(bytes.len);
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = kv[st][i], .len = bytes.len }).uploadAsync(0, bytes, s.handle);
    };
    const qw = try rig.input(std.mem.sliceAsBytes(f.qw));
    const kw = try rig.input(std.mem.sliceAsBytes(f.kw));
    const iw = try rig.input(std.mem.sliceAsBytes(f.iw));
    const ikw = try rig.input(std.mem.sliceAsBytes(f.ikw));
    const inv = try rig.input(std.mem.sliceAsBytes(f.inv));
    const n = f.c.rows;
    var bufs: [2]Bufs = undefined;
    var pages: [2]flash.Pg = .{ .{}, .{} };
    for (0..2) |st| {
        const blocks = cdiv(n, att_rows) + 1;
        const pos = try a.alloc(i32, blocks * 4);
        @memset(pos, 0);
        for (0..blocks) |bi| pos[4 * bi] = @intCast(f.c.lens[st] + bi * att_rows);
        bufs[st] = .{
            .proj = try rig.input(std.mem.sliceAsBytes(f.proj[st])),
            .pos = try rig.input(std.mem.sliceAsBytes(pos)),
            .qw = qw,
            .kw = kw,
            .iw = iw,
            .ikw = ikw,
            .inv = inv,
            .q = try rig.output(n * q_heads * head_dim * 2),
            .iq = try rig.output(n * index_heads * index_dim * 2),
            .kc = kv[st][0],
            .vc = kv[st][1],
            .ikc = kv[st][2],
            .pooled = kv[st][3],
            .sc = try rig.output(att_rows * nbOf(f, st) * 4),
            .ids = try rig.output(fused_rows * idw * 4),
            .nk = try rig.output(fused_rows * 4),
            .spr = try rig.output(fused_rows * 4),
            .po = try rig.scratch(att_rows * nch * q_heads * head_dim * 4),
            .pm = try rig.scratch(att_rows * nch * q_heads * 4),
            .pl = try rig.scratch(att_rows * nch * q_heads * 4),
            .ao = try rig.output(n * out_k * 2),
        };
        if (given orelse placed) |t| pages[st] = .{ .pt = try rig.input(std.mem.sliceAsBytes(t[st])), .ps = @intCast(f.pageElems()), .ptok = @intCast(f.c.ptok) };
    }
    var images: [2]?media.View = .{ null, null };
    if (f.image) |im| for (0..2) |st| {
        images[st] = try media.view(&rig, g.f, im[st]);
    };
    const inputs = try a.dupe(u8, try rig.inputBytes());
    for (0..2) |st| try pass(g, s, select, f.c.lens[st], n, nbOf(f, st), bufs[st], pages[st], images[st]);
    for (0..2) |st| {
        const b = bufs[st];
        r.out[st] = .{
            .q = try a.dupe(u8, try rig.bytes(b.q, n * q_heads * head_dim * 2)),
            .iq = try a.dupe(u8, try rig.bytes(b.iq, n * index_heads * index_dim * 2)),
            .ao = try a.dupe(u8, try rig.bytes(b.ao, n * out_k * 2)),
            .sc = try a.dupe(u8, try rig.bytes(b.sc, att_rows * nbOf(f, st) * 4)),
            .ids = try a.dupe(u8, try rig.bytes(b.ids, fused_rows * idw * 4)),
            .nk = try a.dupe(u8, try rig.bytes(b.nk, fused_rows * 4)),
            .spr = try a.dupe(u8, try rig.bytes(b.spr, fused_rows * 4)),
        };
    }
    if (placed) |ids| {
        r.arena = try a.alloc(u16, r.before.len);
        @memcpy(std.mem.sliceAsBytes(r.arena), try rig.bytes(arena_ptr, r.before.len * 2));
        for (0..2) |st| r.caches[st] = try f.unpage(a, r.arena, ids[st], st);
    } else for (0..2) |st| for (0..4) |i| {
        r.caches[st][i] = try a.alloc(u16, f.caches[st][i].len);
        @memcpy(std.mem.sliceAsBytes(r.caches[st][i]), try rig.bytes(kv[st][i], f.caches[st][i].len * 2));
    };
    try check.expect(try rig.intact() and !rig.overlap() and std.mem.eql(u8, inputs, try rig.inputBytes()), "flashnext pages: a launch changed an input or guard", .{});
    return r;
}

/// The block scores' row stride for stream `st`'s cache (cuda_batch_state.zig scoreStride).
fn nbOf(f: Fixture, st: usize) usize {
    return (cdiv(f.cap[st], ratio) + 15) / 16 * 16;
}

/// Every attention row finite: no pass read a NaN row (an unwritten one, or one of a free page).
fn finite(o: Out) bool {
    const rows: []align(1) const u16 = std.mem.bytesAsSlice(u16, o.ao);
    for (rows) |x| if (x & 0x7f80 == 0x7f80) return false;
    return true;
}

fn same(x: Out, y: Out) !void {
    inline for (.{ "q", "iq", "ao", "sc", "ids", "nk", "spr" }) |field| try check.sameBytes("flashnext pages: paged " ++ field ++ " against contiguous", @field(x, field), @field(y, field));
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try flash.Glue.load(gpu.d);
    defer g.deinit();
    var batch = try cuda.Module.load(gpu.d, cuda.kernels.flashnext_batch);
    defer batch.unload();
    const select = try batch.function("fn_qsa_select");
    try @import("contract_fn_pages_gates.zig").run(gpu, s, &g, select);
    try @import("contract_fn_pages_sums.zig").run(gpu, &g);
    var red_table = false;
    var red_free = false;
    var red_row = false;
    for (cases, 0..) |c, ci| {
        var arena = std.heap.ArenaAllocator.init(gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const f = try Fixture.make(a, c, 0xf1a5 + ci);
        const flat = try launch(gpu, s, &g, select, f, null, null, null, a);
        for (flat.out) |o| try check.expect(finite(o), "flashnext pages: contiguous attention read a NaN row at lens {any}", .{c.lens});
        if (c.image != null) {
            var text = f;
            text.image = null;
            const plain = try launch(gpu, s, &g, select, text, null, null, null, a);
            for (0..2) |st| {
                try check.expect(!std.mem.eql(u8, flat.out[st].q, plain.out[st].q) and !std.mem.eql(u8, flat.out[st].iq, plain.out[st].iq), "flashnext pages: media query angles went unseen", .{});
                try check.expect(!std.mem.eql(u16, flat.caches[st][0], plain.caches[st][0]), "flashnext pages: media key angles went unseen", .{});
                if ((c.lens[st] + c.rows) / ratio > c.lens[st] / ratio) try check.expect(!std.mem.eql(u16, flat.caches[st][3], plain.caches[st][3]), "flashnext pages: media pooled angles went unseen", .{});
            }
        }
        var prng = std.Random.DefaultPrng.init(0x9a6e + ci);
        for (0..3) |rep| {
            // a new placement each repeat: the bytes are the order's alone, never the pages'
            const ids = if (rep == 0) f.ids else try Fixture.shuffled(a, f, prng.random());
            const placed: [2][]const u32 = .{ ids[0], ids[1] };
            const pg = try launch(gpu, s, &g, select, f, placed, null, null, a);
            for (0..2) |st| {
                try same(pg.out[st], flat.out[st]);
                for (0..4) |i| try check.expect(std.mem.eql(u16, pg.caches[st][i], flat.caches[st][i]), "flashnext pages: stream {d}'s {t} cache differs at lens {any}, page {d}", .{ st, std.enums.values(Cache)[i], c.lens, c.ptok });
            }
            const owned = try f.owned(a, placed);
            for (pg.arena, pg.before, owned, 0..) |x, y, o, e| if (!o and x != y)
                return check.expect(false, "flashnext pages: element {d} outside both streams' rows changed (a free page, slot or tail)", .{e});
        }
        // red controls: two of a stream's cached pages swapped in the table it is read through; a page of it a free one
        const n1 = f.cap[1] / c.ptok;
        if (n1 >= 2 and c.lens[1] > 0) {
            const swapped = try a.dupe(u32, f.ids[1]);
            std.mem.swap(u32, &swapped[0], &swapped[n1 - 1]);
            const bad = try launch(gpu, s, &g, select, f, .{ f.ids[0], f.ids[1] }, .{ f.ids[0], swapped }, null, a);
            const moved = !std.mem.eql(u8, bad.out[1].ao, flat.out[1].ao);
            red_table = red_table or moved;
            try check.expect(moved, "flashnext pages: a swapped page table still gave contiguous bytes (control missed) at lens {any}", .{c.lens});
        }
        const spare = for (0..f.pages) |id| {
            if (std.mem.indexOfScalar(u32, f.ids[0], @intCast(id)) == null and std.mem.indexOfScalar(u32, f.ids[1], @intCast(id)) == null) break @as(u32, @intCast(id));
        } else unreachable;
        const st: usize = if (c.lens[0] > 0) 0 else 1; // a stream whose first page holds cached rows its pass reads
        var tables: [2][]const u32 = .{ f.ids[0], f.ids[1] };
        const freed = try a.dupe(u32, f.ids[st]);
        freed[0] = spare;
        tables[st] = freed;
        const poisoned = try launch(gpu, s, &g, select, f, .{ f.ids[0], f.ids[1] }, tables, null, a);
        const p = poisoned.out[st];
        const caught = !finite(p) or !std.mem.eql(u8, p.ao, flat.out[st].ao) or !std.mem.eql(u8, p.sc, flat.out[st].sc);
        red_free = red_free or caught;
        try check.expect(caught, "flashnext pages: a free page in the table went unseen (control missed) at lens {any}", .{c.lens});
        const planted = try launch(gpu, s, &g, select, f, .{ f.ids[0], f.ids[1] }, null, try controlRow(f, flat.out[st].sc), a);
        const row_caught = !finite(planted.out[st]) or !std.mem.eql(u8, planted.out[st].ao, flat.out[st].ao) or !std.mem.eql(u8, planted.out[st].sc, flat.out[st].sc);
        red_row = red_row or row_caught;
        try check.expect(row_caught, "flashnext pages: a planted cached row went unseen at lens {any}", .{c.lens});
        std.debug.print("PASS flashnext pages: {s}, lens {any}, {d} rows a stream, {d}-position pages, slot {d} of {d}: paged == contiguous (queries, rows, scores, selection, caches), three placements, free pages and slots untouched, guards\n", .{ if (c.image != null) "media" else "text", c.lens, c.rows, c.ptok, c.slot, c.slots });
    }
    try check.expect(red_table and red_free and red_row, "flashnext pages: a control never fired", .{});
    std.debug.print("PASS flashnext pages: controls RED as planted (swapped table, a free page read, a planted cached row)\n", .{});
}

test "paged media contract: prompt, verify and sparse rows cover image spans, page edges and the decode suffix" {
    var count: usize = 0;
    var crossing = false;
    var suffix = false;
    var sparse = false;
    var fused = false;
    var decode = false;
    var partial = false;
    var sizes: [6]bool = @splat(false);
    for (cases) |c| if (c.image) |im| {
        count += 1;
        fused = fused or c.rows > att_rows;
        decode = decode or c.rows == 1;
        for ([_]usize{ 32, 64, 96, 128, 256, 512 }, 0..) |size, i| sizes[i] = sizes[i] or c.ptok == size;
        for (c.lens, im.n, im.spans) |start, n, spans| {
            sparse = sparse or start > top * ratio;
            partial = partial or (start + c.rows) % c.ptok != 0;
            suffix = suffix or (start < n and start + c.rows > n);
            for (spans) |span| crossing = crossing or span.at / c.ptok != (span.at + span.tokens() - 1) / c.ptok;
        }
    };
    try std.testing.expectEqual(@as(usize, 6), count);
    try std.testing.expect(crossing and suffix and sparse and fused and decode and partial);
    for (sizes) |present| try std.testing.expect(present);
}

test "paged media contract: shuffled pages round trip, free slots stay poisoned and both address controls differ" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f = try Fixture.make(a, .{ .lens = .{ 31, 64 }, .rows = 17, .ptok = 32, .slot = 12, .slots = 13, .image = media.specs[1] }, 0xf1a5);
    const ids: [2][]const u32 = .{ f.ids[0], f.ids[1] };
    const all = try f.arena(a, ids);
    const owned = try f.owned(a, ids);
    for (all, owned) |x, holds| if (!holds) try std.testing.expectEqual(nan, x);
    for (0..2) |st| {
        const logical = try f.unpage(a, all, ids[st], st);
        for (0..4) |i| try std.testing.expectEqualSlices(u16, f.caches[st][i], logical[i]);
    }
    const swapped = try a.dupe(u32, ids[1]);
    std.mem.swap(u32, &swapped[0], &swapped[swapped.len - 1]);
    const wrong = try f.unpage(a, all, swapped, 1);
    try std.testing.expect(!std.mem.eql(u16, f.caches[1][1], wrong[1]));
    const before = try a.dupe(u16, all);
    const st = plant(f, all, ids, 0);
    try std.testing.expectEqual(@as(usize, 0), st);
    const at = f.placed(ids[st], .v, 0);
    try std.testing.expect(!std.mem.eql(u16, before, all));
    for (all, before, 0..) |x, y, i| try std.testing.expect(if (i >= at and i < at + row_kv) x == nan else x == y);
}

test "paged media contract: the sparse row control chooses a positive cached score and fails without one" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f = try Fixture.make(a, .{ .lens = .{ 2051, 3 }, .rows = 1, .ptok = 512, .slot = 0, .slots = 1 }, 7);
    var dense = f;
    dense.c.lens[0] = 2050;
    try std.testing.expectEqual(@as(usize, 0), try controlRow(dense, &.{}));
    const scores = try a.alloc(u8, 512 * 4);
    @memset(scores, 0);
    try std.testing.expectError(error.ControlHasNoPositiveScore, controlRow(f, scores));
    std.mem.writeInt(u32, scores[4..8], @bitCast(@as(f32, -1)), .little);
    std.mem.writeInt(u32, scores[8..12], @bitCast(std.math.inf(f32)), .little);
    std.mem.writeInt(u32, scores[12..16], @bitCast(@as(f32, 2)), .little);
    try std.testing.expectEqual(@as(usize, 3), try controlRow(f, scores));
    const all = try f.arena(a, .{ f.ids[0], f.ids[1] });
    _ = plant(f, all, .{ f.ids[0], f.ids[1] }, 3);
    const logical = try f.unpage(a, all, f.ids[0], 0);
    for (0..4) |i| try std.testing.expectEqual(i == 3, !std.mem.eql(u16, f.caches[0][i], logical[i]));
    for (logical[3][3 * index_dim ..][0..index_dim]) |x| try std.testing.expectEqual(nan, x);
}

test {
    _ = @import("contract_fn_pages_sums.zig");
}
