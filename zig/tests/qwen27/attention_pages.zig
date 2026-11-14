//! TF_QWEN27_METAL_KV_PAGES kernels against main's: the same rounds, outputs and cache rows bit for bit, pages shuffled
const std = @import("std");
const mtl = @import("metal");
const q = @import("qwen27");
const attn = q.attention;
const planning = attn.planning;

const dim = 256;
const page = attn.page_rows;
const most_rows = 128;
const capacity: u32 = 12288;
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

const Rng = struct {
    s: u64,
    fn next(r: *Rng) u64 {
        r.s ^= r.s >> 12;
        r.s ^= r.s << 25;
        r.s ^= r.s >> 27;
        return r.s *% 0x2545f4914f6cdd1d;
    }
    fn bf16(r: *Rng) u16 {
        const unit: f32 = @as(f32, @floatFromInt(r.next() >> 40)) / @as(f32, 1 << 24);
        return @intCast(@as(u32, @bitCast((unit - 0.5) * 4.0)) >> 16);
    }
    fn below(r: *Rng, n: usize) usize {
        return @intCast(r.next() % n);
    }
};

/// A round: its streams' windows (start, rows, a tree or a chain), the prompt kernel or the decode path, and a keep.
const Case = struct { starts: []const u32, rows: []const u32, tree: bool = false, prompt: bool = false, planted: bool = false };
const cases = [_]Case{
    .{ .starts = &.{0}, .rows = &.{1} },                                                         .{ .starts = &.{255}, .rows = &.{1} },                                           .{ .starts = &.{256}, .rows = &.{1} },
    .{ .starts = &.{511}, .rows = &.{1} },                                                       .{ .starts = &.{8191}, .rows = &.{1} },                                          .{ .starts = &.{12287}, .rows = &.{1} },
    .{ .starts = &.{250}, .rows = &.{16} },                                                      .{ .starts = &.{4090}, .rows = &.{16}, .tree = true },                           .{ .starts = &.{8184}, .rows = &.{16} },
    .{ .starts = &.{ 100, 767, 3000 }, .rows = &.{ 16, 9, 16 }, .tree = true },                  .{ .starts = &.{ 0, 255, 256, 9000 }, .rows = &.{ 3, 6, 16, 1 }, .tree = true }, .{ .starts = &.{0}, .rows = &.{128}, .prompt = true },
    .{ .starts = &.{200}, .rows = &.{128}, .prompt = true },                                     .{ .starts = &.{1000}, .rows = &.{128}, .prompt = true },                        .{ .starts = &.{8100}, .rows = &.{128}, .prompt = true },
    .{ .starts = &.{12160}, .rows = &.{128}, .prompt = true },
    // Inf and NaN in a round's own values, which a window's earlier rows must not see: the masked kernels and scan
                                      .{ .starts = &.{250}, .rows = &.{16}, .planted = true },                         .{ .starts = &.{4090}, .rows = &.{16}, .tree = true, .planted = true },
    .{ .starts = &.{ 100, 767, 3000 }, .rows = &.{ 16, 9, 16 }, .tree = true, .planted = true }, .{ .starts = &.{200}, .rows = &.{128}, .prompt = true, .planted = true },        .{ .starts = &.{8100}, .rows = &.{128}, .prompt = true, .planted = true },
};
const most_streams = 4;

/// One side: main's contiguous caches, or every stream's layer part in one arena of shuffled pages.
const Side = struct {
    op: attn.Attention,
    binding: attn.Binding,
    caches: [most_streams]attn.Cache,
    out: mtl.Buffer,
};

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    const units = device.tensorUnits();
    // --shared-prefix: the M5's plan (prefix kernels over every row's keys) on any chip; the bits check addressing
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const shared = units or (args.len > 1 and std.mem.eql(u8, args[1], "--shared-prefix"));
    const geometry = attn.planning.Geometry{ .shared_prefix = shared };
    const heads = geometry.heads;
    const kvh = geometry.kv_heads;
    if (!units) try q.simd_attention.check(device, queue);
    const limits = attn.Limits{ .rows = most_rows, .streams = most_streams, .keys = capacity, .shared_prefix = shared, .geometry = geometry };
    var scratch = try attn.Scratch.init(device, limits);
    defer scratch.deinit();

    // main's caches: a stream's keys [kv_heads][capacity][dim], its values the same
    const cache_bytes = @as(usize, capacity) * kvh * dim * 2;
    var contiguous: [most_streams][2]mtl.Buffer = undefined;
    for (&contiguous) |*s| for (s) |*b| {
        b.* = try device.buffer(cache_bytes, options);
    };
    // the arena: a page is this layer's keys [kv_heads][256][dim] then its values, ids shuffled across streams
    const part = @as(u64, kvh) * page * dim * 2;
    const page_bytes = 2 * part + 512; // a stride past the two parts, as when other layers share the page
    const per = attn.pageCount(capacity);
    const total = most_streams * per + 5;
    const arena = try device.buffer(@intCast(total * page_bytes), options);
    const tables = try device.buffer(most_streams * per * 4, options);
    var rng = Rng{ .s = 0x9e3779b97f4a7c15 };
    const ids = try a.alloc(u32, total);
    for (ids, 0..) |*id, i| id.* = @intCast(i);
    var i: usize = ids.len;
    while (i > 1) : (i -= 1) std.mem.swap(u32, &ids[i - 1], &ids[rng.below(i)]);
    @memcpy(tables.slice(u32, most_streams * per), ids[0 .. most_streams * per]);
    @memset(arena.slice(u8, total * page_bytes), 0x5a);

    var sides: [2]Side = undefined;
    for (&sides, 0..) |*s, paged| {
        const text = try q.simd_attention.grouped(a, units, true, geometry.group(), kvh, paged == 1);
        const library = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer library.deinit();
        s.op = try attn.Attention.init(device, library);
        s.binding = try attn.Binding.init(device, limits);
        s.out = try device.buffer(most_rows * heads * dim * 2, options);
        for (&s.caches, 0..) |*c, st| c.* = if (paged == 0)
            .{ .keys = .{ .buffer = contiguous[st][0] }, .values = .{ .buffer = contiguous[st][1] }, .capacity = capacity }
        else
            .{ .keys = .{ .buffer = arena }, .values = .{ .buffer = arena, .offset = part }, .capacity = capacity, .pages = .{ .table = .{ .buffer = tables, .offset = st * per * 4 }, .page_bytes = page_bytes } };
    }
    std.debug.print("attention_pages on {s}, tensor units {s}, shared prefix {s}: {d} rounds, {d} pages of {d} bytes shuffled\n", .{ std.mem.span(device.name()), if (units) "yes" else "no", if (shared) "yes" else "no", cases.len, total, page_bytes });

    // the same random rows in both layouts
    for (0..most_streams) |st| for (0..2) |kv| for (0..kvh) |hk| for (0..capacity) |row| {
        const c = contiguous[st][kv].slice(u16, cache_bytes / 2)[(hk * capacity + row) * dim ..][0..dim];
        for (c) |*x| x.* = rng.bf16();
        @memcpy(pagedRow(arena, tables, st, per, page_bytes, part, kv, hk, row), c);
    };
    try same(contiguous, arena, tables, per, page_bytes, part, kvh, 0, "the start");

    const query = try device.buffer(most_rows * heads * dim * 2, options);
    const key = try device.buffer(most_rows * kvh * dim * 2, options);
    const val = try device.buffer(most_rows * kvh * dim * 2, options);
    var compared: usize = 0;
    for (cases, 0..) |cs, n| {
        var parents: [most_rows]i32 = undefined;
        var ws: [most_streams]planning.Window = undefined;
        var rows: usize = 0;
        for (cs.starts, cs.rows, 0..) |start, count, st| {
            for (0..count) |r| parents[rows + r] = if (r == 0) -1 else if (cs.tree and r > 2) @intCast(rng.below(r)) else if (cs.tree) 0 else @intCast(r - 1);
            ws[st] = .{ .parents = parents[rows..][0..count], .start = start, .capacity = capacity };
            rows += count;
        }
        for (query.slice(u16, rows * heads * dim)) |*x| x.* = rng.bf16();
        for (key.slice(u16, rows * kvh * dim), val.slice(u16, rows * kvh * dim)) |*k, *v| {
            k.* = rng.bf16();
            v.* = rng.bf16();
        }
        if (cs.planted) plant(val.slice(u16, rows * kvh * dim), cs.rows, kvh);
        var plan = try attn.Plan.init(a, ws[0..cs.starts.len], geometry);
        defer plan.deinit();
        try scratch.upload(plan);
        for (&sides) |*s| {
            const caches = s.caches[0..cs.starts.len];
            try s.binding.upload(plan, caches);
            @memset(s.out.slice(u16, most_rows * heads * dim), 0xffff);
            const cb = queue.commandBuffer();
            const e = cb.compute(.concurrent);
            try s.op.encode(e, &scratch, &s.binding, plan, .{ .buffer = query }, .{ .buffer = key }, .{ .buffer = val }, .{ .buffer = s.out }, caches, null, cs.prompt, false);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.debug.print("FAIL round {d}: the GPU: {s}\n", .{ n, msg });
                return error.GpuFailed;
            }
        }
        const got = [2][]const u16{ sides[0].out.slice(u16, rows * heads * dim), sides[1].out.slice(u16, rows * heads * dim) };
        if (!std.mem.eql(u16, got[0], got[1])) {
            std.debug.print("FAIL round {d} (starts {any}, rows {any}): attention outputs differ\n", .{ n, cs.starts, cs.rows });
            return error.Differs;
        }
        compared += got[0].len;
        try same(contiguous, arena, tables, per, page_bytes, part, kvh, n, "the append");
        if (cs.tree) {
            // each stream keeps its last row's path, moved to the window's first rows
            var paths: [most_streams][]u32 = undefined;
            var first: usize = 0;
            for (cs.rows, 0..) |count, st| {
                var chain: std.ArrayList(u32) = .empty;
                var at: i32 = @intCast(count - 1);
                while (at >= 0) : (at = parents[first + @as(usize, @intCast(at))]) try chain.append(a, @intCast(at));
                std.mem.reverse(u32, chain.items);
                paths[st] = chain.items;
                first += count;
            }
            for (&sides) |*s| {
                const caches = s.caches[0..cs.starts.len];
                const kept = try s.binding.uploadKeep(plan, caches, paths[0..cs.starts.len]);
                const cb = queue.commandBuffer();
                const e = cb.compute(.concurrent);
                try s.op.keep(e, &scratch, &s.binding, kept, caches);
                e.end();
                cb.commit();
                cb.wait();
                if (cb.failure()) |msg| {
                    std.debug.print("FAIL round {d}'s keep: the GPU: {s}\n", .{ n, msg });
                    return error.GpuFailed;
                }
            }
            try same(contiguous, arena, tables, per, page_bytes, part, kvh, n, "the keep");
        }
    }
    // no page past the tables' was written: the spare pages still hold the fill
    for (ids[most_streams * per ..]) |id| for (arena.slice(u8, total * page_bytes)[id * page_bytes ..][0..page_bytes]) |x| if (x != 0x5a) {
        std.debug.print("FAIL a spare page {d} was written\n", .{id});
        return error.Differs;
    };
    std.debug.print("PASS {d} rounds: {d} output values and every cache row the same bits, paged and contiguous\n", .{ cases.len, compared });
}

/// Each window's later rows: an Inf, a -Inf and a NaN in a few heads' values, which its first row must not see.
fn plant(values: []u16, rows: []const u32, kvh: usize) void {
    var first: usize = 0;
    for (rows) |count| {
        var r: usize = 1;
        while (r < count) : (r += 3) {
            const row = values[((first + r) * kvh + r % kvh) * dim ..][0..dim];
            row[r % dim] = 0x7f80;
            row[(r * 7 + 128) % dim] = 0xff80;
            row[(r * 13 + 5) % dim] = 0x7fc1;
        }
        first += count;
    }
}

/// Stream `st`'s cache `kv` (0 keys, 1 values), head `hk`, row `row` in the arena.
fn pagedRow(arena: mtl.Buffer, tables: mtl.Buffer, st: usize, per: usize, page_bytes: u64, part: u64, kv: usize, hk: usize, row: usize) []u16 {
    const id = tables.slice(u32, (st + 1) * per)[st * per + row / page];
    const at = id * page_bytes + kv * part + (hk * page + row % page) * dim * 2;
    return @alignCast(std.mem.bytesAsSlice(u16, arena.slice(u8, @intCast(at + dim * 2))[@intCast(at)..]));
}

/// Every stream's every row the same in both layouts.
fn same(contiguous: [most_streams][2]mtl.Buffer, arena: mtl.Buffer, tables: mtl.Buffer, per: usize, page_bytes: u64, part: u64, kvh: usize, n: usize, what: []const u8) !void {
    for (0..most_streams) |st| for (0..2) |kv| for (0..kvh) |hk| for (0..capacity) |row| {
        const c = contiguous[st][kv].slice(u16, kvh * capacity * dim)[(hk * capacity + row) * dim ..][0..dim];
        if (!std.mem.eql(u16, c, pagedRow(arena, tables, st, per, page_bytes, part, kv, hk, row))) {
            std.debug.print("FAIL round {d}, after {s}: stream {d} {s} head {d} row {d} differs\n", .{ n, what, st, if (kv == 0) "keys" else "values", hk, row });
            return error.Differs;
        }
    };
}
