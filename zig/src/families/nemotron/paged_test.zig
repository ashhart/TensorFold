//! Nemotron's paged K/V kernels against main's on one GPU: the same bits through shuffled pages; a swapped page is red.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("frags");
const simd_attention = @import("simd_attention.zig");
const paged = sources.nemotron_paged;

const D = 128;
const kvh = 2;
const heads = 32;
const G = heads / kvh;
const page = 256; // TF_PAGE_TOKENS
const layers = 6; // attention layers a page holds, keys then values each
const page_bytes: u64 = layers * 2 * kvh * page * D * 2;
const layer = 3; // a layer past the first, so its part's offset counts
const row_bytes = D * 2;

fn part(values: bool) usize {
    return (layer * 2 + @as(usize, @intFromBool(values))) * kvh * page * D * 2;
}

const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    a: std.mem.Allocator,

    fn pipe(g: Gpu, source: []const u8, name: [:0]const u8) !mtl.Pipeline {
        const lib = try mtl.Library.fromSource(g.device, source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        return mtl.Pipeline.init(g.device, lib, name, false);
    }

    /// A tree source as kernels.zig builds it here: the simdgroup-matrix layout before the M5.
    fn treePipe(g: Gpu, source: []const u8, name: [:0]const u8) !mtl.Pipeline {
        if (g.device.tensorUnits()) return g.pipe(source, name);
        const text = try std.mem.concat(g.a, u8, &.{ "#define TF_SIMD_LAYOUT 1\n", source });
        defer g.a.free(text);
        return g.pipe(text, name);
    }

    /// A decode attention source as kernels.zig builds it here: rewritten to the simdgroup-matrix layout before the M5.
    fn partialPipe(g: Gpu, source: []const u8, name: [:0]const u8) !mtl.Pipeline {
        if (g.device.tensorUnits()) return g.pipe(source, name);
        const text = try simd_attention.rewrite(g.a, source);
        defer g.a.free(text);
        return g.pipe(text, name);
    }

    /// A prompt source with nax.h inlined for this GPU, as prefill_kernels.zig builds it.
    fn prefillPipe(g: Gpu, source: []const u8, name: [:0]const u8) !mtl.Pipeline {
        const text = try frags.source(g.device, g.a, source);
        defer g.a.free(text);
        return g.pipe(text, name);
    }

    fn buf(g: Gpu, bytes: usize) !mtl.Buffer {
        return g.device.buffer(@max(bytes, 4), mtl.ResourceOptions.shared);
    }

    fn finish(_: Gpu, cb: mtl.CommandBuffer) !void {
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
    }
};

fn bf16s(rnd: std.Random, out: []u16) void {
    for (out) |*v| v.* = @truncate(@as(u32, @bitCast(rnd.float(f32) * 2 - 1)) >> 16);
}

/// A stream's keys and values both ways: contiguous [kvh, cap, D] as main keeps them, and in shuffled arena pages.
const Cache = struct {
    cap: usize,
    pages: usize,
    ids: [32]u32 = undefined,
    table: mtl.Buffer,
    arena: mtl.Buffer,
    kc: mtl.Buffer,
    vc: mtl.Buffer,
    before: []u8, // the arena as filled: bytes outside the stream's rows must stay so

    /// `valid` rows alike in both layouts, different noise past them; `swap`: the table's first two entries swap.
    fn init(g: Gpu, rnd: std.Random, cap: usize, valid: usize, swap: bool) !Cache {
        const npages = (cap + page - 1) / page;
        const arena_pages = npages + 3;
        var c: Cache = .{ .cap = cap, .pages = arena_pages, .table = try g.buf(npages * 4), .arena = try g.buf(arena_pages * page_bytes), .kc = try g.buf(kvh * cap * D * 2), .vc = try g.buf(kvh * cap * D * 2), .before = &.{} };
        for (c.ids[0..arena_pages], 0..) |*id, i| id.* = @intCast(i);
        rnd.shuffle(u32, c.ids[0..arena_pages]);
        const t = c.table.slice(u32, npages);
        @memcpy(t, c.ids[0..npages]);
        if (swap) std.mem.swap(u32, &t[0], &t[1]);
        const a8 = c.arena.slice(u8, arena_pages * page_bytes);
        rnd.bytes(a8); // stale bytes, NaNs among them
        bf16s(rnd, c.kc.slice(u16, kvh * cap * D));
        bf16s(rnd, c.vc.slice(u16, kvh * cap * D));
        for (0..valid) |pos| for (0..kvh) |h| for ([_]bool{ false, true }) |values| {
            @memcpy(a8[c.at(pos, h, values)..][0..row_bytes], c.row(pos, h, values));
        };
        c.before = try g.a.dupe(u8, a8);
        return c;
    }

    fn deinit(c: *Cache, g: Gpu) void {
        g.a.free(c.before);
        for ([_]mtl.Buffer{ c.table, c.arena, c.kc, c.vc }) |b| b.deinit();
    }

    /// A row's byte offset in the arena by the true table (a swapped table's writes land elsewhere).
    fn at(c: *const Cache, pos: usize, h: usize, values: bool) usize {
        return c.ids[pos / page] * page_bytes + part(values) + (h * page + pos % page) * row_bytes;
    }

    fn row(c: *const Cache, pos: usize, h: usize, values: bool) []u8 {
        const src = if (values) c.vc else c.kc;
        return src.slice(u8, kvh * c.cap * D * 2)[(h * c.cap + pos) * row_bytes ..][0..row_bytes];
    }

    /// Rows [0, upto) equal in both layouts and every other arena byte as filled.
    fn same(c: *const Cache, g: Gpu, upto: usize) !bool {
        const a8 = c.arena.slice(u8, c.pages * page_bytes);
        const want = try g.a.dupe(u8, c.before);
        defer g.a.free(want);
        for (0..upto) |pos| for (0..kvh) |h| for ([_]bool{ false, true }) |values| {
            @memcpy(want[c.at(pos, h, values)..][0..row_bytes], c.row(pos, h, values));
        };
        return std.mem.eql(u8, want, a8);
    }

    fn bindPaged(c: *const Cache, e: mtl.ComputeEncoder, k: usize, v: usize, table: usize) void {
        e.setBuffer(c.arena, part(false), k);
        e.setBuffer(c.arena, part(true), v);
        e.setBuffer(c.table, 0, table);
        e.setValue(page_bytes, table + 1);
    }
};

/// Equal bytes, and not all zero: a kernel that wrote nothing would equal its twin that wrote nothing.
fn sameBytes(a: mtl.Buffer, b: mtl.Buffer) bool {
    const x = a.slice(u8, a.length());
    return std.mem.indexOfNone(u8, x, &.{0}) != null and std.mem.eql(u8, x, b.slice(u8, b.length()));
}

/// tf_kv_write and its twin write a window's rows after `len`; then tf_kv_compact and its twin keep a path of them.
fn writes(g: Gpu, len: usize, rows: usize, swap: bool) !bool {
    var prng = std.Random.DefaultPrng.init(len * 31 + rows);
    const rnd = prng.random();
    var c = try Cache.init(g, rnd, len + rows + 3, len, swap);
    defer c.deinit(g);
    const qkv_n = heads * D + 2 * kvh * D;
    const qkv = try g.buf(rows * qkv_n * 2);
    defer qkv.deinit();
    bf16s(rnd, qkv.slice(u16, rows * qkv_n));
    const write = try g.pipe(sources.nemotron_glue, "tf_kv_write");
    defer write.deinit();
    const write_paged = try g.pipe(paged.glue, paged.glue_names[0]);
    defer write_paged.deinit();
    const compact = try g.treePipe(sources.nemotron_tree, "tf_kv_compact");
    defer compact.deinit();
    const compact_paged = try g.pipe(paged.glue, paged.glue_names[1]);
    defer compact_paged.deinit();
    // a kept path: each row at or past its index, rising (a tree path's rows in window order)
    var path: [16]i32 = @splat(0);
    var n: usize = 0;
    var r: usize = 0;
    while (r < rows) : (r += 1 + rnd.uintLessThan(usize, 2)) {
        path[n] = @intCast(r);
        n += 1;
    }
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    const dims = [4]u32{ qkv_n, heads * D, @intCast(c.cap), @intCast(len) };
    for ([_]bool{ false, true }) |p| {
        e.setPipeline(if (p) write_paged else write);
        e.setBuffer(qkv, 0, 0);
        if (p) c.bindPaged(e, 1, 2, 4) else {
            e.setBuffer(c.kc, 0, 1);
            e.setBuffer(c.vc, 0, 2);
        }
        e.setValue(dims, 3);
        e.dispatchThreads(mtl.Size.of(D, kvh, rows), mtl.Size.of(D, 1, 1));
    }
    const cdims = [4]u32{ @intCast(len), @intCast(n), @intCast(c.cap), D };
    for ([_]bool{ false, true }) |p| {
        e.setPipeline(if (p) compact_paged else compact);
        if (p) c.bindPaged(e, 0, 1, 4) else {
            e.setBuffer(c.kc, 0, 0);
            e.setBuffer(c.vc, 0, 1);
        }
        e.setBytes(std.mem.sliceAsBytes(path[0..@max(n, 8)]), 2);
        e.setValue(cdims, 3);
        e.dispatchThreads(mtl.Size.of(D, kvh, 1), mtl.Size.of(D, 1, 1));
    }
    e.end();
    try g.finish(cb);
    return c.same(g, len + rows);
}

/// prefill's kv_put and its twin put a prompt chunk's rows at `at` (keep: rows before it copied from `old`).
fn puts(g: Gpu, at: usize, rows: usize, keep: bool, swap: bool) !bool {
    var prng = std.Random.DefaultPrng.init(at * 7 + rows + @intFromBool(keep));
    const rnd = prng.random();
    var c = try Cache.init(g, rnd, at + rows + 5, if (keep) 0 else at, swap);
    defer c.deinit(g);
    var prefill_glue: []const u8 = undefined;
    for (sources.prefill) |f| {
        if (std.mem.eql(u8, f.name, "glue")) prefill_glue = f.text;
    }
    const put = try g.prefillPipe(prefill_glue, "custom_kernel_tf_kv_put_bfloat16_t_bfloat16_t_int32_t_bfloat16_t");
    defer put.deinit();
    const put_paged = try g.pipe(paged.glue, paged.glue_names[2]);
    defer put_paged.deinit();
    const src = try g.buf(rows * kvh * D * 2);
    defer src.deinit();
    bf16s(rnd, src.slice(u16, rows * kvh * D));
    const old = try g.buf(kvh * (at + 1) * D * 2);
    defer old.deinit();
    bf16s(rnd, old.slice(u16, kvh * (at + 1) * D));
    const p = [7]i32{ @intCast(rows), kvh, D, @intCast(c.cap), @intCast(at), @intCast(at + 1), @intFromBool(keep) };
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    for ([_]bool{ false, true }) |pg| for ([_]bool{ false, true }) |values| {
        e.setPipeline(if (pg) put_paged else put);
        e.setBuffer(src, 0, 0);
        e.setBuffer(old, 0, 1);
        e.setBytes(std.mem.asBytes(&p), 2);
        if (pg) {
            e.setBuffer(c.arena, part(values), 3);
            e.setBuffer(c.table, 0, 4);
            e.setValue(page_bytes, 5);
        } else e.setBuffer(if (values) c.vc else c.kc, 0, 3);
        const span = if (keep) at + rows else rows;
        e.dispatchThreads(mtl.Size.of(D, span, kvh), mtl.Size.of(128, 2, 1));
    };
    e.end();
    try g.finish(cb);
    return c.same(g, at + rows);
}

/// attn_partial_N and its twin over `keys` keys for `t` query rows (causal: each row sees up to its own key).
fn partials(g: Gpu, keys: usize, t: usize, causal: bool, swap: bool) !bool {
    var prng = std.Random.DefaultPrng.init(keys * 13 + t + @intFromBool(causal));
    const rnd = prng.random();
    var c = try Cache.init(g, rnd, keys + 2, keys, swap);
    defer c.deinit(g);
    const tiles: usize = @min(t, 16);
    var source: []const u8 = undefined;
    var name: [:0]const u8 = undefined;
    inline for (sources.nemotron.all) |k| {
        if (comptime std.mem.startsWith(u8, k.key, "attn_partial_")) {
            if (std.fmt.parseInt(usize, k.key["attn_partial_".len..], 10) catch 0 == tiles) {
                source = k.source;
                name = k.function;
            }
        }
    }
    const attn = try g.partialPipe(source, name);
    defer attn.deinit();
    const attn_paged = try g.partialPipe(paged.partial, paged.partial_names[tiles - 1]);
    defer attn_paged.deinit();
    const nch = (keys + 511) / 512;
    const rp = G * t;
    const q = try g.buf(kvh * rp * D * 2);
    defer q.deinit();
    bf16s(rnd, q.slice(u16, kvh * rp * D));
    const dims = [5]i32{ @intCast(keys), @intCast(nch), @intCast(t), @intFromBool(causal), @intCast(t) };
    const strides = [4]i64{ @intCast(kvh * c.cap * D), @intCast(c.cap * D), D, 1 };
    var outs: [2][3]mtl.Buffer = undefined;
    for (&outs) |*o| o.* = .{ try g.buf(kvh * nch * rp * D * 4), try g.buf(kvh * nch * rp * 4), try g.buf(kvh * nch * rp * 4) };
    defer for (outs) |o| for (o) |b| b.deinit();
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (outs, 0..) |o, pg| {
        e.setPipeline(if (pg == 1) attn_paged else attn);
        e.setBuffer(q, 0, 0);
        if (pg == 1) c.bindPaged(e, 1, 3, 10) else {
            e.setBuffer(c.kc, 0, 1);
            e.setBuffer(c.vc, 0, 3);
        }
        e.setBytes(std.mem.asBytes(&strides), 2);
        e.setBytes(std.mem.asBytes(&strides), 4);
        e.setValue(@as(f32, 1.0 / @sqrt(@as(f32, D))), 5);
        e.setBytes(std.mem.asBytes(&dims), 6);
        for (o, 7..) |b, slot| e.setBuffer(b, 0, slot);
        e.dispatchThreads(mtl.Size.of(kvh * 32 * tiles, nch, (t + tiles - 1) / tiles), mtl.Size.of(32 * tiles, 1, 1));
    }
    e.end();
    try g.finish(cb);
    for (outs[0], outs[1]) |want, got| if (!sameBytes(want, got)) return false;
    return true;
}

/// tf_tree_tail and its twin: `nt` nodes past `len` cached rows, each over its path's rows gathered by position.
fn tails(g: Gpu, len: usize, nt: usize, swap: bool) !bool {
    var prng = std.Random.DefaultPrng.init(len * 17 + nt);
    const rnd = prng.random();
    var c = try Cache.init(g, rnd, len + nt + 2, len + nt, swap);
    defer c.deinit(g);
    const tail = try g.treePipe(sources.nemotron_tree, "tf_tree_tail");
    defer tail.deinit();
    const tail_paged = try g.treePipe(paged.tree, paged.tree_name);
    defer tail_paged.deinit();
    const maxd = 64;
    var paths: [16 * maxd]i32 = @splat(0);
    var depths: [16]i32 = @splat(0);
    var deepest: usize = 0;
    for (0..nt) |node| {
        depths[node] = @intCast(rnd.uintLessThan(usize, nt));
        deepest = @max(deepest, @as(usize, @intCast(depths[node])) + 1);
        for (0..@as(usize, @intCast(depths[node])) + 1) |j| paths[node * maxd + j] = @intCast(rnd.uintLessThan(usize, nt));
    }
    const pt = len / 64 * 64;
    const nca = (pt + 511) / 512;
    const ncb = (len + deepest) / 512 - pt / 512 + 1;
    const rpa = 16 * nt;
    const q = try g.buf(kvh * rpa * D * 2);
    defer q.deinit();
    bf16s(rnd, q.slice(u16, kvh * rpa * D));
    const shared: [3]mtl.Buffer = .{ try g.buf(kvh * nca * rpa * D * 4), try g.buf(kvh * nca * rpa * 4), try g.buf(kvh * nca * rpa * 4) };
    defer for (shared) |b| b.deinit();
    for (shared, 0..) |b, i| for (b.slice(f32, b.length() / 4)) |*x| {
        x.* = if (i == 2) 1 + rnd.float(f32) else rnd.float(f32) * 2 - 1;
    };
    const dims = [5]i32{ @intCast(len), @intCast(pt), @intCast(nca), @intCast(ncb), @intCast(nt) };
    const strides = [4]i64{ @intCast(kvh * c.cap * D), @intCast(c.cap * D), D, 1 };
    var outs: [2][3]mtl.Buffer = undefined;
    for (&outs) |*o| o.* = .{ try g.buf(kvh * ncb * nt * 16 * D * 4), try g.buf(kvh * ncb * nt * 16 * 4), try g.buf(kvh * ncb * nt * 16 * 4) };
    defer for (outs) |o| for (o) |b| b.deinit();
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (outs, 0..) |o, pg| {
        e.setPipeline(if (pg == 1) tail_paged else tail);
        e.setBuffer(q, 0, 0);
        if (pg == 1) c.bindPaged(e, 1, 3, 15) else {
            e.setBuffer(c.kc, 0, 1);
            e.setBuffer(c.vc, 0, 3);
        }
        e.setBytes(std.mem.asBytes(&strides), 2);
        e.setBytes(std.mem.asBytes(&strides), 4);
        e.setValue(@as(f32, 1.0 / @sqrt(@as(f32, D))), 5);
        e.setBytes(std.mem.asBytes(&dims), 6);
        e.setBytes(std.mem.sliceAsBytes(paths[0 .. nt * maxd]), 7);
        e.setBytes(std.mem.sliceAsBytes(depths[0..@max(nt, 4)]), 8);
        for (shared, 9..) |b, slot| e.setBuffer(b, 0, slot);
        for (o, 12..) |b, slot| e.setBuffer(b, 0, slot);
        e.dispatchThreads(mtl.Size.of(kvh * 32, ncb, nt), mtl.Size.of(32, 1, 1));
    }
    e.end();
    try g.finish(cb);
    for (outs[0], outs[1]) |want, got| if (!sameBytes(want, got)) return false;
    return true;
}

/// tf_attention_nax and its twin: a prompt chunk of `l` rows after `len` cached ones, every key block in its page.
fn prompts(g: Gpu, len: usize, l: usize, swap: bool) !bool {
    var prng = std.Random.DefaultPrng.init(len * 19 + l);
    const rnd = prng.random();
    const kl = len + l;
    var c = try Cache.init(g, rnd, kl + 7, kl, swap);
    defer c.deinit(g);
    const tag = [2]u8{ if (l % 64 == 0) 't' else 'n', if (kl % 32 == 0) 't' else 'n' };
    var which: usize = 0;
    for (paged.prefill_tags, 0..) |t, i| {
        if (t[0] == tag[0] and t[2] == tag[1]) which = i;
    }
    var name: [128]u8 = undefined;
    const printed = try std.fmt.bufPrint(name[0..127], "custom_kernel_tf_attention_nax_bf16_{c}_{c}_t_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_float_bfloat16_t", .{ tag[0], tag[1] });
    name[printed.len] = 0;
    const full = name[0..printed.len :0];
    var nax: []const u8 = undefined;
    for (sources.prefill) |f| {
        if (std.mem.eql(u8, f.name, "attention_nax")) nax = f.text;
    }
    const attn = try g.prefillPipe(nax, full);
    defer attn.deinit();
    const attn_paged = try g.prefillPipe(paged.prefill, paged.prefill_names[which]);
    defer attn_paged.deinit();
    const row = heads * D;
    const q = try g.buf(l * row * 2);
    defer q.deinit();
    bf16s(rnd, q.slice(u16, l * row));
    const nq = (l + 63) / 64;
    const nk = (kl + 31) / 32;
    const kvs = kvh * c.cap * D;
    var p: [32]i32 = @splat(0);
    const head = [_]usize{ 1, heads, l, kl, G, nq, nk, l / 64, kl / 32, l % 64, kl % 32, kl - l };
    const strides = [12]usize{ l * row, D, row, kvs, c.cap * D, D, kvs, c.cap * D, D, l * row, D, row };
    for (head ++ strides, 0..) |v, i| p[i] = @intCast(v);
    var f: [16]f32 = @splat(0);
    f[0] = 1.0 / @sqrt(@as(f32, D));
    const outs = [2]mtl.Buffer{ try g.buf(l * row * 2), try g.buf(l * row * 2) };
    defer for (outs) |b| b.deinit();
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (outs, 0..) |o, pg| {
        e.setPipeline(if (pg == 1) attn_paged else attn);
        e.setBuffer(q, 0, 0);
        if (pg == 1) c.bindPaged(e, 1, 2, 6) else {
            e.setBuffer(c.kc, 0, 1);
            e.setBuffer(c.vc, 0, 2);
        }
        e.setBytes(std.mem.sliceAsBytes(&p), 3);
        e.setBytes(std.mem.sliceAsBytes(&f), 4);
        e.setBuffer(o, 0, 5);
        e.dispatchThreads(mtl.Size.of(nq * 32, heads * 4, 1), mtl.Size.of(32, 4, 1));
    }
    e.end();
    try g.finish(cb);
    return sameBytes(outs[0], outs[1]);
}

fn gpu() ?Gpu {
    if (comptime !@hasDecl(mtl.Device, "init")) return null;
    const device = mtl.Device.init() catch return null;
    const queue = device.queue() catch return null;
    return .{ .device = device, .queue = queue, .a = std.testing.allocator };
}

fn done(g: Gpu) void {
    g.queue.deinit();
    g.device.deinit();
}

test "the twins hold every edit, and main's kernels stay in their sources unedited" {
    try std.testing.expect(std.mem.indexOf(u8, paged.partial, "tK.slice(0, kt)") == null);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, paged.tree, "(kbase + phys * kstep + col)")); // tf_tree_tail's
    try std.testing.expectEqual(@as(usize, 8), std.mem.count(u8, paged.prefill, ", Kb, ks, ") + std.mem.count(u8, paged.prefill, ", Vb, vs, "));
}

// Each shape crosses a page edge, ends in a partial page, or starts at one; the control swaps two of the table's pages.

test "paged kv writes and tree compaction equal main's through shuffled pages; a swapped page is caught" {
    const g = gpu() orelse return error.SkipZigTest;
    defer done(g);
    const ar = mtl.objc.Pool.push();
    defer ar.pop();
    for ([_][2]usize{ .{ 250, 9 }, .{ 255, 1 }, .{ 0, 16 }, .{ 512, 4 }, .{ 700, 13 } }) |s| try std.testing.expect(try writes(g, s[0], s[1], false));
    try std.testing.expect(!try writes(g, 250, 9, true));
}

test "paged prompt kv_put equals main's, with and without kept rows; a swapped page is caught" {
    const g = gpu() orelse return error.SkipZigTest;
    defer done(g);
    const ar = mtl.objc.Pool.push();
    defer ar.pop();
    for ([_][2]usize{ .{ 0, 300 }, .{ 200, 100 }, .{ 256, 256 }, .{ 37, 700 } }) |s| for ([_]bool{ false, true }) |keep| {
        try std.testing.expect(try puts(g, s[0], s[1], keep, false));
    };
    try std.testing.expect(!try puts(g, 200, 100, false, true));
}

test "paged decode attention partials equal attn_partial_N's bits, causal and shared; a swapped page is caught" {
    const g = gpu() orelse return error.SkipZigTest;
    defer done(g);
    const ar = mtl.objc.Pool.push();
    defer ar.pop();
    for ([_][2]usize{ .{ 1, 1 }, .{ 257, 1 }, .{ 700, 3 }, .{ 1100, 16 }, .{ 512, 8 }, .{ 64, 2 }, .{ 1500, 5 } }) |s| try std.testing.expect(try partials(g, s[0], s[1], true, false));
    for ([_][2]usize{ .{ 640, 4 }, .{ 256, 16 } }) |s| try std.testing.expect(try partials(g, s[0], s[1], false, false));
    try std.testing.expect(!try partials(g, 700, 3, true, true));
}

test "a paged tree tail equals tf_tree_tail's bits; a swapped page is caught" {
    const g = gpu() orelse return error.SkipZigTest;
    defer done(g);
    const ar = mtl.objc.Pool.push();
    defer ar.pop();
    for ([_][2]usize{ .{ 250, 9 }, .{ 700, 16 }, .{ 30, 4 }, .{ 1024, 2 } }) |s| try std.testing.expect(try tails(g, s[0], s[1], false));
    try std.testing.expect(!try tails(g, 250, 9, true));
}

test "paged prompt attention equals tf_attention_nax's bits in all four variants; a swapped page is caught" {
    const g = gpu() orelse return error.SkipZigTest;
    defer done(g);
    const ar = mtl.objc.Pool.push();
    defer ar.pop();
    // (len, rows): n_n, t_n, t_t, n_t
    for ([_][2]usize{ .{ 0, 300 }, .{ 200, 64 }, .{ 448, 128 }, .{ 37, 219 }, .{ 900, 77 } }) |s| try std.testing.expect(try prompts(g, s[0], s[1], false));
    try std.testing.expect(!try prompts(g, 200, 64, true));
}
