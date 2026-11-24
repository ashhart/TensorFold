//! attn_rows's join variants on shared pages: every row's partials and merge == main's kernel alone, bit for bit.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const qp = @import("contract_qwen_pages.zig");

const hd = qp.hd;
const q_heads = qp.q_heads;
const kv_heads = qp.kv_heads;
const group = q_heads / kv_heads;
const max_chunks = qp.max_chunks;
const Format = qp.Format;
const Seg = qp.Seg;
const Ancestor = qp.Ancestor;
const ptok = 256; // TF_QWEN_KV_PAGES=256's positions a page
const scale: f32 = 1.0 / 16.0;
const layer: c_int = 1; // the second of a page's two attention layers, as qwen-pages runs
const geo: qp.Geo = .{ .hid = 0, .nk = 0, .nv = 0, .qh = q_heads, .kvh = kv_heads, .nl = 0 };

/// A rows kernel and its join: cache, threads, rows a block (cuda_forward.zig's tile_rows), dynamic shared bytes.
const Kernel = struct { name: [:0]const u8, join: [:0]const u8, format: Format, threads: u32, tile_rows: usize, shared: u32 = 0 };
const kernels = [_]Kernel{
    .{ .name = "tf_q35_attn_rows", .join = "tf_q35_attn_rows_join", .format = .bf16, .threads = 256, .tile_rows = 64 / group, .shared = qp.attn_tiles },
    .{ .name = "tf_q35_attn_rows8", .join = "tf_q35_attn_rows8_join", .format = .fp8, .threads = 256, .tile_rows = 64 / group },
};
/// main's rows kernel, its join (a launch whose streams share takes it: cuda_cache.anyShares) and the merge.
const Fns = struct { rows: cuda.Function, join: cuda.Function, merge: cuda.Function };

/// Paged streams' lengths, all in 640-key chunks (keys in (32768, 40960]); a case takes the first `streams`.
const lens = [_]usize{ 34000, 33000, 35000, 33500, 34700, 33100, 34900, 33900 };
const band_len = 17000; // 384-key chunks: the same page ids as the common prefix, but it must not join
const flat_len = 34200; // pt = 0 (its own contiguous cache) in the 640-key band: it must not join either
const Case = struct { streams: usize, rows: usize };
const cases = [_]Case{ .{ .streams = 2, .rows = 1 }, .{ .streams = 2, .rows = 4 }, .{ .streams = 4, .rows = 1 }, .{ .streams = 4, .rows = 4 }, .{ .streams = 8, .rows = 1 }, .{ .streams = 8, .rows = 4 } };

// The pool's page ids: a prefix all paged streams share, two groups' prefixes, then each stream's own pages.
const common = 60;
const prefix = 121; // 30,976 keys a group shares: no 640-key chunk ends there, so chunk 48 straddles it
const tail = 17; // a stream's own pages past the prefix
const group_base = common;
const own_base = group_base + 2 * (prefix - common);
const band_base = own_base + lens.len * tail;
const band_pages = 7;
const planted_id = band_base + band_pages; // control (a): a copy of page `planted_page` with one value changed
const planted_page = 30;
const pool_pages = planted_id + 1;

fn pageBytes(f: Format) usize {
    return qp.blockBytes(f, ptok, qp.attentions);
}

/// Every row of a block of n positions a head as qwen-pages fills its caches: bf16 in [-1, 1), e4m3 and scales.
fn fillBlock(f: Format, rng: std.Random, out: []u8, n: usize) void {
    for (0..qp.attentions) |at| for (0..2) |s| for (0..kv_heads) |h| for (0..n) |pos| {
        qp.Fixture.fill(f, rng, out, qp.spans(f, n, at, s, h, pos));
    };
}

/// Pages in one device arena, the host copy kept; the first `random` filled, the rest zero until a caller copies.
const Pool = struct {
    format: Format,
    host: []u8,
    dev: cuda.DeviceBuffer,

    fn make(gpu: check.Gpu, a: std.mem.Allocator, f: Format, pages: usize, random: usize, seed: u64) !Pool {
        var prng = std.Random.DefaultPrng.init(seed);
        const host = try a.alloc(u8, pages * pageBytes(f));
        @memset(host, 0);
        for (0..random) |p| fillBlock(f, prng.random(), host[p * pageBytes(f) ..][0..pageBytes(f)], ptok);
        return .{ .format = f, .host = host, .dev = try cuda.DeviceBuffer.fromHost(gpu.d, host) };
    }

    fn page(p: Pool, id: usize) []u8 {
        return p.host[id * pageBytes(p.format) ..][0..pageBytes(p.format)];
    }
};

/// Layer 1's value at (head 0, position 7, dim 3) of a page set to another value: 0.75 or 0.25, e4m3's exponent bit.
fn plant(f: Format, page: []u8) void {
    const sp = qp.spans(f, ptok, 1, 1, 0, 7);
    switch (f) {
        .bf16 => {
            const at = page[sp[0].off + 2 * 3 ..][0..2];
            std.mem.writeInt(u16, at, if (std.mem.readInt(u16, at, .little) == 0x3f40) 0x3e80 else 0x3f40, .little);
        },
        .fp8 => {
            const at = &page[sp[0].off + 3];
            at.* ^= 0x40;
            if (at.* & 0x7f == 0x7f) at.* ^= 1;
        },
    }
}

/// A launch's stream: length, page table (null: its own contiguous cache at `flat`), rows' ancestors and queries.
const Stream = struct { len: usize, table: ?[]const u32 = null, anc: []const Ancestor, q: []const u16, row0: usize = 0, flat: u64 = 0, cap: usize = 0 };

fn chain(a: std.mem.Allocator, n: usize) ![]Ancestor {
    const out = try a.alloc(Ancestor, n);
    for (out, 0..) |*x, r| x.* = .{ .bits = 0, .depth = @intCast(r), .chain = 1 };
    return out;
}

/// Paged stream i's table for `keys` positions: the common pages, its group's ((i / 2) % 2), then its own.
fn streamTable(a: std.mem.Allocator, i: usize, keys: usize) ![]u32 {
    const t = try a.alloc(u32, (keys + ptok - 1) / ptok);
    std.debug.assert(t.len <= prefix + tail);
    const g = i / 2 % 2;
    for (t, 0..) |*x, p| x.* = @intCast(if (p < common) p else if (p < prefix) group_base + g * (prefix - common) + p - common else own_base + i * tail + p - prefix);
    return t;
}

fn bandTable(a: std.mem.Allocator, keys: usize) ![]u32 {
    const t = try a.alloc(u32, (keys + ptok - 1) / ptok);
    std.debug.assert(t.len <= common + band_pages);
    for (t, 0..) |*x, p| x.* = @intCast(if (p < common) p else band_base + p - common);
    return t;
}

/// cuda_cache.anyShares: two streams with rows in the launch start on one page, so it takes the join kernel.
fn joins(st: []const Stream, order: []const usize) bool {
    for (order, 0..) |i, n| if (st[i].table) |t| for (order[0..n]) |j| if (st[j].table) |u| if (t.len > 0 and u.len > 0 and t[0] == u[0]) return true;
    return false;
}

/// A launch's device buffers: segments, rows (`order`'s streams, each's rows in turn), partials and merged rows.
const Staged = struct { m: usize, q: u64, segs: u64, rowseg: u64, rowanc: u64, pm: u64, pl: u64, po: u64, out: u64, join: bool };

fn stage(rig: *rig_.Rig, a: std.mem.Allocator, pool: Pool, st: []const Stream, order: []const usize) !Staged {
    var m: usize = 0;
    for (order) |i| m += st[i].anc.len;
    const segs = try a.alloc(Seg, st.len);
    for (st, segs) |x, *sg| {
        sg.* = .{ .kv = if (x.table == null) x.flat else pool.dev.ptr, .cap = @intCast(x.cap), .len = @intCast(x.len), .row0 = @intCast(x.row0), .rows = @intCast(x.anc.len) };
        if (x.table) |t| {
            sg.pt = try rig.input(std.mem.sliceAsBytes(t));
            sg.ps = if (pool.format == .fp8) pageBytes(.fp8) else pageBytes(.bf16) / 2;
            sg.ptok = ptok;
        }
    }
    const row = q_heads * hd;
    const rowseg = try a.alloc(i32, m);
    const rowanc = try a.alloc(Ancestor, m);
    const q = try a.alloc(u16, m * row);
    var r: usize = 0;
    for (order) |i| for (st[i].anc, 0..) |anc, j| {
        rowseg[r] = @intCast(i);
        rowanc[r] = anc;
        @memcpy(q[r * row ..][0..row], st[i].q[j * row ..][0..row]);
        r += 1;
    };
    var x: Staged = undefined;
    x.m = m;
    x.join = joins(st, order);
    x.q = try rig.input(std.mem.sliceAsBytes(q));
    x.segs = try rig.input(std.mem.sliceAsBytes(segs));
    x.rowseg = try rig.input(std.mem.sliceAsBytes(rowseg));
    x.rowanc = try rig.input(std.mem.sliceAsBytes(rowanc));
    const slots = m * q_heads * max_chunks;
    x.pm = try rig.output(slots * 4);
    x.pl = try rig.output(slots * 4);
    x.po = try rig.output(slots * hd * 4);
    x.out = try rig.output(m * row * 2);
    return x;
}

/// cuda_forward.zig's rows launch: groups of tile_rows rows by MAXCH chunks by key heads.
fn rows(k: Kernel, f: cuda.Function, s: cuda.Stream, x: Staged) !void {
    const none: u64 = 0;
    var args: cuda.Args = .{};
    inline for (.{ geo, x.q, x.segs, x.rowseg, x.rowanc, layer, none, scale, @as(c_int, @intCast(x.m)), x.pm, x.pl, x.po }) |v| args.add(v);
    if (k.format == .bf16) args.add(false);
    const groups = (x.m + k.tile_rows - 1) / k.tile_rows;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(groups), .y = max_chunks, .z = kv_heads }, .block = .{ .x = k.threads }, .shared = k.shared }, s, &args);
}

const Out = struct { pm: []u8, pl: []u8, po: []u8, out: []u8 };
const Part = enum { pm, pl, po, out };

fn part(o: Out, p: Part) []const u8 {
    return switch (p) {
        .pm => o.pm,
        .pl => o.pl,
        .po => o.po,
        .out => o.out,
    };
}

/// A row's bytes in each part: (query head, chunk) floats, their HD values, its merged bf16 row.
fn rowBytes(p: Part) usize {
    return switch (p) {
        .pm, .pl => q_heads * max_chunks * 4,
        .po => q_heads * max_chunks * hd * 4,
        .out => q_heads * hd * 2,
    };
}

/// The rows kernel then tf_q35_attn_merge over `order`'s rows; inputs and guards checked unchanged.
fn launch(gpu: check.Gpu, s: cuda.Stream, k: Kernel, fns: Fns, pool: Pool, st: []const Stream, order: []const usize, a: std.mem.Allocator) !Out {
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer rig.deinit();
    const x = try stage(&rig, a, pool, st, order);
    const before = try a.dupe(u8, try rig.inputBytes());
    try rows(k, if (x.join) fns.join else fns.rows, s, x);
    const none: u64 = 0;
    var args: cuda.Args = .{};
    inline for (.{ geo, x.pm, x.pl, x.po, x.segs, x.rowseg, x.rowanc, none, x.out }) |v| args.add(v);
    try cuda.launch.launch(fns.merge, .{ .grid = .{ .x = q_heads, .y = @intCast(x.m) }, .block = .{ .x = 32 } }, s, &args);
    var o: Out = undefined;
    o.pm = try a.dupe(u8, try rig.bytes(x.pm, x.m * rowBytes(.pm)));
    o.pl = try a.dupe(u8, try rig.bytes(x.pl, x.m * rowBytes(.pl)));
    o.po = try a.dupe(u8, try rig.bytes(x.po, x.m * rowBytes(.po)));
    o.out = try a.dupe(u8, try rig.bytes(x.out, x.m * rowBytes(.out)));
    try check.expect(try rig.intact() and !rig.overlap() and std.mem.eql(u8, before, try rig.inputBytes()), "qwen shared prefix: a launch changed an input or guard", .{});
    return o;
}

/// Rows [at, at + n) of x and [bt, bt + n) of y hold the same bytes in every part.
fn same(x: Out, at: usize, y: Out, bt: usize, n: usize) bool {
    for (std.enums.values(Part)) |p| {
        const w = rowBytes(p);
        if (!std.mem.eql(u8, part(x, p)[at * w ..][0 .. n * w], part(y, p)[bt * w ..][0 .. n * w])) return false;
    }
    return true;
}

/// As `same`, failing with the first difference by part, row, query head, chunk and dim, and the count differing.
fn agree(what: []const u8, x: Out, at: usize, y: Out, bt: usize, n: usize) !void {
    for (std.enums.values(Part)) |p| {
        const w = rowBytes(p);
        const gx = part(x, p)[at * w ..][0 .. n * w];
        const gy = part(y, p)[bt * w ..][0 .. n * w];
        const first = std.mem.indexOfDiff(u8, gx, gy) orelse continue;
        const e: usize = if (p == .out) 2 else 4;
        var count: usize = 0;
        var i: usize = 0;
        while (i < gx.len) : (i += e) count += @intFromBool(!std.mem.eql(u8, gx[i..][0..e], gy[i..][0..e]));
        const k = first / e;
        const per_head: usize = switch (p) {
            .pm, .pl => max_chunks,
            .po => max_chunks * hd,
            .out => hd,
        };
        const row = k / (q_heads * per_head);
        const head = k / per_head % q_heads;
        const chunk = switch (p) {
            .pm, .pl => k % max_chunks,
            .po => k / hd % max_chunks,
            .out => 0,
        };
        const dim = if (p == .po or p == .out) k % hd else 0;
        const got = if (e == 4) std.mem.readInt(u32, gx[k * 4 ..][0..4], .little) else std.mem.readInt(u16, gx[k * 2 ..][0..2], .little);
        const want = if (e == 4) std.mem.readInt(u32, gy[k * 4 ..][0..4], .little) else std.mem.readInt(u16, gy[k * 2 ..][0..2], .little);
        std.debug.print("FAIL qwen shared prefix {s}: {t} first differs at the stream's row {d}, query head {d}, chunk {d}, dim {d}: {x} for {x}; {d} of {d} differ\n", .{ what, p, row, head, chunk, dim, got, want, count, gx.len / e });
        return error.TestFailed;
    }
}

/// Every case on one rows kernel, then control (b); `flat` is the contiguous stream's cache (cap rows a head).
fn kernelCases(gpu: check.Gpu, s: cuda.Stream, k: Kernel, fns: Fns, pool: Pool, flat: u64, flat_cap: usize) !void {
    var red_planted = false;
    for (cases, 0..) |c, ci| {
        var arena = std.heap.ArenaAllocator.init(gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var prng = std.Random.DefaultPrng.init(0x5eed + ci);
        const st = try a.alloc(Stream, c.streams + 2);
        for (st, 0..) |*x, i| {
            const len: usize = if (i < c.streams) lens[i] else if (i == c.streams) band_len else flat_len;
            x.* = .{ .len = len, .anc = try chain(a, c.rows), .q = try qp.randoms(a, prng.random(), c.rows * q_heads * hd) };
            if (i < c.streams) x.table = try streamTable(a, i, len + c.rows);
            if (i == c.streams) x.table = try bandTable(a, len + c.rows);
            if (x.table) |t| x.cap = t.len * ptok else {
                x.flat = flat;
                x.cap = flat_cap;
            }
        }
        // rows of stream 0, the band stream, stream 1, the contiguous stream, then the rest: blocks mix every kind
        const order = try a.alloc(usize, st.len);
        order[0..4].* = .{ 0, c.streams, 1, c.streams + 1 };
        for (2..c.streams) |i| order[i + 2] = i;
        var at: usize = 0;
        for (order) |i| {
            st[i].row0 = at;
            at += c.rows;
        }
        const label = try std.fmt.allocPrint(a, "{s} {d} streams x {d} rows", .{ k.name, c.streams, c.rows });
        const joint = try launch(gpu, s, k, fns, pool, st, order, a);
        const again = try launch(gpu, s, k, fns, pool, st, order, a);
        try agree(try std.fmt.allocPrint(a, "{s}: joint repeat", .{label}), again, 0, joint, 0, at);
        for (st, 0..) |x, i| {
            const solo = try launch(gpu, s, k, fns, pool, st, &[_]usize{i}, a);
            try agree(try std.fmt.allocPrint(a, "{s}: stream {d} (len {d}) joint against alone", .{ label, i, x.len }), joint, x.row0, solo, 0, c.rows);
        }
        // control (a): stream 1 reads one common page from a copy with one value changed; it must show
        const planted = try a.dupe(Stream, st);
        const t1 = try a.dupe(u32, st[1].table.?);
        t1[planted_page] = planted_id;
        planted[1].table = t1;
        const bad = try launch(gpu, s, k, fns, pool, planted, order, a);
        const moved = !same(bad, st[1].row0, joint, st[1].row0, c.rows);
        red_planted = red_planted or moved;
        try check.expect(moved, "qwen shared prefix {s}: a one-value change in stream 1's page left its rows (control missed)", .{label});
        const bad_solo = try launch(gpu, s, k, fns, pool, planted, &[_]usize{1}, a);
        try agree(try std.fmt.allocPrint(a, "{s}: planted stream 1 joint against alone", .{label}), bad, st[1].row0, bad_solo, 0, c.rows);
        for (st, 0..) |x, i| if (i != 1) try agree(try std.fmt.allocPrint(a, "{s}: stream {d} beside the planted page", .{ label, i }), bad, x.row0, joint, x.row0, c.rows);
        std.debug.print("PASS qwen shared prefix {s} (+ a 384-key-chunk stream, a contiguous one): each stream's PM, PL, PO and merge == alone, repeat, planted page seen\n", .{label});
    }
    const red_len = try lenControl(gpu, s, k, fns, pool);
    try check.expect(red_planted and red_len, "qwen shared prefix {s}: a control never fired", .{k.name});
    std.debug.print("PASS qwen shared prefix {s}: controls RED as planted (a changed copy of a shared page, a tree row past its length)\n", .{k.name});
}

/// Control (b): B's table is A's, its length one short of chunk 51's end; its tree row's window key must not join.
fn lenControl(gpu: check.Gpu, s: cuda.Stream, k: Kernel, fns: Fns, pool: Pool) !bool {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xb1);
    const table = try streamTable(a, 0, 34000 + 1);
    const tree = [_]Ancestor{.{ .bits = 0b100, .depth = 0, .chain = 0 }}; // its window key is row 2's, at len + 2
    var st = [2]Stream{
        .{ .len = 34000, .table = table, .anc = try chain(a, 1), .q = try qp.randoms(a, prng.random(), q_heads * hd), .cap = table.len * ptok },
        .{ .len = 51 * 640 + 640 - 1, .table = table, .anc = &tree, .q = try qp.randoms(a, prng.random(), q_heads * hd), .cap = table.len * ptok },
    };
    for ([_][2]usize{ .{ 0, 1 }, .{ 1, 0 } }) |order| {
        st[order[0]].row0 = 0;
        st[order[1]].row0 = 1;
        const joint = try launch(gpu, s, k, fns, pool, &st, &order, a);
        for (st, 0..) |x, i| {
            const solo = try launch(gpu, s, k, fns, pool, &st, &[_]usize{i}, a);
            try agree(try std.fmt.allocPrint(a, "{s} control (b), rows {any}: stream {d} joint against alone", .{ k.name, order, i }), joint, x.row0, solo, 0, 1);
        }
    }
    // what a join would have read: B's window key at its logical position (a chain row), not its tree row's
    var chained = st;
    chained[1].anc = try chain(a, 1);
    const right = try launch(gpu, s, k, fns, pool, &st, &[_]usize{1}, a);
    const wrong = try launch(gpu, s, k, fns, pool, &chained, &[_]usize{1}, a);
    const red = !same(right, 0, wrong, 0, 1);
    try check.expect(red, "qwen shared prefix {s}: a tree row's own window key read alike at its logical position (control missed)", .{k.name});
    std.debug.print("PASS qwen shared prefix {s}: control (b) B len {d} (chunk 51 ends at {d}) on A's pages: joint == alone both row orders\n", .{ k.name, st[1].len, 51 * 640 + 640 });
    return red;
}

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
    defer mod.unload();
    var kv8 = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_kv8);
    defer kv8.unload();
    const merge = try mod.function("tf_q35_attn_merge");
    const bf16: Fns = .{ .rows = try mod.function(kernels[0].name), .join = try mod.function(kernels[0].join), .merge = merge };
    for ([_]cuda.Function{ bf16.rows, bf16.join }) |f| try f.allowDynamicShared(kernels[0].shared);
    if (rest.len == 1 and std.mem.eql(u8, rest[0], "time")) return time(gpu, s, bf16);
    if (rest.len != 0) return error.UnknownCommand;
    for (std.enums.values(Format)) |f| {
        var arena = std.heap.ArenaAllocator.init(gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var pool = try Pool.make(gpu, a, f, pool_pages, planted_id, 0x9001 + @as(u64, @backingInt(f)));
        defer pool.dev.free();
        @memcpy(pool.page(planted_id), pool.page(planted_page));
        plant(f, pool.page(planted_id));
        try pool.dev.upload(planted_id * pageBytes(f), pool.page(planted_id));
        const flat_cap = (flat_len + 4 + ptok - 1) / ptok * ptok;
        const flat_host = try a.alloc(u8, qp.blockBytes(f, flat_cap, qp.attentions));
        var prng = std.Random.DefaultPrng.init(0xf1a7 + @as(u64, @backingInt(f)));
        fillBlock(f, prng.random(), flat_host, flat_cap);
        var flat = try cuda.DeviceBuffer.fromHost(gpu.d, flat_host);
        defer flat.free();
        for (kernels) |k| {
            if (k.format != f) continue;
            const fns: Fns = if (f == .bf16) bf16 else .{ .rows = try kv8.function(k.name), .join = try kv8.function(k.join), .merge = merge };
            try kernelCases(gpu, s, k, fns, pool, flat.ptr, flat_cap);
        }
    }
}

/// tf_q35_attn_rows and its join at 34,817+ keys (8 x 1, 8 x 4, 4 x 1): one prefix's pages, and copies on distinct ids.
fn time(gpu: check.Gpu, s: cuda.Stream, fns: Fns) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const n = 8; // streams at most
    const shared_pages = 136; // 34,816 keys: the prefix ends on a page, each stream's rows on a page of its own
    const len = shared_pages * ptok;
    const copies = shared_pages + n; // the distinct streams' pages start here
    var pool = try Pool.make(gpu, a, .bf16, copies + n * shared_pages, copies, 0x71e);
    defer pool.dev.free();
    for (0..n) |i| for (0..shared_pages) |p| @memcpy(pool.page(copies + i * shared_pages + p), pool.page(p));
    try pool.dev.upload(0, pool.host);
    const k = kernels[0];
    const reps = 30;
    const dropped = 2;
    for ([_][2]usize{ .{ 8, 1 }, .{ 8, 4 }, .{ 4, 1 } }) |shape| {
        const ns = shape[0];
        const rows_n = shape[1];
        var rig = rig_.Rig{ .gpu = gpu, .s = s };
        defer rig.deinit();
        var prng = std.Random.DefaultPrng.init(0x71e + 16 * ns + rows_n);
        var shared: [n]Stream = undefined;
        var distinct: [n]Stream = undefined;
        var order: [n]usize = undefined;
        for (0..ns) |i| {
            const ts = try a.alloc(u32, shared_pages + 1);
            const td = try a.alloc(u32, shared_pages + 1);
            for (0..shared_pages) |p| {
                ts[p] = @intCast(p);
                td[p] = @intCast(copies + i * shared_pages + p);
            }
            ts[shared_pages] = @intCast(shared_pages + i);
            td[shared_pages] = @intCast(shared_pages + i);
            shared[i] = .{ .len = len, .table = ts, .anc = try chain(a, rows_n), .q = try qp.randoms(a, prng.random(), rows_n * q_heads * hd), .row0 = i * rows_n, .cap = ts.len * ptok };
            distinct[i] = shared[i];
            distinct[i].table = td;
            order[i] = i;
        }
        const xs = try stage(&rig, a, pool, shared[0..ns], order[0..ns]);
        const xd = try stage(&rig, a, pool, distinct[0..ns], order[0..ns]);
        // shared through main's kernel and the join, then distinct (no Seg.share) through both
        const runs = [4]struct { x: Staged, f: cuda.Function }{ .{ .x = xs, .f = fns.rows }, .{ .x = xs, .f = fns.join }, .{ .x = xd, .f = fns.rows }, .{ .x = xd, .f = fns.join } };
        var ev: [8]cuda.Event = undefined;
        for (&ev) |*e| e.* = try cuda.Event.init(gpu.d, true);
        defer for (&ev) |*e| e.deinit();
        var ms: [4][reps - dropped]f64 = undefined;
        var got: [4][3][]u8 = undefined;
        for (0..reps) |rep| {
            // the four in turn, each rep starting one later; every run's partials kept from its last rep
            for (0..4) |w| {
                const v = (w + rep) % 4;
                try ev[2 * w].record(s);
                try rows(k, runs[v].f, s, runs[v].x);
                try ev[2 * w + 1].record(s);
                if (rep == reps - 1) inline for (.{ Part.pm, Part.pl, Part.po }, 0..) |p, i| {
                    got[v][i] = try a.dupe(u8, try rig.bytes(@field(runs[v].x, @tagName(p)), runs[v].x.m * rowBytes(p)));
                };
            }
            try s.synchronize();
            if (rep < dropped) continue;
            for (0..4) |w| ms[(w + rep) % 4][rep - dropped] = try cuda.Event.elapsedMs(ev[2 * w], ev[2 * w + 1]);
        }
        const keys = len + rows_n;
        const label = try std.fmt.allocPrint(a, "{s} {d} streams x {d} rows, {d}-{d} keys", .{ k.name, ns, rows_n, keys - rows_n + 1, keys });
        // one prefix's pages and their copies hold the same bytes, so every run gives the one-stream bytes
        for (1..4) |v| for (0..3) |i| try check.sameBytes(try std.fmt.allocPrint(a, "qwen shared prefix time {s}: run {d} part {d} against main's kernel on distinct pages", .{ label, v, i }), got[v][i], got[2][i]);
        std.debug.print("RESULT qwen shared prefix time {s}: shared pages {d:.4} ms a launch through main's kernel, {d:.4} ms joined; distinct pages {d:.4} ms, {d:.4} ms joined; {d} reps after {d} dropped, bytes equal\n", .{ label, check.median(&ms[0]), check.median(&ms[1]), check.median(&ms[2]), check.median(&ms[3]), reps - dropped, dropped });
    }
}
