//! Raw identity A/B for Qwen3.5's whole-cache TC tiles and cross-stream page joins.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const qp = @import("contract_qwen_pages.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const Format = qp.Format;
const Seg = qp.Seg;
const Ancestor = qp.Ancestor;
const hd = qp.hd;
const qh = qp.q_heads;
const kvh = qp.kv_heads;
const group = qh / kvh;
const maxch = qp.max_chunks;
const layers = qp.attentions;
const shared = qp.attn_tiles;
const scale: f32 = 1.0 / 16.0;
const cache_layer: usize = 1;
const layer: c_int = cache_layer;
const geo: qp.Geo = .{ .hid = 0, .nk = 0, .nv = 0, .qh = qh, .kvh = kvh, .nl = 0 };

const Mode = enum { partial_whole, rows_whole, rows_join };
const Form = struct { mode: Mode, format: Format, control: bool = false };
const forms = [_]Form{
    .{ .mode = .partial_whole, .format = .bf16 },
    .{ .mode = .partial_whole, .format = .fp8 },
    .{ .mode = .rows_whole, .format = .bf16 },
    .{ .mode = .rows_whole, .format = .fp8 },
    .{ .mode = .rows_join, .format = .bf16 },
    .{ .mode = .rows_join, .format = .fp8 },
    .{ .mode = .rows_join, .format = .bf16, .control = true },
    .{ .mode = .rows_join, .format = .fp8, .control = true },
};
const short_shapes = [_]c.Shape{
    .{ .rows = 16, .n = qh, .k = 2100 },
    .{ .rows = 64, .n = qh, .k = 700 },
    .{ .rows = 128, .n = qh, .k = 300 },
    .{ .rows = 16, .n = qh, .k = 9000 },
    .{ .rows = 64, .n = qh, .k = 9000 },
    .{ .rows = 64, .n = qh, .k = 5000 }, // 5000/6250/7500/8750: mixed partial and rows in one serving launch
};
const row_shapes = [_]c.Shape{
    .{ .rows = 3, .n = qh, .k = 8191 }, // rows skips strictly below GROUP_LEN
    .{ .rows = 3, .n = qh, .k = 8192 },
    .{ .rows = 16, .n = qh, .k = 9000 },
    .{ .rows = 5, .n = qh, .k = 9000 },
    .{ .rows = 6, .n = qh, .k = 9000 },
    .{ .rows = 3, .n = qh, .k = 32768 },
    .{ .rows = 16, .n = qh, .k = 65535 },
    .{ .rows = 1, .n = qh, .k = 122779 },
    .{ .rows = 64, .n = qh, .k = 9000 },
    .{ .rows = 64, .n = qh, .k = 5000 },
};
const join_shapes = [_]c.Shape{
    .{ .rows = 4, .n = qh, .k = 34000 },
    .{ .rows = 8, .n = qh, .k = 33000 }, // one row per sharing stream
    .{ .rows = 16, .n = qh, .k = 35000 },
    .{ .rows = 64, .n = qh, .k = 9000 },
    .{ .rows = 64, .n = qh, .k = 5000 },
};
const control_shapes = [_]c.Shape{.{ .rows = 8, .n = qh, .k = 34000 }};

fn name(f: Form) []const u8 {
    if (f.control) return if (f.format == .bf16) "control: cuda tf_q35_attn_rows_join changed shared BF16 page" else "control: cuda tf_q35_attn_rows8_join changed shared FP8 page";
    return switch (f.mode) {
        .partial_whole => if (f.format == .bf16) "cuda tf_q35_attn_partial_whole raw PM PL PO OUT" else "cuda tf_q35_attn_partial8_whole raw PM PL PO OUT",
        .rows_whole => if (f.format == .bf16) "cuda tf_q35_attn_rows_whole raw PM PL PO OUT" else "cuda tf_q35_attn_rows8_whole raw PM PL PO OUT",
        .rows_join => if (f.format == .bf16) "cuda tf_q35_attn_rows_join raw PM PL PO OUT" else "cuda tf_q35_attn_rows8_join raw PM PL PO OUT",
    };
}
fn served(f: Form) []const c.Shape {
    if (f.control) return &control_shapes;
    return switch (f.mode) {
        .partial_whole => &short_shapes,
        .rows_whole => &row_shapes,
        .rows_join => &join_shapes,
    };
}
fn replaces(f: Form) []const u8 {
    return switch (f.mode) {
        .partial_whole => if (f.format == .bf16) "tf_q35_attn_partial with page-aware tiles" else "tf_q35_attn_partial8 with page-aware tiles",
        .rows_whole => if (f.format == .bf16) "tf_q35_attn_rows with page-aware tiles" else "tf_q35_attn_rows8 with page-aware tiles",
        .rows_join => if (f.format == .bf16) "tf_q35_attn_rows, every stream's rows independent" else "tf_q35_attn_rows8, every stream's rows independent",
    };
}
fn seen(a: Ancestor) usize {
    return if (a.chain != 0) a.depth + 1 else @popCount(a.bits);
}
fn ptok(shape: c.Shape) usize {
    return switch (shape.k) {
        300 => 16,
        700, 8192, 33000 => 48,
        2100, 35000 => 128,
        9000, 32768, 34000 => 256,
        else => 512,
    };
}
const SpanPair = @TypeOf(qp.spans(.bf16, 0, 0, 0, 0, 0));
fn e4m3Code(v: f32) u8 {
    var x: u8 = 0;
    while (x < 0x7f) : (x += 1) if (core.quant.fp8.e4m3(x) == @abs(v)) break;
    return x | @as(u8, if (v < 0) 0x80 else 0);
}

/// Join cases share a two-layer page arena; whole kernels receive canonical per-stream contiguous caches.
const Host = struct {
    ns: usize,
    ptok: usize,
    page_bytes: usize,
    lens: []usize,
    caps: []usize,
    firsts: []usize,
    rowseg: []i32,
    anc: []Ancestor,
    q: []u16,
    tables: [][]u32,
    bad: []u32,
    arena: []u8,
    caches: [][]u8,

    fn span(h: *const Host, format: Format, si: usize, at: usize, value: bool, head: usize, pos: usize, id: usize) SpanPair {
        if (h.caches.len != 0) return qp.spans(format, h.caps[si], at, @intFromBool(value), head, pos);
        var s = qp.spans(format, h.ptok, at, @intFromBool(value), head, pos % h.ptok);
        for (&s) |*x| x.off += id * h.page_bytes;
        return s;
    }
    fn buffer(h: *Host, si: usize) []u8 {
        return if (h.caches.len != 0) h.caches[si] else h.arena;
    }
    fn put(h: *Host, f: Form, value: bool, pos: usize, e: usize, bf: u16, fp: u8) void {
        const s = h.span(f.format, 0, cache_layer, value, 0, pos, h.tables[0][pos / h.ptok]);
        const out = h.buffer(0);
        switch (f.format) {
            .bf16 => std.mem.writeInt(u16, out[s[0].off + 2 * e ..][0..2], bf, .little),
            .fp8 => {
                out[s[0].off + e] = fp;
                std.mem.writeInt(u32, out[s[1].off..][0..4], @bitCast(@as(f32, 1)), .little);
            },
        }
    }
    /// Equal maxima at a chunk boundary, a larger next-chunk maximum, exact signed zero and subnormals.
    fn adversarial(h: *Host, f: Form) void {
        @memset(h.q[0..hd], 0);
        h.q[0] = c.toBf16(1);
        h.q[1] = 0x8000;
        h.q[2] = 0x0001;
        h.q[3] = 0x8001;
        h.q[4] = 0x007f;
        h.q[5] = 0x807f;
        for (0..h.lens[0] + h.rowseg.len / h.ns) |pos| h.put(f, false, pos, 0, 0, 0);
        const keys = h.lens[0] + seen(h.anc[0]);
        const ck = @max(128, (keys + maxch * 128 - 1) / (maxch * 128) * 128);
        for ([2]usize{ ck - 1, ck }, 0..) |pos, i| {
            h.put(f, false, pos, 0, c.toBf16(16), e4m3Code(16));
            h.put(f, false, pos, 1, if (i == 0) 0x8000 else 0, if (i == 0) 0x80 else 0);
            h.put(f, false, pos, 2, if (i == 0) 0x0001 else 0x8001, if (i == 0) 0x01 else 0x81);
            h.put(f, false, pos, 3, if (i == 0) 0x807f else 0x007f, if (i == 0) 0x87 else 0x07);
            h.put(f, true, pos, 0, c.toBf16(if (i == 0) 1 else 3), e4m3Code(if (i == 0) 1 else 3));
            h.put(f, true, pos, 1, if (i == 0) 0x8000 else 0, if (i == 0) 0x80 else 0);
            h.put(f, true, pos, 2, if (i == 0) 0x0001 else 0x8001, if (i == 0) 0x01 else 0x81);
            h.put(f, true, pos, 3, if (i == 0) 0x8001 else 0x007f, if (i == 0) 0x81 else 0x07);
        }
        if (2 * ck < h.lens[0]) {
            h.put(f, false, 2 * ck, 0, c.toBf16(32), e4m3Code(32));
            h.put(f, false, 2 * ck, 1, 0x8000, 0x80);
            h.put(f, true, 2 * ck, 0, c.toBf16(5), e4m3Code(5));
            h.put(f, true, 2 * ck, 1, 0x0001, 0x01);
        }
    }
    fn nonfinite(h: *Host, f: Form) void {
        const p = h.lens[0];
        const out = h.buffer(0);
        h.q[3] = 0x7fc1;
        h.put(f, true, p - 1, 7, 0x7f80, 0x78);
        if (f.format == .fp8) {
            const s = h.span(.fp8, 0, cache_layer, true, 0, p - 1, h.tables[0][(p - 1) / h.ptok]);
            std.mem.writeInt(u32, out[s[1].off..][0..4], @bitCast(std.math.inf(f32)), .little);
        }
        if (p >= 3) {
            h.put(f, true, p - 2, 11, 0x7f80, 0x78);
            h.put(f, true, p - 3, 11, 0xff80, 0xf8);
        }
        h.put(f, false, p - 1, 3, 0x7fc1, 0x7f);
    }
    fn stress(h: *Host, f: Form) void {
        for (0..h.ns) |si| for (0..kvh) |head| {
            const out = h.buffer(si);
            for (0..h.caps[si]) |pos| for (0..hd) |e| {
                const id = h.tables[si][pos / h.ptok];
                const ks = h.span(f.format, si, cache_layer, false, head, pos, id);
                const vs = h.span(f.format, si, cache_layer, true, head, pos, id);
                const kv: f32 = if (e == 0) 0.25 else 0;
                const vv: f32 = if (pos < 2048) (if (pos % 2 == 0) 128 else -128) else 0;
                switch (f.format) {
                    .bf16 => {
                        std.mem.writeInt(u16, out[ks[0].off + 2 * e ..][0..2], c.toBf16(kv), .little);
                        std.mem.writeInt(u16, out[vs[0].off + 2 * e ..][0..2], c.toBf16(if (e == 0) vv else 0), .little);
                    },
                    .fp8 => {
                        out[ks[0].off + e] = e4m3Code(kv);
                        out[vs[0].off + e] = e4m3Code(if (e == 0) vv else 0);
                        if (e == 0) {
                            std.mem.writeInt(u32, out[ks[1].off..][0..4], @bitCast(@as(f32, 1)), .little);
                            std.mem.writeInt(u32, out[vs[1].off..][0..4], @bitCast(@as(f32, 1)), .little);
                        }
                    },
                }
            };
        };
    }
};

fn makeHost(a: std.mem.Allocator, shape: c.Shape, f: Form, fill: g.Fill, seed: u64) !Host {
    const join = f.mode == .rows_join;
    const ns = if (join) (if (shape.rows <= 8) shape.rows else 4) else if (shape.rows > 64) 1 else (shape.rows + 15) / 16;
    const per = ptok(shape);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var h: Host = .{ .ns = ns, .ptok = per, .page_bytes = qp.blockBytes(f.format, per, layers), .lens = try a.alloc(usize, ns), .caps = try a.alloc(usize, ns), .firsts = try a.alloc(usize, ns), .rowseg = try a.alloc(i32, shape.rows), .anc = try a.alloc(Ancestor, shape.rows), .q = try qp.randoms(a, rng, shape.rows * qh * hd), .tables = try a.alloc([]u32, ns), .bad = undefined, .arena = undefined, .caches = try a.alloc([]u8, if (join) 0 else ns) };
    var np: usize = 0;
    for (0..ns) |si| {
        h.firsts[si] = if (join) si * shape.rows / ns else @min(si * 16, shape.rows);
        const end = if (si + 1 == ns) shape.rows else if (join) (si + 1) * shape.rows / ns else @min((si + 1) * 16, shape.rows);
        h.lens[si] = shape.k + if (join and shape.k != 5000) si * 500 else si * shape.k / ns;
        h.caps[si] = h.lens[si] + end - h.firsts[si];
        const pages = (h.caps[si] + per - 1) / per;
        np = @max(np, pages);
        h.tables[si] = try a.alloc(u32, pages);
        for (h.firsts[si]..end) |r| {
            h.rowseg[r] = @intCast(si);
            const j = r - h.firsts[si];
            if (ns == 1 and shape.rows > 64) {
                h.anc[r] = .{ .bits = 0, .depth = @intCast(j), .chain = 1 };
            } else {
                const parent: ?usize = if (j == 0 or rng.uintLessThan(u32, 6) == 0) null else rng.uintLessThan(usize, j);
                const bits = if (parent) |p| h.anc[h.firsts[si] + p].bits else 0;
                h.anc[r] = .{ .bits = bits | (@as(u64, 1) << @intCast(j)), .depth = 0, .chain = 0 };
            }
        }
    }
    h.arena = try a.alloc(u8, if (join) (ns * np + 1) * h.page_bytes else 0);
    rng.bytes(h.arena);
    for (0..ns) |si| {
        const common = if (join) h.tables[si].len -| 2 else 0;
        for (h.tables[si], 0..) |*id, p| id.* = @intCast(if (p < common) p else si * np + (np - 1 - p));
        if (join) {
            for (0..layers) |at| for (0..2) |value| for (0..kvh) |head| for (0..h.tables[si].len * per) |pos| {
                const id = h.tables[si][pos / per];
                qp.Fixture.fill(f.format, rng, h.arena, h.span(f.format, si, at, value != 0, head, pos, @intCast(id)));
            };
        } else {
            h.caches[si] = try a.alloc(u8, qp.blockBytes(f.format, h.caps[si], layers));
            for (0..layers) |at| for (0..2) |value| for (0..kvh) |head| for (0..h.caps[si]) |pos| {
                qp.Fixture.fill(f.format, rng, h.caches[si], qp.spans(f.format, h.caps[si], at, value, head, pos));
            };
        }
    }
    if (join) {
        h.bad = try a.dupe(u32, h.tables[ns - 1]);
        const copy = ns * np;
        @memcpy(h.arena[copy * h.page_bytes ..][0..h.page_bytes], h.arena[0..h.page_bytes]);
        const plant = qp.spans(f.format, per, cache_layer, 1, 0, 7);
        switch (f.format) {
            .bf16 => {
                const at = h.arena[copy * h.page_bytes + plant[0].off + 6 ..][0..2];
                std.mem.writeInt(u16, at, if (std.mem.readInt(u16, at, .little) == 0x3f40) 0x3e80 else 0x3f40, .little);
            },
            .fp8 => {
                const at = &h.arena[copy * h.page_bytes + plant[0].off + 3];
                at.* ^= 0x40;
                if (at.* & 0x7f == 0x7f) at.* ^= 1;
            },
        }
        h.bad[0] = @intCast(copy);
    } else h.bad = try a.alloc(u32, 0);
    if (fill == .random and !f.control) h.adversarial(f);
    switch (fill) {
        .random => {},
        .nonfinite => h.nonfinite(f),
        .stress => h.stress(f),
    }
    return h;
}

const Bind = struct { q: u64, segs: u64, rowseg: u64, rowanc: u64, pm: u64, pl: u64, po: u64, out: u64 };
fn bind(rig: *rig_.Rig, a: std.mem.Allocator, h: *const Host, f: Form, bad: bool) !Bind {
    const paged = f.mode == .rows_join;
    const kv = if (paged) try rig.input(h.arena) else 0;
    const segs = try a.alloc(Seg, h.ns);
    defer a.free(segs);
    for (segs, 0..) |*s, si| {
        const end = if (si + 1 == h.ns) h.rowseg.len else h.firsts[si + 1];
        const table = if (bad and si + 1 == h.ns) h.bad else h.tables[si];
        s.* = .{ .kv = if (paged) kv else try rig.input(h.caches[si]), .cap = @intCast(if (paged) table.len * h.ptok else h.caps[si]), .len = @intCast(h.lens[si]), .row0 = @intCast(h.firsts[si]), .rows = @intCast(end - h.firsts[si]) };
        if (paged) {
            s.pt = try rig.input(std.mem.sliceAsBytes(table));
            s.ps = if (f.format == .fp8) h.page_bytes else h.page_bytes / 2;
            s.ptok = @intCast(h.ptok);
            s.pn = @intCast(table.len);
        }
    }
    const slots = h.rowseg.len * qh * maxch;
    return .{ .q = try rig.input(std.mem.sliceAsBytes(h.q)), .segs = try rig.input(std.mem.sliceAsBytes(segs)), .rowseg = try rig.input(std.mem.sliceAsBytes(h.rowseg)), .rowanc = try rig.input(std.mem.sliceAsBytes(h.anc)), .pm = try rig.output(slots * 4), .pl = try rig.output(slots * 4), .po = try rig.output(slots * hd * 4), .out = try rig.output(h.rowseg.len * qh * hd * 2) };
}

const Entry = struct {
    rig: rig_.Rig,
    base: rig_.Rig,
    form: Form,
    partial: cuda.Function,
    original_partial: cuda.Function,
    rows: cuda.Function,
    original_rows: cuda.Function,
    merge: cuda.Function,
    msg: [160]u8 = undefined,
    arena: ?std.heap.ArenaAllocator = null,
    host: ?Host = null,
    dev: Bind = undefined,
    base_dev: Bind = undefined,

    fn deinit(e: *Entry) void {
        e.rig.deinit();
        e.base.deinit();
        if (e.arena) |*a| a.deinit();
    }
    fn kernel(e: *Entry) g.Kernel {
        return .{ .ctx = e, .vt = &vt, .control = if (e.form.control) .shapes else null, .spec = .{ .name = name(e.form), .accum = .none, .depth = 0, .out = .raw, .replaces = replaces(e.form), .served = served(e.form), .nan = .{ .stated = "raw PM PL PO OUT, poisoned unwritten slots, NaN and infinities" } } };
    }
    const vt = g.hooks(Entry);
    fn prepare(e: *Entry, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != qh or shape.k == 0) return false;
        e.rig.reset();
        e.base.reset();
        if (e.arena) |*a| a.deinit();
        e.arena = .init(e.rig.gpu.gpa);
        const a = e.arena.?.allocator();
        e.host = try makeHost(a, shape, e.form, fill, seed);
        e.dev = try bind(&e.rig, a, &e.host.?, e.form, e.form.control);
        e.base_dev = try bind(&e.base, a, &e.host.?, e.form, false);
        return true;
    }
    fn launch(e: *Entry, rig: *rig_.Rig, b: Bind, primary: bool) !void {
        const rows = e.host.?.rowseg.len;
        var args: cuda.Args = .{};
        inline for (.{ geo, b.q, b.segs, b.rowseg, b.rowanc, layer, @as(u64, 0), scale }) |v| args.add(v);
        inline for (.{ b.pm, b.pl, b.po }) |v| args.add(v);
        const sm = if (e.form.format == .bf16) shared else 0;
        try cuda.launch.launch(if (primary) e.partial else e.original_partial, .{ .grid = .{ .x = @intCast(rows), .y = maxch, .z = kvh }, .block = .{ .x = 128 }, .shared = sm }, rig.s, &args);
        args = .{};
        inline for (.{ geo, b.q, b.segs, b.rowseg, b.rowanc, layer, @as(u64, 0), scale }) |v| args.add(v);
        args.add(@as(c_int, @intCast(rows)));
        inline for (.{ b.pm, b.pl, b.po }) |v| args.add(v);
        const tiles = 64 / group;
        try cuda.launch.launch(if (primary) e.rows else e.original_rows, .{ .grid = .{ .x = @intCast((rows + tiles - 1) / tiles), .y = maxch, .z = kvh }, .block = .{ .x = 256 }, .shared = sm }, rig.s, &args);
        args = .{};
        inline for (.{ geo, b.pm, b.pl, b.po, b.segs, b.rowseg, b.rowanc, @as(u64, 0), b.out }) |v| args.add(v);
        try cuda.launch.launch(e.merge, .{ .grid = .{ .x = qh, .y = @intCast(rows) }, .block = .{ .x = 32 } }, rig.s, &args);
        try rig.s.synchronize();
        if (!try rig.intact() or rig.overlap()) return error.GuardChanged;
    }
    fn run(e: *Entry) !void {
        try e.launch(&e.rig, e.dev, true);
    }
    fn replaced(e: *Entry) ![]const u8 {
        try e.launch(&e.base, e.base_dev, false);
        return e.base.outputBytes();
    }
    fn count(e: *const Entry) usize {
        const rows = e.host.?.rowseg.len;
        return rows * qh * (maxch * (8 + hd * 4) + hd * 2);
    }
    fn reference(_: *Entry, _: []f64, _: []f64) !void {
        return error.RawIdentityUsesReplacement;
    }
    fn check(e: *Entry, at: g.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const got = try e.rig.outputBytes();
        const want = try e.replaced();
        if (!std.mem.eql(u8, got, want)) return try std.fmt.bufPrint(&e.msg, "nonfinite primary and replacement raw bytes differ", .{});
        const out_n = e.host.?.rowseg.len * qh * hd;
        const out = got[got.len - out_n * 2 ..];
        var finite: usize = 0;
        var nan: usize = 0;
        var inf: usize = 0;
        for (0..out_n) |i| {
            const bits = std.mem.readInt(u16, out[2 * i ..][0..2], .little);
            if (bits & 0x7f80 != 0x7f80) finite += 1 else if (bits & 0x007f == 0) inf += 1 else nan += 1;
        }
        if (finite == 0 or nan == 0 or inf == 0) return try std.fmt.bufPrint(&e.msg, "planted output classes finite={d}, NaN={d}, infinity={d}: no proof", .{ finite, nan, inf });
        return null;
    }
};

/// The six primary exports plus one planted changed-page control for each cache format.
pub const Group = struct {
    a: std.mem.Allocator,
    bf16: ?cuda.Module = null,
    fp8: ?cuda.Module = null,
    items: [forms.len]Entry = undefined,
    made: usize = 0,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        self.* = .{ .a = a };
        errdefer self.destroy();
        self.bf16 = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
        self.fp8 = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_kv8);
        const merge = try self.bf16.?.function("tf_q35_attn_merge");
        for (forms) |f| {
            const mod = if (f.format == .bf16) self.bf16.? else self.fp8.?;
            const whole = f.mode != .rows_join;
            const partial_name: [:0]const u8 = if (f.format == .fp8)
                (if (whole) "tf_q35_attn_partial8_whole" else "tf_q35_attn_partial8")
            else
                (if (whole) "tf_q35_attn_partial_whole" else "tf_q35_attn_partial");
            const rows_name: [:0]const u8 = switch (f.mode) {
                .partial_whole, .rows_whole => if (f.format == .fp8) "tf_q35_attn_rows8_whole" else "tf_q35_attn_rows_whole",
                .rows_join => if (f.format == .fp8) "tf_q35_attn_rows8_join" else "tf_q35_attn_rows_join",
            };
            const original_partial_name: [:0]const u8 = if (f.format == .fp8) "tf_q35_attn_partial8" else "tf_q35_attn_partial";
            const original_rows_name: [:0]const u8 = if (f.format == .fp8) "tf_q35_attn_rows8" else "tf_q35_attn_rows";
            const fns = [4]cuda.Function{ try mod.function(partial_name), try mod.function(original_partial_name), try mod.function(rows_name), try mod.function(original_rows_name) };
            if (f.format == .bf16) for (fns) |function| try function.allowDynamicShared(shared);
            self.items[self.made] = .{ .rig = .{ .gpu = gpu, .s = s }, .base = .{ .gpu = gpu, .s = s }, .form = f, .partial = fns[0], .original_partial = fns[1], .rows = fns[2], .original_rows = fns[3], .merge = merge };
            self.made += 1;
            try kernels.append(a, self.items[self.made - 1].kernel());
        }
        return self;
    }
    pub fn destroy(self: *Group) void {
        for (self.items[0..self.made]) |*e| e.deinit();
        if (self.fp8) |m| m.unload();
        if (self.bf16) |m| m.unload();
        self.a.destroy(self);
    }
};
