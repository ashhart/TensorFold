//! A prompt chunk's working buffers (row r of each is position a + r, each stage out of place) and its row records.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("dsv41").config.Config;
const pipelines = @import("pipelines.zig");
const kv = @import("kv.zig");
const attn = pipelines.attn;
const xm = @import("experts_mma.zig");

pub const Ref = pipelines.glue.Ref;
/// Window rows before a chunk's first row in the linear window buffer: row `back + t - a` holds position t.
pub const back = attn.window - 1;
/// Rows one routed-expert launch carries on the decode row kernels (the loop kept as the MMA experts' reference).
pub const expert_rows = 16;

/// Where each row record starts among the records' ints, for chunks of `rows` rows.
pub const Rec = struct {
    pos: usize,
    cpos: usize,
    dest: usize,
    vis: usize,
    wdest: usize,
    tdest: usize,
    swa: usize,
    comp: usize,
    iota: usize,
    ints: usize,

    pub fn of(rows: usize) Rec {
        var r: Rec = undefined;
        var at: usize = 0;
        inline for (.{ .{ "pos", 1 }, .{ "cpos", 2 }, .{ "dest", 2 }, .{ "vis", 2 }, .{ "wdest", 2 }, .{ "tdest", 1 }, .{ "swa", 8 }, .{ "comp", 8 } }) |f| {
            @field(r, f[0]) = at;
            at += f[1] * rows;
        }
        r.iota = at;
        r.ints = at + back + rows;
        return r;
    }
};

pub const Scratch = struct {
    buf: mtl.Buffer = undefined,
    /// rows a chunk carries, index rows scored at once, the scores' width (positions) and candidate blocks a row
    rows: u32 = 0,
    ix_rows: u32 = 0,
    keys: u32 = 0,
    blocks: u32 = 0,
    rec: Rec = undefined,
    h: Ref = undefined,
    hn: Ref = undefined,
    hq: Ref = undefined,
    qa: Ref = undefined,
    qr: Ref = undefined,
    qrq: Ref = undefined,
    q: Ref = undefined,
    qrope: Ref = undefined,
    kv: Ref = undefined,
    kvn: Ref = undefined,
    kvr: Ref = undefined,
    lin: Ref = undefined,
    cf: Ref = undefined,
    cg: Ref = undefined,
    lat: Ref = undefined,
    latn: Ref = undefined,
    latr: Ref = undefined,
    ik: Ref = undefined,
    ikn: Ref = undefined,
    ikr: Ref = undefined,
    ixq: Ref = undefined,
    ixqr: Ref = undefined,
    ixqc: Ref = undefined,
    ixqs: Ref = undefined,
    ixw: Ref = undefined,
    scores: Ref = undefined,
    bscore: Ref = undefined,
    cmask: Ref = undefined,
    sel: Ref = undefined,
    o: Ref = undefined,
    oq: Ref = undefined,
    oa: Ref = undefined,
    ob: Ref = undefined,
    aout: Ref = undefined,
    x1: Ref = undefined,
    mix_a: [3]Ref = undefined,
    mix_f: [3]Ref = undefined,
    h2: Ref = undefined,
    hn2: Ref = undefined,
    hq2: Ref = undefined,
    logits: Ref = undefined,
    picks: Ref = undefined,
    wts: Ref = undefined,
    ones: Ref = undefined,
    image: Ref = undefined,
    plan: [4]Ref = undefined,
    /// the grouped experts' tile table, act_quant codes and scales, and gate scratch
    xtiles: Ref = undefined,
    xcodes: Ref = undefined,
    xscales: Ref = undefined,
    act: Ref = undefined,
    eout: Ref = undefined,
    yr: Ref = undefined,
    sg: Ref = undefined,
    su: Ref = undefined,
    sact: Ref = undefined,
    sactq: Ref = undefined,
    sout: Ref = undefined,
    mout: Ref = undefined,
    meta: Ref = undefined,

    /// Buffers for chunks of up to `rows` rows; `limit` positions bound the index scores, scored `ix_rows` at a time.
    pub fn init(device: mtl.Device, c: *const Config, rows: u32, ix_rows: u32, limit: u32) !Scratch {
        if (rows == 0 or ix_rows == 0) return error.BadRows;
        var s: Scratch = .{};
        const size = s.layout(c, rows, ix_rows, limit);
        s.buf = try device.buffer(size, mtl.ResourceOptions.shared);
        @memset(s.buf.contents()[0..size], 0);
        _ = s.layout(c, rows, ix_rows, limit);
        @memset(s.f32s(s.ones, rows * c.topk), 1);
        const iota = s.i32s(s.meta, s.rec.ints)[s.rec.iota..];
        for (iota, 0..) |*v, i| v.* = @intCast(i);
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        s.buf.deinit();
    }

    fn layout(s: *Scratch, c: *const Config, rows: u32, ix_rows: u32, limit: u32) usize {
        var off: usize = 0;
        const L = struct {
            fn take(sc: *Scratch, o: *usize, len: usize) Ref {
                const at = std.mem.alignForward(usize, o.*, 256);
                o.* = at + @max(len, 4);
                return .{ .buf = sc.buf, .off = at };
            }
        };
        const R: usize = rows;
        const H: usize = c.hidden;
        const D: usize = c.head_dim;
        const QL: usize = c.q_lora;
        const QH: usize = @as(usize, c.heads) * D;
        const IX: usize = @as(usize, c.index_heads) * c.index_dim;
        const W: usize = c.expert_width;
        const P: usize = @as(usize, rows) * c.topk;
        const xb = xm.bytes(.{ .rows = rows, .k = c.hidden, .n = c.expert_width, .experts = c.experts });
        s.rows = rows;
        s.ix_rows = @min(ix_rows, rows);
        s.keys = limit;
        s.blocks = std.mem.alignForward(u32, (limit + 7) / 8, 4);
        s.rec = Rec.of(R);
        inline for (.{ "h", "hn", "hq", "aout", "h2", "hn2", "hq2", "sout", "mout" }) |f| @field(s, f) = L.take(s, &off, R * H * 2);
        inline for (.{ "qa", "qr", "qrq" }) |f| @field(s, f) = L.take(s, &off, R * QL * 2);
        inline for (.{ "q", "qrope", "o", "oq" }) |f| @field(s, f) = L.take(s, &off, R * QH * 2);
        inline for (.{ "kv", "kvn", "kvr", "lat", "latn", "latr" }) |f| @field(s, f) = L.take(s, &off, R * D * 2);
        s.lin = L.take(s, &off, (back + R) * D * 2);
        inline for (.{ "cf", "cg" }) |f| @field(s, f) = L.take(s, &off, R * D * 4);
        inline for (.{ "ik", "ikn", "ikr" }) |f| @field(s, f) = L.take(s, &off, R * c.index_dim * 2);
        inline for (.{ "ixq", "ixqr" }) |f| @field(s, f) = L.take(s, &off, R * IX * 2);
        s.ixqc = L.take(s, &off, R * IX / 2);
        s.ixqs = L.take(s, &off, R * IX / 32);
        s.ixw = L.take(s, &off, R * c.index_heads * 2);
        s.scores = L.take(s, &off, @as(usize, s.ix_rows) * s.keys * 4);
        s.bscore = L.take(s, &off, @as(usize, s.ix_rows) * s.blocks * 4);
        s.cmask = L.take(s, &off, R * s.blocks);
        s.sel = L.take(s, &off, R * attn.max_selected * 4);
        s.oa = L.take(s, &off, R * @as(usize, c.o_groups) * c.o_lora * 2);
        s.ob = L.take(s, &off, R * @as(usize, c.o_groups) * c.o_lora * 2);
        s.x1 = L.take(s, &off, R * 4 * H * 2);
        for (&s.mix_a, [3]usize{ 16, 16, 64 }) |*m, n| m.* = L.take(s, &off, R * n);
        for (&s.mix_f, [3]usize{ 16, 16, 64 }) |*m, n| m.* = L.take(s, &off, R * n);
        s.logits = L.take(s, &off, R * @as(usize, c.experts) * 4);
        inline for (.{ "picks", "wts" }) |f| @field(s, f) = L.take(s, &off, R * c.topk * 4);
        s.ones = L.take(s, &off, P * 4);
        s.image = L.take(s, &off, R);
        for (&s.plan, [4]usize{ 12 * P, 4, 4 * P, 4 * P }) |*m, n| m.* = L.take(s, &off, n);
        s.xtiles = L.take(s, &off, xb[0]);
        s.xcodes = L.take(s, &off, xb[1]);
        s.xscales = L.take(s, &off, xb[2]);
        s.act = L.take(s, &off, P * W * 2);
        s.eout = L.take(s, &off, P * H * 2);
        s.yr = L.take(s, &off, R * H * 4);
        inline for (.{ "sg", "su", "sact", "sactq" }) |f| @field(s, f) = L.take(s, &off, R * W * 2);
        s.meta = L.take(s, &off, s.rec.ints * 4);
        return off;
    }

    pub fn bytes(s: *const Scratch, r: Ref, n: usize) []u8 {
        return (s.buf.contents() + r.off)[0..n];
    }

    pub fn f32s(s: *const Scratch, r: Ref, n: usize) []f32 {
        return @alignCast(std.mem.bytesAsSlice(f32, s.bytes(r, 4 * n)));
    }

    pub fn i32s(s: *const Scratch, r: Ref, n: usize) []i32 {
        return @alignCast(std.mem.bytesAsSlice(i32, s.bytes(r, 4 * n)));
    }

    /// The records' int at index `at`.
    pub fn metaAt(s: *const Scratch, at: usize) Ref {
        return .{ .buf = s.buf, .off = s.meta.off + 4 * at };
    }

    /// Records of positions [a, a + rows) of one stream; `cap` is the tail ring's rows (its destinations).
    pub fn writeRows(s: *const Scratch, c: *const Config, a: u32, rows: u32, cap: u32) !void {
        if (rows == 0 or rows > s.rows) return error.BadRows;
        const m = s.i32s(s.meta, s.rec.iota);
        @memset(m, -1);
        const R = s.rows;
        const k = s.rec;
        for (0..rows) |r| {
            const pos: u32 = a + @as(u32, @intCast(r));
            m[k.pos + r] = @intCast(pos);
            for ([2]u32{ 1, 2 }, 0..) |ratio, j| {
                const done = (pos + 1) % ratio == 0;
                m[k.cpos + j * R + r] = if (done) @intCast(pos + 1 - ratio) else -1;
                m[k.dest + j * R + r] = if (done) @intCast(pos / ratio) else -1;
                m[k.vis + j * R + r] = @intCast((pos + 1) / ratio);
            }
            const w = kv.windowRows(c, pos);
            m[k.wdest + 2 * r] = @intCast(w[0]);
            m[k.wdest + 2 * r + 1] = @intCast(w[1]);
            m[k.tdest + r] = @intCast(pos % cap);
            const meta = attn.Meta{ .pos = @intCast(pos), .wofs = back, .wbase = @intCast(a), .cofs = 0, .ncomp = 0 };
            var comp = meta;
            comp.ncomp = attn.max_selected;
            @memcpy(m[k.swa + 8 * r ..][0..8], @as(*const [8]i32, @ptrCast(&meta)));
            @memcpy(m[k.comp + 8 * r ..][0..8], @as(*const [8]i32, @ptrCast(&comp)));
        }
    }
};

test "the records' places: each field a run of rows, the iota last" {
    const r = Rec.of(10);
    try std.testing.expectEqual(@as(usize, 10), r.cpos);
    try std.testing.expectEqual(@as(usize, 180), r.comp);
    try std.testing.expectEqual(@as(usize, 260), r.iota);
    try std.testing.expectEqual(@as(usize, 260 + back + 10), r.ints);
}
