//! One launch's working buffers (each stage out of place, so a layer's intermediates stay readable) and row records.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("dsv41").config.Config;
const pipelines = @import("pipelines.zig");
const kv = @import("kv.zig");
const attn = pipelines.attn;
const fp8 = pipelines.fp8;

pub const Ref = pipelines.glue.Ref;
const R = pipelines.max_rows;

/// Index widths of the row records: positions, then per compress ratio (1, 2) publish positions, rows and counts.
pub const Meta = struct {
    pub const pos = 0;
    pub const cpos = R;
    pub const dest = 3 * R;
    pub const vis = 5 * R;
    pub const wdest = 7 * R;
    pub const attn_swa = 9 * R;
    pub const attn_comp = 9 * R + 8 * R;
    /// the launch's row indices 0 .. rows - 1 (a scatter's destinations)
    pub const iota = 9 * R + 16 * R;
    pub const ints = iota + R;
};

pub const Scratch = struct {
    buf: mtl.Buffer = undefined,
    keys: u32 = 0,
    blocks: u32 = 0,
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
    kvq: Ref = undefined,
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
    /// the routed experts' act_quant of hn2: E4M3 values as fp16 [rows][hidden] and fp32 scales [rows][hidden / 32]
    xq: Ref = undefined,
    xqs: Ref = undefined,
    act: Ref = undefined,
    eout: Ref = undefined,
    yr: Ref = undefined,
    sg: Ref = undefined,
    su: Ref = undefined,
    sact: Ref = undefined,
    sactq: Ref = undefined,
    sout: Ref = undefined,
    mout: Ref = undefined,
    fp8s: Ref = undefined,
    /// the hc mix's partial sums (fused.zig)
    hcp: Ref = undefined,
    /// the wide prenorm's partial sums (fused.zig)
    nrp: Ref = undefined,
    meta: Ref = undefined,

    /// Buffers for up to 16 rows of one stream; `limit` positions bound the index scores' width.
    pub fn init(device: mtl.Device, c: *const Config, limit: u32) !Scratch {
        var s: Scratch = .{};
        const size = s.layout(c, limit);
        s.buf = try device.buffer(size, mtl.ResourceOptions.shared);
        @memset(s.buf.contents()[0..size], 0);
        _ = s.layout(c, limit);
        const ones = s.f32s(s.ones, R * c.topk);
        @memset(ones, 1);
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        s.buf.deinit();
    }

    fn layout(s: *Scratch, c: *const Config, limit: u32) usize {
        var off: usize = 0;
        const L = struct {
            fn take(sc: *Scratch, o: *usize, len: usize) Ref {
                const at = std.mem.alignForward(usize, o.*, 256);
                o.* = at + @max(len, 4);
                return .{ .buf = sc.buf, .off = at };
            }
        };
        const H: usize = c.hidden;
        const D: usize = c.head_dim;
        const QL: usize = c.q_lora;
        const QH: usize = @as(usize, c.heads) * D;
        const IX: usize = @as(usize, c.index_heads) * c.index_dim;
        const W: usize = c.expert_width;
        const P: usize = R * c.topk;
        s.keys = limit;
        s.blocks = std.mem.alignForward(u32, (limit + 7) / 8, 4);
        inline for (.{ "h", "hn", "hq", "aout", "h2", "hn2", "hq2", "sout", "mout" }) |f| @field(s, f) = L.take(s, &off, R * H * 2);
        inline for (.{ "qa", "qr", "qrq" }) |f| @field(s, f) = L.take(s, &off, R * QL * 2);
        inline for (.{ "q", "qrope", "o", "oq" }) |f| @field(s, f) = L.take(s, &off, R * QH * 2);
        inline for (.{ "kv", "kvn", "kvr", "kvq", "lat", "latn", "latr" }) |f| @field(s, f) = L.take(s, &off, R * D * 2);
        inline for (.{ "cf", "cg" }) |f| @field(s, f) = L.take(s, &off, R * D * 4);
        inline for (.{ "ik", "ikn", "ikr" }) |f| @field(s, f) = L.take(s, &off, R * c.index_dim * 2);
        inline for (.{ "ixq", "ixqr" }) |f| @field(s, f) = L.take(s, &off, R * IX * 2);
        s.ixqc = L.take(s, &off, R * IX / 2);
        s.ixqs = L.take(s, &off, R * IX / 32);
        s.ixw = L.take(s, &off, R * c.index_heads * 2);
        s.scores = L.take(s, &off, R * @as(usize, s.keys) * 4);
        s.bscore = L.take(s, &off, R * @as(usize, s.blocks) * 4);
        s.cmask = L.take(s, &off, R * @as(usize, s.blocks));
        s.sel = L.take(s, &off, R * attn.max_selected * 4);
        s.oa = L.take(s, &off, R * @as(usize, c.o_groups) * c.o_lora * 2);
        s.ob = L.take(s, &off, R * @as(usize, c.o_groups) * c.o_lora * 2);
        s.x1 = L.take(s, &off, R * 4 * H * 2);
        for (&s.mix_a, [3]usize{ 16, 16, 64 }) |*m, n| m.* = L.take(s, &off, R * n);
        for (&s.mix_f, [3]usize{ 16, 16, 64 }) |*m, n| m.* = L.take(s, &off, R * n);
        s.logits = L.take(s, &off, R * @as(usize, c.experts) * 4);
        inline for (.{ "picks", "wts", "ones" }) |f| @field(s, f) = L.take(s, &off, P * 4);
        s.image = L.take(s, &off, R);
        for (&s.plan, [4]usize{ 12 * P, 4, 4 * P, 4 * P }) |*m, n| m.* = L.take(s, &off, n);
        s.xq = L.take(s, &off, R * H * 2);
        s.xqs = L.take(s, &off, R * H / 8);
        s.act = L.take(s, &off, P * W * 2);
        s.eout = L.take(s, &off, P * H * 2);
        s.yr = L.take(s, &off, R * H * 4);
        inline for (.{ "sg", "su", "sact", "sactq" }) |f| @field(s, f) = L.take(s, &off, R * W * 2);
        s.fp8s = L.take(s, &off, fp8Scratch(c));
        s.hcp = L.take(s, &off, pipelines.fused.mixScratch(R));
        s.nrp = L.take(s, &off, pipelines.fused.normScratch(R));
        s.meta = L.take(s, &off, Meta.ints * 4);
        return off;
    }

    pub fn bytes(s: *const Scratch, r: Ref, n: usize) []u8 {
        return (s.buf.contents() + r.off)[0..n];
    }

    pub fn f32s(s: *const Scratch, r: Ref, n: usize) []f32 {
        return @alignCast(std.mem.bytesAsSlice(f32, s.bytes(r, 4 * n)));
    }

    pub fn u16s(s: *const Scratch, r: Ref, n: usize) []u16 {
        return @alignCast(std.mem.bytesAsSlice(u16, s.bytes(r, 2 * n)));
    }

    pub fn i32s(s: *const Scratch, r: Ref, n: usize) []i32 {
        return @alignCast(std.mem.bytesAsSlice(i32, s.bytes(r, 4 * n)));
    }

    /// The row records at index `at` of Meta.
    pub fn metaAt(s: *const Scratch, at: usize) Ref {
        return .{ .buf = s.buf, .off = s.meta.off + 4 * at };
    }

    /// Records for `rows` consecutive positions from `p0` of one stream, written before the launch's encode.
    pub fn writeRows(s: *const Scratch, c: *const Config, p0: u32, rows: u32) !void {
        return s.writeSpans(c, &.{.{ .p0 = p0, .n = rows }});
    }

    /// Records for several slots' rows: each span's consecutive positions, spans one after another in row order.
    pub fn writeSpans(s: *const Scratch, c: *const Config, spans: []const Span) !void {
        var total: u32 = 0;
        for (spans) |sp| total += sp.n;
        if (total == 0 or total > R) return error.BadRows;
        const m = s.i32s(s.meta, Meta.ints);
        @memset(m, -1);
        var r0: u32 = 0;
        for (spans) |sp| {
            if (sp.n == 0) return error.BadRows;
            for (0..sp.n) |i| writeRow(m, c, sp.p0 + @as(u32, @intCast(i)), r0 + @as(u32, @intCast(i)), @intCast(i));
            r0 += sp.n;
        }
    }
};

/// One slot's consecutive rows of a launch.
pub const Span = struct { p0: u32, n: u32 };

/// Row `r` of the launch at position `pos`; `local` is its index among its own slot's rows (a scatter's destination).
fn writeRow(m: []i32, c: *const Config, pos: u32, r: usize, local: usize) void {
    const nw = kv.windowSlots(c);
    m[Meta.pos + r] = @intCast(pos);
    m[Meta.iota + r] = @intCast(local);
    for ([2]u32{ 1, 2 }, 0..) |ratio, k| {
        const done = (pos + 1) % ratio == 0;
        m[Meta.cpos + k * R + r] = if (done) @intCast(pos + 1 - ratio) else -1;
        m[Meta.dest + k * R + r] = if (done) @intCast(pos / ratio) else -1;
        m[Meta.vis + k * R + r] = @intCast((pos + 1) / ratio);
    }
    const w = kv.windowRows(c, pos);
    m[Meta.wdest + 2 * r] = @intCast(w[0]);
    m[Meta.wdest + 2 * r + 1] = @intCast(w[1]);
    const lo: u32 = if (pos + 1 > c.window) pos + 1 - c.window else 0;
    const meta = attn.Meta{ .pos = @intCast(pos), .wofs = @intCast(lo % nw), .wbase = @intCast(lo), .cofs = 0, .ncomp = 0 };
    var comp = meta;
    comp.ncomp = attn.max_selected;
    @memcpy(m[Meta.attn_swa + 8 * r ..][0..8], @as(*const [8]i32, @ptrCast(&meta)));
    @memcpy(m[Meta.attn_comp + 8 * r ..][0..8], @as(*const [8]i32, @ptrCast(&comp)));
}

/// The fp8 row kernels' scratch for the widest projection a layer runs.
fn fp8Scratch(c: *const Config) usize {
    const kn = [_][2]u32{
        .{ c.hidden, c.q_lora },              .{ c.hidden, c.head_dim },
        .{ c.q_lora, c.heads * c.head_dim },  .{ c.heads / c.o_groups * c.head_dim, c.o_lora },
        .{ c.o_groups * c.o_lora, c.hidden }, .{ c.q_lora, c.index_heads * c.index_dim },
        .{ c.hidden, c.expert_width },        .{ c.expert_width, c.hidden },
    };
    var most: usize = 0;
    for (kn) |p| most = @max(most, fp8.scratchBytes(.{ .k = p[0], .n = p[1], .rows = R }));
    // the DSpark drafter's main_proj reads all the taps of a row
    const taps: u32 = @intCast(c.dspark_targets.len);
    if (c.mtp_layers > 0 and taps > 0) most = @max(most, fp8.scratchBytes(.{ .k = taps * c.hidden, .n = c.hidden, .rows = R }));
    return most;
}
