//! The decode forward on Metal: embed into four streams, every layer (Engram lookups, DSpark taps), hc head, logits.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const ctx_ = @import("ctx.zig");
const weights = @import("weights.zig");
const pipelines = @import("pipelines.zig");
const layer = @import("layer.zig");

const Config = host.config.Config;
const Ctx = ctx_.Ctx;
const Ref = ctx_.Ref;
const g = pipelines.glue;
const cl = @import("calls.zig");
const eng = pipelines.engram;
const cv = ctx_.cv;
const R = pipelines.max_rows;

/// The weights a forward reads: the model's top tensors and every backbone layer in order.
pub const Model = struct { top: *const weights.Top, layers: []const weights.Layer };

/// Where Engram rows come from: the family's hashes over the token history, then a row source (null: index rows).
pub const Lookup = struct { a: std.mem.Allocator, host: *const host.engram_host.Host, source: ?host.engram_host.Source };

/// The Engram call of a launch's rows: hash columns of 256 values in, a key per stream and one value out.
pub fn engramCall(c: *const Config, rows: u32) eng.Call {
    return .{ .rows = rows, .cols = c.hashCols(), .n = (c.hc_mult + 1) * c.hidden, .dim = c.hidden, .eps = c.eps };
}

/// A launch's buffers beside the layer scratch: ids, the ping-pong streams, the Engram stages and the head's.
pub const Bufs = struct {
    buf: mtl.Buffer = undefined,
    ids: Ref = undefined,
    streams: [2]Ref = undefined,
    pres: [2]Ref = undefined,
    /// each Engram layer's rows [rows][cols * 256] bf16, staged by `prepare`
    rows: [4]Ref = undefined,
    emb: Ref = undefined,
    ekv: Ref = undefined,
    gate: Ref = undefined,
    roots: Ref = undefined,
    mask: Ref = undefined,
    hf: Ref = undefined,
    hfn: Ref = undefined,
    /// [rows][vocab] fp32
    logits: Ref = undefined,

    pub fn init(device: mtl.Device, c: *const Config) !Bufs {
        var b: Bufs = .{};
        const size = b.layout(c);
        b.buf = try device.buffer(size, mtl.ResourceOptions.shared);
        @memset(b.buf.contents()[0..size], 0);
        _ = b.layout(c);
        // text only: every row's Engram gate is open (image rows shut it; the prompt path sets them)
        @memset(b.bytes(b.mask, R), 1);
        return b;
    }

    pub fn deinit(b: *Bufs) void {
        b.buf.deinit();
    }

    fn layout(b: *Bufs, c: *const Config) usize {
        var off: usize = 0;
        const L = struct {
            fn take(bb: *Bufs, o: *usize, len: usize) Ref {
                const at = std.mem.alignForward(usize, o.*, 256);
                o.* = at + @max(len, 4);
                return .{ .buf = bb.buf, .off = at };
            }
        };
        const H: usize = c.hidden;
        const call = engramCall(c, R);
        const K: usize = @as(usize, call.cols) * eng.head;
        b.ids = L.take(b, &off, R * 4);
        for (&b.streams) |*s| s.* = L.take(b, &off, R * 4 * H * 2);
        for (&b.pres) |*s| s.* = L.take(b, &off, R * 4 * 4);
        for (b.rows[0..c.engram_layers.len]) |*s| s.* = L.take(b, &off, R * K * 2);
        b.emb = L.take(b, &off, R * K * 2);
        b.ekv = L.take(b, &off, R * @as(usize, call.n) * 2);
        b.gate = L.take(b, &off, R * 4 * 4);
        b.roots = L.take(b, &off, eng.fp8Scratch(call));
        b.mask = L.take(b, &off, R);
        b.hf = L.take(b, &off, R * H * 2);
        b.hfn = L.take(b, &off, R * H * 2);
        b.logits = L.take(b, &off, R * @as(usize, c.vocab) * 4);
        return off;
    }

    pub fn bytes(b: *const Bufs, r: Ref, n: usize) []u8 {
        return (b.buf.contents() + r.off)[0..n];
    }

    /// The launch's logits, row r at [r * vocab].
    pub fn logitsOf(b: *const Bufs, c: *const Config, rows: usize) []const f32 {
        return @alignCast(std.mem.bytesAsSlice(f32, b.bytes(b.logits, rows * c.vocab * 4)));
    }
};

/// Before the encode: the ids, each Engram layer's rows; `lookback`: the ids before, oldest first (short: start).
pub fn prepare(b: *const Bufs, c: *const Config, ids: []const u32, lookback: []const u32, lk: ?Lookup) !void {
    return prepareSegs(b, c, &.{.{ .ids = ids, .lookback = lookback }}, lk);
}

/// One slot's ids of a launch and the ids before them (oldest first; short: the slot's start).
pub const Piece = struct { ids: []const u32, lookback: []const u32 };

/// `prepare` for several slots' rows, pieces one after another in row order.
pub fn prepareSegs(b: *const Bufs, c: *const Config, pieces: []const Piece, lk: ?Lookup) !void {
    var n: usize = 0;
    for (pieces) |pc| {
        if (pc.ids.len == 0) return error.BadRows;
        for (pc.ids) |id| if (id >= c.vocab) return error.BadToken;
        n += pc.ids.len;
    }
    if (n > R) return error.BadRows;
    const all: []u32 = @alignCast(std.mem.bytesAsSlice(u32, b.bytes(b.ids, 4 * n)));
    var at: usize = 0;
    for (pieces) |pc| {
        @memcpy(all[at..][0..pc.ids.len], pc.ids);
        at += pc.ids.len;
    }
    if (c.engram_layers.len == 0) return;
    const l = lk orelse return error.EngramLookupMissing;
    const K = c.hashCols() * c.engram_head_dim;
    for (c.engram_layers.items(), 0..) |id, li| {
        const out: []u16 = @alignCast(std.mem.bytesAsSlice(u16, b.bytes(b.rows[li], 2 * n * K)));
        var r0: usize = 0;
        for (pieces) |pc| {
            try l.host.windowRows(l.a, l.source, pc.ids, pc.lookback, id, 0, 1, c.engram_head_dim, out[r0 * K ..][0 .. pc.ids.len * K]);
            r0 += pc.ids.len;
        }
    }
}

/// The tokens' embedding rows into all four streams of streams[0], and the one-hot pre mix.
pub fn encodeEmbed(x: *const Ctx, e: mtl.ComputeEncoder, m: Model, b: *const Bufs) void {
    g.embed(&x.p.glue, e, m.top.embed, b.ids, b.streams[0], b.pres[0], x.rows, x.c.hidden);
}

/// Layer i: Engram into streams[i % 2], its DSpark tap, then the layer from streams[i % 2] into the other pair.
pub fn encodeLayer(x: *const Ctx, e: mtl.ComputeEncoder, m: Model, b: *const Bufs, i: usize) !void {
    const c = x.c;
    const w = &m.layers[i];
    if (w.layer != i) return error.LayerOrder;
    const a = i % 2;
    if (std.mem.indexOfScalar(u32, c.engram_layers.items(), w.layer)) |li| {
        try engram(x, e, b, w.engram orelse return error.EngramWeightsMissing, li, b.streams[a]);
    }
    if (std.mem.indexOfScalar(u32, c.dspark_targets.items(), w.layer)) |t| {
        const T: u32 = @intCast(c.dspark_targets.len);
        var buf: [R]Ctx = undefined;
        for (x.parts(&buf)) |*px| {
            const taps = px.slot.taps orelse return error.TapsMissing;
            const streams = ctx_.plus(b.streams[a], @as(usize, px.r0) * 4 * c.hidden * 2);
            g.hcmean(&x.p.glue, e, streams, .{ .buf = taps, .off = t * c.hidden * 2 }, px.rows, c.hidden, T * c.hidden);
        }
    }
    try layer.encode(x, e, w, .{ .x = b.streams[a], .pre = b.pres[a], .x_out = b.streams[1 - a], .pre_out = b.pres[1 - a] });
}

/// Engram's step on bf16 rows (the values engram's embed makes): act_quant first, then the projection; adds in place.
fn engram(x: *const Ctx, e: mtl.ComputeEncoder, b: *const Bufs, w: weights.Engram, li: usize, streams: Ref) !void {
    const p = &x.p.engram;
    const call = engramCall(x.c, x.rows);
    if (w.wkv.k != call.cols * eng.head or w.wkv.n != call.n) return error.BadEngramShape;
    try eng.encodeAq(p, e, cv(eng.Ref, b.rows[li]), cv(eng.Ref, b.emb), call);
    e.barrier();
    try eng.encodeWkv(p, e, cv(eng.Ref, b.emb), cv(eng.Ref, w.wkv.w), cv(eng.Ref, w.wkv.s), cv(eng.Ref, b.roots), cv(eng.Ref, b.ekv), call);
    e.barrier();
    try eng.encodeGate(p, e, cv(eng.Ref, streams), cv(eng.Ref, b.ekv), cv(eng.Ref, w.q), cv(eng.Ref, w.k), cv(eng.Ref, b.mask), cv(eng.Ref, b.gate), call);
    e.barrier();
    try eng.encodeAdd(p, e, cv(eng.Ref, streams), cv(eng.Ref, b.ekv), cv(eng.Ref, b.gate), cv(eng.Ref, streams), call);
    e.barrier();
}

/// The streams after the last layer collapsed by its pre mix, the final norm, the head: logits [rows][vocab] fp32.
pub fn encodeHead(x: *const Ctx, e: mtl.ComputeEncoder, m: Model, b: *const Bufs) void {
    const c = x.c;
    const gp = &x.p.glue;
    const last = m.layers.len % 2;
    cl.hcpre(x.p, e, b.streams[last], b.pres[last], b.hf, x.rows, c.hidden);
    e.barrier();
    g.rmsnorm(gp, e, b.hf, m.top.norm, b.hfn, x.rows, c.hidden, c.eps);
    e.barrier();
    g.mv(gp, e, b.hfn, m.top.head, b.logits, x.rows, c.hidden, c.vocab, true);
}

/// The whole decode forward in one encoder (serial or concurrent), after `prepare` and the scratch's row records.
pub fn encode(x: *const Ctx, e: mtl.ComputeEncoder, m: Model, b: *const Bufs) !void {
    if (m.layers.len != x.c.layers) return error.LayerOrder;
    if (x.rows == 0 or x.rows > R or x.rows != x.total()) return error.BadRows;
    encodeEmbed(x, e, m, b);
    e.barrier();
    for (0..m.layers.len) |i| try encodeLayer(x, e, m, b, i);
    encodeHead(x, e, m, b);
}

/// Command buffers a round of `run` is cut into: the GPU runs one while the host encodes the next.
pub const buffers = 5;
/// Layers in a round's first command buffer, so the GPU starts after encoding only these.
pub const first_layers = 1;

/// The end of the command buffer starting at layer `lo` of `n`: `first_layers`, then the rest in even parts.
pub fn chunkEnd(lo: usize, n: usize) usize {
    if (lo == 0) return @min(first_layers, n);
    const per = (n - first_layers + buffers - 2) / (buffers - 1);
    return @min(lo + @max(per, 1), n);
}

/// The decode forward in about `buffers` command buffers, each committed once encoded; waits for the last.
pub fn run(x: *const Ctx, q: mtl.Queue, m: Model, b: *const Bufs) !void {
    if (m.layers.len != x.c.layers) return error.LayerOrder;
    if (x.rows == 0 or x.rows > R or x.rows != x.total()) return error.BadRows;
    var cbs: [max_buffers]mtl.CommandBuffer = undefined;
    var n: usize = 0;
    defer for (cbs[0..n]) |cb| cb.release();
    var lo: usize = 0;
    while (true) {
        const hi = chunkEnd(lo, m.layers.len);
        if (n == max_buffers) return error.TooManyBuffers;
        const cb = q.commandBuffer();
        const e = cb.compute(.concurrent);
        const ok = encodeRange(x, e, m, b, lo, hi);
        e.end();
        ok catch |err| {
            if (n > 0) cbs[n - 1].wait();
            return err;
        };
        cb.commit();
        cbs[n] = cb.retain();
        n += 1;
        if (hi == m.layers.len) break;
        lo = hi;
    }
    cbs[n - 1].wait();
    for (cbs[0..n]) |cb| if (cb.failure()) |msg| {
        std.debug.print("dsv41 metal window failed: {s}\n", .{msg});
        return error.CommandBufferFailed;
    };
}

const max_buffers = 16;

/// Layers lo .. hi of the forward, the embedding before layer 0 and the head after the last.
fn encodeRange(x: *const Ctx, e: mtl.ComputeEncoder, m: Model, b: *const Bufs, lo: usize, hi: usize) !void {
    if (lo == 0) {
        encodeEmbed(x, e, m, b);
        e.barrier();
    }
    for (lo..hi) |i| try encodeLayer(x, e, m, b, i);
    if (hi == m.layers.len) encodeHead(x, e, m, b);
}

test "a round's command buffers: one layer, then four even parts of the rest" {
    var lo: usize = 0;
    var ends: [8]usize = undefined;
    var n: usize = 0;
    while (lo < 43) : (n += 1) {
        lo = chunkEnd(lo, 43);
        ends[n] = lo;
    }
    try std.testing.expectEqualSlices(usize, &.{ 1, 12, 23, 34, 43 }, ends[0..n]);
    try std.testing.expectEqual(@as(usize, 1), chunkEnd(0, 1));
}

test "an Engram call of the release: 24 hash columns, a key per stream and the value" {
    const c: Config = .{};
    const call = engramCall(&c, 3);
    try std.testing.expectEqual(@as(u32, 24), call.cols);
    try std.testing.expectEqual(@as(u32, 25600), call.n);
}
