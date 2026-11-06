//! A prompt's buffers beside the layer scratch: a chunk's ids, streams, Engram stages, head and taps; a slot's tail.
const std = @import("std");
const mtl = @import("metal");
const Config = @import("dsv41").config.Config;
const ctx_ = @import("ctx.zig");
const forward = @import("forward.zig");
const pipelines = @import("pipelines.zig");

const Ref = ctx_.Ref;
const eng = pipelines.engram;

/// A chunk's host-written inputs: ids, Engram gate mask, Engram rows (two sets: the next fills while one runs).
pub const Set = struct { ids: Ref = undefined, mask: Ref = undefined, rows: [4]Ref = undefined };

pub const Bufs = struct {
    buf: mtl.Buffer = undefined,
    /// the set the encoded chunk reads (one of `sets`)
    ids: Ref = undefined,
    /// the ping-pong streams [rows][4][hidden] bf16 and pre mixes [rows][4] fp32
    streams: [2]Ref = undefined,
    pres: [2]Ref = undefined,
    /// each Engram layer's rows [rows][cols * 256] bf16
    rows: [4]Ref = undefined,
    emb: Ref = undefined,
    ekv: Ref = undefined,
    gate: Ref = undefined,
    mask: Ref = undefined,
    sets: [2]Set = .{ .{}, .{} },
    hf: Ref = undefined,
    hfn: Ref = undefined,
    /// the head's rows [head][vocab] fp32 and the taps' [taps][targets][hidden] bf16, the prompt's last rows
    logits: Ref = undefined,
    taps: Ref = undefined,

    pub fn init(device: mtl.Device, c: *const Config, rows: u32, head: u32, taps: u32) !Bufs {
        var b: Bufs = .{};
        const size = b.layout(c, rows, head, taps);
        b.buf = try device.buffer(size, mtl.ResourceOptions.shared);
        @memset(b.buf.contents()[0..size], 0);
        _ = b.layout(c, rows, head, taps);
        // text only: every row's Engram gate is open
        for (b.sets) |st| @memset(b.bytes(st.mask, rows), 1);
        b.use(0);
        return b;
    }

    /// The chunk inputs encoded launches read: set `i`.
    pub fn use(b: *Bufs, i: usize) void {
        b.ids = b.sets[i].ids;
        b.mask = b.sets[i].mask;
        b.rows = b.sets[i].rows;
    }

    pub fn deinit(b: *Bufs) void {
        b.buf.deinit();
    }

    fn layout(b: *Bufs, c: *const Config, rows: u32, head: u32, taps: u32) usize {
        var off: usize = 0;
        const L = struct {
            fn take(bb: *Bufs, o: *usize, len: usize) Ref {
                const at = std.mem.alignForward(usize, o.*, 256);
                o.* = at + @max(len, 4);
                return .{ .buf = bb.buf, .off = at };
            }
        };
        const R: usize = rows;
        const H: usize = c.hidden;
        const call = forward.engramCall(c, rows);
        const K: usize = @as(usize, call.cols) * eng.head;
        for (&b.sets) |*st| {
            st.ids = L.take(b, &off, R * 4);
            st.mask = L.take(b, &off, R);
            for (st.rows[0..c.engram_layers.len]) |*s| s.* = L.take(b, &off, R * K * 2);
        }
        for (&b.streams) |*s| s.* = L.take(b, &off, R * 4 * H * 2);
        for (&b.pres) |*s| s.* = L.take(b, &off, R * 4 * 4);
        b.emb = L.take(b, &off, R * K * 2);
        b.ekv = L.take(b, &off, R * @as(usize, call.n) * 2);
        b.gate = L.take(b, &off, R * 4 * 4);
        b.hf = L.take(b, &off, @as(usize, head) * H * 2);
        b.hfn = L.take(b, &off, @as(usize, head) * H * 2);
        b.logits = L.take(b, &off, @as(usize, head) * c.vocab * 4);
        b.taps = L.take(b, &off, @as(usize, taps) * c.dspark_targets.len * H * 2);
        return off;
    }

    pub fn bytes(b: *const Bufs, r: Ref, n: usize) []u8 {
        return (b.buf.contents() + r.off)[0..n];
    }
};

/// A slot's tail: layer d - 1's output (the streams entering the decoder) for the last `cap` positions, a ring.
pub const Tail = struct {
    /// [cap][4][hidden] bf16 and [cap][4] fp32, position p at row p % cap
    x: mtl.Buffer,
    pre: mtl.Buffer,
    cap: u32,
    /// positions whose encoder layers ran (the next prompt row), and those whose decoder layers ran too
    end: u32 = 0,
    done: u32 = 0,

    pub fn init(device: mtl.Device, c: *const Config, cap: u32) !Tail {
        if (cap == 0) return error.BadRows;
        const xb = @as(usize, cap) * 4 * c.hidden * 2;
        const x = try device.buffer(xb, mtl.ResourceOptions.shared);
        errdefer x.deinit();
        const pre = try device.buffer(@as(usize, cap) * 16, mtl.ResourceOptions.shared);
        @memset(x.contents()[0..xb], 0);
        @memset(pre.contents()[0 .. @as(usize, cap) * 16], 0);
        return .{ .x = x, .pre = pre, .cap = cap };
    }

    pub fn deinit(t: *Tail) void {
        t.x.deinit();
        t.pre.deinit();
    }

    /// The other tail's rows (in a blit on `cb`) and positions; both tails have the same rows.
    pub fn copyFrom(t: *Tail, cb: mtl.CommandBuffer, from: *const Tail) !void {
        if (t.cap != from.cap) return error.BadRows;
        const b = cb.blit();
        b.copy(from.x, 0, t.x, 0, t.x.length());
        b.copy(from.pre, 0, t.pre, 0, t.pre.length());
        b.end();
        t.end = from.end;
        t.done = from.done;
    }

    /// Decode steps ran every layer up to position `n` (a finished prompt's tail moves with them).
    pub fn advance(t: *Tail, n: u32) !void {
        if (t.end != t.done or n < t.end) return error.Unfinished;
        t.end = n;
        t.done = n;
    }
};
