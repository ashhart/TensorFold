//! The slot kernels' contract inputs: a launch's rows split over slots, each slot's state in buffers of its own.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_glue_ref.zig");

const c = contract.spec;
const dm = tf.dsv41_metal;
const cp = dm.compress;
const fu = dm.fused;
const at = dm.attn;
const D = 512;

/// Slots a launch holds at most here, and its most rows (the decode tier and one past its seam).
pub const most = 4;
pub const rows_max = 17;
/// Rows a slot's kept compressor inputs hold, so a 17-row slot fits.
pub const keep = 32;
/// Window slots of a ring: the attention window plus a launch's rows, mirrored.
const nw = 144;
/// Compressed rows a slot holds, positions a rope table covers, and a slot's publish rows.
const crows = 64;
const npos = 384;
const prows = 20;

/// Indices into `ins`: the shared row records and tables.
pub const pos = 2;
pub const dest = 3;
pub const iota = 4;
pub const rowseg = 5;
const cs = 6;
const sk = 7;
const win0 = 8;

pub const Kind = enum { kv, pool, rt, pack, attn, attn_pf };
pub const Elem = enum { bf16, f32, raw };
pub const Out = struct { ref: rig_.Ref, len: usize, init: []const u8, elem: Elem };

/// The pipelines a probe launches: the slot kernels and the per-slot ones, from one TF_CTL.
pub const Pipes = struct {
    fused: fu.Pipelines,
    compress: cp.Pipelines,
    attn: at.Pipelines,
    glue: dm.glue_tmp.Pipelines,

    pub fn load(a: std.mem.Allocator, device: mtl.Device, ctl: u8) !Pipes {
        var p: Pipes = undefined;
        const fuse = try text(a, ctl, sources.dsv41.fused);
        defer a.free(fuse);
        p.fused = try fu.Pipelines.loadSource(device, fuse);
        errdefer p.fused.deinit();
        const comp = try text(a, ctl, sources.dsv41.compress);
        defer a.free(comp);
        p.compress = try cp.Pipelines.loadSource(device, comp);
        errdefer p.compress.deinit();
        const att = try text(a, ctl, sources.dsv41.csa2_attn);
        defer a.free(att);
        p.attn = try at.Pipelines.loadSource(device, att);
        errdefer p.attn.deinit();
        p.glue = try dm.glue_tmp.Pipelines.load(device, sources.dsv41.glue_tmp);
        return p;
    }

    fn text(a: std.mem.Allocator, ctl: u8, src: []const u8) ![]u8 {
        if (ctl == 0) return a.dupe(u8, src);
        return std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ ctl, src });
    }

    pub fn deinit(p: *Pipes) void {
        p.fused.deinit();
        p.compress.deinit();
        p.attn.deinit();
        p.glue.deinit();
    }
};

/// The kernels' TfDsv41Seg.
const Seg = extern struct { window: u64 = 0, pool: u64 = 0, keys: u64 = 0, carry: u64 = 0, inputs: u64 = 0, comp: u64 = 0, ikeys: u64 = 0, base: u32, rows: u32, nkeys: u32 = 0, start: u32 = 0 };

fn addr(r: rig_.Ref) u64 {
    return r.buf.gpuAddress() + r.off;
}

fn bf(rng: std.Random, scale: f32) u16 {
    return c.toBf16((rng.float(f32) * 2 - 1) * scale);
}

pub const Lay = struct {
    a: std.mem.Allocator,
    kind: Kind,
    rows: usize,
    slots: usize,
    heads: usize = 0,
    size: [most]usize = undefined,
    base: [most]usize = undefined,
    ins: std.ArrayList(rig_.Ref) = .empty,
    outs: std.ArrayList(Out) = .empty,
    plants: usize = 0,

    /// The rows split unevenly over `k` slots, each at its own positions; inputs, state and outputs placed.
    pub fn make(a: std.mem.Allocator, rig: *rig_.Rig, kind: Kind, shape: c.Shape, nf: bool, seed: u64) !Lay {
        var L: Lay = .{ .a = a, .kind = kind, .rows = shape.rows, .slots = shape.k, .heads = shape.n };
        for (0..L.slots) |j| L.size[j] = L.rows / L.slots + @intFromBool(j < L.rows % L.slots);
        if (L.slots > 1 and L.size[1] > 1) {
            L.size[1] -= 1;
            L.size[0] += 1;
        }
        var b: usize = 0;
        for (0..L.slots) |j| {
            L.base[j] = b;
            b += L.size[j];
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = L.rows;
        const p = try a.alloc(i32, n);
        const local = try a.alloc(i32, n);
        const slot = try a.alloc(i32, n);
        for (0..L.slots) |j| for (0..L.size[j]) |i| {
            p[L.base[j] + i] = @intCast(200 + 37 * j + i);
            local[L.base[j] + i] = @intCast(i);
            slot[L.base[j] + i] = @intCast(j);
        };
        const width: usize = if (kind == .pack) 128 else D;
        const main: []u8 = switch (kind) {
            .kv, .rt, .pack => blk: {
                const x = try a.alloc(u16, n * width);
                if (kind == .kv) {
                    for (x) |*v| v.* = bf(rng, 2);
                    if (nf and n > 2) {
                        x[1 * D + 5] = c.toBf16(std.math.nan(f32));
                        x[2 * D + 300] = c.toBf16(std.math.inf(f32));
                        L.plants = 2;
                    }
                } else {
                    ref.fillQuant(if (kind == .rt) .fp4_e4m3 else .fp4_e8m0, x, if (kind == .rt) 16 else 32, seed);
                    if (nf) L.plants = ref.plant(x, n, width, 0, width);
                }
                break :blk std.mem.sliceAsBytes(x);
            },
            .pool => blk: {
                const x = try a.alloc(f32, n * D);
                for (x) |*v| v.* = (rng.float(f32) * 2 - 1) * 4;
                if (nf and n > 1) {
                    x[1 * D + 3] = std.math.nan(f32);
                    L.plants = 1;
                }
                break :blk std.mem.sliceAsBytes(x);
            },
            .attn, .attn_pf => blk: {
                const x = try a.alloc(u16, n * L.heads * D);
                for (x) |*v| v.* = bf(rng, 1);
                if (nf and n > 1) {
                    for (0..16) |i| x[1 * L.heads * D + i] = c.toBf16(std.math.nan(f32));
                    L.plants = 16;
                }
                break :blk std.mem.sliceAsBytes(x);
            },
        };
        try L.ins.append(a, try rig.input(main));
        const second: []u8 = switch (kind) {
            .kv => blk: {
                const w = try a.alloc(f32, D);
                for (w) |*v| v.* = 0.5 + rng.float(f32);
                break :blk std.mem.sliceAsBytes(w);
            },
            .pool => blk: {
                const s = try a.alloc(f32, n * D);
                for (s) |*v| v.* = (rng.float(f32) * 2 - 1) * 3;
                if (nf and n > 2) {
                    s[2 * D + 4] = std.math.inf(f32);
                    L.plants += 1;
                }
                break :blk std.mem.sliceAsBytes(s);
            },
            .attn, .attn_pf => blk: {
                const ix = try a.alloc(i32, n * 512);
                for (ix, 0..) |*v, i| v.* = if (i % 7 == 3) -1 else @intCast(rng.uintLessThan(u32, crows));
                break :blk std.mem.sliceAsBytes(ix);
            },
            else => std.mem.sliceAsBytes(try a.alloc(u32, 1)),
        };
        try L.ins.append(a, try rig.input(second));
        try L.ins.append(a, try rig.input(std.mem.sliceAsBytes(p)));
        try L.ins.append(a, try rig.input(try L.records(p)));
        try L.ins.append(a, try rig.input(std.mem.sliceAsBytes(local)));
        try L.ins.append(a, try rig.input(std.mem.sliceAsBytes(slot)));
        const tab = try a.alloc(f32, npos * 64);
        for (tab) |*v| v.* = rng.float(f32) * 2 - 1;
        try L.ins.append(a, try rig.input(std.mem.sliceAsBytes(tab)));
        const sinks = try a.alloc(f32, @max(L.heads, 1));
        for (sinks) |*v| v.* = rng.float(f32) * 2 - 1;
        try L.ins.append(a, try rig.input(std.mem.sliceAsBytes(sinks)));
        if (kind == .attn or kind == .attn_pf) {
            for (0..2 * L.slots) |i| {
                const keys = try a.alloc(u16, @as(usize, if (i < L.slots) 2 * nw else crows) * D);
                for (keys) |*v| v.* = bf(rng, 1);
                // a key three positions before row 0's: window row wofs + (t - lo)
                if (nf and i == 0) keys[@as(usize, @intCast(@mod(@max(p[0] - 127, 0), nw) + 124)) * D + 9] = c.toBf16(std.math.inf(f32));
                try L.ins.append(a, try rig.input(std.mem.sliceAsBytes(keys)));
            }
        }
        try L.places(rig, rng);
        return L;
    }

    /// kv: window rows [rows][2]; rt and pack: publish rows (every fourth skipped); attn: TfAttnMeta [rows].
    fn records(L: *const Lay, p: []const i32) ![]u8 {
        const n = L.rows;
        switch (L.kind) {
            .kv => {
                const d = try L.a.alloc(i32, 2 * n);
                for (p, 0..) |v, r| {
                    d[2 * r] = @mod(v, nw);
                    d[2 * r + 1] = @mod(v, nw) + nw;
                }
                return std.mem.sliceAsBytes(d);
            },
            .attn, .attn_pf => {
                const m = try L.a.alloc(i32, 8 * n);
                @memset(m, 0);
                for (p, 0..) |v, r| {
                    const lo = @max(v - 127, 0);
                    m[8 * r ..][0..5].* = .{ v, @mod(lo, nw), lo, 0, 512 };
                }
                return std.mem.sliceAsBytes(m);
            },
            else => {
                const d = try L.a.alloc(i32, n);
                for (0..L.slots) |j| for (0..L.size[j]) |i| {
                    d[L.base[j] + i] = if (i % 4 == 3) -1 else @intCast(i);
                };
                return std.mem.sliceAsBytes(d);
            },
        }
    }

    fn out(L: *Lay, rig: *rig_.Rig, len: usize, init: []const u8, elem: Elem) !void {
        try L.outs.append(L.a, .{ .ref = try rig.output(len), .len = len, .init = init, .elem = elem });
    }

    fn fill(L: *const Lay, len: usize, v: u8) ![]u8 {
        const b = try L.a.alloc(u8, len);
        @memset(b, v);
        return b;
    }

    /// The outputs in order: shared rows first, then each slot's state; each with the bytes a run starts from.
    fn places(L: *Lay, rig: *rig_.Rig, rng: std.Random) !void {
        const n = L.rows;
        switch (L.kind) {
            .kv => {
                for (0..3) |_| try L.out(rig, n * D * 2, try L.fill(n * D * 2, 0xff), .bf16);
                for (0..L.slots) |_| try L.out(rig, 2 * nw * D * 2, try L.fill(2 * nw * D * 2, 0xff), .bf16);
            },
            .pool => {
                try L.out(rig, n * D * 2, try L.fill(n * D * 2, 0xff), .bf16);
                for (0..L.slots) |_| {
                    const st = try L.a.alloc(f32, 4 * D);
                    for (st, 0..) |*v, i| v.* = if (i < 2 * D) (rng.float(f32) * 2 - 1) * 4 else (rng.float(f32) * 2 - 1) * 3;
                    try L.out(rig, 4 * D * 4, std.mem.sliceAsBytes(st), .f32);
                    try L.out(rig, 2 * keep * D * 4, try L.fill(2 * keep * D * 4, 0xff), .f32);
                }
            },
            .rt => for (0..L.slots) |_| try L.out(rig, prows * D * 2, try L.fill(prows * D * 2, 0xff), .bf16),
            .pack => for (0..L.slots) |_| try L.out(rig, prows * 68, try L.fill(prows * 68, 0), .raw),
            .attn, .attn_pf => try L.out(rig, n * L.heads * D * 2, try L.fill(n * L.heads * D * 2, 0xff), .bf16),
        }
    }

    /// Slot j's state among `outs`.
    pub fn state(L: *const Lay, outs: []const Out, j: usize) [2]rig_.Ref {
        return switch (L.kind) {
            .kv => .{ outs[3 + j].ref, outs[3 + j].ref },
            .pool => .{ outs[1 + 2 * j].ref, outs[2 + 2 * j].ref },
            .rt, .pack => .{ outs[j].ref, outs[j].ref },
            .attn, .attn_pf => .{ outs[0].ref, outs[0].ref },
        };
    }

    /// The launch's slot table over `outs` (and the attention's read-only rings), in a scratch buffer.
    pub fn table(L: *const Lay, rig: *rig_.Rig, outs: []const Out) !rig_.Ref {
        const t = try rig.scratch(most * @sizeOf(Seg));
        const segs = std.mem.bytesAsSlice(Seg, rig.bytes(t, most * @sizeOf(Seg)));
        @memset(rig.bytes(t, most * @sizeOf(Seg)), 0);
        for (0..L.slots) |j| {
            var s: Seg = .{ .base = @intCast(L.base[j]), .rows = @intCast(L.size[j]) };
            const st = L.state(outs, j);
            switch (L.kind) {
                .kv => s.window = addr(st[0]),
                .pool => {
                    s.carry = addr(st[0]);
                    s.inputs = addr(st[1]);
                },
                .rt => s.pool = addr(st[0]),
                .pack => s.keys = addr(st[0]),
                .attn, .attn_pf => {
                    s.window = addr(L.ins.items[win0 + j]);
                    s.comp = addr(L.ins.items[win0 + L.slots + j]);
                },
            }
            segs[j] = s;
        }
        return t;
    }

    /// Copies of the outputs at their starting bytes, for the per-slot launches.
    pub fn twins(L: *const Lay, rig: *rig_.Rig) ![]Out {
        const t = try L.a.alloc(Out, L.outs.items.len);
        for (L.outs.items, t) |o, *w| {
            w.* = o;
            w.ref = try rig.scratch(o.len);
            @memcpy(rig.bytes(w.ref, o.len), o.init);
        }
        return t;
    }

    pub fn gather(L: *const Lay, rig: *rig_.Rig, outs: []const Out) ![]const u8 {
        var b: std.ArrayList(u8) = .empty;
        for (outs) |o| try b.appendSlice(L.a, rig.bytes(o.ref, o.len));
        return b.items;
    }

    pub fn restore(L: *const Lay, rig: *rig_.Rig) void {
        for (L.outs.items) |o| @memcpy(rig.bytes(o.ref, o.len), o.init);
    }

    pub fn outBytes(L: *const Lay) usize {
        var n: usize = 0;
        for (L.outs.items) |o| n += o.len;
        return n;
    }

    /// Elements a run changed to a non-finite value (or, raw, to the NaN byte 255).
    pub fn nonfinite(L: *const Lay, rig: *rig_.Rig) usize {
        var n: usize = 0;
        for (L.outs.items) |o| {
            const b = rig.bytes(o.ref, o.len);
            switch (o.elem) {
                .bf16 => for (std.mem.bytesAsSlice(u16, b), std.mem.bytesAsSlice(u16, o.init)) |v, w| {
                    n += @intFromBool(v != w and (v & 0x7f80) == 0x7f80);
                },
                .f32 => for (std.mem.bytesAsSlice(u32, b), std.mem.bytesAsSlice(u32, o.init)) |v, w| {
                    n += @intFromBool(v != w and (v & 0x7f800000) == 0x7f800000);
                },
                .raw => for (b, o.init) |v, w| {
                    n += @intFromBool(v != w and v == 0xff);
                },
            }
        }
        return n;
    }

    pub fn whole(_: *const Lay, x: rig_.Ref) cp.Ref {
        return .{ .buf = x.buf, .off = x.off };
    }

    pub fn plus(_: *const Lay, x: rig_.Ref, bytes: usize) cp.Ref {
        return .{ .buf = x.buf, .off = x.off + bytes };
    }

    /// The window key launch of slot j (or of every row, when the table supplies the rings).
    pub fn kvBind(L: *const Lay, outs: []const Out, j: usize) fu.Kv {
        const r0 = L.base[j];
        const in = L.ins.items;
        const F = struct {
            fn of(x: rig_.Ref, bytes: usize) fu.Ref {
                return .{ .buf = x.buf, .off = x.off + bytes };
            }
        };
        return .{ .kv = F.of(in[0], r0 * D * 2), .w = F.of(in[1], 0), .kvn = F.of(outs[0].ref, r0 * D * 2), .kvr = F.of(outs[1].ref, r0 * D * 2), .kvq = F.of(outs[2].ref, r0 * D * 2), .ring = F.of(L.state(outs, j)[0], 0), .pos = F.of(in[pos], r0 * 4), .table = F.of(in[cs], 0), .dest = F.of(in[dest], r0 * 8), .reps = 2 };
    }

    /// The decode attention of slot j's rows from row r0 (the table supplies the rings when every row runs).
    pub fn attnBind(L: *const Lay, outs: []const Out, j: usize, r0: usize) at.Bind {
        const in = L.ins.items;
        const H = L.heads;
        const F = struct {
            fn of(x: rig_.Ref, bytes: usize) at.Ref {
                return .{ .buf = x.buf, .off = x.off + bytes };
            }
        };
        return .{ .q = F.of(in[0], r0 * H * D * 2), .w = F.of(in[win0 + j], 0), .c = F.of(in[win0 + L.slots + j], 0), .i = F.of(in[1], r0 * 512 * 4), .m = F.of(in[dest], r0 * 32), .sk = F.of(in[sk], 0), .cs = F.of(in[cs], 0), .o = F.of(outs[0].ref, r0 * H * D * 2) };
    }
};

/// The index keys' packing: 68-byte rows, codes then the scales from byte 64, at the publish rows `d`.
pub fn packArgs(d: cp.Ref) cp.Pack {
    return .{ .blocks = 128 / 32, .in_stride = 128, .code_row = 68, .scale_row = 68, .scale_off = 64, .dest = d };
}
