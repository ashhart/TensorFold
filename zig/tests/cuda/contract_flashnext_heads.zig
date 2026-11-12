//! Flash Next's DeltaNet twins on one rank's heads: each kernel's bytes equal its full original's on those heads.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const rig_ = @import("contract_rig.zig");
const lay = @import("flashnext_gdn_heads.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const parts = flash.batch_state;
const Heads = lay.Heads;
const Part = lay.Part;
pub const Kind = lay.Kind;
const dk = lay.dk;
const dv = lay.dv;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// The prompt pass's rows on one rank's heads (n: its value heads x dv): the fused seam, the default and the most.
const half_served = [_]c.Shape{
    .{ .rows = 64, .n = Heads.half.nv * dv, .k = dk },
    .{ .rows = parts.default_chunk, .n = Heads.half.nv * dv, .k = dk },
    .{ .rows = parts.max_chunk, .n = Heads.half.nv * dv, .k = dk },
};

const ranks = 2;

const names = blk: {
    var n: [std.enums.values(Kind).len][ranks][]const u8 = undefined;
    for (std.enums.values(Kind)) |k| for (0..ranks) |r| {
        n[@backingInt(k)][r] = std.fmt.comptimePrint("cuda flashnext DeltaNet {s} on rank {d}'s 8 key and 24 value heads", .{ lay.symbol(k, true), r });
    };
    break :blk n;
};

const originals = blk: {
    var n: [std.enums.values(Kind).len][]const u8 = undefined;
    for (std.enums.values(Kind)) |k| n[@backingInt(k)] = std.fmt.comptimePrint("{s} on the same heads of the full layout", .{lay.symbol(k, false)});
    break :blk n;
};

/// rows = token rows; n = the candidate's value heads x dv; k = dk.
pub fn spec(kind: Kind, rank: usize) c.Spec {
    return .{ .name = names[@backingInt(kind)][rank], .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "every output byte, the state's included, equal to the full kernel's on these heads, non-finite inputs included" }, .replaces = originals[@backingInt(kind)], .tiles = .{ .{ .size = 16 }, .{}, .{} }, .in_place = kind == .rows or kind == .fused, .served = &half_served };
}

/// A buffer the kernel reads: what its rows hold (null: not cut, the norm weight), its element bytes and its values.
const In = struct { part: ?Part, elem: usize = 4, lo: f32, hi: f32, decay: bool = false };
/// A buffer the kernel writes: what its rows hold, its element bytes, and whether it is the state read in place.
const Out = struct { part: Part, elem: usize = 4, state: bool = false };

const proj: In = .{ .part = .proj, .elem = 2, .lo = -1, .hi = 1 };
const conv: In = .{ .part = .conv, .elem = 2, .lo = -1, .hi = 1 };
const cw: In = .{ .part = .taps, .elem = 2, .lo = -0.5, .hi = 0.5 };
const gates: In = .{ .part = .gates, .lo = -1, .hi = 1 };
const norm_w: In = .{ .part = null, .elem = 2, .lo = 0.5, .hi = 1.5 };
const keys: In = .{ .part = .keys, .lo = -0.1, .hi = 0.1 };
const chain_in = [_]In{ keys, keys, .{ .part = .values, .lo = -1, .hi = 1 }, .{ .part = .gb, .lo = 0, .hi = 1, .decay = true } };
const state: In = .{ .part = .state, .lo = -0.5, .hi = 0.5 };

fn inputs(kind: Kind) []const In {
    return switch (kind) {
        .front => &.{ proj, conv, cw, gates, gates },
        .norm => &.{ .{ .part = .values, .lo = -1, .hi = 1 }, proj, norm_w },
        .rows => &chain_in,
        .rows_to, .tpr => &(chain_in ++ [_]In{state}),
        .fused => &.{ proj, conv, cw, gates, gates, norm_w },
    };
}

fn outputs(kind: Kind) []const Out {
    return switch (kind) {
        .front => &.{ .{ .part = .keys }, .{ .part = .keys }, .{ .part = .values }, .{ .part = .gb } },
        .norm => &.{ .{ .part = .values, .elem = 2 }, .{ .part = .sums } },
        .rows => &.{ .{ .part = .values }, .{ .part = .state, .state = true } },
        .rows_to, .tpr => &.{ .{ .part = .values }, .{ .part = .state } },
        .fused => &.{ .{ .part = .values, .elem = 2 }, .{ .part = .sums }, .{ .part = .state, .state = true }, .{ .part = .values } },
    };
}

/// Bytes of a buffer in the layout of heads h: the norm weight (no part) is dv values whatever the heads.
fn bytesOf(part: ?Part, elem: usize, m: usize, h: Heads) usize {
    const p = part orelse return dv * elem;
    return lay.bytes(p, elem, m, h);
}

fn put(host: []u8, in: In, at: usize, v: f32) void {
    if (in.elem == 2) std.mem.bytesAsSlice(u16, host)[at] = c.toBf16(v) else std.mem.bytesAsSlice(f32, host)[at] = v;
}

pub const Rank = struct {
    rig: rig_.Rig,
    kind: Kind,
    /// The candidate's heads: one rank's (its twins), cut from the full layout.
    heads: Heads = .half,
    rank: usize,
    f: *const lay.Fns,
    rows: usize = 0,
    /// The full-layout inputs and, by the same slot, the candidate's: rank r's heads gathered from them.
    wide: [6]u64 = @splat(0),
    ins: [6]u64 = @splat(0),
    outs: [4]u64 = @splat(0),
    nout: usize = 0,
    /// The full outputs the original writes, back to back, poisoned before each baseline.
    base: u64 = 0,
    /// The candidate's state read and written in place (0: none), and the starting bytes of it and of its full twin.
    state: u64 = 0,
    fresh: ?[]u8 = null,
    wide_fresh: ?[]u8 = null,
    want: std.ArrayList(u8) = .empty,
    seen: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Rank) void {
        self.drop();
        self.want.deinit(self.rig.gpu.gpa);
        self.seen.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    fn drop(self: *Rank) void {
        if (self.fresh) |f| self.rig.gpu.gpa.free(f);
        if (self.wide_fresh) |f| self.rig.gpu.gpa.free(f);
        self.fresh = null;
        self.wide_fresh = null;
    }

    pub fn kernel(self: *Rank) g.Kernel {
        return .{ .spec = spec(self.kind, self.rank), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Rank);

    /// Full-layout values of a buffer, with gate 1's plants at known places of rank r's heads.
    fn values(self: *Rank, in: In, r: std.Random, bad: bool) ![]u8 {
        const m = self.rows;
        const host = try self.rig.gpu.gpa.alloc(u8, bytesOf(in.part, in.elem, m, .full));
        const n = host.len / in.elem;
        for (0..n) |i| {
            var v = in.lo + r.float(f32) * (in.hi - in.lo);
            if (in.decay and i % 2 == 0) v = -v; // as the full fixture: a log decay, then beta in (0, 1)
            put(host, in, i, v);
        }
        const part = in.part orelse return host;
        if (!bad) return host;
        const h = self.heads;
        switch (part) {
            .proj => {
                self.plant(host, in, 0, 3, std.math.nan(f32)); // q of the rank's first key head
                self.plant(host, in, m / 2, 2 * h.nk * dk + 7, std.math.inf(f32)); // v
                self.plant(host, in, m - 1, h.convDim() + 5, -std.math.inf(f32)); // z
            },
            .keys => self.plant(host, in, 0, 3, std.math.nan(f32)),
            .values => self.plant(host, in, m / 2, 5, std.math.inf(f32)),
            .state => self.plant(host, in, 0, dv * dk + 7, -std.math.inf(f32)),
            else => {},
        }
        return host;
    }

    /// Element i of row `row` of the rank's local layout, placed in the full-layout buffer.
    fn plant(self: *Rank, host: []u8, in: In, row: usize, i: usize, v: f32) void {
        const part = in.part.?;
        put(host, in, row * lay.width(part, .full) + lay.fullIndex(part, self.heads, self.rank, i), v);
    }

    fn local(self: *Rank, host: []const u8, part: ?Part, elem: usize) ![]u8 {
        const a = self.rig.gpu.gpa;
        const p = part orelse return a.dupe(u8, host);
        return lay.gather(a, host, elem, p, self.heads, self.rank);
    }

    pub fn prepare(self: *Rank, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        self.drop();
        self.state = 0;
        self.nout = 0;
        self.rows = shape.rows;
        if (shape.rows == 0 or shape.n != self.heads.valueDim() or shape.k != dk) return false;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6865616473);
        const r = prng.random();
        const a = self.rig.gpu.gpa;
        const bad = fill == .nonfinite;
        for (inputs(self.kind), 0..) |in, i| {
            const host = try self.values(in, r, bad);
            defer a.free(host);
            self.wide[i] = try self.rig.input(host);
            const mine = try self.local(host, in.part, in.elem);
            defer a.free(mine);
            self.ins[i] = try self.rig.input(mine);
        }
        var total: usize = 0;
        for (outputs(self.kind), 0..) |o, i| {
            self.outs[i] = try self.rig.output(bytesOf(o.part, o.elem, self.rows, self.heads));
            self.nout += 1;
            total += bytesOf(o.part, o.elem, self.rows, .full);
            if (!o.state) continue;
            // the state read and written where it lies: every launch of either kernel starts from these bytes
            self.state = self.outs[i];
            self.wide_fresh = try self.values(state, r, bad);
            self.fresh = try self.local(self.wide_fresh.?, .state, 4);
        }
        self.base = try self.rig.scratch(total);
        return true;
    }

    /// The kernel of this kind built for heads h, on inputs `in` into outputs `o`, as the pass binds it.
    fn launch(self: *Rank, h: Heads, in: []const u64, o: []const u64) !void {
        const f = self.f.of(self.kind, h);
        const s = self.rig.s;
        const m = self.rows;
        switch (self.kind) {
            .front => try lay.front(f, s, h, m, in[0], in[1], in[2], in[3], in[4], o[0], o[1], o[2], o[3]),
            .norm => try lay.norm(f, s, h, m, in[0], in[1], in[2], o[0], o[1]),
            .rows => try lay.chain(f, s, .rows, h, m, in[0], in[1], in[2], in[3], o[1], o[1], o[0]),
            .rows_to, .tpr => try lay.chain(f, s, self.kind, h, m, in[0], in[1], in[2], in[3], in[4], o[1], o[0]),
            .fused => try lay.fused(f, s, h, m, in[0], in[1], in[2], in[3], in[4], o[2], o[2], in[5], o[0], o[1], o[3]),
        }
    }

    fn upload(self: *Rank, ptr: u64, bytes: []const u8) !void {
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = ptr, .len = bytes.len };
        try b.upload(0, bytes);
    }

    pub fn run(self: *Rank) !void {
        if (self.fresh) |f| try self.upload(self.state, f);
        try self.launch(self.heads, &self.ins, self.outs[0..self.nout]);
        try self.rig.s.synchronize();
    }

    /// The full original on the full inputs into its own outputs, rank r's heads gathered in the candidate's order.
    pub fn replaced(self: *Rank) ![]const u8 {
        const a = self.rig.gpu.gpa;
        if (self.fresh) |f| try self.upload(self.state, f);
        const list = outputs(self.kind);
        var o: [4]u64 = undefined;
        var at = self.base;
        for (list, 0..) |x, i| {
            o[i] = at;
            at += bytesOf(x.part, x.elem, self.rows, .full);
        }
        const ff = try a.alloc(u8, at - self.base);
        defer a.free(ff);
        @memset(ff, 0xff);
        try self.upload(self.base, ff);
        for (list, 0..) |x, i| if (x.state) try self.upload(o[i], self.wide_fresh.?);
        try self.launch(.full, &self.wide, o[0..list.len]);
        try self.rig.s.synchronize();
        self.want.clearRetainingCapacity();
        for (list, 0..) |x, i| {
            const got = try self.rig.bytes(o[i], bytesOf(x.part, x.elem, self.rows, .full));
            const mine = try lay.gather(a, got, x.elem, x.part, self.heads, self.rank);
            defer a.free(mine);
            try self.want.appendSlice(a, mine);
        }
        return self.want.items;
    }

    /// Gate 1's stated rule: the outputs on the planted inputs equal the original's on the same heads.
    pub fn check(self: *Rank, gate: g.Gate) !?[]const u8 {
        _ = gate;
        self.seen.clearRetainingCapacity();
        try self.seen.appendSlice(self.rig.gpu.gpa, try self.rig.outputBytes());
        const want = try self.replaced();
        if (std.mem.eql(u8, self.seen.items, want)) return null;
        return try std.fmt.bufPrint(&self.msg, "bytes differ from the full kernel's from byte {d}", .{std.mem.indexOfDiff(u8, self.seen.items, want).?});
    }

    pub fn count(self: *Rank) usize {
        var total: usize = 0;
        for (outputs(self.kind)) |x| total += bytesOf(x.part, x.elem, self.rows, self.heads);
        return total;
    }

    pub fn reference(_: *Rank, _: []f64, _: []f64) !void {
        return error.NoFloat64Reference;
    }
};
