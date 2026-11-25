//! Hy4-flash paged MLA attention contract: sliding windows, ring caches, causal spans, and index lists.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const hy4 = @import("hy4_mla_paged_cuda");
const rig_ = @import("contract_rig.zig");
const set = @import("contract_hy4_mla_paged.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const sumIn = contract.rows.sumIn;
const lat = hy4.latent;
const rope = hy4.rope;
const row = hy4.row;
const chunk = 32; // tf_hy4::kKC
const window = 512;
const ring = 1024;
const u: f64 = 0x1p-24;

/// Keys by a row's window (linear or 1,024-slot ring), causal span or list; `rows`: spans, a cache a row.
pub const Form = enum { window, ring, dense, list, rows };

/// rows x a row's outputs (heads x 512) x the keys the last row sees (a list's entries).
fn depth(s: c.Shape) usize {
    return row + s.k + s.k / chunk + 8;
}

const window_shapes = [_]c.Shape{ .{ .rows = 16, .n = 16 * lat, .k = window }, .{ .rows = 1, .n = 64 * lat, .k = window }, .{ .rows = 5, .n = 32 * lat, .k = 300 } };
const ring_shapes = [_]c.Shape{ .{ .rows = 16, .n = 32 * lat, .k = window }, .{ .rows = 3, .n = 64 * lat, .k = window } };
const dense_shapes = [_]c.Shape{ .{ .rows = 8, .n = 16 * lat, .k = 2048 }, .{ .rows = 1, .n = 32 * lat, .k = 1500 }, .{ .rows = 33, .n = 8 * lat, .k = 700 } };
const rows_shapes = [_]c.Shape{ .{ .rows = 8, .n = 16 * lat, .k = 1024 }, .{ .rows = 3, .n = 32 * lat, .k = 700 }, .{ .rows = 16, .n = 8 * lat, .k = 300 } };
const list_shapes = [_]c.Shape{ .{ .rows = 4, .n = 32 * lat, .k = 2048 }, .{ .rows = 1, .n = 16 * lat, .k = 2048 }, .{ .rows = 9, .n = 8 * lat, .k = 2048 } };

pub fn spec(form: Form, fault: set.Fault) c.Spec {
    const keys: c.Tile = .{ .size = chunk };
    var s: c.Spec = switch (form) {
        .window => .{ .name = "cuda hy4_mla_paged tf_hy4_mla_attn (sliding window 512, sinks)", .accum = .f32, .depth = depth, .ops = 8, .out = .f32, .tiles = .{ .{}, .{}, keys }, .served = &window_shapes },
        .ring => .{ .name = "cuda hy4_mla_paged tf_hy4_mla_attn (sliding window 512 over a 1,024-slot ring, sinks)", .accum = .f32, .depth = depth, .ops = 8, .out = .f32, .served = &ring_shapes },
        .dense => .{ .name = "cuda hy4_mla_paged tf_hy4_mla_attn (causal span to 2,048 keys, sinks)", .accum = .f32, .depth = depth, .ops = 8, .out = .f32, .tiles = .{ .{}, .{}, keys }, .served = &dense_shapes },
        .list => .{ .name = "cuda hy4_mla_paged tf_hy4_mla_attn (the indexer's list of 2,048 keys, sinks)", .accum = .f32, .depth = depth, .ops = 8, .out = .f32, .tiles = .{ .{}, .{}, keys }, .served = &list_shapes },
        .rows => .{ .name = "cuda hy4_mla_paged tf_hy4_mla_attn_rows (causal spans, a cache a row, sinks)", .accum = .f32, .depth = depth, .ops = 8, .out = .f32, .tiles = .{ .{}, .{}, keys }, .served = &rows_shapes },
    };
    if (fault != .none) s.name = set.controlName(if (form == .rows) "attn_rows" else "attn", fault);
    return s;
}

pub const Attn = struct {
    rig: rig_.Rig,
    form: Form,
    f: cuda.Function,
    fault: set.Fault = .none,
    fill: g.Fill = .random,
    shape: c.Shape = undefined,
    heads: usize = 0,
    slots: usize = 0,
    nidx: usize = 0,
    ql: []f32 = &.{}, // [rows][heads][512]
    qr: []u16 = &.{}, // [rows][heads][64]
    cache: []u16 = &.{}, // [slots][576]
    pos: []i32 = &.{},
    idx: []i32 = &.{}, // [rows][nidx]
    sinks: []f32 = &.{},
    tol: []f64 = &.{},
    in: [6]u64 = @splat(0),
    caches: u64 = 0, // rows: device u64 [rows], row r's cache (in[2] or the second cache past it)
    out: u64 = 0,

    pub fn deinit(self: *Attn) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Attn) void {
        const a = self.rig.gpu.gpa;
        a.free(self.ql);
        a.free(self.qr);
        a.free(self.cache);
        a.free(self.pos);
        a.free(self.idx);
        a.free(self.sinks);
        a.free(self.tol);
        self.ql = &.{};
        self.qr = &.{};
        self.cache = &.{};
        self.pos = &.{};
        self.idx = &.{};
        self.sinks = &.{};
        self.tol = &.{};
    }

    pub fn kernel(self: *Attn) g.Kernel {
        return .{ .spec = spec(self.form, self.fault), .ctx = self, .vt = &vt, .control = self.fault.gate() };
    }

    const vt = g.hooks(Attn);

    pub fn count(self: *const Attn) usize {
        return self.shape.rows * self.heads * lat;
    }

    fn windowed(self: *const Attn) bool {
        return self.form == .window or self.form == .ring;
    }

    /// The last row's position: a full window starts past position 0, a short one at it; a list's rows lie far in.
    fn lastPos(self: *const Attn, k: usize) usize {
        return switch (self.form) {
            .window => if (k == window) 700 else k - 1,
            .ring => 1500,
            .dense, .rows => k - 1,
            .list => 3000,
        };
    }

    /// The positions row r sees, in the kernel's walk order, into `out` (its length returned).
    fn seen(self: *const Attn, r: usize, out: []usize) usize {
        const t: usize = @intCast(self.pos[r]);
        var n: usize = 0;
        if (self.form == .list) {
            for (self.idx[r * self.nidx ..][0..self.nidx]) |p| if (p >= 0 and p <= t) {
                out[n] = @intCast(p);
                n += 1;
            };
            return n;
        }
        const lo = if (self.windowed() and t + 1 > window) t + 1 - window else 0;
        for (lo..t + 1) |p| {
            out[n] = p;
            n += 1;
        }
        return n;
    }

    fn slot(self: *const Attn, p: usize) usize {
        return if (self.form == .ring) p % ring else p;
    }

    /// Row r's cache row for position p: the second cache past the first for a `rows` form's odd rows.
    fn cacheRow(self: *const Attn, r: usize, p: usize) usize {
        return (if (self.form == .rows) (r % 2) * self.slots else 0) + self.slot(p);
    }

    pub fn prepare(self: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0 or shape.n % (hy4.heads_tile * lat) != 0) return false;
        if (self.windowed() and shape.k > window) return false;
        const last = self.lastPos(shape.k);
        if (last + 1 < shape.rows or (self.form != .list and last + 1 < shape.k)) return false;
        self.free();
        self.rig.reset();
        self.fill = fill;
        self.shape = shape;
        self.heads = shape.n / lat;
        self.nidx = if (self.form == .list) shape.k else 0;
        self.slots = if (self.form == .ring) ring else last + 1;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        self.pos = try a.alloc(i32, rows);
        for (self.pos, 0..) |*p, r| p.* = @intCast(if (self.form == .rows) last - (r * 37) % @min(last + 1, 64) else last + 1 - rows + r);
        self.ql = try a.alloc(f32, rows * self.heads * lat);
        self.qr = try a.alloc(u16, rows * self.heads * rope);
        self.cache = try a.alloc(u16, (if (self.form == .rows) 2 * self.slots else self.slots) * row);
        self.sinks = try a.alloc(f32, self.heads);
        self.idx = try a.alloc(i32, rows * self.nidx);
        for (self.ql) |*v| v.* = (rng.float(f32) * 2 - 1) * 4;
        for (self.qr) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 4);
        for (self.cache) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (self.sinks) |*v| v.* = (rng.float(f32) * 2 - 1) * 2;
        if (self.form == .list) for (0..rows) |r| try self.choose(rng, r, fill != .stress);
        switch (fill) {
            .random => {},
            .nonfinite => self.plant(),
            .stress => try self.stress(rng),
        }
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.ql));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.qr));
        self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.cache));
        self.in[3] = try self.rig.input(std.mem.sliceAsBytes(self.pos));
        self.in[4] = if (self.nidx > 0) try self.rig.input(std.mem.sliceAsBytes(self.idx)) else 0;
        self.in[5] = try self.rig.input(std.mem.sliceAsBytes(self.sinks));
        if (self.form == .rows) {
            const ptrs = try a.alloc(u64, rows);
            defer a.free(ptrs);
            for (ptrs, 0..) |*q, r| q.* = self.in[2] + self.cacheRow(r, 0) * row * 2;
            self.caches = try self.rig.input(std.mem.sliceAsBytes(ptrs));
        }
        self.out = try self.rig.output(self.count() * 4);
        return true;
    }

    /// Row r's list: its own position and other distinct earlier ones in random order; `masked` adds a -1 and t + 3.
    fn choose(self: *Attn, rng: std.Random, r: usize, masked: bool) !void {
        const t: usize = @intCast(self.pos[r]);
        const list = self.idx[r * self.nidx ..][0..self.nidx];
        const a = self.rig.gpu.gpa;
        const all = try a.alloc(i32, t + 1);
        defer a.free(all);
        for (all, 0..) |*p, i| p.* = @intCast(i);
        rng.shuffle(i32, all[0..t]);
        @memcpy(list[0 .. list.len - 1], all[0 .. list.len - 1]);
        list[list.len - 1] = @intCast(t);
        rng.shuffle(i32, list);
        if (masked and list.len > 12) {
            list[5] = -1;
            list[9] = @intCast(t + 3);
        }
    }

    /// Plants NaN in row 0 head 1 ql, +Inf in row 1 head 2 qr, +-Inf in row 2 head 0 ql, +Inf in last row's own key.
    fn plant(self: *Attn) void {
        const h = self.heads;
        self.ql[(0 * h + 1) * lat + 7] = std.math.nan(f32);
        self.qr[(1 * h + 2) * rope + 3] = c.toBf16(std.math.inf(f32));
        self.ql[(2 * h + 0) * lat + 0] = std.math.inf(f32);
        self.ql[(2 * h + 0) * lat + 1] = -std.math.inf(f32);
        const t: usize = @intCast(self.pos[self.shape.rows - 1]);
        self.cache[self.cacheRow(self.shape.rows - 1, t) * row + 9] = c.toBf16(std.math.inf(f32));
    }

    /// Gate 4: q and sinks 0 (weights 1, l = n + 1); eight 1s plus +-10240|14336 cancel in fp32 and break fp16/bf16.
    fn stress(self: *Attn, rng: std.Random) !void {
        @memset(self.ql, 0);
        @memset(self.qr, 0);
        @memset(self.sinks, 0);
        for (0..self.cache.len / row) |k| @memset(self.cache[k * row ..][0..lat], 0);
        const r = self.shape.rows - 1;
        if (self.form == .list) for (0..r) |i| @memcpy(self.idx[i * self.nidx ..][0..self.nidx], self.idx[r * self.nidx ..][0..self.nidx]);
        const a = self.rig.gpu.gpa;
        const keys = try a.alloc(usize, self.slots + self.nidx + 1);
        defer a.free(keys);
        const n = self.seen(r, keys);
        const half = n / 2;
        for (0..half) |j| for (0..lat) |col| {
            const v: f32 = if (j < 8) 1 else 2048 * @as(f32, @floatFromInt(5 + 2 * rng.uintLessThan(u32, 2)));
            self.cache[self.cacheRow(r, keys[j]) * row + col] = c.toBf16(v);
            self.cache[self.cacheRow(r, keys[half + j]) * row + col] = c.toBf16(if (j < 8) 0 else -v);
        };
    }

    pub fn run(self: *Attn) !void {
        const keys: hy4.Keys = .{ .pos = self.in[3], .window = if (self.windowed()) window else 0, .idx = self.in[4], .nidx = self.nidx, .ring = if (self.form == .ring) ring else 0 };
        const cache = if (self.form == .rows) self.caches else self.in[2];
        try hy4.attn(self.f, self.rig.s, self.in[0], self.in[1], cache, keys, self.in[5], self.out, self.shape.rows, self.heads);
        try self.rig.s.synchronize();
    }

    /// Row r, head h: sinked softmax reference (NaN kept in M), `acc` on P*V, and per-column fp32 error bound `tol`.
    fn attend(self: *const Attn, r: usize, h: usize, acc: ?c.Accum, order: g.Order, o: []f64, tol: ?[]f64) !void {
        const a = self.rig.gpu.gpa;
        const keys = try a.alloc(usize, self.slots + self.nidx + 1);
        defer a.free(keys);
        const n = self.seen(r, keys);
        const buf = try a.alloc(f64, 4 * n + 1);
        defer a.free(buf);
        const s = buf[0..n];
        const e = buf[n .. 2 * n];
        const p = buf[2 * n .. 3 * n];
        const col = buf[3 * n ..][0..n];
        const at = r * self.heads + h;
        const sink: f64 = self.sinks[h];
        var most = sink;
        for (keys[0..n], s, e) |key, *sv, *ev| {
            const kr = self.cache[self.cacheRow(r, key) * row ..][0..row];
            var dot: f64 = 0;
            var abs: f64 = 0;
            for (0..row) |cc| {
                const q: f64 = if (cc < lat) self.ql[at * lat + cc] else c.fromBf16(self.qr[at * rope + cc - lat]);
                const x = q * c.fromBf16(kr[cc]);
                dot += x;
                abs += @abs(x);
            }
            sv.* = dot * hy4.scale;
            ev.* = c.gamma(row, u) * abs * hy4.scale;
            most = if (std.math.isNan(most) or std.math.isNan(sv.*)) std.math.nan(f64) else @max(most, sv.*);
        }
        const steps = (n + chunk - 1) / chunk;
        const drift = @as(f64, @floatFromInt(4 * steps + 4)) * u + @abs(most - sink) * u; // exp and the rescales
        var l: f64 = @exp(sink - most);
        var worst: f64 = 0;
        for (p, s, e) |*pv, sv, ev| {
            pv.* = @exp(sv - most);
            l += pv.*;
            worst = @max(worst, ev + @abs(sv - most) * u + drift);
        }
        const rel_l = worst + c.gamma(@floatFromInt(2 * steps + 8), u);
        const chain = c.gamma(@floatFromInt(n + steps + 1), u);
        for (o, 0..) |*od, d| {
            var spread: f64 = 0;
            for (col, p, keys[0..n], s, e) |*x, w, key, sv, ev| {
                const v: f64 = c.fromBf16(self.cache[self.cacheRow(r, key) * row + d]);
                x.* = w * v;
                spread += @abs(w * v) * (ev + @abs(sv - most) * u + drift + chain);
            }
            od.* = (if (acc) |k| sumIn(k, order, col) else sumIn(.none, .sequential, col)) / l;
            if (tol) |t| t[d] = @abs(od.*) * (rel_l + 2 * u) + spread / l + 0x1p-126;
        }
    }

    pub fn reference(self: *Attn, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        a.free(self.tol);
        self.tol = try a.alloc(f64, want.len);
        for (0..self.shape.rows) |r| for (0..self.heads) |h| {
            const at = (r * self.heads + h) * lat;
            try self.attend(r, h, null, .sequential, want[at..][0..lat], self.tol[at..][0..lat]);
        };
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn bounds(self: *Attn, _: []const f64, _: []const f64, allow: []f64) !void {
        if (self.fill == .stress) return;
        for (allow, self.tol) |*l, t| l.* = if (std.math.isFinite(t)) t else std.math.floatMax(f64);
    }

    pub fn simulate(self: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..self.shape.rows) |r| for (0..self.heads) |h| {
            const at = (r * self.heads + h) * lat;
            try self.attend(r, h, acc, order, out[at..][0..lat], null);
        };
    }
};
