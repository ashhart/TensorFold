//! Inkling 2's attention launches under the contract: sliding chunks, global chunks (fold, split, prompt), the merge.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ink = @import("inkling2");
const rig_ = @import("contract_rig.zig");
const ref = core.contract.inkling2_attn_ref;

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const K = ink.kernels;
const D = ref.D;
const heads = ref.heads;
const kv = ref.kv;
const n_out = heads * D;
const global_ring = 4096; // global keys to 1M wrap a 4,096-slot ring (key p in slot p % ring): no 2 GiB cache needed

/// The launch kernels.attend picks: sliding chunks, a block a global chunk, a block a sub-chunk, one stream's prompt.
pub const Path = enum { sliding, fold, split, prompt };

/// A control's planted fault, through the device's inputs only: the served kernels run as they are.
pub const Plant = enum { none, late_bias, no_sink, q_ulp };

pub const fold_name = "inkling2 attn tf_ink2_attn_fold + tf_ink2_merge (global layers, 16,384-key chunks)";

/// Unused beside `bounds`: the reference derives each output's allowance from the device's own steps.
fn depth(_: c.Shape) usize {
    return 16 * 69 + 2 * (16 + 64) + 8;
}

const sliding_shapes = [_]c.Shape{ .{ .rows = 16, .n = n_out, .k = 4096 }, .{ .rows = 1, .n = n_out, .k = 1048576 }, .{ .rows = 2048, .n = n_out, .k = 2048 }, .{ .rows = 2048, .n = n_out, .k = 1048576 }, .{ .rows = 128, .n = n_out, .k = 300000 }, .{ .rows = 7, .n = n_out, .k = 1030 } };
const fold_shapes = [_]c.Shape{ .{ .rows = 16, .n = n_out, .k = 20000 }, .{ .rows = 1, .n = n_out, .k = 1048576 }, .{ .rows = 128, .n = n_out, .k = 300000 }, .{ .rows = 70, .n = n_out, .k = 16390 }, .{ .rows = 2, .n = n_out, .k = 1048576 } };
const split_shapes = [_]c.Shape{ .{ .rows = 16, .n = n_out, .k = 20000 }, .{ .rows = 1, .n = n_out, .k = 1048576 }, .{ .rows = 8, .n = n_out, .k = 1048576 }, .{ .rows = 64, .n = n_out, .k = 70000 }, .{ .rows = 33, .n = n_out, .k = 1030 } };
const prompt_shapes = [_]c.Shape{ .{ .rows = 130, .n = n_out, .k = 20000 }, .{ .rows = 1, .n = n_out, .k = 1048576 }, .{ .rows = 2048, .n = n_out, .k = 2048 }, .{ .rows = 2048, .n = n_out, .k = 18432 }, .{ .rows = 256, .n = n_out, .k = 1048576 } };

/// rows x 24 heads x 128 outputs; k: the first stream's context (its last row's position + 1).
pub fn spec(path: Path) c.Spec {
    const no: c.Tile = .{};
    const rows32: c.Tile = .{ .size = 32 }; // 192 (row, head) pairs: three 64-pair tiles
    const s: c.Spec = .{ .name = "", .accum = .f32, .depth = depth, .ops = 12, .out = .bf16, .served = &fold_shapes };
    var x = s;
    switch (path) {
        .sliding => {
            x.name = "inkling2 attn tf_ink2_attn + tf_ink2_merge (sliding layers: window 1,024, 256-key chunks)";
            x.served = &sliding_shapes;
            x.tiles = .{ rows32, no, .{ .size = 1024 } };
        },
        .fold => {
            x.name = fold_name;
            x.tiles = .{ rows32, no, .{ .size = 16384 } };
        },
        .split => {
            x.name = "inkling2 attn tf_ink2_attn split + tf_ink2_merge (global layers, a block a 1,024-key sub-chunk)";
            x.served = &split_shapes;
            x.tiles = .{ rows32, no, .{ .size = 1024 } };
            x.replaces = fold_name;
        },
        .prompt => {
            x.name = "inkling2 attn tf_ink2_attn_prompt + tf_ink2_merge (one stream's prompt rows, 128 pairs a block)";
            x.served = &prompt_shapes;
            x.tiles = .{ .{ .size = 64 }, no, .{ .size = 16384 } };
            x.replaces = fold_name;
        },
    }
    return x;
}

/// Gate 4's value at key p of column col: period 1,024, its second half the first's negation a key along, + 1 on 8.
fn periodic(p: usize, col: usize) f64 {
    const i = p % 1024;
    if (i < 512) return @floatFromInt(129 + (i * 73 + col * 31) % 127);
    const j = i - 512;
    return -@as(f64, @floatFromInt(129 + ((j + 1) % 512 * 73 + col * 31) % 127)) + @as(f64, if (j < 8) 1 else 0);
}

pub const Attn = struct {
    rig: rig_.Rig,
    k: *const K.Kernels,
    path: Path,
    plant: Plant = .none,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    layer: K.Layer = undefined,
    streams: usize = 1,
    n_tiles: u32 = 0,
    q: []u16 = &.{},
    kc: []u16 = &.{},
    vc: []u16 = &.{},
    tab: []u16 = &.{},
    sink: []u16 = &.{},
    gate: []u16 = &.{},
    pos: []usize = &.{},
    stream: []usize = &.{},
    tol: []f64 = &.{},
    dev: Dev = .{},

    const Dev = struct { q: u64 = 0, q_run: u64 = 0, kc: u64 = 0, vc: u64 = 0, tab: u64 = 0, tab_run: u64 = 0, sink: u64 = 0, sink_run: u64 = 0, gate: u64 = 0, pos: u64 = 0, seg_of: u64 = 0, segs: u64 = 0, tiles: u64 = 0, po: u64 = 0, pml: u64 = 0, out: u64 = 0, base: u64 = 0 };

    pub fn deinit(self: *Attn) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Attn) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ "q", "kc", "vc", "tab", "sink", "gate" }) |f| {
            a.free(@field(self, f));
            @field(self, f) = &.{};
        }
        a.free(self.pos);
        a.free(self.stream);
        a.free(self.tol);
        self.pos = &.{};
        self.stream = &.{};
        self.tol = &.{};
    }

    pub fn kernel(self: *Attn) g.Kernel {
        var s = spec(self.path);
        s.name = switch (self.plant) {
            .none => s.name,
            .late_bias => "control: inkling2 attn sliding rows read each distance's bias one distance late",
            .no_sink => "control: inkling2 attn global fold rows merged without their sink",
            .q_ulp => "control: inkling2 attn prompt rows with one q value a bf16 step up (bit equality must see it)",
        };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.plant == .none) null else .shapes };
    }

    const vt = g.hooks(Attn);

    pub fn count(self: *const Attn) usize {
        return self.shape.rows * n_out;
    }

    fn call(self: *const Attn) ref.Call {
        return .{ .q = self.q, .k = self.kc, .v = self.vc, .tab = self.tab, .sink = self.sink, .gate = self.gate, .pos = self.pos, .stream = self.stream, .ring = self.layer.ring, .window = self.layer.window, .exact = self.fill == .stress };
    }

    /// Two streams for decode windows (2..128 rows; the second at half the context), else one; prompt: one.
    pub fn prepare(self: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != n_out or shape.rows == 0 or shape.k < shape.rows) return false;
        if (self.path == .split and shape.rows > K.split_rows) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.fill = fill;
        const a = self.rig.gpu.gpa;
        const rows = shape.rows;
        self.layer = if (self.path == .sliding) K.Layer.sliding(1024, 2048) else .{ .window = 0, .ring = global_ring, .chunk = K.global_chunk, .tau = true };
        const ring: usize = self.layer.ring;
        self.streams = if (self.path != .prompt and rows >= 2 and rows <= 128) 2 else 1;
        const n1 = if (self.streams == 2) rows / 2 else 0;
        const n0 = rows - n1;
        const last1 = @max(n1, shape.k / 2);
        const segs = [2]K.Seg{ .{ .row0 = 0, .n = @intCast(n0), .p0 = @intCast(shape.k - n0), .off = 0 }, .{ .row0 = @intCast(n0), .n = @intCast(n1), .p0 = @intCast(last1 - n1), .off = @intCast(ring * kv * D) } };
        self.pos = try a.alloc(usize, rows);
        self.stream = try a.alloc(usize, rows);
        for (0..rows) |r| {
            const st: usize = if (r < n0) 0 else 1;
            self.stream[r] = st;
            self.pos[r] = @as(usize, @intCast(segs[st].p0)) + r - @as(usize, @intCast(segs[st].row0));
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        try self.fillAll(rng, ring);
        if (fill == .nonfinite) self.plantNonfinite(ring);
        const tiles = try a.alloc(K.Tile, K.tileCount(rows, ref.group) + 2);
        defer a.free(tiles);
        self.n_tiles = @intCast(try K.tiles(segs[0..self.streams], ref.group, tiles));
        const pos32 = try a.alloc(i32, 2 * rows);
        defer a.free(pos32);
        for (0..rows) |r| {
            pos32[r] = @intCast(self.pos[r]);
            pos32[rows + r] = @intCast(self.stream[r]);
        }
        const d = &self.dev;
        d.q = try self.rig.input(std.mem.sliceAsBytes(self.q));
        d.kc = try self.rig.input(std.mem.sliceAsBytes(self.kc));
        d.vc = try self.rig.input(std.mem.sliceAsBytes(self.vc));
        d.tab = try self.rig.input(std.mem.sliceAsBytes(self.tab));
        d.sink = try self.rig.input(std.mem.sliceAsBytes(self.sink));
        d.gate = try self.rig.input(std.mem.sliceAsBytes(self.gate));
        d.pos = try self.rig.input(std.mem.sliceAsBytes(pos32[0..rows]));
        d.seg_of = try self.rig.input(std.mem.sliceAsBytes(pos32[rows..]));
        d.segs = try self.rig.input(std.mem.sliceAsBytes(segs[0..self.streams]));
        d.tiles = try self.rig.input(std.mem.sliceAsBytes(tiles[0..self.n_tiles]));
        try self.planted();
        const per: usize = if (self.path == .split) K.subs else 1;
        const parts = rows * heads * K.slotsFor(self.layer, shape.k) * per;
        d.po = try self.rig.scratch(parts * D * 4);
        d.pml = try self.rig.scratch(parts * 8);
        d.out = try self.rig.output(rows * n_out * 2);
        if (self.path == .split or self.path == .prompt) d.base = try self.rig.scratch(rows * n_out * 2);
        return true;
    }

    /// q, k clipped as prep leaves them, v rms-normed, the table rel . proj rounded once; stress: gate 4's rows.
    fn fillAll(self: *Attn, rng: std.Random, ring: usize) !void {
        const a = self.rig.gpu.gpa;
        const rows = self.shape.rows;
        const stress = self.fill == .stress;
        self.q = try a.alloc(u16, rows * n_out);
        self.kc = try a.alloc(u16, self.streams * kv * ring * D);
        self.vc = try a.alloc(u16, self.kc.len);
        self.tab = try a.alloc(u16, rows * heads * ref.extent);
        self.sink = try a.alloc(u16, rows * heads);
        self.gate = try a.alloc(u16, rows * n_out);
        const amp: f64 = if (self.fill == .nonfinite) 0.3 else 3; // non-finite rows: no p underflows, so 0 x inf never meets
        var x: [D]f64 = undefined;
        for (0..rows * heads) |rh| {
            for (&x) |*v| v.* = if (stress) 0 else rng.floatNorm(f64) * amp;
            const f = ref.clipFactor(&x, ref.qk_clip);
            for (self.q[rh * D ..][0..D], x) |*o, v| o.* = ref.bits(v * f); // q/k L2 clip, then one bf16 rounding
        }
        for (0..self.streams * kv * ring) |slot| {
            for (&x) |*v| v.* = rng.floatNorm(f64) * 2.5;
            const f = ref.clipFactor(&x, ref.qk_clip);
            for (self.kc[slot * D ..][0..D], x) |*o, v| o.* = ref.bits(v * f);
            var ss: f64 = 0;
            for (&x) |*v| {
                v.* = rng.floatNorm(f64);
                ss += v.* * v.*;
            }
            const inv = 1 / @sqrt(ss / D + ref.eps);
            const col0 = (slot / ring % kv) * D;
            for (self.vc[slot * D ..][0..D], x, 0..) |*o, v, dd| o.* = ref.bits(if (stress) periodic(slot % ring, col0 + dd) else v * inv);
        }
        var proj: [ref.d_rel * ref.extent]f64 = undefined;
        for (&proj) |*p| p.* = ref.rn(rng.floatNorm(f64) * 0.5);
        var rel: [ref.d_rel]f64 = undefined;
        for (0..rows * heads) |rh| {
            for (&rel) |*v| v.* = rng.floatNorm(f64);
            const f = ref.clipFactor(&rel, ref.rel_clip);
            for (&rel) |*v| v.* = ref.rn(v.* * f);
            for (self.tab[rh * ref.extent ..][0..ref.extent], 0..) |*o, dd| {
                var acc: f64 = 0;
                for (rel, 0..) |rv, j| acc += rv * proj[j * ref.extent + dd];
                o.* = if (stress) 0 else ref.bits(acc); // the bf16 table: rel . proj rounded once (transformers' bf16 matmul)
            }
            self.sink[rh] = if (stress) 0xff80 else ref.bits(8 + 16 * rng.float(f64));
            for (self.gate[rh * D ..][0..D]) |*o| o.* = ref.bits(if (stress) 32 else rng.floatNorm(f64) * 2);
        }
    }

    /// NaN q, a +inf sink, both gate infinities, +inf in a key and +inf, -inf in values every row of stream 0 sees.
    fn plantNonfinite(self: *Attn, ring: usize) void {
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const p0 = self.pos[0];
        const at = struct {
            fn f(kvh: usize, slot: usize, dd: usize, r: usize) usize {
                return (kvh * r + slot) * D + dd;
            }
        }.f;
        self.q[(0 * heads + 1) * D + 5] = nan;
        if (self.shape.rows > 1) self.sink[1 * heads + 2] = inf;
        if (self.shape.rows > 2) {
            self.gate[(2 * heads + 3) * D + 0] = inf;
            self.gate[(2 * heads + 3) * D + 1] = inf | 0x8000;
        }
        self.kc[at(2, p0 % ring, 7, ring)] = inf;
        self.vc[at(1, p0 % ring, 9, ring)] = inf;
        self.vc[at(1, p0 % ring, 10, ring)] = inf;
        if (p0 > 0) self.vc[at(1, (p0 - 1) % ring, 10, ring)] = inf | 0x8000;
    }

    /// The device's own inputs: a control's faulted copy, else the true ones.
    fn planted(self: *Attn) !void {
        const a = self.rig.gpu.gpa;
        const d = &self.dev;
        d.q_run = d.q;
        d.tab_run = d.tab;
        d.sink_run = d.sink;
        switch (self.plant) {
            .none => {},
            .late_bias => {
                const t = try a.dupe(u16, self.tab);
                defer a.free(t);
                for (0..t.len / ref.extent) |rh| for (0..ref.extent) |dd| {
                    t[rh * ref.extent + dd] = if (dd + 1 < ref.extent) self.tab[rh * ref.extent + dd + 1] else 0;
                };
                d.tab_run = try self.rig.input(std.mem.sliceAsBytes(t));
            },
            .no_sink => {
                const t = try a.alloc(u16, self.sink.len);
                defer a.free(t);
                @memset(t, 0xff80);
                d.sink_run = try self.rig.input(std.mem.sliceAsBytes(t));
            },
            .q_ulp => {
                const t = try a.dupe(u16, self.q);
                defer a.free(t);
                t[0] +%= 1;
                d.q_run = try self.rig.input(std.mem.sliceAsBytes(t));
            },
        }
    }

    /// kernels.attend as the forward calls it; `path` picks its branch through `streams` and the partial room.
    fn launch(self: *Attn, path: Path, q: u64, tab: u64, sink: u64, out: u64) !void {
        const d = self.dev;
        var r: K.Rows = .{ .rows = @intCast(self.shape.rows), .heads = heads, .kv_heads = kv, .pos = d.pos, .seg_of = d.seg_of, .segs = d.segs, .tiles = d.tiles, .n_tiles = self.n_tiles, .last_pos = @intCast(self.shape.k - 1), .streams = @intCast(self.streams) };
        const stride = K.slotsFor(self.layer, self.shape.k);
        var slots: usize = 0; // no partial room for sub-chunks: global calls take the fold or the prompt kernel
        switch (path) {
            .sliding, .prompt => {},
            .fold => r.streams = 2, // the dispatcher's multi-stream branch: a block a chunk, its sub-chunks folded in order
            .split => slots = self.shape.rows * heads * stride * K.subs,
        }
        try K.attend(self.k, self.rig.s, r, self.layer, stride, slots, q, .{ .k = d.kc, .v = d.vc }, tab, d.po, d.pml, sink, d.gate, out);
    }

    pub fn run(self: *Attn) !void {
        try self.launch(self.path, self.dev.q_run, self.dev.tab_run, self.dev.sink_run, self.dev.out);
        try self.rig.s.synchronize();
    }

    /// The fold's bytes on the true inputs: the kernel the split and prompt launches must equal bit for bit.
    pub fn replaced(self: *Attn) ![]const u8 {
        try self.launch(.fold, self.dev.q, self.dev.tab, self.dev.sink, self.dev.base);
        return self.rig.bytes(self.dev.base, self.count() * 2);
    }

    pub fn reference(self: *Attn, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        a.free(self.tol);
        self.tol = try a.alloc(f64, want.len);
        const cl = self.call();
        try Pool.run(a, .{ .call = &cl, .want = want, .tol = self.tol }, self.shape.rows * heads);
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn bounds(self: *Attn, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, self.tol) |*l, t| l.* = if (std.math.isFinite(t)) t else std.math.floatMax(f64);
    }

    /// Gate 4 (stress rows only): q = 0 gives every head of a key head the same row, so one replay serves its group.
    pub fn simulate(self: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        std.debug.assert(self.fill == .stress);
        const a = self.rig.gpu.gpa;
        const cl = self.call();
        try Pool.run(a, .{ .call = &cl, .want = out, .tol = &.{}, .sim = .{ .acc = acc, .order = order } }, self.shape.rows * kv);
    }
};

/// The reference's (row, head) pairs (gate 4: (row, key head) pairs) over up to six threads, each its own buffers.
const Pool = struct {
    call: *const ref.Call,
    want: []f64,
    tol: []f64,
    sim: ?struct { acc: c.Accum, order: g.Order } = null,

    const Job = struct {
        pool: *const Pool,
        first: usize,
        step: usize,
        items: usize,
        ks: *ref.Keys,
        p: []f64,
        col: []f64,

        pub fn run(j: *Job) void {
            const pool = j.pool;
            var i = j.first;
            while (i < j.items) : (i += j.step) {
                if (pool.sim) |s| {
                    const r = i / kv;
                    const h0 = (i % kv) * ref.group;
                    const o = pool.want[(r * heads + h0) * D ..][0..D];
                    ref.simulate(pool.call, r, h0, s.acc, s.order, j.p, j.col, o);
                    for (1..ref.group) |e| @memcpy(pool.want[(r * heads + h0 + e) * D ..][0..D], o);
                } else {
                    const at = i * D;
                    ref.row(pool.call, i / heads, i % heads, j.ks, pool.want[at..][0..D], pool.tol[at..][0..D]);
                }
            }
        }
    };

    fn run(a: std.mem.Allocator, pool: Pool, items: usize) !void {
        const n = contract.reference_threads.count(6, items);
        var jobs: [contract.reference_threads.maximum]Job = undefined;
        var keys: usize = 0;
        for (pool.call.pos) |p| keys = @max(keys, p + 1);
        var made: usize = 0;
        defer for (jobs[0..made]) |j| {
            a.destroy(j.ks);
            a.free(j.p);
            a.free(j.col);
        };
        for (jobs[0..n], 0..) |*j, w| {
            const big = if (pool.sim != null) keys else 0;
            j.* = .{ .pool = &pool, .first = w, .step = n, .items = items, .ks = try a.create(ref.Keys), .p = try a.alloc(f64, big), .col = try a.alloc(f64, big) };
            made += 1;
        }
        contract.reference_threads.run(Job, jobs[0..n], .normal);
    }
};
