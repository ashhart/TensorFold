//! Inkling 2's HIP attention launches under the contract at Mini's 24/8 heads: sliding, fold, split, prompt, the merge.
const std = @import("std");
const core = @import("core");
const K = @import("inkling2_hip");
const Rig = @import("contract_rig.zig").Rig;
const Profile = @import("contract_entry.zig").Profile;
const fill_ = @import("contract_inkling2_fill.zig");

const contract = core.contract;
const ref = contract.inkling2_attn_ref;
const c = contract.spec;
const g = contract.gates;
const D = ref.D;
const heads = fill_.heads;
const kv = fill_.kv;
const n_out = heads * D;

/// The launch kernels.attend picks: sliding chunks, a block a global chunk, a block a sub-chunk, one stream's prompt.
pub const Path = enum { sliding, fold, split, prompt };

/// A control's planted fault, through the device's inputs or launch only: the served kernels run as they are.
pub const Plant = enum { none, late_bias, no_sink, q_ulp, window_one, pre_sigmoid };

pub const fold_name = "inkling2 hip attn tf_ink2_attn_fold + tf_ink2_merge (global layers, 16,384-key chunks)";

/// Unused beside `bounds`: the reference derives each output's allowance from the device's own steps.
fn depth(_: c.Shape) usize {
    return 16 * 65 + 2 * (16 + 64) + 8;
}

fn sh(rows: usize, k: usize) c.Shape {
    return .{ .rows = rows, .n = n_out, .k = k };
}

// served shapes: decode windows, long global spans to 1M keys on a 4,096-slot ring that wraps, and prompt chunks
const sliding_full = [_]c.Shape{ sh(16, 4096), sh(1, 1048576), sh(2048, 2048), sh(2048, 1048576), sh(128, 300000), sh(7, 1030), sh(1, 1), sh(3, 3) };
const fold_full = [_]c.Shape{ sh(16, 20000), sh(1, 1048576), sh(128, 300000), sh(70, 16390), sh(2, 1048576), sh(16, 16), sh(5, 1024), sh(5, 1025), sh(3, 3) };
const split_full = [_]c.Shape{ sh(16, 20000), sh(1, 1048576), sh(8, 1048576), sh(64, 70000), sh(33, 1030) };
const prompt_full = [_]c.Shape{ sh(130, 20000), sh(1, 1048576), sh(2048, 2048), sh(2048, 18432), sh(256, 1048576) };
const sliding_smoke = [_]c.Shape{ sh(16, 4096), sh(7, 1030), sh(2048, 2048) };
// smoke keeps one 1M-key row: gate 4 sums over the longest served span, where bf16 sums miss by over 100x
const fold_smoke = [_]c.Shape{ sh(16, 20000), sh(70, 16390), sh(1, 1048576) };
const split_smoke = [_]c.Shape{ sh(16, 20000), sh(33, 1030), sh(1, 1048576) };
const prompt_smoke = [_]c.Shape{ sh(130, 20000), sh(2048, 2048), sh(1, 1048576) };
// three streams in one launch: at least a row each
const apart_full = [_]c.Shape{ sh(16, 20000), sh(3, 1048576), sh(64, 70000), sh(9, 1030) };
const apart_smoke = [_]c.Shape{ sh(16, 20000), sh(9, 1030), sh(3, 1048576) };

fn served(path: Path, apart: bool, profile: Profile) []const c.Shape {
    const full = profile == .full;
    if (apart) return if (full) &apart_full else &apart_smoke;
    return switch (path) {
        .sliding => if (full) &sliding_full else &sliding_smoke,
        .fold => if (full) &fold_full else &fold_smoke,
        .split => if (full) &split_full else &split_smoke,
        .prompt => if (full) &prompt_full else &prompt_smoke,
    };
}

// exact P: rows below position 64, one tile each; the outputs equal float64 bit for bit
const exact_shapes = [_]c.Shape{ sh(16, 64), sh(1, 64), sh(64, 64), sh(3, 40) };

/// The exact-P registrations: bf16 P's own bits decide every output, so an unrounded P goes red at shapes.
fn exactSpec(path: Path) c.Spec {
    const why = "64 keys of gate 4's values: neither fp16 nor bf16 provably misses here; the 1M-key registrations prove fp32";
    var x: c.Spec = .{ .name = "", .accum = .f32, .depth = depth, .ops = 12, .out = .bf16, .served = &exact_shapes, .sums = .{ .f16 = 0, .bf16 = 0, .why = why } };
    x.tiles = .{ .{ .size = 4 }, .{}, .{} };
    x.name = if (path == .sliding) "inkling2 hip attn sliding rows at bf16 P's own bits (q 0, no sink, gate 1, integer v)" else "inkling2 hip attn global fold rows at bf16 P's own bits (q 0, no sink, gate 1, integer v)";
    return x;
}

/// rows x 24 heads x 128 outputs; k: the first stream's context (its last row's position + 1).
pub fn spec(path: Path, apart: bool, profile: Profile) c.Spec {
    const no: c.Tile = .{};
    const rows4: c.Tile = if (apart) .{ .size = 4, .min = 3 } else .{ .size = 4 }; // 12 pairs: three 4-pair tiles
    var x: c.Spec = .{ .name = "", .accum = .f32, .depth = depth, .ops = 12, .out = .bf16, .served = served(path, apart, profile) };
    switch (path) {
        .sliding => {
            x.name = "inkling2 hip attn tf_ink2_attn + tf_ink2_merge (sliding layers: window 1,024, 256-key chunks)";
            x.tiles = .{ rows4, no, .{ .size = 1024 } };
        },
        .fold => {
            x.name = fold_name;
            x.tiles = .{ rows4, no, .{ .size = 16384 } };
        },
        .split => {
            x.name = "inkling2 hip attn tf_ink2_attn split + tf_ink2_merge (global layers, a block a 1,024-key sub-chunk)";
            x.tiles = .{ rows4, no, .{ .size = 1024 } };
            x.replaces = fold_name;
        },
        .prompt => {
            x.name = "inkling2 hip attn tf_ink2_attn_prompt + tf_ink2_merge (one stream's prompt rows, 8 pairs a block)";
            x.tiles = .{ .{ .size = 8 }, no, .{ .size = 16384 } };
            x.replaces = fold_name;
        },
    }
    if (apart) {
        x.name = switch (path) {
            .sliding => "inkling2 hip attn sliding: three streams' rows and caches in one launch",
            .fold => "inkling2 hip attn global fold: three streams' rows and caches in one launch",
            .split => "inkling2 hip attn global split: three streams' rows and caches in one launch",
            .prompt => unreachable,
        };
        x.replaces = "the same launches one stream at a time, at each stream's row and cache offsets";
    }
    return x;
}

pub const Attn = struct {
    rig: Rig,
    k: *const K.Kernels,
    path: Path,
    plant: Plant = .none,
    apart: bool = false,
    exact_p: bool = false, // q 0, no sink, gate 1, integer v: each output bf16(fp32(sum bf16(P) v / sum bf16(P)))
    profile: Profile = .full,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    lay: fill_.Layout = undefined,
    data: fill_.Data = .{},
    tol: []f64 = &.{},
    dev: Dev = .{},
    alone: [3]Alone = undefined,

    const Dev = struct { q: u64 = 0, q_run: u64 = 0, kc: u64 = 0, vc: u64 = 0, tab: u64 = 0, tab_run: u64 = 0, sink: u64 = 0, sink_run: u64 = 0, gate: u64 = 0, gate_run: u64 = 0, pos: u64 = 0, seg_of: u64 = 0, segs: u64 = 0, tiles: u64 = 0, n_tiles: u32 = 0, po: u64 = 0, pml: u64 = 0, out: u64 = 0, base: u64 = 0 };
    /// One stream's own launch: its segment from row 0, and its tiles.
    const Alone = struct { segs: u64, tiles: u64, n_tiles: u32 };

    pub fn deinit(self: *Attn) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Attn) void {
        self.data.free(self.rig.a);
        self.rig.a.free(self.tol);
        self.tol = &.{};
    }

    pub fn kernel(self: *Attn) g.Kernel {
        var s = if (self.exact_p) exactSpec(self.path) else spec(self.path, self.apart, self.profile);
        s.name = switch (self.plant) {
            .none => s.name,
            .late_bias => "control: inkling2 hip attn sliding rows read each distance's bias one distance late",
            .no_sink => "control: inkling2 hip attn global fold rows merged without their sink",
            .q_ulp => "control: inkling2 hip attn prompt rows with one q value a bf16 step up (bit equality must see it)",
            .window_one => "control: inkling2 hip attn sliding rows with a window one key long",
            .pre_sigmoid => "control: inkling2 hip attn sliding rows given sigmoid(sink) and sigmoid(gate), as #4387's glue did",
        };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.plant == .none) null else .shapes };
    }

    const vt = g.hooks(Attn);

    pub fn count(self: *const Attn) usize {
        return self.shape.rows * n_out;
    }

    fn call(self: *const Attn) ref.Call {
        const d = &self.data;
        return .{ .q = d.q, .k = d.kc, .v = d.vc, .tab = d.tab, .sink = d.sink, .gate = d.gate, .pos = d.pos, .stream = d.stream, .ring = self.lay.layer.ring, .window = self.lay.layer.window, .exact = self.fill == .stress, .heads = heads, .kv = kv, .chain = .hip };
    }

    /// The rows' streams (fill_.Layout), their inputs, and the device's buffers; false for a shape the path refuses.
    pub fn prepare(self: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != n_out or shape.rows == 0 or shape.k < shape.rows) return false;
        if (self.path == .split and shape.rows > K.split_rows) return false;
        if (self.apart and shape.rows < 3) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        self.fill = fill;
        const a = self.rig.a;
        const rows = shape.rows;
        const layer: K.Layer = if (self.path == .sliding) K.Layer.sliding(1024, 2048) else .{ .window = 0, .ring = fill_.global_ring, .chunk = K.global_chunk, .tau = true };
        const streams: usize = if (self.apart) 3 else if (self.path != .prompt and rows >= 2 and rows <= 128) 2 else 1;
        self.lay = fill_.Layout.of(layer, rows, shape.k, streams);
        if (self.exact_p and fill == .random) try self.data.makeExactP(a, self.lay, seed) else try self.data.make(a, self.lay, fill, seed);
        const d = &self.dev;
        const td = &self.data;
        d.q = try self.rig.input(std.mem.sliceAsBytes(td.q));
        d.kc = try self.rig.input(std.mem.sliceAsBytes(td.kc));
        d.vc = try self.rig.input(std.mem.sliceAsBytes(td.vc));
        d.tab = try self.rig.input(std.mem.sliceAsBytes(td.tab));
        d.sink = try self.rig.input(std.mem.sliceAsBytes(td.sink));
        d.gate = try self.rig.input(std.mem.sliceAsBytes(td.gate));
        const pos32 = try a.alloc(i32, rows);
        defer a.free(pos32);
        for (pos32, td.pos) |*p, v| p.* = @intCast(v);
        d.pos = try self.rig.input(std.mem.sliceAsBytes(pos32));
        const segs = self.lay.segs[0..streams];
        d.segs = try self.rig.input(std.mem.sliceAsBytes(segs));
        const tiles = try a.alloc(K.Tile, K.tileCount(rows, fill_.group) + streams);
        defer a.free(tiles);
        d.n_tiles = @intCast(try K.tiles(segs, fill_.group, tiles));
        d.tiles = try self.rig.input(std.mem.sliceAsBytes(tiles[0..d.n_tiles]));
        if (self.apart) for (segs, &self.alone) |s, *x| {
            var one = s;
            one.row0 = 0;
            x.segs = try self.rig.input(std.mem.asBytes(&one));
            x.n_tiles = @intCast(try K.tiles(&.{one}, fill_.group, tiles));
            x.tiles = try self.rig.input(std.mem.sliceAsBytes(tiles[0..x.n_tiles]));
        };
        try self.planted();
        const per: usize = if (self.path == .split) K.subs else 1;
        const parts = rows * heads * K.slotsFor(layer, shape.k) * per;
        d.po = try self.rig.scratch(parts * D * 4);
        d.pml = try self.rig.scratch(parts * 8);
        d.out = try self.rig.output(rows * n_out * 2);
        if (self.apart or self.path == .split or self.path == .prompt) d.base = try self.rig.scratch(rows * n_out * 2);
        return true;
    }

    /// The device's own inputs: a control's faulted copy, else the true ones.
    fn planted(self: *Attn) !void {
        const a = self.rig.a;
        const d = &self.dev;
        const td = &self.data;
        d.q_run = d.q;
        d.tab_run = d.tab;
        d.sink_run = d.sink;
        d.gate_run = d.gate;
        switch (self.plant) {
            .none, .window_one => {},
            .late_bias => {
                const t = try a.dupe(u16, td.tab);
                defer a.free(t);
                for (0..t.len / ref.extent) |rh| for (0..ref.extent) |dd| {
                    t[rh * ref.extent + dd] = if (dd + 1 < ref.extent) td.tab[rh * ref.extent + dd + 1] else 0;
                };
                d.tab_run = try self.rig.input(std.mem.sliceAsBytes(t));
            },
            .no_sink => {
                const t = try a.alloc(u16, td.sink.len);
                defer a.free(t);
                @memset(t, 0xff80);
                d.sink_run = try self.rig.input(std.mem.sliceAsBytes(t));
            },
            .q_ulp => {
                const t = try a.dupe(u16, td.q);
                defer a.free(t);
                t[0] +%= 1;
                d.q_run = try self.rig.input(std.mem.sliceAsBytes(t));
            },
            .pre_sigmoid => {
                const s = try a.alloc(u16, td.sink.len);
                defer a.free(s);
                for (s, td.sink) |*o, v| o.* = ref.bits(sigmoid(ref.bf(v)));
                d.sink_run = try self.rig.input(std.mem.sliceAsBytes(s));
                const t = try a.alloc(u16, td.gate.len);
                defer a.free(t);
                for (t, td.gate) |*o, v| o.* = ref.bits(sigmoid(ref.bf(v)));
                d.gate_run = try self.rig.input(std.mem.sliceAsBytes(t));
            },
        }
    }

    fn sigmoid(x: f64) f64 {
        return 1 / (1 + @exp(-x));
    }

    /// One launch's inputs: rows from `row0` (a stream alone) or all of them.
    const Call = struct { r: K.Rows, row0: usize = 0, q: u64, tab: u64, sink: u64, gate: u64, out: u64 };

    /// kernels.attend as the forward calls it; `path` picks its branch through `streams` and the partial room.
    fn launch(self: *Attn, path: Path, x: Call) !void {
        const d = self.dev;
        var r = x.r;
        var layer = self.lay.layer;
        if (self.plant == .window_one) layer.window = 1;
        const stride = K.slotsFor(self.lay.layer, self.shape.k);
        var slots: usize = 0; // no partial room for sub-chunks: global calls take the fold or the prompt kernel
        switch (path) {
            .sliding, .prompt => {},
            .fold => r.streams = @max(r.streams, 2), // the dispatcher's multi-stream branch
            .split => slots = @as(usize, r.rows) * heads * stride * K.subs,
        }
        const b = x.row0 * heads;
        try K.attend(self.k, self.rig.s, r, layer, stride, slots, x.q + b * D * 2, .{ .k = d.kc, .v = d.vc }, x.tab + b * ref.extent * 2, d.po, d.pml, x.sink + b * 2, x.gate + b * D * 2, x.out + b * D * 2);
    }

    fn rowsAll(self: *const Attn) K.Rows {
        const d = self.dev;
        return .{ .rows = @intCast(self.shape.rows), .heads = heads, .kv_heads = kv, .pos = d.pos, .seg_of = 0, .segs = d.segs, .tiles = d.tiles, .n_tiles = d.n_tiles, .last_pos = @intCast(self.shape.k - 1), .streams = @intCast(self.lay.streams) };
    }

    pub fn run(self: *Attn) !void {
        const d = self.dev;
        try self.launch(self.path, .{ .r = self.rowsAll(), .q = d.q_run, .tab = d.tab_run, .sink = d.sink_run, .gate = d.gate_run, .out = d.out });
        try self.rig.s.synchronize();
    }

    /// The baseline on the true inputs: each stream alone (apart), else the fold the split and prompt must equal.
    pub fn replaced(self: *Attn) ![]const u8 {
        const d = self.dev;
        if (!self.apart) {
            try self.launch(.fold, .{ .r = self.rowsAll(), .q = d.q, .tab = d.tab, .sink = d.sink, .gate = d.gate, .out = d.base });
            return self.rig.bytes(d.base, self.count() * 2);
        }
        for (self.lay.segs[0..3], self.alone) |s, x| {
            const row0: usize = @intCast(s.row0);
            const r: K.Rows = .{ .rows = @intCast(s.n), .heads = heads, .kv_heads = kv, .pos = d.pos + row0 * 4, .seg_of = 0, .segs = x.segs, .tiles = x.tiles, .n_tiles = x.n_tiles, .last_pos = @intCast(s.p0 + s.n - 1) };
            try self.launch(self.path, .{ .r = r, .row0 = row0, .q = d.q, .tab = d.tab, .sink = d.sink, .gate = d.gate, .out = d.base });
        }
        return self.rig.bytes(d.base, self.count() * 2);
    }

    pub fn reference(self: *Attn, want: []f64, mag: []f64) !void {
        const a = self.rig.a;
        a.free(self.tol);
        self.tol = try a.alloc(f64, want.len);
        const cl = self.call();
        if (self.exact_p and self.fill == .random) {
            for (0..self.shape.rows * heads) |rh| fill_.exactP(&cl, rh / heads, rh % heads, want[rh * D ..][0..D]);
            @memset(self.tol, 0); // bit for bit
        } else try fill_.Pool.run(a, .{ .call = &cl, .want = want, .tol = self.tol }, self.shape.rows * heads);
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn bounds(self: *Attn, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, self.tol) |*l, t| l.* = if (std.math.isFinite(t)) t else std.math.floatMax(f64);
    }

    /// Gate 4 (stress rows only): q = 0 gives every head of a key head the same row, so one replay serves its group.
    pub fn simulate(self: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        std.debug.assert(self.fill == .stress);
        const cl = self.call();
        try fill_.Pool.run(self.rig.a, .{ .call = &cl, .want = out, .tol = &.{}, .sim = .{ .acc = acc, .order = order } }, self.shape.rows * kv);
    }
};
