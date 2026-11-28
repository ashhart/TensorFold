//! tf_hy4_rotary_table (fp32 phases, each pair's axis picking its position, bf16 cos and sin) and tf_hy4_rotate.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const pj = @import("metal_hy4_project.zig");

const fl = h.fl;
const c = contract.spec;
const g = contract.gates;
const Range = fl.Range;
const bf = fl.bf;
const rope = h.hy.rope;
const mw = h.hy.mla_weights;
const pairs = rope.pairs;
const width = 2 * pairs;
const huge = std.math.floatMax(f64);

/// Rows: the call's rows; n: the pairs; k: row 0's time position (the height and width positions trail it).
fn tableSpec(label: []const u8) c.Spec {
    const served = [_]c.Shape{ .{ .rows = h.window_rows, .n = pairs, .k = 1000 }, .{ .rows = h.decode_rows, .n = pairs, .k = 1_030_000 }, .{ .rows = h.prompt_rows, .n = pairs, .k = 0 }, .{ .rows = h.window_rows, .n = pairs, .k = 262_000 } };
    return .{ .name = label, .accum = .none, .depth = fl.noSums, .out = .bf16, .served = &served };
}

pub const Table = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    layer: h.hy.config.Attention,
    ctl: ?*const pj.Control,
    params: rope.Parameters,
    live: rope.Parameters = undefined,
    label: [224]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    pos: []i32 = &.{},
    ins: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, layer: h.hy.config.Attention, ctl: ?*const pj.Control) !Table {
        const cfg = try h.text(a);
        defer cfg.deinit();
        const params = try cfg.value.text_config.rotary(layer);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pj.pipe(a, device, "tf_hy4_rotary_table", ctl), .layer = layer, .ctl = ctl, .params = params };
    }
    pub fn deinit(t: *Table) void {
        t.a.free(t.pos);
        t.rig.deinit();
        t.pipe.deinit();
    }
    pub fn kernel(t: *Table) g.Kernel {
        const variant = if (t.layer == .sliding_attention) "sliding layers (theta 1e4)" else "sparse layers (theta 1e7)";
        t.name = pj.name(&t.label, "tf_hy4_rotary_table", variant, t.ctl);
        return .{ .spec = tableSpec(t.name), .ctx = t, .vt = &table_vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const table_vt = g.hooks(Table);

    pub fn prepare(t: *Table, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != pairs or shape.k + shape.rows > 1 << 20) return false;
        t.rig.reset();
        t.a.free(t.pos);
        t.rows = shape.rows;
        t.live = t.params;
        t.pos = try t.a.alloc(i32, t.rows * 3);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (0..t.rows) |r| {
            const time: u32 = @intCast(shape.k + r);
            t.pos[r * 3] = @intCast(time);
            for (1..3) |axis| t.pos[r * 3 + axis] = @intCast(time - rng.uintLessThan(u32, @min(time, @as(u32, 4095)) + 1));
        }
        if (fill == .nonfinite) for (fl.planted, 1..) |v, pair| {
            t.live.frequencies[pair] = v;
        };
        t.ins = try t.rig.input(std.mem.sliceAsBytes(t.pos));
        t.out = try t.rig.output(t.rows * width * 2);
        return true;
    }

    pub fn run(t: *Table) !void {
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.ins.buf, t.ins.off, 0);
        p.e.setBuffer(t.out.buf, t.out.off, 1);
        p.e.setValue(t.live, 2);
        p.e.dispatchThreads(mtl.Size.of(pairs, t.rows, 1), mtl.Size.of(pairs, 1, 1));
        try t.rig.finish(p);
    }

    pub fn count(t: *Table) usize {
        return t.rows * width;
    }

    /// Output i's angle: the fp32 product of its axis's position (exact in fp32) and the pair's fp32 frequency.
    fn phase(t: *const Table, i: usize) f64 {
        const r = i / width;
        const pair = (i % width) / 2;
        const p: f64 = @floatFromInt(t.pos[r * 3 + t.live.axes[pair]]);
        return @as(f32, @floatCast(p * t.live.frequencies[pair]));
    }
    fn trig(t: *const Table, i: usize) f64 {
        return if (i % 2 == 0) @cos(t.phase(i)) else @sin(t.phase(i));
    }

    pub fn reference(t: *Table, want: []f64, mag: []f64) !void {
        for (want, 0..) |*w, i| w.* = bf(t.trig(i));
        @memset(mag, 0);
    }

    /// precise::cos and precise::sin within 4 ulps of the true value at the fp32 angle, then the bf16 store.
    pub fn intervals(t: *Table, low: []f64, high: []f64) !void {
        for (low, high, 0..) |*l, *hi, i| {
            const v = t.trig(i);
            const r = if (std.math.isFinite(v)) pj.ulps(Range.point(v), 4).bf16() else Range{ .lo = -huge, .hi = huge };
            l.* = r.lo;
            hi.* = r.hi;
        }
    }

    pub fn bounds(t: *Table, want: []const f64, _: []const f64, allow: []f64) !void {
        try fl.allowFrom(t.a, t, want, allow);
    }
};

const Form = enum { mla, index };

/// Strides and first rotary words: the query's after 192, the absorbed query's after 512, the indexer's last 64 of 128.
const Geo = struct { in_stride: usize, in_off: usize, out_stride: usize, out_off: usize };
fn geo(f: Form) Geo {
    const iw = h.hy.index_recall.width;
    return switch (f) {
        .mla => .{ .in_stride = mw.query_stride, .in_off = mw.nope, .out_stride = mw.absorbed_stride, .out_off = mw.latent },
        .index => .{ .in_stride = iw, .in_off = iw - width, .out_stride = iw, .out_off = iw - width },
    };
}

/// Rows: the call's rows; n: its heads (sparse 32, sliding 64; the indexer's 16, its group keys 1 a group row).
fn rotateSpec(label: []const u8, f: Form) c.Spec {
    const mla = [_]c.Shape{ .{ .rows = h.window_rows, .n = 32, .k = 0 }, .{ .rows = h.decode_rows, .n = 32, .k = 0 }, .{ .rows = h.prompt_rows, .n = 32, .k = 0 }, .{ .rows = h.window_rows, .n = 64, .k = 0 }, .{ .rows = h.decode_rows, .n = 64, .k = 0 }, .{ .rows = h.prompt_rows, .n = 64, .k = 0 } };
    const ih = h.hy.index_recall.heads;
    const index = [_]c.Shape{ .{ .rows = h.window_rows, .n = ih, .k = 0 }, .{ .rows = h.decode_rows, .n = ih, .k = 0 }, .{ .rows = h.prompt_rows, .n = ih, .k = 0 }, .{ .rows = 1, .n = 1, .k = 0 }, .{ .rows = 3, .n = 1, .k = 0 }, .{ .rows = h.prompt_rows / 4 + 1, .n = 1, .k = 0 } };
    return .{ .name = label, .accum = .none, .depth = fl.noSums, .out = .bf16, .in_place = f == .index, .served = if (f == .mla) &mla else &index };
}

pub const Rotate = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    form: Form,
    ctl: ?*const pj.Control,
    label: [224]u8 = undefined,
    name: []const u8 = "",
    rows: usize = 0,
    heads: usize = 0,
    x: []u16 = &.{},
    tab: []u16 = &.{},
    shown: std.ArrayList(u8) = .empty,
    xin: rig_.Ref = undefined,
    tin: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, form: Form, ctl: ?*const pj.Control) !Rotate {
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try pj.pipe(a, device, "tf_hy4_rotate", ctl), .form = form, .ctl = ctl };
    }
    pub fn deinit(t: *Rotate) void {
        t.free();
        t.shown.deinit(t.a);
        t.rig.deinit();
        t.pipe.deinit();
    }
    fn free(t: *Rotate) void {
        t.a.free(t.x);
        t.a.free(t.tab);
        t.x = &.{};
        t.tab = &.{};
    }
    pub fn kernel(t: *Rotate) g.Kernel {
        const variant = if (t.form == .mla) "mla query (stride 256 to the absorbed query's 576)" else "indexer in place (stride 128)";
        t.name = pj.name(&t.label, "tf_hy4_rotate", variant, t.ctl);
        return .{ .spec = rotateSpec(t.name, t.form), .ctx = t, .vt = &rotate_vt, .control = if (t.ctl) |x| x.gate else null };
    }
    const rotate_vt = blk: {
        var v = g.hooks(Rotate);
        v.output = output;
        v.intact = intact;
        v.overlap = overlap;
        break :blk v;
    };
    fn of(ctx: *anyopaque) *Rotate {
        return @ptrCast(@alignCast(ctx));
    }
    fn ends(t: *const Rotate) usize {
        return t.rows * t.heads;
    }
    fn outBytes(t: *Rotate) []u8 {
        return t.rig.bytes(t.out, t.ends() * geo(t.form).out_stride * 2);
    }

    pub fn prepare(t: *Rotate, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n > 64) return false;
        t.rig.reset();
        t.free();
        t.rows = shape.rows;
        t.heads = shape.n;
        const q = geo(t.form);
        t.x = try t.a.alloc(u16, t.ends() * q.in_stride);
        t.tab = try t.a.alloc(u16, t.rows * width);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        pj.uniform(rng, t.x, 3);
        pj.angles(rng, t.tab);
        if (fill == .nonfinite) {
            // NaN in row 0's a, +Inf in row 1's b, -Inf in row 2's a, both infinities in row 3's pair 5
            const at = [_][2]usize{ .{ 0, 0 }, .{ 1, 3 }, .{ 2, 6 }, .{ 3, 10 }, .{ 3, 11 } };
            const vals = [_]f32{ fl.nan32, fl.inf32, -fl.inf32, fl.inf32, -fl.inf32 };
            for (at, vals) |p, v| if (p[0] < t.rows) {
                t.x[(p[0] * t.heads + p[0] % t.heads) * q.in_stride + q.in_off + p[1]] = c.toBf16(v);
            };
        }
        t.tin = try t.rig.input(std.mem.sliceAsBytes(t.tab));
        if (t.form == .mla) {
            t.xin = try t.rig.input(std.mem.sliceAsBytes(t.x));
            t.out = try t.rig.output(t.ends() * q.out_stride * 2);
        } else {
            t.out = try t.rig.output(t.x.len * 2);
            t.xin = t.out;
        }
        return true;
    }

    pub fn run(t: *Rotate) !void {
        const q = geo(t.form);
        if (t.form == .index) @memcpy(t.outBytes(), std.mem.sliceAsBytes(t.x));
        const p = t.rig.begin();
        p.e.setPipeline(t.pipe);
        p.e.setBuffer(t.xin.buf, t.xin.off + q.in_off * 2, 0);
        p.e.setBuffer(t.tin.buf, t.tin.off, 1);
        p.e.setBuffer(t.out.buf, t.out.off + q.out_off * 2, 2);
        p.e.setValue(rope.View{ .rows = @intCast(t.rows), .heads = @intCast(t.heads), .input_stride = @intCast(q.in_stride), .output_stride = @intCast(q.out_stride) }, 3);
        p.e.dispatchThreads(mtl.Size.of(pairs, t.heads, t.rows), mtl.Size.of(pairs, 1, 1));
        try t.rig.finish(p);
    }

    /// The rotated elements only, head by head.
    fn output(ctx: *anyopaque) anyerror![]const u8 {
        const t = of(ctx);
        const q = geo(t.form);
        t.shown.clearRetainingCapacity();
        const all = t.outBytes();
        for (0..t.ends()) |rh| try t.shown.appendSlice(t.a, all[(rh * q.out_stride + q.out_off) * 2 ..][0 .. width * 2]);
        return t.shown.items;
    }

    /// Guards intact, and every element a run does not rotate as it was: unwritten (MLA) or the input's (in place).
    fn intact(ctx: *anyopaque) anyerror!bool {
        const t = of(ctx);
        if (!try t.rig.intact()) return false;
        const q = geo(t.form);
        const all = std.mem.bytesAsSlice(u16, t.outBytes());
        for (0..t.ends()) |rh| for (0..q.out_stride) |e| {
            if (e >= q.out_off and e < q.out_off + width) continue;
            const was: u16 = if (t.form == .mla) 0xffff else t.x[rh * q.in_stride + e];
            if (all[rh * q.out_stride + e] != was) return false;
        };
        return true;
    }

    fn overlap(ctx: *anyopaque) bool {
        const t = of(ctx);
        return t.form == .index or t.rig.overlap();
    }

    pub fn count(t: *Rotate) usize {
        return t.ends() * width;
    }

    /// Output i before its fp32 add's rounding: a pair of the input times the table's row.
    fn exact(t: *const Rotate, i: usize) f64 {
        const q = geo(t.form);
        const rh = i / width;
        const col = (i % width) & ~@as(usize, 1);
        const at = rh * q.in_stride + q.in_off + col;
        const tb = (rh / t.heads) * width + col;
        const v = pj.turn(c.fromBf16(t.x[at]), c.fromBf16(t.x[at + 1]), c.fromBf16(t.tab[tb]), c.fromBf16(t.tab[tb + 1]));
        return v[i % 2];
    }

    pub fn reference(t: *Rotate, want: []f64, mag: []f64) !void {
        for (want, 0..) |*w, i| w.* = bf(t.exact(i));
        @memset(mag, 0);
    }

    pub fn intervals(t: *Rotate, low: []f64, high: []f64) !void {
        for (low, high, 0..) |*l, *hi, i| {
            const r = pj.turned(t.exact(i));
            l.* = r.lo;
            hi.* = r.hi;
        }
    }

    pub fn bounds(t: *Rotate, want: []const f64, _: []const f64, allow: []f64) !void {
        try fl.allowFrom(t.a, t, want, allow);
    }
};

pub fn collect(s: *pj.Suite, only: ?[]const u8) !void {
    var buf: [224]u8 = undefined;
    const layers = [_]h.hy.config.Attention{ .sliding_attention, .deepseek_sparse_attention };
    for (layers) |layer| if (fl.wanted(only, pj.name(&buf, "tf_hy4_rotary_table", "", null))) {
        try s.tables.add(s.a, s.kernels, try Table.init(s.a, s.device, s.queue, layer, null));
    };
    if (fl.wanted(only, pj.name(&buf, "tf_hy4_rotary_table", "", null)))
        try s.tables.add(s.a, s.kernels, try Table.init(s.a, s.device, s.queue, .sliding_attention, pj.control("tf_hy4_rotary_table", "time-axis")));
    if (!fl.wanted(only, pj.name(&buf, "tf_hy4_rotate", "", null))) return;
    for ([_]Form{ .mla, .index }) |f| try s.turns.add(s.a, s.kernels, try Rotate.init(s.a, s.device, s.queue, f, null));
    try s.turns.add(s.a, s.kernels, try Rotate.init(s.a, s.device, s.queue, .mla, pj.control("tf_hy4_rotate", "turned-back")));
}
