//! Packed expert output contracts: independent BF16 rounding and exact split-dispatch bytes.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const u = @import("inkling2_util.zig");
const rig_ = @import("rig.zig");
const fixture = @import("inkling2_expert_fixture.zig");
const math = @import("inkling2_expert_math.zig");
const ck = tf.checkpoint_metal;
const Store = tf.inkling2_metal.experts.Store;
const c = u.c;
const g = u.g;
const Format = math.Format;
const topk = fixture.topk;
const Range = math.Range;

// The actual Mini projection widths, with single-token and prompt dispatches.
const served = [_]c.Shape{
    .{ .rows = 8, .n = fixture.inter, .k = fixture.hidden },
    .{ .rows = 1, .n = fixture.inter, .k = fixture.hidden },
    .{ .rows = 17, .n = fixture.inter, .k = fixture.hidden },
};

const rule = "dense zero weights propagate nonfinite inputs; a nonfinite active route poisons the sum; inactive slots are ignored";

fn round(v: f64, acc: c.Accum) f64 {
    return switch (acc) {
        .f32 => math.fp32(v),
        .bf16 => u.bf(v),
        .f16 => @as(f16, @floatCast(v)),
        .none => v,
    };
}

fn sumStress(t: [4]f64, acc: c.Accum, order: g.Order) f64 {
    const a = round(t[0], acc);
    const b = round(t[1], acc);
    const d = round(t[2], acc);
    const e = round(t[3], acc);
    return switch (order) {
        .sequential => round(round(round(a + b, acc) + d, acc) + e, acc),
        .pairwise => round(round(a + b, acc) + round(d + e, acc), acc),
        // The fourth term shares the first term's lane (K block 32); the last term added is in lane 1.
        .strided => round(round(round(a + e, acc) + d, acc) + b, acc),
    };
}

const Probe = struct {
    rig: rig_.Rig,
    arena: std.heap.ArenaAllocator,
    format: Format,
    label: []const u8,
    source: ?[]const u8,
    store: ?Store = null,
    shape: c.Shape = undefined,
    fill: g.Fill = undefined,
    x: []u16 = undefined,
    ids: []u32 = undefined,
    weights: []f32 = undefined,
    input: rig_.Ref = undefined,
    route_ids: rig_.Ref = undefined,
    route_weights: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    msg: [240]u8 = undefined,

    fn deinit(p: *Probe) void {
        if (p.store) |*s| s.deinit();
        if (p.source) |text| p.rig.a.free(text);
        p.rig.deinit();
        p.arena.deinit();
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != fixture.inter or shape.k != fixture.hidden) return false;
        if (p.store) |*s| s.deinit();
        p.store = null;
        const a = u.reset(&p.arena, &p.rig);
        p.shape = shape;
        p.fill = fill;
        p.x = try a.alloc(u16, shape.rows * shape.k);
        p.ids = try a.alloc(u32, shape.rows * topk);
        p.weights = try a.alloc(f32, shape.rows * topk);
        fixture.rows(p.format, shape, fill, seed, p.x, p.ids, p.weights);
        p.input = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.route_ids = try p.rig.input(std.mem.sliceAsBytes(p.ids));
        p.route_weights = try p.rig.input(std.mem.sliceAsBytes(p.weights));
        p.out = try p.rig.output(shape.rows * shape.k * 2);
        var cp = ck.Checkpoint.init(a);
        defer cp.deinit();
        try fixture.tensors(p.rig.a, &p.rig, &cp, p.format, fill, shape);
        const cfg = fixture.configuration(shape);
        p.store = try Store.loadWith(p.rig.a, p.rig.device, &cp, &cfg, @intCast(shape.rows), .{
            .source = p.source orelse sources.inkling2_experts,
        });
        return true;
    }

    pub fn count(p: *Probe) usize {
        return p.shape.rows * p.shape.k;
    }

    pub fn run(p: *Probe) !void {
        const s = &p.store.?;
        @memset(p.rig.bytes(p.out, p.count() * 2), 0xff);
        try s.run(p.rig.queue, 0, u.view(p.input), u.view(p.route_ids), u.view(p.route_weights), u.view(p.out), @intCast(p.shape.rows));
    }

    fn factor(p: *const Probe, gate_up: bool, expert: usize) f64 {
        return if (p.format == .nvfp4)
            math.factor(fixture.wAmax(p.fill, gate_up, expert), fixture.denominator(gate_up))
        else
            1;
    }

    fn projectPair(p: *const Probe, x: []const u16, col: usize, expert: usize, outer: f64, band: bool) Range {
        var proj: [2]f64 = undefined;
        for (&proj, 0..) |*dst, i| {
            var taps: [4]fixture.Index = undefined;
            const terms = fixture.support(p.format, p.fill, true, 2 * col + i, expert, x.len, &taps);
            var dot: f64 = 0;
            for (terms) |term| dot += u.val(x[term.at]) * math.weight(p.format, term.code, term.scale, outer);
            dst.* = u.bf(dot);
        }
        return if (band)
            math.activation(proj[0], proj[1], fixture.gate_cap, fixture.up_cap)
        else
            .point(u.bf(math.swiglu(proj[0], proj[1], fixture.gate_cap, fixture.up_cap)));
    }

    fn projectDown(p: *const Probe, mid: []const Range, col: usize, expert: usize, outer: f64, acc: c.Accum, order: g.Order) Range {
        var buf: [4]fixture.Index = undefined;
        const taps = fixture.support(p.format, p.fill, false, col, expert, mid.len, &buf);
        if (taps.len == 1) {
            const t = taps[0];
            return mid[t.at].mul(math.weight(p.format, t.code, t.scale, outer)).bf16();
        }

        // Four exact products test the reduced-precision replays without tying FP64 to the launch schedule.
        var terms: [4]f64 = undefined;
        for (taps, &terms) |t, *v| v.* = mid[t.at].lo * math.weight(p.format, t.code, t.scale, outer);
        const dot = if (acc == .none)
            terms[0] + terms[1] + terms[2] + terms[3]
        else
            sumStress(terms, acc, order);
        return .point(u.bf(dot));
    }

    fn rowReference(p: *const Probe, row: usize, out: []Range, acc: c.Accum, order: g.Order, band: bool) void {
        const x = p.x[row * p.shape.k ..][0..p.shape.k];
        const ids = p.ids[row * topk ..][0..topk];
        const weights = p.weights[row * topk ..][0..topk];
        // The dense zeros in both gate/up rows multiply NaN or infinity at the planted input columns.
        const bad_input = p.fill == .nonfinite and row < 4;
        @memset(out, .point(0));
        var mid: [fixture.inter]Range = undefined;
        // Eager Transformers visits experts in ascending index order, independent of route slot.
        for (0..fixture.experts) |expert| {
            const slot = std.mem.indexOfScalar(u32, ids, @intCast(expert)) orelse continue;
            if (bad_input or std.math.isNan(weights[slot]) or (p.fill == .nonfinite and expert == 8)) {
                @memset(out, .point(std.math.nan(f64)));
                return;
            }
            const up = p.factor(true, expert);
            const down = p.factor(false, expert);
            for (&mid, 0..) |*v, col| v.* = p.projectPair(x, col, expert, up, band);
            for (out, 0..) |*v, col| {
                const projected = p.projectDown(&mid, col, expert, down, acc, order);
                v.* = v.add(projected.mul(weights[slot]));
            }
        }
        for (out) |*v| v.* = v.bf16();
    }

    fn oracle(p: *const Probe, out: []f64, acc: c.Accum, order: g.Order) void {
        var want: [fixture.hidden]Range = undefined;
        for (0..p.shape.rows) |row| {
            p.rowReference(row, &want, acc, order, false);
            for (out[row * p.shape.k ..][0..p.shape.k], want) |*v, w| v.* = w.lo;
        }
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        p.oracle(want, .none, .sequential);
        for (want, mag) |v, *m| m.* = @abs(v);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        p.oracle(out, acc, order);
    }

    fn shifted(ref: rig_.Ref, offset: usize) u.View {
        return .{ .buffer = ref.buf, .offset = ref.off + offset };
    }

    fn checkSplit(p: *Probe, expected: []const u8) !?[]const u8 {
        const s = &p.store.?;
        const out = try p.rig.scratch(expected.len);
        const got = p.rig.bytes(out, expected.len);
        @memset(got, 0xff);
        const row_bytes = p.shape.k * @sizeOf(u16);
        const route_bytes = topk * @sizeOf(u32);
        const pass = p.rig.begin();
        var i = p.shape.rows;
        while (i != 0) {
            i -= 1;
            s.encode(pass.e, 0, shifted(p.input, i * row_bytes), shifted(p.route_ids, i * route_bytes), shifted(p.route_weights, i * route_bytes), shifted(out, i * row_bytes), 1) catch |err| {
                pass.e.end();
                pass.pool.pop();
                return err;
            };
        }
        try p.rig.finish(pass);
        if (std.mem.indexOfDiff(u8, expected, got)) |i_byte|
            return try std.fmt.bufPrint(&p.msg, "reverse one-row dispatch differs from batch at row {d}, byte {d}", .{ i_byte / row_bytes, i_byte % row_bytes });
        return null;
    }

    pub fn check(p: *Probe, gate: g.Gate) !?[]const u8 {
        const got = p.rig.bytes(p.out, p.count() * @sizeOf(u16));
        var want: [fixture.hidden]Range = undefined;
        for (0..p.shape.rows) |row| {
            p.rowReference(row, &want, .none, .sequential, true);
            for (want, 0..) |w, col| {
                const i = row * p.shape.k + col;
                const seen = u.outBf(got, i);
                const class = c.classOf(w.lo);
                if (c.classOf(seen) != class or (class == .finite and !(w.lo <= seen and seen <= w.hi)))
                    return try std.fmt.bufPrint(&p.msg, "row {d} column {d}: {e} outside BF16 [{e}, {e}]", .{ row, col, seen, w.lo, w.hi });
                if (gate == .nonfinite and row == 4 and std.mem.readInt(u16, got[2 * i ..][0..2], .little) != 0)
                    return "the all-inactive token must store positive zero";
            }
        }
        if (gate == .shapes and p.shape.rows > 1) {
            if (try p.checkSplit(got)) |reason| return reason;
        }
        if (!try p.rig.intact()) return "the batch or split dispatch wrote beyond its bound buffer views";
        return null;
    }

    const vt = g.hooks(Probe);

    fn kernel(p: *Probe, control: ?g.Gate) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = control, .spec = .{
            .name = p.label,
            .accum = .f32,
            .depth = depth,
            .out = .bf16,
            .nan = .{ .stated = rule },
            .served = &served,
        } };
    }
};

fn depth(shape: c.Shape) usize {
    return 2 * (32 + (std.math.divCeil(usize, shape.k, 16 * 32) catch unreachable) + 5) + topk;
}

const Mutation = struct { format: Format, gate: g.Gate, label: []const u8, from: []const u8, to: []const u8 };
const mutations = [_]Mutation{
    .{
        .format = .nvfp4,
        .gate = .shapes,
        .label = "control: inkling2 experts NVFP4 misreads the low nibble",
        .from = "i2x_bf4((low * scale) * factor)",
        .to = "i2x_bf4((high * scale) * factor)",
    },
    .{
        .format = .nvfp4,
        .gate = .shapes,
        .label = "control: inkling2 experts NVFP4 skips decoded-weight BF16 rounding",
        .from = "i2x_bf4((low * scale) * factor)",
        .to = "(low * scale) * factor",
    },
    .{
        .format = .mxfp8,
        .gate = .shapes,
        .label = "control: inkling2 experts MXFP8 skips projection BF16 rounding before SwiGLU",
        .from = "const float4 v = i2x_bf4(accum);",
        .to = "const float4 v = accum;",
    },
    .{
        .format = .mxfp8,
        .gate = .shapes,
        .label = "control: inkling2 experts SwiGLU sigmoid sees the capped gate",
        .from = "precise::exp(-gate)",
        .to = "precise::exp(-precise::tanh(precise::divide(gate, gate_cap)) * gate_cap)",
    },
    .{
        .format = .mxfp8,
        .gate = .nonfinite,
        .label = "control: inkling2 experts MXFP8 treats UE8M0 byte zero as a zero scale",
        .from = "max(s, 1u) << 23",
        .to = "s << 23",
    },
    .{
        .format = .mxfp8,
        .gate = .shapes,
        .label = "control: inkling2 experts MXFP8 down projection writes its first two columns swapped",
        .from = "    dst[0] = bfloat(accum.x);\n    dst[1] = bfloat(accum.y);",
        .to = "    dst[0] = bfloat(accum.y);\n    dst[1] = bfloat(accum.x);",
    },
    .{
        .format = .nvfp4,
        .gate = .shapes,
        .label = "control: inkling2 experts combine rounds the FP32 route product to BF16",
        .from = "sum += float(down[first + slot * a.hidden]) * gammas[routes + slot];",
        .to = "sum += float(bfloat(float(down[first + slot * a.hidden]) * gammas[routes + slot]));",
    },
    .{
        .format = .mxfp8,
        .gate = .shapes,
        .label = "control: inkling2 experts combine uses route-slot order instead of expert order",
        .from = "const uint slot = (order >> (4u * rank)) & 0xfu;",
        .to = "const uint slot = rank; if (ids[routes + slot] >= a.experts) continue;",
    },
    .{
        .format = .nvfp4,
        .gate = .sums,
        .label = "control: inkling2 experts NVFP4 rounds lane sums to BF16",
        .from = "    accum += part;",
        .to = "    accum = i2x_bf4(accum + part);",
    },
    .{
        .format = .mxfp8,
        .gate = .sums,
        .label = "control: inkling2 experts MXFP8 rounds lane sums to BF16",
        .from = "    accum += part;",
        .to = "    accum = i2x_bf4(accum + part);",
    },
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    held: std.ArrayList(*Probe) = .empty,

    fn add(s: *Suite, only: ?[]const u8, label: []const u8, format: Format, mutation: ?Mutation) !void {
        if (only) |text| if (std.mem.indexOf(u8, label, text) == null) return;
        const p = try s.a.create(Probe);
        errdefer s.a.destroy(p);
        const source = if (mutation) |m| try u.mutate(s.a, sources.inkling2_experts, m.from, m.to) else null;
        p.* = .{ .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .arena = .init(s.a), .format = format, .label = label, .source = source };
        errdefer p.deinit();
        try s.held.append(s.a, p);
        errdefer _ = s.held.pop();
        try s.kernels.append(s.a, p.kernel(if (mutation) |m| m.gate else null));
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        try s.add(only, "metal inkling2 experts checkpoint NVFP4", .nvfp4, null);
        try s.add(only, "metal inkling2 experts checkpoint MXFP8", .mxfp8, null);
        for (mutations) |m| try s.add(only, m.label, m.format, m);
    }

    pub fn deinit(s: *Suite) void {
        for (s.held.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.held.deinit(s.a);
    }
};
