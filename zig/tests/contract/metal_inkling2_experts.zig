//! Device contracts for both original Inkling 2 Mini expert formats through the Store used by the graph.
//! Check both quantizers' codes, scales and row maxima, gamma-weighted BF16 SwiGLU, each BF16 down row,
//! and the final FP32 route-slot reduction. The sparse fixture has the real projection widths and top-k.
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
const max_k = 2304;

// rows, intermediate output width, hidden input width. K-group tails cross the 32-lane accumulation tile.
const served = [_]c.Shape{
    .{ .rows = 8, .n = 1152, .k = 2304 },
    .{ .rows = 1, .n = 1152, .k = 2304 },
    .{ .rows = 1, .n = 32, .k = 64 },
    .{ .rows = 3, .n = 544, .k = 1056 },
    .{ .rows = 17, .n = 1152, .k = 2304 },
};

const rule = "NaN or infinity in an activation poisons its active rows; invalid routes are zero; UE8M0 byte 255 is NaN";

const Quantized = struct {
    values: [max_k]u8 = undefined,
    scales: [max_k / 16]u8 = undefined,
    amax: f64 = undefined,

    fn quant(q: *Quantized, format: Format, x: []const u16) void {
        q.amax = math.quant(format, x, q.values[0..x.len], q.scales[0 .. x.len / format.block()]);
    }
};

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
            .shared_scratch = true,
        });
        return true;
    }

    pub fn count(p: *Probe) usize {
        return p.shape.rows * p.shape.k;
    }

    pub fn run(p: *Probe) !void {
        const s = &p.store.?;
        @memset(s.scratch.buffer.contents()[0..s.scratchBytes()], 0xff);
        @memset(p.rig.bytes(p.out, p.count() * 2), 0xff);
        try s.run(p.rig.queue, 0, u.view(p.input), u.view(p.route_ids), u.view(p.route_weights), u.view(p.out), @intCast(p.shape.rows));
    }

    fn dot(p: *const Probe, q: *const Quantized, gate_up: bool, index: usize, expert: usize, acc: c.Accum, order: g.Order) f64 {
        const f = p.format;
        const block = f.block();
        var buf: [4]fixture.Index = undefined;
        const support = fixture.support(f, p.fill, gate_up, index, expert, if (gate_up) p.shape.k else p.shape.n, &buf);
        var terms: [4]f64 = undefined;
        for (support, 0..) |term, i| {
            terms[i] = math.dotTerm(f, q.values[term.at], q.scales[term.at / block], term.code, term.scale);
        }
        const raw = if (support.len == 1) terms[0] else sumStress(terms, acc, order);
        return math.dotOutput(f, raw, q.amax, fixture.wAmax(expert), fixture.den);
    }

    fn activation(p: *const Probe, q: *const Quantized, pair: usize, expert: usize, col: usize, acc: c.Accum, order: g.Order) f64 {
        const gate = p.dot(q, true, 2 * col, expert, acc, order);
        const up = p.dot(q, true, 2 * col + 1, expert, acc, order);
        return math.swiglu(gate, up, p.weights[pair], fixture.gate_cap, fixture.up_cap);
    }

    fn oracle(p: *Probe, out: []f64, acc: c.Accum, order: g.Order) void {
        const k = p.shape.k;
        const n = p.shape.n;
        @memset(out, 0);
        var qi: Quantized = .{};
        var qa: Quantized = .{};
        var mid: [max_k]u16 = undefined;
        for (0..p.shape.rows) |row| {
            qi.quant(p.format, p.x[row * k ..][0..k]);
            const dst = out[row * k ..][0..k];
            for (0..topk) |j| {
                const pair = row * topk + j;
                const expert = p.ids[pair];
                if (expert >= fixture.experts) continue;
                for (mid[0..n], 0..) |*w, col| w.* = u.word(p.activation(&qi, pair, expert, col, acc, order));
                qa.quant(p.format, mid[0..n]);
                for (dst, 0..) |*v, col| v.* += p.dot(&qa, false, col, expert, acc, order);
            }
            for (dst) |*v| v.* = u.bf(v.*);
        }
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        p.oracle(want, .none, .sequential);
        for (want, mag) |v, *m| m.* = @abs(v);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        p.oracle(out, acc, order);
    }

    fn quantCheck(p: *Probe, q: *const Quantized, stage: usize, row: usize, width: usize) !?[]const u8 {
        const s = &p.store.?;
        const scratch = s.scratch;
        const qr = scratch.quants[stage];
        const bytes = scratch.buffer.contents();
        const blocks = width / p.format.block();
        const got_a = c.read(.f32, bytes[qr.amax .. qr.amax + (row + 1) * 4], row);
        if (c.classOf(got_a) != c.classOf(q.amax) or (std.math.isFinite(q.amax) and got_a != q.amax))
            return try std.fmt.bufPrint(&p.msg, "quantizer {d} row {d}: amax {e} != {e}", .{ stage, row, got_a, q.amax });
        const sc = bytes[qr.scales + row * blocks ..][0..blocks];
        for (sc, 0..) |got, b| if (got != q.scales[b]) {
            return try std.fmt.bufPrint(&p.msg, "quantizer {d} row {d}, block {d}: scale {x} != {x}", .{ stage, row, b, got, q.scales[b] });
        };
        const nv = p.format == .nvfp4;
        const pitch = if (nv) width / 2 else width;
        const codes = bytes[qr.values + row * pitch ..][0..pitch];
        for (0..width) |i| {
            const sh: u3 = @intCast(4 * (i % 2));
            const got = if (nv) (codes[i / 2] >> sh) & 15 else codes[i];
            if (got != q.values[i])
                return try std.fmt.bufPrint(&p.msg, "quantizer {d} row {d}, column {d}: code {x} != {x}", .{ stage, row, i, got, q.values[i] });
        }
        return null;
    }

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const s = &p.store.?;
        const k = p.shape.k;
        const n = p.shape.n;
        const bytes = s.scratch.buffer.contents()[0..s.scratchBytes()];
        const gpu_mid = bytes[s.scratch.stage..][0 .. p.shape.rows * topk * n * 2];
        const gpu_down = bytes[s.scratch.down..][0 .. p.shape.rows * topk * k * 2];
        const gpu_out = p.rig.bytes(p.out, p.count() * 2);
        var qi: Quantized = .{};
        var qa: Quantized = .{};
        var mid: [max_k]u16 = undefined;
        var total: [max_k]f64 = undefined;
        var magnitude: [max_k]f64 = undefined;
        for (0..p.shape.rows) |row| {
            qi.quant(p.format, p.x[row * k ..][0..k]);
            if (try p.quantCheck(&qi, 0, row, k)) |why| return why;
            @memset(total[0..k], 0);
            @memset(magnitude[0..k], 0);
            for (0..topk) |j| {
                const pair = row * topk + j;
                const expert = p.ids[pair];
                const active = expert < fixture.experts;
                for (mid[0..n], 0..) |*w, col| {
                    w.* = std.mem.readInt(u16, gpu_mid[2 * (pair * n + col) ..][0..2], .little);
                    const got = u.val(w.*);
                    const ref = if (active) p.activation(&qi, pair, expert, col, .none, .sequential) else 0;
                    // Exact sparse dots; allow 32 FP32 units for Metal vs host exp/tanh and product roundings, then BF16.
                    const error_band = 32 * 0x1p-24 * @abs(ref);
                    if (!bounded(got, ref, error_band))
                        return try std.fmt.bufPrint(&p.msg, "pair {d} SwiGLU column {d}: {e} outside BF16 band around {e}", .{ pair, col, got, ref });
                }
                // Midpoint uncertainty ends here: independently check quantization of the verified BF16 activations.
                qa.quant(p.format, mid[0..n]);
                if (try p.quantCheck(&qa, 1, pair, n)) |why| return why;
                for (0..k) |col| {
                    const got = u.outBf(gpu_down, pair * k + col);
                    const want = if (active) p.dot(&qa, false, col, expert, .none, .sequential) else 0;
                    if (c.classOf(got) != c.classOf(want) or (std.math.isFinite(want) and got != want))
                        return try std.fmt.bufPrint(&p.msg, "pair {d} down column {d}: {e} != {e}", .{ pair, col, got, want });
                    total[col] += want;
                    magnitude[col] += @abs(want);
                }
            }
            for (0..k) |col| {
                const got = u.outBf(gpu_out, row * k + col);
                const limit = c.gamma(topk, c.unit(.f32)) * magnitude[col];
                if (!bounded(got, total[col], limit))
                    return try std.fmt.bufPrint(&p.msg, "row {d} combine column {d}: {e} outside BF16 band around {e}", .{ row, col, got, total[col] });
            }
        }
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

fn bounded(got: f64, want: f64, allowance: f64) bool {
    if (c.classOf(got) != c.classOf(want)) return false;
    if (!std.math.isFinite(want)) return true;
    return got >= u.bf(want - allowance) and got <= u.bf(want + allowance);
}

fn depth(shape: c.Shape) usize {
    return 2 * (32 + (std.math.divCeil(usize, shape.k, 16 * 32) catch unreachable) + 5) + topk;
}

const Mutation = struct { format: Format, gate: g.Gate, label: []const u8, from: []const u8, to: []const u8 };
const mutations = [_]Mutation{
    .{ .format = .nvfp4, .gate = .nonfinite, .label = "control: inkling2 experts NVFP4 drops a non-finite row amax",
        .from = "row_amax[row] = bad ? NAN : amax;", .to = "row_amax[row] = bad ? 0.0f : amax;" },
    .{ .format = .nvfp4, .gate = .shapes, .label = "control: inkling2 experts NVFP4 ties choose four",
        .from = "const bool use4 = e4[0] < e6[0];", .to = "const bool use4 = e4[0] <= e6[0];" },
    .{ .format = .nvfp4, .gate = .shapes, .label = "control: inkling2 experts NVFP4 swapped adjacent nibble order",
        .from = "out[b * 8u + j] = uchar(lo | (hi << 4));", .to = "out[b * 8u + j] = uchar(hi | (lo << 4));" },
    .{ .format = .nvfp4, .gate = .shapes, .label = "control: inkling2 experts NVFP4 missing raw BF16 rounding",
        .from = "float4 v = i2x_bf4(accum);", .to = "float4 v = NV ? accum : i2x_bf4(accum);" },
    .{ .format = .mxfp8, .gate = .shapes, .label = "control: inkling2 experts MXFP8 exponent divisor 480",
        .from = "bm * (1.0f / 448.0f)", .to = "bm * (1.0f / 480.0f)" },
    .{ .format = .mxfp8, .gate = .shapes, .label = "control: inkling2 experts omits gamma before SwiGLU rounding",
        .from = "return ((gate * sigmoid) * up) * gamma;", .to = "return (gate * sigmoid) * up;" },
    .{ .format = .mxfp8, .gate = .nonfinite, .label = "control: inkling2 experts MXFP8 treats UE8M0 zero as zero scale",
        .from = "return (xs == 255u || ws == 255u) ? NAN : ldexp(dot, int(xs) + int(ws) - 254);",
        .to = "return (xs == 255u || ws == 255u) ? NAN : (xs == 0u || ws == 0u ? 0.0f : ldexp(dot, int(xs) + int(ws) - 254));" },
    .{ .format = .mxfp8, .gate = .sums, .label = "control: inkling2 experts MXFP8 rounds lane sums to BF16",
        .from = "accum += part;", .to = "accum = i2x_bf4(accum + part);" },
    .{ .format = .nvfp4, .gate = .sums, .label = "control: inkling2 experts NVFP4 rounds lane sums to BF16",
        .from = "accum += part;", .to = "accum = i2x_bf4(accum + part);" },
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
        p.* = .{ .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .arena = .init(s.a),
            .format = format, .label = label, .source = source };
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
