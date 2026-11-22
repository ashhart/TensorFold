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
const experts = tf.inkling2_metal.experts;
const Store = experts.Store;
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

// A batch past the grouped threshold, so one dispatch takes the grouped path while its rows replay one at a time.
const big_rows = experts.group_min_rows + 8;
// A decode window past the dedup threshold: tiles of up to eight pairs per expert, replayed one row at a time.
const small_rows = 12;

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
        p.store = try Store.loadWith(p.rig.a, p.rig.device, &cp, &cfg, @intCast(@max(shape.rows, big_rows)), .{
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

    /// One grouped dispatch of `n` rows against the same rows dispatched one at a time: their bytes are equal.
    fn agree(p: *Probe, s: *const Store, n: usize, x: []const u16, ids: []const u32, weights: []const f32, what: []const u8) !?[]const u8 {
        const row_bytes = p.shape.k * @sizeOf(u16);
        const input = try p.rig.input(std.mem.sliceAsBytes(x));
        const route_ids = try p.rig.input(std.mem.sliceAsBytes(ids));
        const route_weights = try p.rig.input(std.mem.sliceAsBytes(weights));
        const batch = try p.rig.scratch(n * row_bytes);
        const split = try p.rig.scratch(n * row_bytes);
        @memset(p.rig.bytes(batch, n * row_bytes), 0xff);
        @memset(p.rig.bytes(split, n * row_bytes), 0xff);
        try s.run(p.rig.queue, 0, u.view(input), u.view(route_ids), u.view(route_weights), u.view(batch), @intCast(n));
        const pass = p.rig.begin();
        for (0..n) |i| {
            s.encode(pass.e, 0, shifted(input, i * row_bytes), shifted(route_ids, i * topk * 4), shifted(route_weights, i * topk * 4), shifted(split, i * row_bytes), 1) catch |err| {
                pass.e.end();
                pass.pool.pop();
                return err;
            };
        }
        try p.rig.finish(pass);
        const want = p.rig.bytes(split, n * row_bytes);
        const got = p.rig.bytes(batch, n * row_bytes);
        const at = std.mem.indexOfDiff(u8, want, got) orelse return null;
        return try std.fmt.bufPrint(&p.msg, "grouped batch of {d} rows ({s}) differs from one-row dispatch at row {d}, byte {d}", .{ n, what, at / row_bytes, at % row_bytes });
    }

    /// The prepared rows repeated to `big_rows` (nonfinite inputs included), or with every pair on one expert.
    fn checkGrouped(p: *Probe, n: usize, skew: bool) !?[]const u8 {
        const a = p.arena.allocator();
        const k = p.shape.k;
        const x = try a.alloc(u16, n * k);
        const ids = try a.alloc(u32, n * topk);
        const weights = try a.alloc(f32, n * topk);
        for (0..n) |r| {
            const from = r % p.shape.rows;
            @memcpy(x[r * k ..][0..k], p.x[from * k ..][0..k]);
            @memcpy(ids[r * topk ..][0..topk], p.ids[from * topk ..][0..topk]);
            @memcpy(weights[r * topk ..][0..topk], p.weights[from * topk ..][0..topk]);
        }
        if (skew) for (ids, weights, 0..) |*id, *w, i| {
            id.* = 1;
            w.* = @as(f32, @floatFromInt(1 + i % 7)) / 13;
        };
        return p.agree(&p.store.?, n, x, ids, weights, if (skew) "every pair on one expert" else "prepared rows repeated");
    }

    /// Dense random weights and BF16 activations make FP32 sums inexact, so the chain order shows in the bytes.
    fn checkDense(p: *Probe, n: usize, one: bool) !?[]const u8 {
        const a = p.arena.allocator();
        var cp = ck.Checkpoint.init(a);
        defer cp.deinit();
        try fixture.dense(p.rig.a, &p.rig, &cp, p.format, 0x5eed, p.shape);
        const cfg = fixture.configuration(p.shape);
        var dense = try Store.loadWith(p.rig.a, p.rig.device, &cp, &cfg, big_rows, .{ .source = p.source orelse sources.inkling2_experts });
        defer dense.deinit();
        var prng = std.Random.DefaultPrng.init(0xd0d0);
        const rng = prng.random();
        const k = p.shape.k;
        const x = try a.alloc(u16, n * k);
        for (x) |*v| v.* = u.word((rng.float(f64) - 0.5) * 6);
        const ids = try a.alloc(u32, n * topk);
        const weights = try a.alloc(f32, n * topk);
        for (0..n) |r| {
            var order: [fixture.experts]u32 = undefined;
            for (&order, 0..) |*v, i| v.* = @intCast(i);
            rng.shuffle(u32, &order);
            for (0..topk) |slot| {
                ids[r * topk + slot] = if (one) 3 else order[slot];
                weights[r * topk + slot] = 0.1 + rng.float(f32);
            }
        }
        return p.agree(&dense, n, x, ids, weights, if (one) "dense, every pair on one expert" else "dense random routes");
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
            for ([_]usize{ big_rows, small_rows }) |n| {
                if (try p.checkGrouped(n, false)) |reason| return reason;
                if (try p.checkGrouped(n, true)) |reason| return reason;
                if (try p.checkDense(n, false)) |reason| return reason;
                if (try p.checkDense(n, true)) |reason| return reason;
            }
        }
        if (gate == .nonfinite) {
            for ([_]usize{ big_rows, small_rows }) |n| if (try p.checkGrouped(n, false)) |reason| return reason;
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
        .from = "i2x_bf2((low * scale) * factor)",
        .to = "i2x_bf2((high * scale) * factor)",
    },
    .{
        .format = .nvfp4,
        .gate = .shapes,
        .label = "control: inkling2 experts NVFP4 skips decoded-weight BF16 rounding",
        .from = "i2x_bf2((low * scale) * factor)",
        .to = "(low * scale) * factor",
    },
    .{
        .format = .mxfp8,
        .gate = .shapes,
        .label = "control: inkling2 experts MXFP8 skips projection BF16 rounding before SwiGLU",
        .from = "const float2 v = i2x_bf2(accum[r]);",
        .to = "const float2 v = accum[r];",
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
        .from = "      dst[0] = bfloat(accum[r].x);\n      dst[1] = bfloat(accum[r].y);",
        .to = "      dst[0] = bfloat(accum[r].y);\n      dst[1] = bfloat(accum[r].x);",
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
        .gate = .shapes,
        .label = "control: inkling2 experts MMA NVFP4 decodes the high nibble for the low weight",
        .from = "bfloat((e.x * sc) * outer)",
        .to = "bfloat((e.y * sc) * outer)",
    },
    .{
        .format = .nvfp4,
        .gate = .shapes,
        .label = "control: inkling2 experts grouped tiles ignore their live pair count",
        .from = "min(a.tile, count - at)",
        .to = "a.tile",
    },
    .{
        .format = .mxfp8,
        .gate = .shapes,
        .label = "control: inkling2 experts grouped plan keeps only four tiles of an expert",
        .from = "for (uint at = 0u; at < count; at += a.tile)",
        .to = "for (uint at = 0u; at < min(count, 4u * a.tile); at += a.tile)",
    },
    .{
        .format = .mxfp8,
        .gate = .nonfinite,
        .label = "control: inkling2 experts MMA MXFP8 treats UE8M0 byte zero as a zero scale",
        .from = "select(1.0f, 0.5f, raw.code == 0u)",
        .to = "1.0f",
    },
    .{
        .format = .mxfp8,
        .gate = .shapes,
        .label = "control: inkling2 experts MMA MXFP8 down writes a pair's two columns swapped",
        .from = "out[ulong(order[tile.y + p0]) * a.out_width + col] = bfloat(r0);",
        .to = "out[ulong(order[tile.y + p0]) * a.out_width + col] = bfloat(r1);",
    },
    .{
        .format = .nvfp4,
        .gate = .shapes,
        .label = "control: inkling2 experts MMA NVFP4 rounds lane sums to BF16",
        .from = "acc[j][si * per_stage + i].thread_elements()[1] += part[j].thread_elements()[1];",
        .to = "acc[j][si * per_stage + i].thread_elements()[1] = float(bfloat(acc[j][si * per_stage + i].thread_elements()[1] + part[j].thread_elements()[1]));",
    },
    .{
        .format = .nvfp4,
        .gate = .shapes,
        .label = "control: inkling2 experts MMA NVFP4 carries the lane sum through the block's MMAs",
        .from = "part[j] = i2m_frag(0.0f);",
        .to = "{ part[j] = acc[j][si * per_stage + i]; acc[j][si * per_stage + i] = i2m_frag(0.0f); }",
    },
    .{
        .format = .nvfp4,
        .gate = .shapes,
        .label = "control: inkling2 experts MMA NVFP4 sums a block's K octets in reverse",
        .from = "const uint ss = i * bsteps + s;",
        .to = "const uint ss = i * bsteps + (bsteps - 1u - s);",
    },
    .{
        .format = .nvfp4,
        .gate = .shapes,
        .label = "control: inkling2 experts MMA NVFP4 combines the lane-tree passes in a different order",
        .from = "for (uint s = I2M_NP / 2u; s != 0u; s >>= 1u)",
        .to = "for (uint s = 1u; s < I2M_NP; s <<= 1u)",
    },
    .{
        .format = .mxfp8,
        .gate = .shapes,
        .label = "control: inkling2 experts MMA MXFP8 combines the chains of a pass in a different order",
        .from = "for (uint s = I2M_NC / 2u; s != 0u; s >>= 1u) {",
        .to = "for (uint s = 1u; s < I2M_NC; s <<= 1u) {",
    },
    .{
        .format = .nvfp4,
        .gate = .shapes,
        .label = "control: inkling2 experts MMA NVFP4 runs one pair tile of a threadgroup's simdgroup",
        .from = "const uint active = min(PT, tiles_live > pg * PT ? tiles_live - pg * PT : 0u);",
        .to = "const uint active = min(PT, 1u);",
    },
    .{
        .format = .mxfp8,
        .gate = .shapes,
        .label = "control: inkling2 experts MMA MXFP8 SwiGLU swaps gate and up",
        .from = "r0 = i2x_swiglu(r0, up0, a.gate_cap, a.up_cap);",
        .to = "r0 = i2x_swiglu(up0, r0, a.gate_cap, a.up_cap);",
    },
    .{
        .format = .nvfp4,
        .gate = .sums,
        .label = "control: inkling2 experts NVFP4 rounds lane sums to BF16",
        .from = "accum[r] += part[r];",
        .to = "accum[r] = i2x_bf2(accum[r] + part[r]);",
    },
    .{
        .format = .mxfp8,
        .gate = .sums,
        .label = "control: inkling2 experts MXFP8 rounds lane sums to BF16",
        .from = "accum[r] += part[r];",
        .to = "accum[r] = i2x_bf2(accum[r] + part[r]);",
    },
    .{
        .format = .nvfp4,
        .gate = .shapes,
        .label = "control: inkling2 experts dedup NVFP4 multiplies a tile's low weights by its first pair's activations",
        .from = "part[r] += w_low * float(xs[r][block * 16u + 2u * i]);",
        .to = "part[r] += w_low * float(xs[0][block * 16u + 2u * i]);",
    },
    .{
        .format = .mxfp8,
        .gate = .shapes,
        .label = "control: inkling2 experts dedup MXFP8 multiplies a tile's weights by its first pair's activations",
        .from = "part[r] += decoded * float(xs[r][block * 32u + i]);",
        .to = "part[r] += decoded * float(xs[0][block * 32u + i]);",
    },
    .{
        .format = .nvfp4,
        .gate = .shapes,
        .label = "control: inkling2 experts dedup NVFP4 reads every pair's activations from the tile's first pair",
        .from = "xs[r] = x + ulong(pairs[r] / a.src_div) * a.k;",
        .to = "xs[r] = x + ulong(pairs[0] / a.src_div) * a.k;",
    },
    .{
        .format = .mxfp8,
        .gate = .shapes,
        .label = "control: inkling2 experts dedup MXFP8 stores every pair of a tile on the first pair's row",
        .from = "device bfloat* dst = out + ulong(pairs[r]) * a.out_width + col;",
        .to = "device bfloat* dst = out + ulong(pairs[0]) * a.out_width + col;",
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
