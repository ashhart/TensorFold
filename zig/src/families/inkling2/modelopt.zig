//! Inkling 2 Mini's routed experts: stored NVFP4/MXFP8 codes, bf16 activations and fp32 routing.
const std = @import("std");
const cuda = @import("cuda");
const experts = @import("experts.zig");
const core = @import("core");

pub const Format = enum(i32) { nvfp4 = 0, mxfp8 = 1 };
pub const max_slots = 32;
const max_experts = 512;

pub const Projection = struct {
    codes: u64,
    scales: u64,
    amax: u64 = 0,
    denominator: u64 = 0,
    n: u32,
    k: u32,
};

/// One physical layer. Loop visits share these device pointers.
pub const Layer = struct {
    format: Format,
    experts: u32,
    w13: Projection,
    w2: Projection,
};

pub const Weight = extern struct {
    codes: u64,
    scales: u64,
    amax: u64,
    denominator: u64,
    experts: i64,
    n: i64,
    k: i64,
    group: i32,
    format: i32,
    row0: i32,
    row_step: i32,
};

pub const Kernels = struct {
    mod: cuda.Module,
    plan: cuda.Function,
    decode: cuda.Function,
    prompt: cuda.Function,
    act: cuda.Function,
    combine: cuda.Function,
    resident: usize,

    pub fn load(d: *const cuda.Driver, sms: usize) !Kernels {
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_modelopt);
        errdefer mod.unload();
        const decode = try mod.function("tf_ink2moe_up");
        return .{
            .mod = mod,
            .plan = try mod.function("tf_ink2moe_plan"),
            .decode = decode,
            .prompt = try mod.function("tf_ink2moe_prompt_up"),
            .act = try mod.function("tf_ink2moe_act"),
            .combine = try mod.function("tf_ink2moe_combine"),
            .resident = @max(1, try decode.occupancy(128, 0)) * sms,
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }
};

fn aligned(n: usize) usize {
    return (n + 255) / 256 * 256;
}

pub const Layout = struct {
    gate: u64,
    up: u64,
    act: u64,
    down: u64,
    items: u64,
    item_count: u64,
    members: u64,
    bytes: usize,
};

/// Reconstructs the stable intermediate addresses inside one scratch allocation.
pub fn layout(base: u64, rows: usize, slots: usize, hidden: usize, intermediate: usize) Layout {
    const pairs = rows * slots;
    const items_max = (pairs + 15) / 16 + @min(pairs, max_experts);
    var at = base;
    const take = struct {
        fn f(p: *u64, bytes: usize) u64 {
            const r = p.*;
            p.* = r + aligned(bytes);
            return r;
        }
    }.f;
    const gate = take(&at, pairs * intermediate * 2);
    const up = take(&at, pairs * intermediate * 2);
    const act = take(&at, pairs * intermediate * 2);
    const down = take(&at, pairs * hidden * 2);
    const items = take(&at, items_max * 12);
    const item_count = take(&at, 4);
    const members = take(&at, pairs * 4);
    return .{ .gate = gate, .up = up, .act = act, .down = down, .items = items, .item_count = item_count, .members = members, .bytes = at - base };
}

/// Gate, up, activation and per-expert outputs, plus the ascending-expert group plan.
pub fn scratchBytes(rows: usize, slots: usize, hidden: usize, intermediate: usize) !usize {
    if (rows == 0 or rows > 2048 or slots == 0 or slots > max_slots) return error.BadExperts;
    return layout(0, rows, slots, hidden, intermediate).bytes;
}

fn i32d(v: usize) i32 {
    return @intCast(v);
}

pub fn weight(p: Projection, l: Layer, row0: i32, row_step: i32) Weight {
    return .{
        .codes = p.codes,
        .scales = p.scales,
        .amax = p.amax,
        .denominator = p.denominator,
        .experts = l.experts,
        .n = p.n,
        .k = p.k,
        .group = if (l.format == .nvfp4) 16 else 32,
        .format = @backingInt(l.format),
        .row0 = row0,
        .row_step = row_step,
    };
}

/// Output columns a decode warp of tf_ink2moe_up owns (modelopt.cu kDecodeTiles n8 tiles); the prompt kernel owns 32.
const decode_columns: usize = 8;

/// `up` (decode only): w is w13 in row order, gate rows to `out` and up rows to `up`.
fn projection(s: cuda.Stream, fn_: cuda.Function, prompt: bool, resident: usize, x: u64, x_stride: usize, slots: usize, w: Weight, items: u64, item_count: u64, members: u64, out: u64, up: u64, item_bound: usize) !void {
    const kg: usize = @intCast(@divExact(w.k, 32));
    const outputs: usize = @intCast(if (w.row_step == 2) @divExact(w.n, 2) else w.n);
    const nb = outputs / if (prompt) @as(usize, 32) else decode_columns;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(i32d(x_stride));
    a.add(i32d(slots));
    a.add(w);
    a.add(i32d(kg));
    a.add(i32d(nb));
    for ([_]u64{ items, item_count, members, out }) |v| a.add(v);
    a.add(i32d(if (up != 0) outputs / 2 else outputs));
    a.add(@as(i32, -1));
    if (!prompt) a.add(up);
    const bound = item_bound * nb;
    const grid = if (prompt) (bound + 3) / 4 else @min((bound + 3) / 4, resident);
    if (grid == 0) return;
    try cuda.launch.launch(fn_, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &a);
}

/// Run one MoE layer. `pick` is row-major int32 expert IDs; routing `weights` are fp32.
pub fn run(k: *const Kernels, s: cuda.Stream, l: Layer, caps: experts.Caps, x: u64, x_stride: usize, pick: u64, weights: u64, rows: usize, slots: usize, out: u64, scratch: u64) !void {
    if (rows == 0 or rows > 2048 or slots == 0 or slots > max_slots or l.experts == 0 or l.experts > max_experts) return error.BadExperts;
    if (l.w13.n % 64 != 0 or l.w13.k % 32 != 0 or l.w2.n % 32 != 0 or l.w2.k % 32 != 0) return error.BadExperts;
    if (l.w13.k != l.w2.n or l.w13.n / 2 != l.w2.k or x_stride < l.w13.k) return error.BadExperts;
    const hidden: usize = l.w2.n;
    const intermediate: usize = l.w2.k;
    const pairs = rows * slots;
    const b = layout(scratch, rows, slots, hidden, intermediate);
    const listed = rows >= 32;
    const tile: usize = if (listed) 64 else 16;
    const item_bound = (pairs + tile - 1) / tile + @min(pairs, @as(usize, l.experts));

    var g: cuda.Args = .{};
    for ([_]u64{ pick, b.items, b.members, b.item_count }) |v| g.add(v);
    for ([_]usize{ rows, slots, l.experts, tile }) |v| g.add(i32d(v));
    try cuda.launch.launch(k.plan, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 32 } }, s, &g);

    const fn_ = if (listed) k.prompt else k.decode;
    if (listed) {
        try projection(s, fn_, true, k.resident, x, x_stride, slots, weight(l.w13, l, 0, 2), b.items, b.item_count, b.members, b.gate, 0, item_bound);
        try projection(s, fn_, true, k.resident, x, x_stride, slots, weight(l.w13, l, 1, 2), b.items, b.item_count, b.members, b.up, 0, item_bound);
    } else {
        try projection(s, fn_, false, k.resident, x, x_stride, slots, weight(l.w13, l, 0, 1), b.items, b.item_count, b.members, b.gate, b.up, item_bound);
    }

    var aa: cuda.Args = .{};
    for ([_]u64{ b.gate, b.up, b.act }) |v| aa.add(v);
    aa.add(i32d(pairs * intermediate));
    aa.add(caps.gate);
    aa.add(caps.up);
    try cuda.launch.launch(k.act, .{ .grid = .{ .x = @intCast((pairs * intermediate + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &aa);
    try projection(s, fn_, listed, k.resident, b.act, intermediate, 0, weight(l.w2, l, 0, 1), b.items, b.item_count, b.members, b.down, 0, item_bound);
    try combine(k, s, pick, weights, b.down, out, rows, hidden, slots, l.experts);
}

/// Weighted sum over per-expert BF16 outputs in ascending expert-ID order (A12), rounding once to BF16.
pub fn combine(k: *const Kernels, s: cuda.Stream, pick: u64, weights: u64, down: u64, out: u64, rows: usize, hidden: usize, slots: usize, expert_count: usize) !void {
    if (rows == 0 or rows > 2048 or slots == 0 or slots > max_slots or expert_count == 0 or expert_count > max_experts) return error.BadExperts;
    var ca: cuda.Args = .{};
    for ([_]u64{ pick, weights, down, out }) |v| ca.add(v);
    for ([_]i32{ i32d(rows), i32d(hidden), i32d(slots), i32d(expert_count) }) |v| ca.add(v);
    const count = rows * hidden;
    try cuda.launch.launch(k.combine, .{ .grid = .{ .x = @intCast((count + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &ca);
}

fn put(e: anytype, bytes: []const u8) !u64 {
    const p = try e.alloc(bytes.len);
    try e.arena.upload(p - e.arena.ptr, bytes);
    return p;
}

fn loadProjection(e: anytype, ck: *core.Checkpoint, prefix: []const u8, name: []const u8, format: Format, count: usize, n: usize, k: usize) !Projection {
    var wb: [256]u8 = undefined;
    var sb: [256]u8 = undefined;
    const weight_name = try std.fmt.bufPrint(&wb, "{s}{s}_weight", .{ prefix, name });
    const scale_name = try std.fmt.bufPrint(&sb, "{s}{s}_weight_scale", .{ prefix, name });
    const w = switch (format) {
        .nvfp4 => try ck.expect(weight_name, .u8, &.{ count, n, k / 2 }),
        .mxfp8 => try ck.expect(weight_name, .f8_e4m3, &.{ count, n, k }),
    };
    const scales = switch (format) {
        .nvfp4 => try ck.expect(scale_name, .f8_e4m3, &.{ count, n, k / 16 }),
        .mxfp8 => try ck.expect(scale_name, .u8, &.{ count, n, k / 32 }),
    };
    var p: Projection = .{ .codes = try put(e, w.bytes), .scales = try put(e, scales.bytes), .n = @intCast(n), .k = @intCast(k) };
    if (format == .nvfp4) {
        var ab: [256]u8 = undefined;
        var db: [256]u8 = undefined;
        const amax = try ck.expect(try std.fmt.bufPrint(&ab, "{s}{s}_weight_amax", .{ prefix, name }), .f32, &.{count});
        const denominator = try ck.expect(try std.fmt.bufPrint(&db, "{s}{s}_weight_outer_scale_denominator", .{ prefix, name }), .f32, &.{1});
        p.amax = try put(e, amax.bytes);
        p.denominator = try put(e, denominator.bytes);
    }
    return p;
}

/// Upload the checkpoint's two stored projections without a repack or an activation-quantization side input.
pub fn load(e: anytype, ck: *core.Checkpoint, prefix: []const u8, format: Format, count: usize, hidden: usize, intermediate: usize) !Layer {
    if (count == 0 or count > max_experts or hidden % 32 != 0 or intermediate % 32 != 0) return error.BadExperts;
    return .{
        .format = format,
        .experts = @intCast(count),
        .w13 = try loadProjection(e, ck, prefix, "w13", format, count, 2 * intermediate, hidden),
        .w2 = try loadProjection(e, ck, prefix, "w2", format, count, hidden, intermediate),
    };
}

test "Mini's expert scratch holds all tensor-core stages and the group plan" {
    const bytes = try scratchBytes(2048, 8, 2304, 1152);
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(Weight));
    try std.testing.expect(bytes > 2048 * 8 * (3 * 1152 + 2304) * 2);
    try std.testing.expectEqual(@as(i32, 0), @backingInt(Format.nvfp4));
    try std.testing.expectEqual(@as(i32, 1), @backingInt(Format.mxfp8));
    try std.testing.expectError(error.BadExperts, scratchBytes(0, 8, 2304, 1152));
}

/// One-member warp: A/B lanes explicit; exact rationals expose scalar/mma FP32 reduction diffs despite BF16 rounding.
const FragmentHostProbe = struct {
    const expert_count = 2;
    const hidden = 64; // two 32-column output blocks in the down projection
    const intermediate = 64; // two gate/up blocks and two 32-input groups in down
    const gate_cap = 4;
    const up_cap = 25;

    const Matrix = struct {
        codes: []const u8,
        scales: []const u8,
        n: usize,
        k: usize,

        /// Checkpoint NVFP4 -> BF16, with outer scale one to isolate fragment addressing.
        fn decoded(m: Matrix, e: usize, row: usize, col: usize) u16 {
            const z = (e * m.n + row) * m.k + col;
            const nib = (m.codes[z / 2] >> @as(u3, @intCast(4 * (col & 1)))) & 15;
            const magnitude = ([8]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 })[nib & 7];
            const q = if ((nib & 8) != 0) -magnitude else magnitude;
            const scale_code = m.scales[z / 16];
            const se = (scale_code >> 3) & 15;
            const sm = scale_code & 7;
            const sign: u32 = if ((scale_code & 128) != 0) 0x80000000 else 0;
            const scale: f32 = if (se == 0)
                @as(f32, @floatFromInt(sm)) * 0x1p-9 * (if (sign != 0) @as(f32, -1) else 1)
            else
                @bitCast(sign | ((@as(u32, se) + 120) << 23) | (@as(u32, sm) << 20));
            return core.bf16_round.bits(q * scale);
        }
    };

    fn value(bits: u16) f32 {
        return core.bf16_round.value(bits);
    }

    fn fill(codes: []u8, scales: []u8, e: usize, n: usize, k: usize, one_per_row: bool) void {
        const order = [16]u8{ 2, 10, 4, 13, 1, 7, 14, 0, 3, 9, 5, 12, 6, 15, 11, 8 };
        const scale_codes = [4]u8{ 0x28, 0x30, 0x2c, 0x34 }; // 1/4, 1/2, 3/8, 3/4
        for (0..n) |row| for (0..k) |col| {
            const z = (e * n + row) * k + col;
            const index = (7 * row + 11 * col + (row * col) % 13 + 3 * e) & 15;
            var code = order[index];
            // Down's one nonzero rotates through all K positions over the two output tiles.
            if (one_per_row and col != (19 * row + 7 * e) % k) code = 0;
            if (one_per_row and col == (19 * row + 7 * e) % k and (code & 7) == 0) code = 2;
            if ((col & 1) == 0) codes[z / 2] = code else codes[z / 2] |= code << 4;
            if ((col & 15) == 0) scales[z / 16] = scale_codes[(3 * row + col / 16 + e) & 3];
        };
    }

    /// The arithmetic baseline: dequantize row-major weights and evaluate x @ W directly.
    fn scalarProjection(m: Matrix, e: usize, row0: usize, row_step: usize, x: []const u16, out: []u16) void {
        for (out, 0..) |*o, col| {
            var sum: f32 = 0;
            const row = row0 + row_step * col;
            for (0..m.k) |k| sum += value(x[k]) * value(m.decoded(e, row, k));
            o.* = core.bf16_round.bits(sum);
        }
    }

    /// Served RT=1 (ten max): materialize uint2/uint4 A/B; m16n8k16 D gathers PTX lanes, B column 2*t+n, not lane gq.
    fn fragmentProjection(m: Matrix, e: usize, row0: usize, row_step: usize, x: []const u16, out: []u16, members: usize) void {
        const width = out.len / members;
        for (0..width / 32) |cb| {
            var acc: [32][4][4]f32 = std.mem.zeroes([32][4][4]f32);
            for (0..m.k / 32) |g| {
                // [lane][row half][q][uint2 member][BF16 in the member]
                var a: [32][2][2][2][2]f32 = std.mem.zeroes([32][2][2][2][2]f32);
                // Four ld_w uint4s: [n8 tile][lane][eight BF16 values in x,y,z,w].
                var b: [4][32][8]f32 = undefined;
                for (0..32) |lane| {
                    const gq = lane >> 2;
                    const t = lane & 3;
                    for (0..2) |h| for (0..2) |q| for (0..2) |pair| for (0..2) |element| {
                        const i = gq + 8 * h;
                        // x = X + token*x_stride + 4*t; uint2 spans two adjacent pairs.
                        const col = 32 * g + 16 * q + 4 * t + 2 * pair + element;
                        a[lane][h][q][pair][element] = if (i < members) value(x[col]) else 0;
                    };
                    for (0..4) |j| for (0..8) |i| {
                        const row = row0 + row_step * (cb * 32 + 8 * j + gq);
                        const col = g * 32 + 4 * t + (i & 3) + 16 * (i >> 2);
                        b[j][lane][i] = value(m.decoded(e, row, col));
                    };
                }
                for (0..2) |q| for (0..4) |j| for (0..32) |lane| {
                    const gq = lane >> 2;
                    const t = lane & 3;
                    for (0..2) |h| for (0..2) |n| {
                        const output_col = 2 * t + n;
                        var dot: f32 = 0;
                        for (0..16) |kk| {
                            // uint2 A/B permute mma K as 4*t+(0,1,2,3), valid only together; gather both source lanes.
                            const a_lane = 4 * gq + kk / 4;
                            const b_lane = 4 * output_col + kk / 4;
                            const av = a[a_lane][h][q][(kk & 3) / 2][kk & 1];
                            const bv = b[j][b_lane][4 * q + (kk & 3)];
                            dot += av * bv;
                        }
                        acc[lane][j][2 * h + n] += dot;
                    };
                };
            }
            // Every valid member uses its own lane group's D row, including h=1,q=1.
            for (0..32) |lane| for (0..2) |h| {
                const member = (lane >> 2) + 8 * h;
                if (member >= members) continue;
                for (0..4) |j| for (0..2) |n| {
                    const col = cb * 32 + 8 * j + 2 * (lane & 3) + n;
                    out[member * width + col] = core.bf16_round.bits(acc[lane][j][2 * h + n]);
                };
            };
        }
    }

    fn reference(m13: Matrix, m2: Matrix, x: []const u16) [hidden]u16 {
        var gate: [expert_count * intermediate]u16 = undefined;
        var up: [expert_count * intermediate]u16 = undefined;
        var act: [expert_count * intermediate]u16 = undefined;
        var down: [expert_count * hidden]u16 = undefined;
        var out: [hidden]u16 = undefined;
        for (0..expert_count) |e| {
            scalarProjection(m13, e, 0, 2, x, gate[e * intermediate ..][0..intermediate]);
            scalarProjection(m13, e, 1, 2, x, up[e * intermediate ..][0..intermediate]);
            for (0..intermediate) |j| {
                const raw = value(gate[e * intermediate + j]);
                const gc = gate_cap * @as(f32, @floatCast(std.math.tanh(raw / gate_cap)));
                const uc = up_cap * @as(f32, @floatCast(std.math.tanh(value(up[e * intermediate + j]) / up_cap)));
                const sigmoid: f32 = 1 / (1 + @exp(-raw));
                act[e * intermediate + j] = core.bf16_round.bits(gc * sigmoid * uc);
            }
            scalarProjection(m2, e, 0, 1, act[e * intermediate ..][0..intermediate], down[e * hidden ..][0..hidden]);
        }
        // Routing is slot 0 -> expert 1 (+3/4), slot 1 -> expert 0 (-3/8).
        for (&out, 0..) |*o, col| {
            var total: f32 = 0;
            total += -0.375 * value(down[col]);
            total += 0.75 * value(down[hidden + col]);
            o.* = core.bf16_round.bits(total);
        }
        return out;
    }

    fn model(m13: Matrix, m2: Matrix, x: []const u16) [hidden]u16 {
        var gate: [expert_count * intermediate]u16 = undefined;
        var up: [expert_count * intermediate]u16 = undefined;
        var act: [expert_count * intermediate]u16 = undefined;
        var down: [expert_count * hidden]u16 = undefined;
        var out: [hidden]u16 = undefined;
        for (0..expert_count) |e| {
            fragmentProjection(m13, e, 0, 2, x, gate[e * intermediate ..][0..intermediate], 1);
            fragmentProjection(m13, e, 1, 2, x, up[e * intermediate ..][0..intermediate], 1);
            for (0..intermediate) |j| {
                const raw = value(gate[e * intermediate + j]);
                const gc = gate_cap * @as(f32, @floatCast(std.math.tanh(raw / gate_cap)));
                const uc = up_cap * @as(f32, @floatCast(std.math.tanh(value(up[e * intermediate + j]) / up_cap)));
                const sigmoid: f32 = 1 / (1 + @exp(-raw));
                act[e * intermediate + j] = core.bf16_round.bits(gc * sigmoid * uc);
            }
            fragmentProjection(m2, e, 0, 1, act[e * intermediate ..][0..intermediate], down[e * hidden ..][0..hidden], 1);
        }
        for (&out, 0..) |*o, col| {
            var total: f32 = 0;
            total += -0.375 * value(down[col]);
            total += 0.75 * value(down[hidden + col]);
            o.* = core.bf16_round.bits(total);
        }
        return out;
    }

    fn checkMembers(m13: Matrix, x: []const u16) !void {
        const members = 10;
        var tensor: [members * intermediate]u16 = undefined;
        var scalar: [intermediate]u16 = undefined;
        scalarProjection(m13, 0, 0, 2, x, &scalar);
        fragmentProjection(m13, 0, 0, 2, x, &tensor, members);
        var first_difference: ?usize = null;
        for (tensor, 0..) |got, i| if (got != scalar[i % intermediate]) {
            first_difference = i;
            break;
        };
        try std.testing.expectEqual(@as(?usize, null), first_difference);
    }

    fn check() !void {
        var codes13: [expert_count * 2 * intermediate * hidden / 2]u8 = undefined;
        var scales13: [expert_count * 2 * intermediate * hidden / 16]u8 = undefined;
        var codes2: [expert_count * hidden * intermediate / 2]u8 = undefined;
        var scales2: [expert_count * hidden * intermediate / 16]u8 = undefined;
        for (0..expert_count) |e| {
            fill(&codes13, &scales13, e, 2 * intermediate, hidden, false);
            fill(&codes2, &scales2, e, hidden, intermediate, true);
        }
        const m13: Matrix = .{ .codes = &codes13, .scales = &scales13, .n = 2 * intermediate, .k = hidden };
        const m2: Matrix = .{ .codes = &codes2, .scales = &scales2, .n = hidden, .k = intermediate };
        var x: [hidden]u16 = undefined;
        for (&x, 0..) |*v, i| v.* = core.bf16_round.bits(@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 128);
        const got = model(m13, m2, &x);
        const want = reference(m13, m2, &x);
        var first_difference: ?usize = null;
        for (got, want, 0..) |g, w, i| if (g != w) {
            first_difference = i;
            break;
        };
        try std.testing.expectEqual(@as(?usize, null), first_difference);
        try checkMembers(m13, &x);
    }
};

test "one token: m16n8k16 fragment pipeline matches independent BF16 dequantization" {
    try FragmentHostProbe.check();
}
