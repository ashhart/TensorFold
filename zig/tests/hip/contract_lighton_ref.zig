//! Independent FP64 formulas and explicit BF16 points for LightOnOCR's HIP operators.
const std = @import("std");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;
const bf = core.bf16_round;
const rows = core.contract.rows;
pub const Kind = enum { tables, rotary, merge, gelu, rms, text_qk, rows_to_heads, heads_to_rows, attention };
pub const Profile = enum { smoke, full };
pub const Entry = struct { kind: Kind, projector: bool = false, span: bool = false, control: bool = false };
pub const entries = [_]Entry{
    .{ .kind = .tables },                  .{ .kind = .rotary },                                   .{ .kind = .merge },                   .{ .kind = .gelu },
    .{ .kind = .rms },                     .{ .kind = .rms, .projector = true },                   .{ .kind = .text_qk },                 .{ .kind = .rows_to_heads },
    .{ .kind = .heads_to_rows },           .{ .kind = .attention },                                .{ .kind = .tables, .control = true }, .{ .kind = .rotary, .control = true },
    .{ .kind = .merge, .control = true },  .{ .kind = .gelu, .control = true },                    .{ .kind = .rms, .control = true },    .{ .kind = .text_qk, .control = true },
    .{ .kind = .attention, .span = true }, .{ .kind = .attention, .span = true, .control = true },
};

pub fn symbol(e: Entry) [:0]const u8 {
    if (e.control) return switch (e.kind) {
        .tables => "control_lo_tables_axes",
        .rotary => "control_lo_rotary_unrounded",
        .merge => "control_lo_merge_patchmajor",
        .gelu => "control_lo_gelu_tanh",
        .rms => "control_lo_rms_unrounded",
        .text_qk => "control_lo_text_qk_unrounded",
        .attention => "control_lo_attention_leak",
        else => unreachable,
    };
    return switch (e.kind) {
        .tables => "tf_lo_tables",
        .rotary => "tf_lo_rotary",
        .merge => "tf_lo_merge",
        .gelu => "tf_lo_gelu",
        .rms => "tf_lo_rms",
        .text_qk => "tf_lo_text_qk",
        .rows_to_heads => "tf_lo_rows_to_heads",
        .heads_to_rows => "tf_lo_heads_to_rows",
        .attention => "tf_lo_attention64",
    };
}

fn none(_: c.Shape) usize {
    return 0;
}
fn normDepth(s: c.Shape) usize {
    return (s.n + 255) / 256 + 13;
}
fn headDepth(_: c.Shape) usize {
    return 12;
}
fn attentionDepth(s: c.Shape) usize {
    return 8 * s.k + 128;
}

/// Spatial ops: gh,gw,width; norms/pointwise: rows,width,1; text/layout: rows,heads,dim; attention: heads,64,keys.
pub fn spec(e: Entry, profile: Profile) c.Spec {
    const names = [_][]const u8{
        "hip lighton_ocr tf_lo_tables",        "hip lighton_ocr tf_lo_rotary",        "hip lighton_ocr tf_lo_merge",
        "hip lighton_ocr tf_lo_gelu",          "hip lighton_ocr tf_lo_rms",           "hip lighton_ocr tf_lo_text_qk",
        "hip lighton_ocr tf_lo_rows_to_heads", "hip lighton_ocr tf_lo_heads_to_rows", "hip lighton_ocr tf_lo_attention64",
    };
    const smoke = profile == .smoke;
    return .{
        .name = if (e.control) symbol(e) else if (e.projector) "hip lighton_ocr tf_lo_rms (projector epsilon)" else if (e.span) "hip lighton_ocr tf_lo_attention64 (isolated image span)" else names[@backingInt(e.kind)],
        .accum = if (e.kind == .rms or e.kind == .text_qk or e.kind == .attention) .f32 else .none,
        .depth = switch (e.kind) {
            .rms => normDepth,
            .text_qk => headDepth,
            .attention => attentionDepth,
            else => none,
        },
        .ops = if (e.kind == .attention) 16 else if (e.kind == .tables) 4 else 8,
        .mids = if (e.kind == .rms) 1 else if (e.kind == .text_qk) 4 else 0,
        .out = if (e.kind == .rows_to_heads) .f32 else .bf16,
        .served = switch (e.kind) {
            .tables, .rotary => if (smoke) &.{ .{ .rows = 3, .n = 5, .k = 64 }, .{ .rows = 8, .n = 9, .k = 64 } } else &.{ .{ .rows = 110, .n = 110, .k = 64 }, .{ .rows = 73, .n = 109, .k = 64 }, .{ .rows = 3, .n = 5, .k = 64 } },
            .merge => if (smoke) &.{ .{ .rows = 3, .n = 5, .k = 7 }, .{ .rows = 8, .n = 9, .k = 1024 } } else &.{ .{ .rows = 110, .n = 110, .k = 1024 }, .{ .rows = 73, .n = 109, .k = 1024 }, .{ .rows = 3, .n = 5, .k = 7 } },
            .gelu => if (smoke) &.{.{ .rows = 3, .n = 1024, .k = 1 }} else &.{ .{ .rows = 3025, .n = 1024, .k = 1 }, .{ .rows = 1, .n = 1024, .k = 1 } },
            .rms => if (smoke) &.{.{ .rows = 8, .n = 1024, .k = 1 }} else &.{ .{ .rows = 12100, .n = 1024, .k = 1 }, .{ .rows = 128, .n = 1024, .k = 1 }, .{ .rows = 1, .n = 1024, .k = 1 } },
            .text_qk => &.{ .{ .rows = 1, .n = 16, .k = 128 }, .{ .rows = 128, .n = 16, .k = 128 } },
            .rows_to_heads, .heads_to_rows => &.{ .{ .rows = 1, .n = 16, .k = 128 }, .{ .rows = 128, .n = 16, .k = 128 }, .{ .rows = 17, .n = 16, .k = 64 } },
            .attention => if (smoke) &.{ .{ .rows = 16, .n = 64, .k = 65 }, .{ .rows = 3, .n = 64, .k = 129 } } else &.{ .{ .rows = 16, .n = 64, .k = 12100 }, .{ .rows = 16, .n = 64, .k = 65 }, .{ .rows = 3, .n = 64, .k = 129 } },
        },
        .sums = if (e.kind == .attention) .{ .f16 = 0, .bf16 = 0, .why = "finite-span online attention stress margins are reported; BF16 output can hide reduced sums" } else .{ .f16 = 40, .bf16 = 0, .why = "FP16 squared integers overflow; BF16 norm rounding can hide reduced BF16 square sums" },
    };
}

pub const Data = struct {
    a: std.mem.Allocator,
    e: Entry,
    shape: c.Shape,
    fill: g.Fill,
    b: [5][]u16 = .{ &.{}, &.{}, &.{}, &.{}, &.{} },
    f: [3][]f32 = .{ &.{}, &.{}, &.{} },
    pos: []i32 = &.{},
    want: []f64,
    mag: []f64,
    low: []f64,
    high: []f64,
    total: usize = 0,
    start: usize = 2,
    period: usize = 8,

    pub fn init(a: std.mem.Allocator, e: Entry, shape: c.Shape, fill: g.Fill, seed: u64) !Data {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return error.Shape;
        const spatial = e.kind == .tables or e.kind == .rotary or e.kind == .merge;
        if (spatial and (shape.rows > 110 or shape.n > 110)) return error.Shape;
        if ((e.kind == .tables or e.kind == .rotary) and shape.k != 64) return error.Shape;
        if (e.kind == .merge and (shape.rows < 2 or shape.n < 2 or shape.k > 1024)) return error.Shape;
        if ((e.kind == .rms or e.kind == .gelu) and (shape.n > 4096 or shape.rows > 12100 or shape.k != 1)) return error.Shape;
        if ((e.kind == .text_qk or e.kind == .rows_to_heads or e.kind == .heads_to_rows) and (shape.rows > 128 or shape.n != 16 or (shape.k != 64 and shape.k != 128))) return error.Shape;
        if (e.kind == .text_qk and shape.k != 128) return error.Shape;
        if (e.kind == .attention and (shape.rows > 16 or shape.n != 64 or shape.k > 12100)) return error.Shape;
        const n = switch (e.kind) {
            .tables => shape.rows * shape.n * 64,
            .rotary => 3 * shape.rows * shape.n * 16 * 64,
            .merge => (shape.rows / 2) * (shape.n / 2) * shape.k * 4,
            .gelu, .rms => shape.rows * shape.n,
            .text_qk => shape.rows * 32 * 128,
            .rows_to_heads, .heads_to_rows => shape.rows * shape.n * shape.k,
            .attention => shape.rows * shape.k * 64,
        };
        var d: Data = .{ .a = a, .e = e, .shape = shape, .fill = fill, .want = try a.alloc(f64, n), .mag = try a.alloc(f64, n), .low = try a.alloc(f64, n), .high = try a.alloc(f64, n) };
        var random = std.Random.DefaultPrng.init(seed);
        const rng = random.random();
        switch (e.kind) {
            .tables => {
                d.f[0] = try a.alloc(f32, 32);
                frequencies(d.f[0], 10000, true);
                if (fill == .nonfinite) d.f[0][0..3].* = .{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            },
            .rotary => {
                for (d.b[0..3]) |*v| v.* = try words(a, n / 3, rng, 4);
                d.b[3] = try words(a, shape.rows * shape.n * 32, rng, 0.5);
                d.b[4] = try words(a, d.b[3].len, rng, 0.5);
                d.b[0][0] = 0x3e2c;
                d.b[0][32] = 0x3ed0;
                d.b[3][0] = 0x3f35;
                d.b[4][0] = 0x3f35;
                if (fill == .nonfinite) for (d.b[0..3]) |v| {
                    v[2..5].* = .{ 0x7fc1, 0x7f80, 0xff80 };
                };
            },
            .merge => {
                d.b[0] = try words(a, shape.rows * shape.n * shape.k, rng, 3);
                if (fill == .nonfinite) for (d.b[0], 0..) |*v, i| {
                    v.* = ([_]u16{ 0x7fc1, 0x7f80, 0xff80, 0x8000, 0 })[i % 5];
                };
            },
            .gelu, .rms => {
                d.b[0] = try words(a, n, rng, 4);
                if (e.kind == .rms) {
                    d.f[0] = try a.alloc(f32, shape.n);
                    for (d.f[0]) |*w| w.* = bf.round32(0.5 + 4 * rng.float(f32));
                    if (fill == .stress) {
                        for (d.b[0]) |*x| x.* = bf.bits(@as(f32, @floatFromInt(32 + rng.uintLessThan(u32, 13))));
                        @memset(d.f[0], 1);
                    }
                }
                if (fill == .nonfinite) d.b[0][0..3].* = .{ 0x7fc1, 0x7f80, 0xff80 };
            },
            .text_qk => {
                d.b[0] = try words(a, shape.rows * 16 * 128, rng, 2);
                d.b[1] = try words(a, d.b[0].len / 2, rng, 2);
                d.b[2] = try words(a, d.b[1].len, rng, 2);
                for (d.f[0..2]) |*f| {
                    f.* = try a.alloc(f32, 128);
                    for (f.*) |*w| w.* = bf.round32(0.8 + 0.4 * rng.float(f32));
                }
                d.f[2] = try a.alloc(f32, 64);
                frequencies(d.f[2], 1000000, false);
                d.pos = try a.alloc(i32, shape.rows);
                for (d.pos, 0..) |*p, i| p.* = @intCast(16383 - i);
                if (fill == .stress) {
                    for (d.b[0..3]) |b| @memset(b, bf.bits(64));
                    for (d.f[0..2]) |f| @memset(f, 1);
                    @memset(d.f[2], 0);
                }
                if (fill == .nonfinite) for (d.b[0..3], 0..) |b, i| {
                    b[5] = ([_]u16{ 0x7fc1, 0x7f80, 0xff80 })[i];
                };
                if (fill == .random) {
                    @memset(d.b[0][0..128], bf.bits(1));
                    d.f[0][0] = bf.value(0x3e2c);
                    d.f[0][64] = bf.value(0x3ed0);
                    d.pos[0] = 1;
                    d.f[2][0] = @floatCast(std.math.pi / 4.0);
                }
            },
            .rows_to_heads => {
                d.b[0] = try words(a, n, rng, 4);
                if (fill == .nonfinite) d.b[0][0..4].* = .{ 0x7fc1, 0x7f80, 0xff80, 0x8000 };
            },
            .heads_to_rows => {
                d.f[0] = try a.alloc(f32, n);
                for (d.f[0]) |*v| v.* = rng.floatNorm(f32);
                if (fill == .nonfinite) d.f[0][0..4].* = .{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), -0.0 };
            },
            .attention => {
                d.start = if (e.span) 2 else 0;
                d.total = shape.k + @as(usize, if (e.span) 4 else 0);
                d.period = if (fill == .stress) 64 else 8;
                for (d.b[0..3]) |*b| b.* = try a.alloc(u16, shape.rows * d.total * 64);
                for (0..shape.rows) |head| {
                    const pattern = try words(a, 3 * d.period * 64, rng, 0.5);
                    for (0..d.total) |row| for (0..64) |col| for (0..3) |part| {
                        var v = pattern[(part * d.period + row % d.period) * 64 + col];
                        if (fill == .stress) v = if (part < 2) 0 else bf.bits(if (row % 64 < 16) @as(f32, 2048) else if (row % 64 < 48) @as(f32, 1) else -2048);
                        if (fill == .nonfinite and part == 0 and head == 0 and row % 8 == 0 and col == 1) v = 0x7fc1;
                        if (fill == .nonfinite and part == 2 and head == 2 and row % 8 == 3 and col == 2) v = 0x7f80;
                        if (row < d.start or row >= d.start + shape.k) v = if (part == 2) bf.bits(32) else 0;
                        d.b[part][(head * d.total + row) * 64 + col] = v;
                    };
                }
            },
        }
        try d.calculate(null, .sequential, d.want, d.mag, d.low, d.high);
        return d;
    }

    pub fn calculate(d: Data, acc: ?c.Accum, order: rows.Order, want: []f64, mag: []f64, low: []f64, high: []f64) !void {
        @memset(mag, 0);
        const sh = d.shape;
        switch (d.e.kind) {
            .tables => for (want, 0..) |*v, i| {
                const half = want.len / 2;
                const row = (i % half) / 32;
                const j = i % 32;
                const angle: f32 = @as(f32, @floatFromInt(if (j < 16) row / sh.n else row % sh.n)) * d.f[0][j];
                v.* = if (i < half) @cos(@as(f64, angle)) else @sin(@as(f64, angle));
            },
            .rotary => for (0..3) |part| for (0..sh.rows * sh.n) |row| for (0..16) |head| for (0..64) |col| {
                const source = (row * 16 + head) * 64 + col;
                const dest = part * (want.len / 3) + (head * sh.rows * sh.n + row) * 64 + col;
                want[dest] = if (part == 2) bf.value(d.b[2][source]) else rotate(bf.value(d.b[part][source]), bf.value(d.b[part][source - col + (col + 32) % 64]), bf.value(d.b[3][row * 32 + col % 32]), bf.value(d.b[4][row * 32 + col % 32]), col < 32);
            },
            .merge => for (want, 0..) |*v, i| {
                const token = i / (sh.k * 4);
                const lane = i % 4;
                const ch = (i / 4) % sh.k;
                const row = (token / (sh.n / 2)) * 2 + lane / 2;
                const col = (token % (sh.n / 2)) * 2 + lane % 2;
                v.* = bf.value(d.b[0][(row * sh.n + col) * sh.k + ch]);
            },
            .gelu => for (want, low, high, d.b[0]) |*v, *lo, *hi, x| {
                v.* = core.act_ref.geluErf(bf.value(x), 1, 0, 0);
                const range = core.act_ref.geluErfRange(bf.value(x), 1);
                lo.* = range[0];
                hi.* = range[1];
            },
            .rms => {
                const eps: f64 = if (d.e.projector) 1e-6 else 1e-5;
                for (0..sh.rows) |row| try norm(d.a, d.b[0][row * sh.n ..][0..sh.n], d.f[0], eps, acc, order, want[row * sh.n ..][0..sh.n], low[row * sh.n ..][0..sh.n], high[row * sh.n ..][0..sh.n]);
            },
            .text_qk => {
                const scratch = try d.a.alloc(f64, 6 * 128);
                defer d.a.free(scratch);
                const y, const lo, const hi = .{ scratch[0..128], scratch[128..256], scratch[256..384] };
                for (0..2) |part| for (0..sh.rows) |row| for (0..@as(usize, if (part == 0) 16 else 8)) |head| {
                    const heads_: usize = if (part == 0) 16 else 8;
                    const at = (row * heads_ + head) * 128;
                    try norm(d.a, d.b[part][at..][0..128], d.f[part], 1e-6, acc, order, y, lo, hi);
                    for (0..128) |col| {
                        const dst = if (part == 0) at + col else sh.rows * 16 * 128 + at + col;
                        const angle: f32 = @as(f32, @floatFromInt(d.pos[row])) * d.f[2][col % 64];
                        const cosine = @cos(@as(f64, angle));
                        const sine = @sin(@as(f64, angle));
                        const cs = bf.bf16(cosine);
                        const sn = bf.bf16(sine);
                        want[dst] = rotate(y[col], y[(col + 64) % 128], cs, sn, col < 64);
                        const range = rotateRange(.{ lo[col], hi[col] }, .{ lo[(col + 64) % 128], hi[(col + 64) % 128] }, trigRange(cosine), trigRange(sine), col < 64);
                        low[dst] = range[0];
                        high[dst] = range[1];
                    }
                };
                const vo = sh.rows * 24 * 128;
                for (d.b[2], 0..) |x, i| {
                    want[vo + i] = bf.value(x);
                    low[vo + i] = want[vo + i];
                    high[vo + i] = want[vo + i];
                }
            },
            .rows_to_heads, .heads_to_rows => for (want, 0..) |*v, i| {
                const row = i / (sh.n * sh.k);
                const head = (i / sh.k) % sh.n;
                const col = i % sh.k;
                if (d.e.kind == .heads_to_rows) v.* = bf.bf16(d.f[0][(head * sh.rows + row) * sh.k + col]) else {
                    const src_row = (i / sh.k) % sh.rows;
                    const src_head = i / (sh.rows * sh.k);
                    v.* = bf.value(d.b[0][(src_row * sh.n + src_head) * sh.k + col]);
                }
            },
            .attention => try d.attend(acc, order, want, mag),
        }
        if (d.e.kind != .gelu and d.e.kind != .rms and d.e.kind != .text_qk) for (want, mag, low, high) |v, m, *lo, *hi| {
            const bound = c.bound(spec(d.e, .full), sh, v, m);
            lo.* = v - bound;
            hi.* = v + bound;
        };
    }

    fn attend(d: Data, acc: ?c.Accum, order: rows.Order, want: []f64, mag: []f64) !void {
        const sh = d.shape;
        const n = sh.k;
        const tmp = try d.a.alloc(f64, n + 64 * 3);
        defer d.a.free(tmp);
        const terms = tmp[0..n];
        const scores = tmp[n .. n + 64];
        const probs = tmp[n + 64 .. n + 128];
        const counts = try d.a.alloc(usize, d.period);
        defer d.a.free(counts);
        @memset(counts, 0);
        for (d.start..d.start + n) |i| counts[i % d.period] += 1;
        for (0..sh.rows) |head| for (0..d.period) |qclass| {
            const qr = d.start + (qclass + d.period - d.start % d.period) % d.period;
            if (qr >= d.start + n) continue;
            var maximum = -std.math.inf(f64);
            var dot_mag: f64 = 0;
            for (0..d.period) |kc| {
                const kr = d.start + (kc + d.period - d.start % d.period) % d.period;
                if (kr >= d.start + n) {
                    scores[kc] = -std.math.inf(f64);
                    continue;
                }
                var sum: f64 = 0;
                var size: f64 = 0;
                var dot: [64]f64 = undefined;
                for (&dot, 0..) |*v, col| {
                    v.* = @as(f64, bf.value(d.b[0][(head * d.total + qr) * 64 + col])) * 0.125 * bf.value(d.b[1][(head * d.total + kr) * 64 + col]);
                    sum += v.*;
                    size += @abs(v.*);
                }
                scores[kc] = if (acc) |a| rows.sumIn(a, order, &dot) else sum;
                maximum = if (std.math.isNan(scores[kc]) or std.math.isNan(maximum)) std.math.nan(f64) else @max(maximum, scores[kc]);
                dot_mag = @max(dot_mag, size);
            }
            var denominator: f64 = 0;
            for (0..d.period) |kc| {
                probs[kc] = @exp(scores[kc] - maximum);
                denominator += probs[kc] * @as(f64, @floatFromInt(counts[kc]));
            }
            if (acc) |a| {
                for (terms, 0..) |*v, i| v.* = probs[(d.start + i) % d.period];
                denominator = rows.sumIn(a, order, terms);
            }
            var answer: [64]f64 = undefined;
            var sizes: [64]f64 = undefined;
            for (&answer, &sizes, 0..) |*v, *size, col| {
                var numerator: f64 = 0;
                size.* = 0;
                for (0..d.period) |kc| {
                    if (counts[kc] == 0) continue;
                    const kr = d.start + (kc + d.period - d.start % d.period) % d.period;
                    const product = probs[kc] * bf.value(d.b[2][(head * d.total + kr) * 64 + col]);
                    numerator += product * @as(f64, @floatFromInt(counts[kc]));
                    size.* += @abs(product) * @as(f64, @floatFromInt(counts[kc]));
                }
                if (acc) |a| {
                    for (terms, 0..) |*t, i| t.* = probs[(d.start + i) % d.period] * bf.value(d.b[2][(head * d.total + d.start + i) * 64 + col]);
                    numerator = rows.sumIn(a, order, terms);
                }
                v.* = numerator / denominator;
                size.* = size.* / denominator * (1 + 2 * dot_mag);
            }
            var row = qr;
            while (row < d.start + n) : (row += d.period) for (0..64) |col| {
                const at = ((row - d.start) * sh.rows + head) * 64 + col;
                want[at] = answer[col];
                mag[at] = sizes[col];
            };
        };
    }
};

fn words(a: std.mem.Allocator, n: usize, rng: std.Random, scale: f32) ![]u16 {
    const out = try a.alloc(u16, n);
    for (out) |*v| v.* = bf.bits((rng.float(f32) * 2 - 1) * scale);
    return out;
}
fn frequencies(out: []f32, theta: f64, axial: bool) void {
    for (out, 0..) |*v, i| {
        const power = if (axial) (if (i < 16) 4 * i else 4 * (i - 16) + 2) else 2 * i;
        v.* = @floatCast(1 / std.math.pow(f64, theta, @as(f64, @floatFromInt(power)) / @as(f64, @floatFromInt(out.len * 2))));
    }
}
fn rotate(x: f64, partner: f64, cs: f64, sn: f64, negative: bool) f64 {
    return bf.bf16(bf.bf16(x * cs) + bf.bf16((if (negative) -partner else partner) * sn));
}
fn trigRange(value: f64) [2]f64 {
    const error_ = 4 * 0x1p-24 * @abs(value);
    return .{ bf.bf16(value - error_), bf.bf16(value + error_) };
}
fn rotateRange(x: [2]f64, partner: [2]f64, cs: [2]f64, sn: [2]f64, negative: bool) [2]f64 {
    var lo = std.math.inf(f64);
    var hi = -std.math.inf(f64);
    for (x) |a| for (partner) |b| for (cs) |cv| for (sn) |sv| {
        const v = rotate(a, b, cv, sv, negative);
        if (std.math.isNan(v)) return @splat(std.math.nan(f64));
        lo = @min(lo, v);
        hi = @max(hi, v);
    };
    return .{ lo, hi };
}
fn norm(a: std.mem.Allocator, x: []const u16, weight: []const f32, eps: f64, acc: ?c.Accum, order: rows.Order, out: []f64, low: []f64, high: []f64) !void {
    const terms = try a.alloc(f64, x.len);
    defer a.free(terms);
    var ss: f64 = 0;
    for (terms, x) |*t, v| {
        const value: f64 = bf.value(v);
        t.* = value * value;
        ss += t.*;
    }
    if (acc) |kind| ss = rows.sumIn(kind, order, terms);
    const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(x.len)) + eps);
    const spread = c.gamma(@floatFromInt((x.len + 255) / 256 + 13), c.unit(.f32)) / 2 + 8 * c.unit(.f32);
    for (out, low, high, x, weight) |*v, *lo, *hi, h, w| {
        const value = @as(f64, bf.value(h)) * inv;
        v.* = bf.bf16(bf.bf16(value) * w);
        const a_ = bf.bf16(bf.bf16(value * (1 - spread)) * w);
        const z = bf.bf16(bf.bf16(value * (1 + spread)) * w);
        lo.* = if (std.math.isNan(a_) or std.math.isNan(z)) std.math.nan(f64) else @min(a_, z);
        hi.* = if (std.math.isNan(a_) or std.math.isNan(z)) std.math.nan(f64) else @max(a_, z);
    }
}
