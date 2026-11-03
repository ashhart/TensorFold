// The GDN contract's reference layer: the served shapes, the specs and the f64 references the gates check.
const std = @import("std");
const contract = @import("contract");
const fx = @import("stage4_fixture.zig");

const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

const Model = fx.Model;
pub const heads: usize = @intCast(Model.heads); // 16
pub const dk: usize = @intCast(Model.head_dim); // 128
pub const linear_dim: usize = @intCast(Model.linear_dim); // 2048
pub const channels: usize = @intCast(Model.conv_channels); // 6144
pub const taps: usize = @intCast(Model.taps); // 4
pub const state_elems: usize = @intCast(Model.state_elements); // 262144 f32

// The served prefill chunk: snapshots appear from 9 tokens on, one interior boundary per additional chunk.
pub const chunk_fixed: u64 = 8;
pub const timeout_ns: u64 = 30 * std.time.ns_per_s;
pub const poison: u16 = 0xffff;
pub const guard_words: usize = 8;
pub const max_tokens: usize = 4096; // host allocation cap on generated edges

pub const Which = enum { split_soa, split_g, conv_prefill, conv_decode, prefill_y, prefill_state, decode_y, decode_state };

pub fn splitDepth(_: c.Shape) usize {
    return 40; // the 32-lane square sum, the stage 4 fixture's stated depth
}
pub fn gateDepth(_: c.Shape) usize {
    return 1;
}
pub fn tapsDepth(_: c.Shape) usize {
    return taps;
}
// Recurrence depth adds about 130 f32 roundings per token plus one output-dot depth; stage 4 derives 130 per update.
pub fn recurYDepth(s: c.Shape) usize {
    return s.rows * 130 + 130;
}
pub fn recurStateDepth(s: c.Shape) usize {
    return s.rows * 130 + 2;
}
// Decode repeats prefill per-token depth; each 130-rounding step includes the 4-op raw-gates prologue.
pub fn decodeYDepth(s: c.Shape) usize {
    return s.rows * 260 + 130;
}
pub fn decodeStateDepth(s: c.Shape) usize {
    return s.rows * 132 + 2;
}
// A snapshot slot's depth: `chunk_fixed` tokens from the chunk's start, plus the store.
pub fn snapDepth() f64 {
    return @floatFromInt(@as(usize, @intCast(chunk_fixed)) * 130 + 2);
}

const token_tile = c.Tile{ .size = 8, .step = 1, .min = 1 };

fn servedOf(which: Which) []const c.Shape {
    return switch (which) {
        .split_soa, .split_g, .conv_prefill, .conv_decode, .decode_y, .decode_state => &.{
            .{ .rows = 8, .n = heads, .k = dk },
            .{ .rows = 1, .n = heads, .k = dk },
            .{ .rows = 32, .n = heads, .k = dk },
        },
        .prefill_y, .prefill_state => &.{
            .{ .rows = 4, .n = heads, .k = dk },
            .{ .rows = 1, .n = heads, .k = dk },
            .{ .rows = 32, .n = heads, .k = dk },
            .{ .rows = 130, .n = heads, .k = dk },
        },
    };
}

const soa_sums_why = "the 32-lane square sum reaches the output only through its inverse square root, and the stated budget's magnitude term at the output's scale is wider than any fp16 miss the stress can engineer: gate 3's shapes bound the sum to f32 quality instead (a bf16-summed ss misses by ~2^-8 relative, far past the observed worst of bound)";
const conv_sums_why = "these stress inputs do not separate bf16 tap sums beyond the rounded output bound; fp16 overflows";
const recur_sums_why = "the kv dot's bf16 error enters the state only scaled by beta and the budget's magnitude term already carries the f32 chain: the fp16 margin proves fp32 sums";

pub fn spec(which: Which) c.Spec {
    return switch (which) {
        .split_soa => .{
            .name = "vulkan gdn split soa+beta (bf16)",
            .accum = .f32,
            .depth = splitDepth,
            .ops = 5,
            .mids = 3,
            .out = .bf16,
            .tiles = .{ token_tile, .{}, .{} },
            .served = servedOf(which),
            .sums = .{ .f16 = 0, .bf16 = 0, .why = soa_sums_why },
        },
        .split_g => .{
            .name = "vulkan gdn split g (f32)",
            .accum = .none, // pointwise gate chain: no sums (CUDA gdn_prep's gb precedent)
            .depth = gateDepth,
            .ops = 4,
            .mids = 2,
            .out = .f32,
            .tiles = .{ token_tile, .{}, .{} },
            .served = servedOf(which),
        },
        .conv_prefill => .{
            .name = "vulkan gdn conv prefill out+state (bf16)",
            .accum = .f32,
            .depth = tapsDepth,
            .out = .bf16,
            .tiles = .{ token_tile, .{}, .{} },
            .served = servedOf(which),
            .sums = .{ .bf16 = 0, .why = conv_sums_why },
        },
        .conv_decode => .{
            .name = "vulkan gdn conv decode out+state (bf16)",
            .accum = .f32,
            .depth = tapsDepth,
            .out = .bf16,
            .tiles = .{ token_tile, .{}, .{} },
            .served = servedOf(which),
            .sums = .{ .bf16 = 0, .why = conv_sums_why },
        },
        .prefill_y => .{
            .name = "vulkan gdn prefill y (bf16)",
            .accum = .f32,
            .depth = recurYDepth,
            .ops = 4,
            .out = .bf16,
            .tiles = .{ token_tile, .{}, .{} },
            .served = servedOf(which),
            .sums = .{ .bf16 = 0, .why = recur_sums_why },
        },
        .prefill_state => .{
            .name = "vulkan gdn prefill hf+snap (f32)",
            .accum = .f32,
            .depth = recurStateDepth,
            .ops = 2,
            .out = .f32,
            .tiles = .{ token_tile, .{}, .{} },
            .served = servedOf(which),
            .sums = .{ .bf16 = 0, .why = recur_sums_why },
        },
        .decode_y => .{
            .name = "vulkan gdn decode y (bf16)",
            .accum = .f32,
            .depth = decodeYDepth,
            .ops = 4,
            .out = .bf16,
            .tiles = .{ token_tile, .{}, .{} },
            .served = servedOf(which),
            .sums = .{ .bf16 = 0, .why = recur_sums_why },
        },
        .decode_state => .{
            .name = "vulkan gdn decode state (f32)",
            .accum = .f32,
            .depth = decodeStateDepth,
            .ops = 2,
            .out = .f32,
            .tiles = .{ token_tile, .{}, .{} },
            .served = servedOf(which),
            .sums = .{ .bf16 = 0, .why = recur_sums_why },
        },
    };
}

pub fn f32Of(pairs: []const u16, word: usize) f64 {
    return @as(f32, @bitCast(@as(u32, pairs[2 * word]) | (@as(u32, pairs[2 * word + 1]) << 16)));
}

pub fn bf(bits: u16) f64 {
    return c.fromBf16(bits);
}

pub fn putF32(pairs: []u16, word: usize, v: f32) void {
    const bits: u32 = @bitCast(v);
    pairs[2 * word] = @truncate(bits);
    pairs[2 * word + 1] = @truncate(bits >> 16);
}

// `walk` matches fx.RecurWalk/fx.Decode bitwise (null `acc`=f64); writes y/state, snapshots, pre-update `max_state`.
pub fn walk(
    tokens: usize,
    soa_bits: []const u16,
    g_pairs: []const u16,
    beta_bits: []const u16,
    h0_bits: []const u16,
    st: []f64,
    want_y: ?[]f64,
    want_state: ?[]f64,
    max_state: *f64,
    snap_chunk: ?usize,
    want_snap: ?[]f64,
    acc: ?c.Accum,
    order: ?g.Order,
) void {
    const section = tokens * linear_dim;
    for (0..state_elems) |i| st[i] = f32Of(h0_bits, i);
    var terms: [dk]f64 = undefined;
    for (0..tokens) |t| {
        for (0..heads) |h| {
            const gv = f32Of(g_pairs, t * heads + h);
            const beta = bf(beta_bits[t * heads + h]);
            const qsec = 0 * section + (t * heads + h) * dk;
            const ksec = 1 * section + (t * heads + h) * dk;
            const vsec = 2 * section + (t * heads + h) * dk;
            for (0..dk) |row| {
                for (0..dk) |i| {
                    const s = st[h * dk * dk + row * dk + i];
                    max_state.* = @max(max_state.*, @abs(s));
                    terms[i] = (s * gv) * bf(soa_bits[ksec + i]);
                }
                const kv = if (acc) |a| rows.sumIn(a, order.?, &terms) else seq(&terms);
                const delta = (bf(soa_bits[vsec + row]) - kv) * beta;
                for (0..dk) |i| {
                    const idx = h * dk * dk + row * dk + i;
                    const ns = (st[idx] * gv) + delta * bf(soa_bits[ksec + i]);
                    st[idx] = ns;
                    terms[i] = ns * bf(soa_bits[qsec + i]);
                }
                const o = if (acc) |a| rows.sumIn(a, order.?, &terms) else seq(&terms);
                if (want_y) |wy| wy[(t * heads + h) * dk + row] = o;
            }
        }
        if (want_snap) |ws| if (snap_chunk) |chk| {
            if ((t + 1) % chk == 0 and t + 1 < tokens) {
                const slot = (t + 1) / chk - 1;
                @memcpy(ws[slot * state_elems ..][0..state_elems], st[0..state_elems]);
            }
        };
    }
    if (want_state) |ws| @memcpy(ws[0..state_elems], st[0..state_elems]);
}

fn seq(terms: []const f64) f64 {
    var sum: f64 = 0;
    for (terms) |v| sum += v;
    return sum;
}

// Split bounds use `want/mag`: soa then beta; mag is row `vmax^2` for soa and `|want|` for beta.
pub fn splitReference(
    tokens: usize,
    conv_out_bits: []const u16,
    a_bits: []const u16,
    b_bits: []const u16,
    a_log_bits: []const u16,
    dt_bits: []const u16,
    want: []f64,
    mag: []f64,
) void {
    const soa_elems = 3 * tokens * linear_dim;
    for (0..tokens) |t| {
        for (0..3 * heads) |head| {
            const hv: usize = head % heads;
            var vals: [dk]f64 = undefined;
            var ss: f64 = 0;
            var vmax: f64 = 0;
            for (0..dk) |i| {
                const cv = bf(conv_out_bits[t * channels + head * dk + i]);
                const sig = 1.0 / (1.0 + @exp(-cv));
                vals[i] = cv * sig;
                ss += vals[i] * vals[i];
                vmax = @max(vmax, @abs(vals[i]));
            }
            const is_q = head < heads;
            const is_k = head >= heads and head < 2 * heads;
            const inv: f64 = 1.0 / @sqrt(ss / 128.0 + 1e-6 / 128.0);
            const scale_f: f64 = if (is_q) 1.0 / 128.0 else @sqrt(1.0 / 128.0);
            const dst_base = (head / heads) * (tokens * linear_dim) + (t * heads + hv) * dk;
            for (0..dk) |i| {
                want[dst_base + i] = if (is_q or is_k) scale_f * (vals[i] * inv) else vals[i];
                mag[dst_base + i] = vmax * vmax;
            }
        }
        for (0..heads) |hv| {
            const ref = fx.gateRefAsh(a_bits[t * heads + hv], b_bits[t * heads + hv], f32Of(a_log_bits, hv), dt_bits[hv]);
            want[soa_elems + t * heads + hv] = ref.beta;
            mag[soa_elems + t * heads + hv] = @abs(ref.beta);
        }
    }
}

// Conv uses `in_word` on state then projection rows; it writes T rows and shifted carry when `want_state` is set.
pub fn convReference(
    tokens: usize,
    weight_bits: []const u16,
    ctx: anytype,
    want: []f64,
    mag: []f64,
    want_state: ?[]f64,
) void {
    for (0..channels) |ch| {
        for (0..tokens) |t| {
            var acc: f64 = 0;
            for (0..taps) |k| {
                acc += bf(ctx.inWord(t + k, ch)) * bf(weight_bits[ch * taps + k]);
            }
            want[t * channels + ch] = acc;
            mag[t * channels + ch] = @abs(acc);
        }
        if (want_state) |ws| for (0..3) |j| {
            ws[j * channels + ch] = bf(ctx.inWord(tokens + j, ch));
        };
    }
}

// The prefill conv's assembled-input reader: rows 0..2 are the stored conv state, rows 3.. the projection output.
pub const PrefillInput = struct {
    state_bits: []const u16,
    qkv_bits: []const u16,
    pub fn inWord(self: PrefillInput, row: usize, ch: usize) u16 {
        return if (row < 3) self.state_bits[row * channels + ch] else self.qkv_bits[(row - 3) * channels + ch];
    }
};

// The decode conv's assembled-input reader at token `t`: rows 0..2 of that step's carried state, row 3 the token's x.
pub const DecodeInput = struct {
    state_bits: []const u16,
    x_bits: []const u16,
    t: usize,
    pub fn inWord(self: DecodeInput, row: usize, ch: usize) u16 {
        return if (row < 3) self.state_bits[row * channels + ch] else self.x_bits[self.t * channels + ch];
    }
};

// The g/beta pair one decode token's raw-gates prologue produces (the Ash chain), as the walk reads them.
pub fn decodeGates(a_bits: []const u16, b_bits: []const u16, a_log_bits: []const u16, dt_bits: []const u16, t: usize, g_pairs: []u16, beta_bits: []u16) void {
    for (0..heads) |h| {
        const ref = fx.gateRefAsh(a_bits[t * heads + h], b_bits[t * heads + h], f32Of(a_log_bits, h), dt_bits[h]);
        putF32(g_pairs, h, @floatCast(ref.g));
        beta_bits[h] = c.toBf16(@floatCast(ref.beta));
    }
}

// The shared gate harness expects a rig object; as in HIP, it exposes bytes while the fixture owns device buffers.
pub const Rig = struct {
    output: []const u8 = &.{},
    inputs: []const u8 = &.{},
    guards_ok: bool = false,
    pub fn outputBytes(r: Rig) []const u8 {
        return r.output;
    }
    pub fn inputBytes(r: Rig) []const u8 {
        return r.inputs;
    }
    pub fn intact(r: Rig) bool {
        return r.guards_ok;
    }
    pub fn overlap(_: Rig) bool {
        return false; // every view here is out of place; the wrapper's refuseOverlap refuses aliasing at bind
    }
};
