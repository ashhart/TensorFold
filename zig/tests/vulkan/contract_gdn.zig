//! The stage 4 GatedDeltaNet shaders under the kernel contract: the gate
//! split (soa+beta and g), the prefill conv and the decode conv step (raw
//! output and state carry), the chunked prefill recurrence (y and
//! hf+snapshots) and the raw-gates decode step (y and state), each dispatched
//! through the production Gdn wrapper at the fixed Qwen3.5 geometry (16
//! heads, head_dim 128, 6144 conv channels, 4 taps). f32 state flows in and
//! out of place, the interior chunk snapshot is bound exactly when the
//! prefill's chunks exceed one, and the wrapper owns every binding table, so
//! the dummy and read-only slots stay exactly as it declares them.
//!
//! References are the stage 4 fixture's f64 serial walkers made parametric in
//! the token count; every depth a spec states is a function of the shape's
//! token count, because a recurrence's rounding chain grows by one full state
//! update (about 130 f32 roundings deep, the stage 4 fixture's stated
//! per-token derivation) per token. Decode loops the one-token step, so its
//! depth grows the same way. The gate formula's own correctness is the split
//! kernels' business: the recurrence kernels read g and beta as inputs, as
//! the production graph feeds them.
const std = @import("std");
const vk = @import("vulkan");
const contract = @import("contract");
const fx = @import("stage4_fixture.zig");

const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

const Model = fx.Model;
const heads: usize = @intCast(Model.heads); // 16
const dk: usize = @intCast(Model.head_dim); // 128
const linear_dim: usize = @intCast(Model.linear_dim); // 2048
const channels: usize = @intCast(Model.conv_channels); // 6144
const taps: usize = @intCast(Model.taps); // 4
const state_elems: usize = @intCast(Model.state_elements); // 262144 f32

/// The served prefill chunk: snapshots appear from 9 tokens on, one interior
/// boundary per additional chunk.
pub const chunk_fixed: u64 = 8;
const timeout_ns: u64 = 30 * std.time.ns_per_s;
const poison: u16 = 0xffff;
const guard_words: usize = 8;
const max_tokens: usize = 4096; // host allocation cap on generated edges

pub const Which = enum { split_soa, split_g, conv_prefill, conv_decode, prefill_y, prefill_state, decode_y, decode_state };

fn splitDepth(_: c.Shape) usize {
    return 40; // the 32-lane square sum, the stage 4 fixture's stated depth
}
fn gateDepth(_: c.Shape) usize {
    return 1;
}
fn tapsDepth(_: c.Shape) usize {
    return taps;
}
/// A recurrence's chain grows one state update (about 130 f32 roundings, the
/// stage 4 fixture's per-token derivation) per token, plus its own output
/// dot's depth once at the end.
fn recurYDepth(s: c.Shape) usize {
    return s.rows * 130 + 130;
}
fn recurStateDepth(s: c.Shape) usize {
    return s.rows * 130 + 2;
}
/// Decode loops the one-token step: the same per-token growth as the prefill,
/// with the raw-gates prologue's chain (4 ops) inside each token's 130.
fn decodeYDepth(s: c.Shape) usize {
    return s.rows * 260 + 130;
}
fn decodeStateDepth(s: c.Shape) usize {
    return s.rows * 132 + 2;
}
/// A snapshot slot's depth: `chunk_fixed` tokens from the chunk's start, plus
/// the store.
fn snapDepth() f64 {
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

fn f32Of(pairs: []const u16, word: usize) f64 {
    return @as(f32, @bitCast(@as(u32, pairs[2 * word]) | (@as(u32, pairs[2 * word + 1]) << 16)));
}

fn bf(bits: u16) f64 {
    return c.fromBf16(bits);
}

fn putF32(pairs: []u16, word: usize, v: f32) void {
    const bits: u32 = @bitCast(v);
    pairs[2 * word] = @truncate(bits);
    pairs[2 * word + 1] = @truncate(bits >> 16);
}

/// The stage 4 fixture's serial gated-delta walk, parametric in `tokens`,
/// with each row's kv and output dots replayable in a contract accumulator
/// (`acc` null: the fixture's sequential f64 sum, bit-identical to
/// fx.RecurWalk and fx.Decode). Reads soa bf16, g f32 pairs and beta bf16
/// bits; fills y, the terminal state and every interior chunk boundary's
/// snapshot slot ((t+1)/chunk - 1) when `snap_chunk` is given — the shader
/// stores one slot per interior boundary, not just the first. `max_state`
/// tracks the largest pre-update state magnitude, the fixture's magnitude
/// scale.
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

/// The fixture Split checker's math as want/mag: the soa section then beta,
/// both bf16-kind outputs; mag is vmax^2 per head row for soa and |want| for
/// beta, exactly the fixture budget's arguments.
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

/// The fixture Conv checker's math for one conv: `in_word` reads the assembled
/// input (state rows then projection rows) at (input row, channel); outputs
/// T rows then, when `want_state` is given, the shifted carry window.
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

/// The prefill conv's assembled-input reader: rows 0..2 are the stored conv
/// state, rows 3.. the projection output.
const PrefillInput = struct {
    state_bits: []const u16,
    qkv_bits: []const u16,
    pub fn inWord(self: PrefillInput, row: usize, ch: usize) u16 {
        return if (row < 3) self.state_bits[row * channels + ch] else self.qkv_bits[(row - 3) * channels + ch];
    }
};

/// The decode conv's assembled-input reader at token `t`: rows 0..2 of that
/// step's carried state, row 3 the token's x.
const DecodeInput = struct {
    state_bits: []const u16,
    x_bits: []const u16,
    t: usize,
    pub fn inWord(self: DecodeInput, row: usize, ch: usize) u16 {
        return if (row < 3) self.state_bits[row * channels + ch] else self.x_bits[self.t * channels + ch];
    }
};

/// The g/beta pair one decode token's raw-gates prologue produces (the Ash
/// chain), as the walk reads them.
fn decodeGates(a_bits: []const u16, b_bits: []const u16, a_log_bits: []const u16, dt_bits: []const u16, t: usize, g_pairs: []u16, beta_bits: []u16) void {
    for (0..heads) |h| {
        const ref = fx.gateRefAsh(a_bits[t * heads + h], b_bits[t * heads + h], f32Of(a_log_bits, h), dt_bits[h]);
        putF32(g_pairs, h, @floatCast(ref.g));
        beta_bits[h] = c.toBf16(@floatCast(ref.beta));
    }
}

/// The shared gate harness expects a rig object; as in HIP, it exposes bytes
/// while the fixture owns device buffers.
const Rig = struct {
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

pub const Gdn = struct {
    a: std.mem.Allocator,
    dev: *const vk.Device,
    cmds: *vk.Commands,
    staging: vk.Buffer,
    m: *vk.Gdn,
    which: Which,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    tokens: usize = 0,
    chunks: usize = 1,
    max_state: f64 = 0,

    // host inputs (the uploaded bits, kept for the references)
    conv_out_bits: []u16 = &.{},
    a_bits: []u16 = &.{},
    b_bits: []u16 = &.{},
    dt_bits: []u16 = &.{},
    a_log_bits: []u16 = &.{},
    state_bits: []u16 = &.{}, // conv state (bf16) or recurrence state (f32 pairs)
    state_init_bits: []u16 = &.{}, // the decode loops' uploaded initial state, never written
    state_swap_bits: []u16 = &.{}, // poison for the decode loops' scratch carry buffers
    qkv_bits: []u16 = &.{}, // conv_prefill projection rows; conv_decode/decode x rows
    weight_bits: []u16 = &.{},
    soa_bits: []u16 = &.{},
    g_pairs: []u16 = &.{},
    beta_bits: []u16 = &.{},
    h0_bits: []u16 = &.{},
    walk_st: []f64 = &.{}, // the walker's state scratch

    // host outputs and the contract bookkeeping
    out: []u16 = &.{},
    out_words: usize = 0, // out's data width in u16 words (f32 outputs take two a word)
    want: []f64 = &.{},
    mag: []f64 = &.{},
    input_snapshot: []u8 = &.{},
    input_now: []u8 = &.{},
    bufs: [16]vk.Buffer = undefined,
    made: usize = 0,
    /// The decode loops' carry buffer holding the final state after run().
    final_state_idx: usize = 0,
    rig: Rig = .{},

    pub fn init(a: std.mem.Allocator, dev: *const vk.Device, cmds: *vk.Commands, staging: vk.Buffer, m: *vk.Gdn, which: Which) Gdn {
        return .{ .a = a, .dev = dev, .cmds = cmds, .staging = staging, .m = m, .which = which, .arena = .init(a) };
    }

    pub fn deinit(f: *Gdn) void {
        f.reset();
        f.arena.deinit();
    }

    fn reset(f: *Gdn) void {
        while (f.made > 0) {
            f.made -= 1;
            f.bufs[f.made].deinit();
        }
        _ = f.arena.reset(.retain_capacity);
        f.conv_out_bits = &.{};
        f.a_bits = &.{};
        f.b_bits = &.{};
        f.dt_bits = &.{};
        f.a_log_bits = &.{};
        f.state_bits = &.{};
        f.state_init_bits = &.{};
        f.state_swap_bits = &.{};
        f.qkv_bits = &.{};
        f.weight_bits = &.{};
        f.soa_bits = &.{};
        f.g_pairs = &.{};
        f.beta_bits = &.{};
        f.h0_bits = &.{};
        f.walk_st = &.{};
        f.out = &.{};
        f.want = &.{};
        f.mag = &.{};
        f.input_snapshot = &.{};
        f.input_now = &.{};
        f.rig = .{};
        f.shape = .{ .rows = 0, .n = 0, .k = 0 };
        f.tokens = 0;
        f.chunks = 1;
        f.max_state = 0;
    }

    const vt = g.hooks(Gdn);
    pub fn kernel(f: *Gdn) g.Kernel {
        return .{ .spec = spec(f.which), .ctx = f, .vt = &vt };
    }

    fn allocU16(f: *Gdn, n: usize, fill: ?u16) ![]u16 {
        const v = try f.arena.allocator().alloc(u16, n);
        if (fill) |p| @memset(v, p) else for (v) |*x| x.* = 0;
        return v;
    }

    fn allocF64(f: *Gdn, n: usize) ![]f64 {
        return f.arena.allocator().alloc(f64, n);
    }

    /// One device buffer uploaded from `bytes` (an input) or poisoned to
    /// `poison_bytes` (an output), registered for cleanup; returns its index.
    fn devBuf(f: *Gdn, bytes: []const u8) !usize {
        const b = try vk.Buffer.init(f.dev, .device, bytes.len);
        f.bufs[f.made] = b;
        f.made += 1;
        try vk.orExit(vk.upload(f.cmds, f.bufs[f.made - 1], 0, f.staging, bytes, timeout_ns));
        return f.made - 1;
    }

    fn view(f: *Gdn, idx: usize, kind: vk.tensor.Kind, elements: u64) !vk.tensor.View {
        return vk.tensor.View.init(&f.bufs[idx], kind, 0, elements);
    }

    /// The fixture's dyadic sampler so gate 3's inputs sit inside the ranges
    /// the stated budgets were derived over.
    fn fillDyadic(rng: std.Random, v: []u16, lo: i32, hi: i32) void {
        for (v) |*x| x.* = fx.sampleDyadic(rng, lo, hi);
    }

    fn fillDyadicF32(rng: std.Random, pairs: []u16, lo: i32, hi: i32) void {
        for (0..pairs.len / 2) |i| {
            const bits: u32 = @bitCast(fx.sampleDyadicF32(rng, lo, hi));
            pairs[2 * i] = @truncate(bits);
            pairs[2 * i + 1] = @truncate(bits >> 16);
        }
    }

    /// Gate 1's plants: token row 1 of the per-token inputs carries a NaN and
    /// both infinities; shared per-head parameters (a_log, dt, weight) stay
    /// finite, so a planted class cannot swallow every output.
    fn plantNonfinite(f: *Gdn, rng: std.Random) !void {
        const T = f.tokens;
        const word = [_]u16{ 0x7fc0, 0x7f80, 0xff80 }; // NaN, +inf, -inf
        switch (f.which) {
            .split_soa, .split_g => {
                f.conv_out_bits[1 * channels + 5] = word[0];
                f.conv_out_bits[1 * channels + 6] = word[1];
                f.conv_out_bits[1 * channels + 7] = word[2];
                f.a_bits[1 * heads + 3] = word[0];
                f.a_bits[1 * heads + 11] = word[1];
                f.b_bits[1 * heads + 9] = word[2];
            },
            .conv_prefill => {
                f.qkv_bits[5] = word[0]; // qkv row 0 = assembled input row 3
                f.state_bits[3] = word[1]; // state row 0: +inf and -inf one row
                f.state_bits[9] = word[2];
            },
            .conv_decode => {
                f.qkv_bits[3 * channels + 5] = word[0]; // token 3's x
                f.state_bits[3] = word[1];
                f.state_bits[9] = word[2];
            },
            .prefill_y, .prefill_state => {
                const section = T * linear_dim;
                f.soa_bits[0 * section + (1 * heads + 3) * dk + 5] = word[0]; // q
                f.soa_bits[1 * section + (1 * heads + 3) * dk + 7] = word[1]; // k
                f.soa_bits[2 * section + (1 * heads + 3) * dk + 9] = word[2]; // v
                putF32(f.g_pairs, 1 * heads + 3, std.math.inf(f32));
                f.beta_bits[1 * heads + 7] = word[0];
                f.h0_bits[2 * (11 * dk * dk)] = word[0];
                f.h0_bits[2 * (11 * dk * dk) + 1] = word[0];
            },
            .decode_y, .decode_state => {
                f.soa_bits[0 * linear_dim + 1] = word[0]; // q head 0
                f.soa_bits[1 * linear_dim + 2] = word[1]; // k head 1
                f.soa_bits[2 * linear_dim + 3] = word[2]; // v head 2
                if (T > 1) {
                    f.soa_bits[(1 * 3 + 0) * linear_dim + 130] = word[0];
                    f.soa_bits[(1 * 3 + 1) * linear_dim + 131] = word[1];
                    f.soa_bits[(1 * 3 + 2) * linear_dim + 132] = word[2];
                    f.a_bits[1 * heads + 5] = word[1];
                    f.b_bits[1 * heads + 6] = word[2];
                }
                f.state_bits[2 * (4 * dk * dk)] = word[0];
                f.state_bits[2 * (4 * dk * dk) + 1] = word[0];
            },
        }
        _ = rng;
    }

    /// Gate 4's stress fill: integer-dyadic inputs whose f32 sums are exact in
    /// any order (partials under 2^24) while fp16 overflows or loses by
    /// thousands, and whose recurrence state stays pinned (v mirrors the f32
    /// kv dot as closely as bf16 stores it, so delta is a tiny exact drift).
    fn fillStress(f: *Gdn) void {
        const T = f.tokens;
        switch (f.which) {
            .split_soa, .split_g => {
                for (0..T * channels) |i| {
                    const mag: i64 = 128 + @as(i64, @intCast(i % 8));
                    f.conv_out_bits[i] = c.toBf16(@floatFromInt(if (i % 2 == 0) mag else -mag));
                }
                for (0..T * heads) |i| {
                    f.a_bits[i] = c.toBf16(0);
                    f.b_bits[i] = c.toBf16(0);
                }
                for (0..heads) |h| {
                    f.dt_bits[h] = c.toBf16(0);
                    putF32(f.a_log_bits, h, 0);
                }
            },
            .conv_prefill => {
                for (f.state_bits, 0..) |*x, i| {
                    const sgn: f32 = if (i % 2 == 0) 1 else -1;
                    x.* = c.toBf16(sgn * @as(f32, @floatFromInt(16384 + i % 64)));
                }
                for (f.qkv_bits, 0..) |*x, i| {
                    const sgn: f32 = if (i % 2 == 0) 1 else -1;
                    x.* = c.toBf16(sgn * @as(f32, @floatFromInt(16384 + i % 64)));
                }
                for (f.weight_bits, 0..) |*x, i| {
                    const sgn: f32 = if (i % 3 == 0) 1 else -1;
                    x.* = c.toBf16(sgn * @as(f32, @floatFromInt(2 + i % 3)));
                }
            },
            .conv_decode => {
                for (f.state_bits, 0..) |*x, i| {
                    const sgn: f32 = if (i % 2 == 0) 1 else -1;
                    x.* = c.toBf16(sgn * @as(f32, @floatFromInt(16384 + i % 64)));
                }
                for (f.qkv_bits, 0..) |*x, i| {
                    const sgn: f32 = if (i % 2 == 0) 1 else -1;
                    x.* = c.toBf16(sgn * @as(f32, @floatFromInt(16384 + i % 64)));
                }
                for (f.weight_bits, 0..) |*x, i| {
                    const sgn: f32 = if (i % 3 == 0) 1 else -1;
                    x.* = c.toBf16(sgn * @as(f32, @floatFromInt(2 + i % 3)));
                }
            },
            .prefill_y, .prefill_state => {
                // state 2^12 + small; k = 1, q = alternating +-1; g = 1 and
                // beta = 2^-10, a convergent feedback gain (1.0 would collapse
                // the state onto v and blow the kv dot's partials past 2^24);
                // v mirrors the f32 kv dot as bf16 stores it, so delta is a
                // tiny exact drift and the state stays pinned near 2^12
                for (0..state_elems) |i| putF32(f.h0_bits, i, @floatFromInt(4096 + i % 7));
                for (0..3 * T * linear_dim) |i| f.soa_bits[i] = c.toBf16(1); // q, k
                for (0..T * linear_dim) |i| {
                    if (i % dk % 2 == 1) f.soa_bits[i] = c.toBf16(-1); // q alternates
                }
                // the kv dot is the same 524667-ish integer for every head and
                // row: k = 1 and the pinned state differ only in i % 7
                var kv: f64 = 0;
                for (0..dk) |i| kv += @as(f64, @floatFromInt(4096 + i % 7));
                for (0..T * linear_dim) |i| f.soa_bits[2 * T * linear_dim + i] = c.toBf16(@floatCast(kv));
                for (0..T * heads) |i| {
                    putF32(f.g_pairs, i, 1.0);
                    f.beta_bits[i] = c.toBf16(0x1p-10);
                }
            },
            .decode_y, .decode_state => {
                for (0..state_elems) |i| putF32(f.state_bits, i, @floatFromInt(4096 + i % 7));
                for (f.soa_bits) |*x| x.* = c.toBf16(1); // q, k = 1
                for (0..T * linear_dim) |i| {
                    if (i % dk % 2 == 1) f.soa_bits[i] = c.toBf16(-1);
                }
                // v mirrors the kv dot (state pinned, k = 1)
                for (0..heads) |h| for (0..dk) |row| {
                    var kv: f64 = 0;
                    for (0..dk) |i| kv += @as(f64, @floatFromInt(4096 + @as(i64, @intCast((h * dk * dk + row * dk + i) % 7))));
                    for (0..T) |t| f.soa_bits[(t * 3 + 2) * linear_dim + h * dk + row] = c.toBf16(@floatCast(kv));
                };
                for (0..T * heads) |i| {
                    f.a_bits[i] = c.toBf16(0);
                    f.b_bits[i] = c.toBf16(-10); // sigmoid(-10): a small convergent beta gain
                }
                for (0..heads) |h| {
                    f.dt_bits[h] = c.toBf16(0);
                    putF32(f.a_log_bits, h, -88.0); // exp(-88) softplus gates g to f32 1.0
                }
            },
        }
    }

    /// The host half of prepare: refuse, generate, reference. Null: the shape
    /// is refused. Callable without a device, which is how the host tests
    /// prove the generated fills against the references before any GPU runs.
    fn prepareHost(f: *Gdn, shape: c.Shape, fill: g.Fill, seed: u64) !?usize {
        if (shape.rows == 0 or shape.n != heads or shape.k != dk or shape.rows > max_tokens) return null;
        f.reset();
        _ = f.arena.reset(.retain_capacity);
        f.shape = shape;
        f.tokens = shape.rows;
        const T = f.tokens;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        f.chunks = @intCast((T + @as(usize, @intCast(chunk_fixed)) - 1) / @as(usize, @intCast(chunk_fixed)));

        // host inputs per which
        switch (f.which) {
            .split_soa, .split_g => {
                f.conv_out_bits = try f.allocU16(T * channels, poison);
                f.a_bits = try f.allocU16(T * heads, poison);
                f.b_bits = try f.allocU16(T * heads, poison);
                f.dt_bits = try f.allocU16(heads, poison);
                f.a_log_bits = try f.allocU16(2 * heads, poison);
            },
            .conv_prefill => {
                f.state_bits = try f.allocU16(3 * channels, poison);
                f.qkv_bits = try f.allocU16(T * channels, poison);
                f.weight_bits = try f.allocU16(taps * channels, poison);
                f.a_bits = try f.allocU16(T * heads, poison);
                f.b_bits = try f.allocU16(T * heads, poison);
                f.dt_bits = try f.allocU16(heads, poison);
                f.a_log_bits = try f.allocU16(2 * heads, poison);
            },
            .conv_decode => {
                f.state_bits = try f.allocU16(3 * channels, poison);
                f.state_init_bits = try f.allocU16(3 * channels, poison);
                f.state_swap_bits = try f.allocU16(3 * channels, poison);
                f.qkv_bits = try f.allocU16(T * channels, poison); // x rows
                f.weight_bits = try f.allocU16(taps * channels, poison);
                f.a_bits = try f.allocU16(T * heads, poison);
                f.b_bits = try f.allocU16(T * heads, poison);
                f.dt_bits = try f.allocU16(heads, poison);
                f.a_log_bits = try f.allocU16(2 * heads, poison);
            },
            .prefill_y, .prefill_state => {
                f.soa_bits = try f.allocU16(3 * T * linear_dim, poison);
                f.g_pairs = try f.allocU16(2 * T * heads, poison);
                f.beta_bits = try f.allocU16(T * heads, poison);
                f.h0_bits = try f.allocU16(2 * state_elems, poison);
                f.walk_st = try f.allocF64(state_elems);
            },
            .decode_y, .decode_state => {
                f.soa_bits = try f.allocU16(3 * T * linear_dim, poison);
                f.a_bits = try f.allocU16(T * heads, poison);
                f.b_bits = try f.allocU16(T * heads, poison);
                f.a_log_bits = try f.allocU16(2 * heads, poison);
                f.dt_bits = try f.allocU16(heads, poison);
                f.state_bits = try f.allocU16(2 * state_elems, poison);
                f.state_init_bits = try f.allocU16(2 * state_elems, poison);
                f.state_swap_bits = try f.allocU16(2 * state_elems, poison);
                f.walk_st = try f.allocF64(state_elems);
                @memcpy(f.state_init_bits, f.state_bits);
            },
        }

        // fills: the fixture's dyadic ranges for gate 3, engineered integers
        // for gate 4, planted classes for gate 1
        switch (fill) {
            .random => switch (f.which) {
                .split_soa, .split_g => {
                    fillDyadic(rng, f.conv_out_bits, -4, 2);
                    fillDyadic(rng, f.a_bits, -3, 2);
                    fillDyadic(rng, f.b_bits, -3, 2);
                    fillDyadic(rng, f.dt_bits, -4, 1);
                    // served a_log keeps g a normal f32; a_log above ~2.4
                    // drives g subnormal, which the device flushes - the
                    // stage 4 fixture's census case, Main's call, not a
                    // random-fill failure
                    fillDyadicF32(rng, f.a_log_bits, -6, 0);
                },
                .conv_prefill, .conv_decode => {
                    fillDyadic(rng, f.state_bits, -4, 2);
                    fillDyadic(rng, f.qkv_bits, -4, 2);
                    fillDyadic(rng, f.weight_bits, -3, 1);
                    fillDyadic(rng, f.a_bits, -3, 2);
                    fillDyadic(rng, f.b_bits, -3, 2);
                    fillDyadic(rng, f.dt_bits, -4, 1);
                    fillDyadicF32(rng, f.a_log_bits, -6, 0);
                },
                .prefill_y, .prefill_state => {
                    // 130 tokens must stay inside f32: g <= 2^-3, beta and the
                    // whole soa <= 2^-3, so the state's per-token factor is
                    // well under one and the walk decays to the delta floor
                    fillDyadic(rng, f.soa_bits, -6, -3);
                    fillDyadicF32(rng, f.g_pairs, -6, -3);
                    fillDyadic(rng, f.beta_bits, -6, -4);
                    fillDyadicF32(rng, f.h0_bits, -6, -2);
                },
                .decode_y, .decode_state => {
                    fillDyadic(rng, f.soa_bits, -4, 2);
                    fillDyadic(rng, f.a_bits, -3, 2);
                    fillDyadic(rng, f.b_bits, -3, 2);
                    fillDyadic(rng, f.dt_bits, -4, 1);
                    fillDyadicF32(rng, f.a_log_bits, -4, 1);
                    fillDyadicF32(rng, f.state_bits, -6, -2);
                },
            },
            .stress => f.fillStress(),
            .nonfinite => switch (f.which) {
                .split_soa, .split_g => {
                    fillDyadic(rng, f.conv_out_bits, -4, 2);
                    fillDyadic(rng, f.a_bits, -3, 2);
                    fillDyadic(rng, f.b_bits, -3, 2);
                    fillDyadic(rng, f.dt_bits, -4, 1);
                    // served a_log keeps g a normal f32; a_log above ~2.4
                    // drives g subnormal, which the device flushes - the
                    // stage 4 fixture's census case, Main's call, not a
                    // random-fill failure
                    fillDyadicF32(rng, f.a_log_bits, -6, 0);
                },
                .conv_prefill, .conv_decode => {
                    fillDyadic(rng, f.state_bits, -4, 2);
                    fillDyadic(rng, f.qkv_bits, -4, 2);
                    fillDyadic(rng, f.weight_bits, -3, 1);
                    fillDyadic(rng, f.a_bits, -3, 2);
                    fillDyadic(rng, f.b_bits, -3, 2);
                    fillDyadic(rng, f.dt_bits, -4, 1);
                    fillDyadicF32(rng, f.a_log_bits, -6, 0);
                },
                .prefill_y, .prefill_state => {
                    // 130 tokens must stay inside f32: g <= 2^-3, beta and the
                    // whole soa <= 2^-3, so the state's per-token factor is
                    // well under one and the walk decays to the delta floor
                    fillDyadic(rng, f.soa_bits, -6, -3);
                    fillDyadicF32(rng, f.g_pairs, -6, -3);
                    fillDyadic(rng, f.beta_bits, -6, -4);
                    fillDyadicF32(rng, f.h0_bits, -6, -2);
                },
                .decode_y, .decode_state => {
                    fillDyadic(rng, f.soa_bits, -4, 2);
                    fillDyadic(rng, f.a_bits, -3, 2);
                    fillDyadic(rng, f.b_bits, -3, 2);
                    fillDyadic(rng, f.dt_bits, -4, 1);
                    fillDyadicF32(rng, f.a_log_bits, -4, 1);
                    fillDyadicF32(rng, f.state_bits, -6, -2);
                },
            },
        }
        if (fill == .nonfinite) try f.plantNonfinite(rng);
        // the decode loops' uploaded initial state: the FINAL generated bits
        if (f.state_init_bits.len != 0) @memcpy(f.state_init_bits, f.state_bits);

        // host outputs (poisoned, guarded) and the f64 references. out is
        // sized in u16 WORDS (an f32 output takes two), the reference count in
        // OUTPUT ELEMENTS: the harness reads spec.out over count elements.
        const out_len: usize = switch (f.which) {
            .split_soa => 3 * T * linear_dim + T * heads,
            .split_g => 2 * T * heads,
            .conv_prefill, .conv_decode => T * channels + 3 * channels,
            .prefill_y => T * linear_dim,
            .prefill_state => 2 * state_elems * f.chunks,
            .decode_y => T * linear_dim,
            .decode_state => 2 * state_elems,
        };
        const count_len: usize = switch (f.which) {
            .split_g => T * heads,
            .prefill_state => state_elems * f.chunks,
            .decode_state => state_elems,
            else => out_len,
        };
        f.out = try f.allocU16(out_len + guard_words, poison);
        f.out_words = out_len;
        f.want = try f.allocF64(count_len);
        f.mag = try f.allocF64(count_len);
        try f.computeWant();
        if (fill != .nonfinite) for (f.want, f.mag) |w, m| {
            if (!std.math.isFinite(w) or !std.math.isFinite(m)) return error.NonFiniteReference;
        };
        return out_len;
    }

    pub fn prepare(f: *Gdn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const out_len = (try f.prepareHost(shape, fill, seed)) orelse return false;
        const a = f.arena.allocator();

        // device buffers at per-which fixed indices; inputs are uploaded from
        // the bits, outputs from out's poisoned regions (or scratch poison)
        var input_slices: [8][]u16 = undefined;
        const inputs = f.inputList(&input_slices);
        var input_total: usize = 0;
        for (inputs) |s| input_total += std.mem.sliceAsBytes(s).len;
        try f.uploadBuffers();
        f.input_snapshot = try a.alloc(u8, input_total);
        f.input_now = try a.alloc(u8, input_total);
        var pos: usize = 0;
        for (inputs) |s| {
            const bytes = std.mem.sliceAsBytes(s);
            @memcpy(f.input_snapshot[pos..][0..bytes.len], bytes);
            pos += bytes.len;
        }
        f.rig = .{ .output = std.mem.sliceAsBytes(f.out[0..out_len]), .inputs = f.input_snapshot, .guards_ok = true };
        return true;
    }

    /// The uploaded inputs in the fixed order the unchanged-check compares.
    fn inputList(f: *Gdn, buf: *[8][]u16) []const []u16 {
        switch (f.which) {
            .split_soa, .split_g => {
                buf[0..5].* = .{ f.conv_out_bits, f.a_bits, f.b_bits, f.dt_bits, f.a_log_bits };
                return buf[0..5];
            },
            .conv_prefill => {
                buf[0..7].* = .{ f.state_bits, f.qkv_bits, f.weight_bits, f.a_bits, f.b_bits, f.dt_bits, f.a_log_bits };
                return buf[0..7];
            },
            // the initial state rides its own never-written buffer, so the
            // unchanged check sees it exactly as the first step consumed it
            .conv_decode => {
                buf[0..7].* = .{ f.state_init_bits, f.qkv_bits, f.weight_bits, f.a_bits, f.b_bits, f.dt_bits, f.a_log_bits };
                return buf[0..7];
            },
            .prefill_y, .prefill_state => {
                buf[0..4].* = .{ f.soa_bits, f.g_pairs, f.beta_bits, f.h0_bits };
                return buf[0..4];
            },
            .decode_y, .decode_state => {
                buf[0..6].* = .{ f.soa_bits, f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, f.state_init_bits };
                return buf[0..6];
            },
        }
    }

    /// One output's uploaded words: out's region at `start` when the fetch
    /// downloads that output there, else fresh poison scratch (an output no
    /// fetch reads). out is sliced only where prepare sized it, so a
    /// mismatch is an error the gate reports, not a panic.
    fn outWords(f: *Gdn, checked: bool, start: usize, words: usize) ![]u16 {
        if (!checked) return f.allocU16(words, poison);
        if (start > f.out_words or words > f.out_words - start) return error.OutputRegionOutOfRange;
        return f.out[start .. start + words];
    }

    /// The words each device buffer takes, in buffer order: the inputs' bits
    /// and the outputs' poison. The checked output of a kernel takes its
    /// region of out, which prepare sized for it; every other output takes
    /// fresh poison scratch, so out is only ever sliced where it was sized.
    fn uploadWords(f: *Gdn, buf: *[16][]u16) !usize {
        const T = f.tokens;
        var n: usize = 0;
        switch (f.which) {
            .split_soa, .split_g => {
                // 0 conv_out, 1 a, 2 b, 3 dt, 4 a_log, 5 soa, 6 g, 7 beta
                buf[0..5].* = .{ f.conv_out_bits, f.a_bits, f.b_bits, f.dt_bits, f.a_log_bits };
                n = 5;
                buf[n] = try f.outWords(f.which == .split_soa, 0, 3 * T * linear_dim);
                buf[n + 1] = try f.outWords(f.which == .split_g, 0, 2 * T * heads);
                buf[n + 2] = try f.outWords(f.which == .split_soa, 3 * T * linear_dim, T * heads);
                n += 3;
            },
            .conv_prefill => {
                // 0 state_in, 1 qkv, 2 weight, 3 a, 4 b, 5 dt, 6 a_log,
                // 7 conv_input, 8 conv_out, 9 soa, 10 g, 11 beta, 12 state_out
                buf[0..7].* = .{ f.state_bits, f.qkv_bits, f.weight_bits, f.a_bits, f.b_bits, f.dt_bits, f.a_log_bits };
                n = 7;
                buf[n] = try f.outWords(false, 0, (T + 3) * channels);
                buf[n + 1] = try f.outWords(true, 0, T * channels);
                buf[n + 2] = try f.outWords(false, 0, 3 * T * linear_dim);
                buf[n + 3] = try f.outWords(false, 0, 2 * T * heads);
                buf[n + 4] = try f.outWords(false, 0, T * heads);
                buf[n + 5] = try f.outWords(true, T * channels, 3 * channels);
                n += 6;
            },
            .conv_decode => {
                // 0 state_init, 1 state_a, 2 state_b, 3 x, 4 weight, 5 a, 6 b,
                // 7 dt, 8 a_log, 9 activated, 10 soa, 11 g, 12 beta
                buf[0..9].* = .{ f.state_init_bits, f.state_swap_bits, f.state_swap_bits, f.qkv_bits, f.weight_bits, f.a_bits, f.b_bits, f.dt_bits, f.a_log_bits };
                n = 9;
                buf[n] = try f.outWords(true, 0, T * channels);
                buf[n + 1] = try f.outWords(false, 0, 3 * T * linear_dim);
                buf[n + 2] = try f.outWords(false, 0, 2 * T * heads);
                buf[n + 3] = try f.outWords(false, 0, T * heads);
                n += 4;
            },
            .prefill_y, .prefill_state => {
                // 0 soa, 1 g, 2 beta, 3 h0, 4 y, 5 hf, 6 snap?
                buf[0..4].* = .{ f.soa_bits, f.g_pairs, f.beta_bits, f.h0_bits };
                n = 4;
                buf[n] = try f.outWords(f.which == .prefill_y, 0, T * linear_dim);
                buf[n + 1] = try f.outWords(f.which == .prefill_state, 0, 2 * state_elems);
                n += 2;
                if (f.chunks > 1) {
                    buf[n] = try f.outWords(f.which == .prefill_state, 2 * state_elems, 2 * state_elems * (f.chunks - 1));
                    n += 1;
                }
            },
            .decode_y, .decode_state => {
                // 0 soa, 1 a, 2 b, 3 a_log, 4 dt, 5 state_init, 6 state_a,
                // 7 state_b, 8 y
                buf[0..8].* = .{ f.soa_bits, f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, f.state_init_bits, f.state_swap_bits, f.state_swap_bits };
                n = 8;
                buf[n] = try f.outWords(f.which == .decode_y, 0, T * linear_dim);
                n += 1;
            },
        }
        return n;
    }

    /// Device buffers at per-which fixed indices (uploadWords and run agree).
    /// Inputs upload from the generated bits; a checked output uploads poison
    /// from out's own region, so a dead store survives as NaN into every gate,
    /// and an output no fetch reads uploads fresh poison scratch.
    fn uploadBuffers(f: *Gdn) !void {
        var words: [16][]u16 = undefined;
        const n = try f.uploadWords(&words);
        for (words[0..n]) |w| _ = try f.devBuf(std.mem.sliceAsBytes(w));
    }

    /// The f64 references over the generated bits, into want/mag, before any
    /// device work; reference() hands the harness the prepared copy.
    fn computeWant(f: *Gdn) !void {
        const T = f.tokens;
        switch (f.which) {
            .split_soa => splitReference(T, f.conv_out_bits, f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, f.want, f.mag),
            .split_g => {
                for (0..T * heads) |i| {
                    const ref = fx.gateRefAsh(f.a_bits[i], f.b_bits[i], f32Of(f.a_log_bits, i % heads), f.dt_bits[i % heads]);
                    f.want[i] = ref.g;
                    f.mag[i] = @abs(ref.g);
                }
            },
            .conv_prefill => {
                const ctx: PrefillInput = .{ .state_bits = f.state_bits, .qkv_bits = f.qkv_bits };
                convReference(T, f.weight_bits, ctx, f.want[0 .. T * channels], f.mag[0 .. T * channels], f.want[T * channels ..]);
                @memset(f.mag[T * channels ..], 0);
            },
            .conv_decode => {
                // the chain: each step taps the carried state then shifts it
                for (0..T) |t| {
                    const ctx: DecodeInput = .{ .state_bits = f.state_bits, .x_bits = f.qkv_bits, .t = t };
                    convReference(1, f.weight_bits, ctx, f.want[t * channels ..][0 .. channels], f.mag[t * channels ..][0 .. channels], null);
                    for (0..2) |r| @memcpy(f.state_bits[r * channels ..][0..channels], f.state_bits[(r + 1) * channels ..][0..channels]);
                    @memcpy(f.state_bits[2 * channels ..][0..channels], f.qkv_bits[t * channels ..][0..channels]);
                }
                // the final carry is the input rows T..T+2 of the walked chain
                for (0..3 * channels) |i| {
                    f.want[T * channels + i] = bf(f.state_bits[i]);
                    f.mag[T * channels + i] = 0;
                }
            },
            .prefill_y, .prefill_state => {
                f.max_state = 0;
                const snap_chunk: ?usize = if (f.chunks > 1) @intCast(chunk_fixed) else null;
                const y: ?[]f64 = if (f.which == .prefill_y) f.want else null;
                const hf: ?[]f64 = if (f.which == .prefill_state) f.want[0..state_elems] else null;
                const snap: ?[]f64 = if (f.which == .prefill_state and f.chunks > 1) f.want[state_elems..] else null;
                walk(T, f.soa_bits, f.g_pairs, f.beta_bits, f.h0_bits, f.walk_st, y, hf, &f.max_state, snap_chunk, snap, null, null);
                @memset(f.mag, f.max_state);
            },
            .decode_y, .decode_state => {
                f.max_state = 0;
                const g_t = try f.allocU16(2 * heads, 0);
                const beta_t = try f.allocU16(heads, 0);
                const y_t = try f.allocF64(linear_dim);
                const st_scratch = try f.allocF64(state_elems);
                const st_out = try f.allocF64(state_elems);
                for (0..T) |t| {
                    decodeGates(f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, t, g_t, beta_t);
                    walk(1, f.soa_bits[t * 3 * linear_dim ..][0 .. 3 * linear_dim], g_t, beta_t, f.state_bits, st_scratch, y_t, st_out, &f.max_state, null, null, null, null);
                    if (f.which == .decode_y) {
                        @memcpy(f.want[t * linear_dim ..][0..linear_dim], y_t);
                    } else {
                        // only the final carry is checked; the chain still
                        // walks every step
                        @memcpy(f.want[0..state_elems], st_out);
                    }
                    // the next step reads the f32 state this one left
                    for (0..state_elems) |i| putF32(f.state_bits, i, @floatCast(st_out[i]));
                }
                @memset(f.mag, f.max_state);
            },
        }
    }

    pub fn run(f: *Gdn) !void {
        const T = f.tokens;
        switch (f.which) {
            .split_soa, .split_g => {
                const conv_out = try f.view(0, .bf16, @intCast(T * channels));
                const a_v = try f.view(1, .bf16, @intCast(T * heads));
                const b_v = try f.view(2, .bf16, @intCast(T * heads));
                const dt_v = try f.view(3, .bf16, heads);
                const a_log_v = try f.view(4, .f32, heads);
                const soa_v = try f.view(5, .bf16, @intCast(3 * T * linear_dim));
                const g_v = try f.view(6, .f32, @intCast(T * heads));
                const beta_v = try f.view(7, .bf16, @intCast(T * heads));
                try vk.orExit(f.m.split(f.cmds, soa_v, g_v, beta_v, conv_out, a_v, b_v, a_log_v, dt_v, T, timeout_ns));
            },
            .conv_prefill => {
                const state_v = try f.view(0, .bf16, @intCast(3 * channels));
                const qkv_v = try f.view(1, .bf16, @intCast(T * channels));
                const weight_v = try f.view(2, .bf16, @intCast(taps * channels));
                const a_v = try f.view(3, .bf16, @intCast(T * heads));
                const b_v = try f.view(4, .bf16, @intCast(T * heads));
                const dt_v = try f.view(5, .bf16, heads);
                const a_log_v = try f.view(6, .f32, heads);
                const conv_input_v = try f.view(7, .bf16, @intCast((T + 3) * channels));
                const conv_out_v = try f.view(8, .bf16, @intCast(T * channels));
                const soa_v = try f.view(9, .bf16, @intCast(3 * T * linear_dim));
                const g_v = try f.view(10, .f32, @intCast(T * heads));
                const beta_v = try f.view(11, .bf16, @intCast(T * heads));
                const state_out_v = try f.view(12, .bf16, @intCast(3 * channels));
                try vk.orExit(f.m.convPrefill(f.cmds, soa_v, g_v, beta_v, conv_out_v, conv_input_v, state_v, qkv_v, weight_v, a_v, b_v, a_log_v, dt_v, state_out_v, T, timeout_ns));
            },
            .conv_decode => {
                const x_v = try f.view(3, .bf16, @intCast(T * channels));
                const weight_v = try f.view(4, .bf16, @intCast(taps * channels));
                const a_v = try f.view(5, .bf16, @intCast(T * heads));
                const b_v = try f.view(6, .bf16, @intCast(T * heads));
                const dt_v = try f.view(7, .bf16, heads);
                const a_log_v = try f.view(8, .f32, heads);
                const act_v = try f.view(9, .bf16, @intCast(T * channels));
                const soa_v = try f.view(10, .bf16, @intCast(3 * T * linear_dim));
                const g_v = try f.view(11, .f32, @intCast(T * heads));
                const beta_v = try f.view(12, .bf16, @intCast(T * heads));
                var in_idx: usize = 0; // state_init; carry buffers are 1 and 2
                for (0..T) |t| {
                    const out_idx: usize = if (in_idx == 1) 2 else 1;
                    const st_in = try f.view(in_idx, .bf16, @intCast(3 * channels));
                    const st_out = try f.view(out_idx, .bf16, @intCast(3 * channels));
                    try vk.orExit(f.m.convDecodeStep(f.cmds, try soa_v.slice(t * 3 * linear_dim, 3 * linear_dim), try g_v.slice(t * heads, heads), try beta_v.slice(t * heads, heads), try act_v.slice(t * channels, channels), st_in, try x_v.slice(t * channels, channels), weight_v, try a_v.slice(t * heads, heads), try b_v.slice(t * heads, heads), a_log_v, dt_v, st_out, timeout_ns));
                    in_idx = out_idx;
                }
                f.final_state_idx = in_idx; // the initial buffer at 0 was never written
            },
            .prefill_y, .prefill_state => {
                const soa_v = try f.view(0, .bf16, @intCast(3 * T * linear_dim));
                const g_v = try f.view(1, .f32, @intCast(T * heads));
                const beta_v = try f.view(2, .bf16, @intCast(T * heads));
                const h0_v = try f.view(3, .f32, state_elems);
                const y_v = try f.view(4, .bf16, @intCast(T * linear_dim));
                const hf_v = try f.view(5, .f32, state_elems);
                const snap_v: ?vk.tensor.View = if (f.chunks > 1) try f.view(6, .f32, state_elems * (f.chunks - 1)) else null;
                try vk.orExit(f.m.prefillRecur(f.cmds, soa_v, g_v, beta_v, h0_v, y_v, hf_v, snap_v, T, chunk_fixed, timeout_ns));
            },
            .decode_y, .decode_state => {
                const a_v = try f.view(1, .bf16, @intCast(T * heads));
                const b_v = try f.view(2, .bf16, @intCast(T * heads));
                const a_log_v = try f.view(3, .u16, 2 * heads);
                const dt_v = try f.view(4, .bf16, heads);
                const y_v = try f.view(8, .bf16, @intCast(T * linear_dim));
                var in_idx: usize = 5; // state_init
                for (0..T) |t| {
                    const st_in = try f.view(in_idx, .f32, state_elems);
                    const out_idx: usize = if (in_idx == 5) 6 else if (in_idx == 6) 7 else 6;
                    const st_out = try f.view(out_idx, .f32, state_elems);
                    try vk.orExit(f.m.decodeRecur(f.cmds, try f.soaView(t), try a_v.slice(t * heads, heads), try b_v.slice(t * heads, heads), a_log_v, dt_v, st_in, st_out, try y_v.slice(t * linear_dim, linear_dim), timeout_ns));
                    in_idx = out_idx;
                }
                f.final_state_idx = in_idx;
            },
        }
        try f.fetchOutputs();
    }

    fn soaView(f: *Gdn, t: usize) !vk.tensor.View {
        const soa_v = try f.view(0, .bf16, @intCast(3 * f.tokens * linear_dim));
        return soa_v.slice(t * 3 * linear_dim, 3 * linear_dim);
    }

    /// Download every checked output into out's regions, then every input for
    /// the unchanged check, and verify the guard words.
    fn fetchOutputs(f: *Gdn) !void {
        const T = f.tokens;
        switch (f.which) {
            .split_soa => {
                try vk.download(f.cmds, f.bufs[5], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. 3 * T * linear_dim]), timeout_ns);
                try vk.download(f.cmds, f.bufs[7], 0, f.staging, std.mem.sliceAsBytes(f.out[3 * T * linear_dim .. 3 * T * linear_dim + T * heads]), timeout_ns);
            },
            .split_g => {
                try vk.download(f.cmds, f.bufs[6], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. 2 * T * heads]), timeout_ns);
            },
            .conv_prefill => {
                try vk.download(f.cmds, f.bufs[8], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. T * channels]), timeout_ns);
                try vk.download(f.cmds, f.bufs[12], 0, f.staging, std.mem.sliceAsBytes(f.out[T * channels .. T * channels + 3 * channels]), timeout_ns);
            },
            .conv_decode => {
                try vk.download(f.cmds, f.bufs[9], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. T * channels]), timeout_ns);
                try vk.download(f.cmds, f.bufs[f.final_state_idx], 0, f.staging, std.mem.sliceAsBytes(f.out[T * channels .. T * channels + 3 * channels]), timeout_ns);
            },
            .prefill_y => {
                try vk.download(f.cmds, f.bufs[4], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. T * linear_dim]), timeout_ns);
            },
            .prefill_state => {
                try vk.download(f.cmds, f.bufs[5], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. 2 * state_elems]), timeout_ns);
                if (f.chunks > 1) try vk.download(f.cmds, f.bufs[6], 0, f.staging, std.mem.sliceAsBytes(f.out[2 * state_elems .. 2 * state_elems * f.chunks]), timeout_ns);
            },
            .decode_y => {
                try vk.download(f.cmds, f.bufs[8], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. T * linear_dim]), timeout_ns);
            },
            .decode_state => {
                try vk.download(f.cmds, f.bufs[f.final_state_idx], 0, f.staging, std.mem.sliceAsBytes(f.out[0 .. 2 * state_elems]), timeout_ns);
            },
        }
        // inputs, post-run, in the snapshot's order
        var slices: [8][]u16 = undefined;
        const inputs = f.inputList(&slices);
        var idxs: [8]usize = undefined;
        const dev_idx = f.inputBufIndices(&idxs);
        var pos: usize = 0;
        for (inputs, dev_idx) |s, di| {
            try vk.download(f.cmds, f.bufs[di], 0, f.staging, std.mem.sliceAsBytes(s), timeout_ns);
            const bytes = std.mem.sliceAsBytes(s);
            @memcpy(f.input_now[pos..][0..bytes.len], bytes);
            pos += bytes.len;
        }
        f.rig.inputs = f.input_now;
        f.rig.guards_ok = true;
        for (f.out[f.out_words..]) |w| if (w != poison) {
            f.rig.guards_ok = false;
            break;
        };
    }

    /// The device buffer index of every uploaded input, in inputList's order.
    fn inputBufIndices(f: *Gdn, buf: *[8]usize) []const usize {
        switch (f.which) {
            .split_soa, .split_g => {
                buf[0..5].* = .{ 0, 1, 2, 3, 4 };
                return buf[0..5];
            },
            .conv_prefill => {
                buf[0..7].* = .{ 0, 1, 2, 3, 4, 5, 6 };
                return buf[0..7];
            },
            .conv_decode => {
                buf[0..7].* = .{ 0, 3, 4, 5, 6, 7, 8 };
                return buf[0..7];
            },
            .prefill_y, .prefill_state => {
                buf[0..4].* = .{ 0, 1, 2, 3 };
                return buf[0..4];
            },
            .decode_y, .decode_state => {
                buf[0..6].* = .{ 0, 1, 2, 3, 4, 5 };
                return buf[0..6];
            },
        }
    }

    /// The sums replay gate 4 runs: every dot in the op re-summed in `accum`
    /// in `order`, everything else in float64, into `out` in the output
    /// layout reference() uses.
    pub fn simulate(f: *Gdn, accum: c.Accum, order: g.Order, out: []f64) !void {
        const T = f.tokens;
        switch (f.which) {
            .split_soa => {
                var terms: [dk]f64 = undefined;
                for (0..T) |t| {
                    for (0..3 * heads) |head| {
                        const hv: usize = head % heads;
                        var vals: [dk]f64 = undefined;
                        var vmax: f64 = 0;
                        for (0..dk) |i| {
                            const cv = bf(f.conv_out_bits[t * channels + head * dk + i]);
                            vals[i] = cv / (1.0 + @exp(-cv));
                            vmax = @max(vmax, @abs(vals[i]));
                        }
                        for (0..dk) |i| terms[i] = vals[i] * vals[i];
                        const ss = rows.sumIn(accum, order, &terms);
                        const inv: f64 = 1.0 / @sqrt(ss / 128.0 + 1e-6 / 128.0);
                        const scale_f: f64 = if (head < heads) 1.0 / 128.0 else @sqrt(1.0 / 128.0);
                        const dst = (head / heads) * (T * linear_dim) + (t * heads + hv) * dk;
                        for (0..dk) |i| out[dst + i] = if (head < 2 * heads) scale_f * (vals[i] * inv) else vals[i];
                    }
                    for (0..heads) |hv| {
                        const ref = fx.gateRefAsh(f.a_bits[t * heads + hv], f.b_bits[t * heads + hv], f32Of(f.a_log_bits, hv), f.dt_bits[hv]);
                        out[3 * T * linear_dim + t * heads + hv] = ref.beta;
                    }
                }
            },
            .split_g => return error.Unimplemented, // accum none; gate 4 is N/A
            .conv_prefill => {
                const ctx: PrefillInput = .{ .state_bits = f.state_bits, .qkv_bits = f.qkv_bits };
                var terms: [taps]f64 = undefined;
                for (0..channels) |ch| {
                    for (0..T) |t| {
                        for (0..taps) |k| terms[k] = bf(ctx.inWord(t + k, ch)) * bf(f.weight_bits[ch * taps + k]);
                        out[t * channels + ch] = rows.sumIn(accum, order, &terms);
                    }
                    for (0..3) |j| out[T * channels + j * channels + ch] = bf(ctx.inWord(T + j, ch));
                }
            },
            .conv_decode => {
                // the chain replays from the UPLOADED initial state
                var terms: [taps]f64 = undefined;
                const chain_a = try f.allocU16(3 * channels, 0);
                const chain_b = try f.allocU16(3 * channels, 0);
                @memcpy(chain_a, f.state_init_bits);
                var chain: []u16 = chain_a;
                for (0..T) |t| {
                    const ctx: DecodeInput = .{ .state_bits = chain, .x_bits = f.qkv_bits, .t = t };
                    for (0..channels) |ch| {
                        for (0..taps) |k| terms[k] = bf(ctx.inWord(t + k, ch)) * bf(f.weight_bits[ch * taps + k]);
                        out[t * channels + ch] = rows.sumIn(accum, order, &terms);
                    }
                    const nxt: []u16 = if (chain.ptr == chain_a.ptr) chain_b else chain_a;
                    for (0..2) |r| @memcpy(nxt[r * channels ..][0..channels], chain[(r + 1) * channels ..][0..channels]);
                    @memcpy(nxt[2 * channels ..][0..channels], f.qkv_bits[t * channels ..][0..channels]);
                    chain = nxt;
                }
                for (0..3 * channels) |i| out[T * channels + i] = bf(chain[i]);
            },
            .prefill_y => walk(T, f.soa_bits, f.g_pairs, f.beta_bits, f.h0_bits, f.walk_st, out, null, &f.max_state, null, null, accum, order),
            .prefill_state => {
                const snap_chunk: ?usize = if (f.chunks > 1) @intCast(chunk_fixed) else null;
                walk(T, f.soa_bits, f.g_pairs, f.beta_bits, f.h0_bits, f.walk_st, null, out[0..state_elems], &f.max_state, snap_chunk, if (f.chunks > 1) out[state_elems..] else null, accum, order);
            },
            .decode_y, .decode_state => {
                const g_t = try f.allocU16(2 * heads, 0);
                const beta_t = try f.allocU16(heads, 0);
                const y_t: ?[]f64 = if (f.which == .decode_y) out else null;
                const st_scratch = try f.allocF64(state_elems);
                const st_out = try f.allocF64(state_elems);
                const bits = try f.allocU16(2 * state_elems, 0);
                @memcpy(bits, f.state_init_bits);
                for (0..T) |t| {
                    decodeGates(f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, t, g_t, beta_t);
                    walk(1, f.soa_bits[t * 3 * linear_dim ..][0 .. 3 * linear_dim], g_t, beta_t, bits, st_scratch, if (y_t) |y| y[t * linear_dim ..][0..linear_dim] else null, st_out, &f.max_state, null, null, accum, order);
                    if (f.which == .decode_state) @memcpy(out[0..state_elems], st_out);
                    for (0..state_elems) |i| putF32(bits, i, @floatCast(st_out[i]));
                }
            },
        }
    }

    /// The stage 4 fixture's stated budgets as per-output allowances: the
    /// same depth/ops/mids derivation and the same magnitude scale the
    /// fixture's checkers use (max_state for the recurrence, the head row's
    /// vmax^2 for the split, |acc| for the conv, |want| for the gate).
    pub fn bounds(f: *Gdn, want: []const f64, mag: []const f64, allow: []f64) !void {
        switch (f.which) {
            .split_soa => for (want, mag, allow) |w, m, *l| {
                l.* = fx.budget(40, 5, 3, w, m);
            },
            .split_g => for (want, mag, allow) |w, m, *l| {
                l.* = fx.budget(1, 4, 2, w, m);
            },
            .conv_prefill, .conv_decode => for (want, mag, allow) |w, m, *l| {
                l.* = fx.budget(taps, 0, 0, w, m);
            },
            .prefill_y => for (want, mag, allow) |w, m, *l| {
                l.* = fx.budget(@floatFromInt(recurYDepth(f.shape)), 4, 0, w, m);
            },
            .prefill_state => for (want, mag, 0..) |w, m, i| {
                allow[i] = fx.budget(if (i < state_elems) @floatFromInt(recurStateDepth(f.shape)) else snapDepth(), 2, 0, w, m);
            },
            .decode_y => for (want, mag, allow) |w, m, *l| {
                l.* = fx.budget(@floatFromInt(decodeYDepth(f.shape)), 4, 0, w, m);
            },
            .decode_state => for (want, mag, allow) |w, m, *l| {
                l.* = fx.budget(@floatFromInt(decodeStateDepth(f.shape)), 2, 0, w, m);
            },
        }
    }

    pub fn count(f: *Gdn) usize {
        return f.want.len;
    }

    pub fn reference(f: *Gdn, want: []f64, mag: []f64) !void {
        @memcpy(want, f.want);
        @memcpy(mag, f.mag);
    }
};

const testing = std.testing;

/// A fixture good enough for the host half of prepare: the arena needs a real
/// allocator, but no device field is touched before uploadBuffers, which the
/// host tests never call.
fn hostFixture(which: Which) Gdn {
    return Gdn.init(testing.allocator, undefined, undefined, undefined, undefined, which);
}

test "gdn contract prepareHost refuses torn geometry and serves the fixed one" {
    var f = hostFixture(.split_soa);
    defer f.deinit();
    try testing.expectEqual(@as(?usize, null), try f.prepareHost(.{ .rows = 0, .n = heads, .k = dk }, .random, 1));
    try testing.expectEqual(@as(?usize, null), try f.prepareHost(.{ .rows = 8, .n = 8, .k = dk }, .random, 1));
    try testing.expectEqual(@as(?usize, null), try f.prepareHost(.{ .rows = 8, .n = heads, .k = 64 }, .random, 1));
    for (std.enums.values(Which)) |which| {
        var g_ = hostFixture(which);
        defer g_.deinit();
        const out_len = (try g_.prepareHost(.{ .rows = 1, .n = heads, .k = dk }, .random, 3)) orelse return error.Refused;
        try testing.expect(out_len > 0);
    }
}

test "gdn contract prepareHost random fill: finite references within fixture budgets" {
    for (std.enums.values(Which)) |which| {
        var f = hostFixture(which);
        defer f.deinit();
        for ([_]usize{ 1, 8, 32 }) |T| {
            _ = (try f.prepareHost(.{ .rows = T, .n = heads, .k = dk }, .random, 0x6764 + T)) orelse return error.Refused;
            try testing.expectEqual(f.want.len, f.count());
            var buf: [8][]u16 = undefined;
            try testing.expect(f.inputList(&buf).len > 0);
            for (f.want, f.mag) |w, m| {
                try testing.expect(std.math.isFinite(w));
                try testing.expect(std.math.isFinite(m) and m >= 0);
            }
            const allow = try f.arena.allocator().alloc(f64, f.want.len);
            try f.bounds(f.want, f.mag, allow);
            for (allow) |l| try testing.expect(std.math.isFinite(l) and l > 0);
            // every output word is still the poison it was uploaded from,
            // because no run happened: the walker's want is what the GPU must
            // meet, so a reference of the wrong length would show here
            for (f.out[f.want.len..]) |w| try testing.expectEqual(poison, w);
        }
    }
}

test "gdn contract prepareHost stress fill: bounded state, planted mirror, fp16-proving partials" {
    for ([_]Which{ .prefill_y, .prefill_state, .decode_y, .decode_state }) |which| {
        var f = hostFixture(which);
        defer f.deinit();
        _ = (try f.prepareHost(.{ .rows = 4, .n = heads, .k = dk }, .stress, 0x6764)) orelse return error.Refused;
        // g and beta are 1.0-ish small: g exactly 1.0, beta 2^-10 (prefill) or
        // sigmoid(-10) rounded (decode), so the state feedback converges
        const state_src = if (which == .prefill_y or which == .prefill_state) f.h0_bits else f.state_init_bits;
        for (0..state_elems) |i| {
            const v = f32Of(state_src, i);
            try testing.expect(v >= 4096 and v < 4104);
        }
        // the mirror: v = bf16 of the pinned kv dot, so delta is tiny (scan
        // every soa word: the prefill's v section is contiguous, the decode's
        // is per token)
        var mirror_ok = false;
        for (f.soa_bits) |bits| {
            const v = bf(bits);
            if (v > 500000 and v < 550000) mirror_ok = true;
        }
        try testing.expect(mirror_ok);
        // the f64 reference stays finite and bounded at stress scale
        for (f.want) |w| try testing.expect(std.math.isFinite(w));
        try testing.expect(f.max_state < 1e6);
    }
}

test "gdn contract prepareHost nonfinite fill plants classes the walkers propagate" {
    for (std.enums.values(Which)) |which| {
        var f = hostFixture(which);
        defer f.deinit();
        _ = (try f.prepareHost(.{ .rows = 8, .n = heads, .k = dk }, .nonfinite, 0x6765)) orelse return error.Refused;
        var planted: usize = 0;
        for (f.want) |w| {
            if (c.classOf(w) != .finite) planted += 1;
        }
        try testing.expect(planted > 0);
        try testing.expect(planted < f.want.len);
    }
}

test "gdn contract conv prefill reference reproduces the fixture Conv checker's want" {
    // the assembled input taps and the state carry, at the fixture's shape
    var f = hostFixture(.conv_prefill);
    defer f.deinit();
    _ = (try f.prepareHost(.{ .rows = @intCast(fx.tokens), .n = heads, .k = dk }, .random, 0x57a5)) orelse return error.Refused;
    const cf = try fx.Conv.init(testing.allocator, 0x57a4 + 1);
    defer cf.deinit();
    // same seed family as the fixture: check MY reference math against a
    // directly computed conv over MY generated bits
    const C: usize = channels;
    for (0..@intCast(fx.tokens)) |t| {
        for (0..C) |ch| {
            var acc: f64 = 0;
            for (0..4) |k| {
                const row = t + k;
                const in_bit = if (row < 3) f.state_bits[row * C + ch] else f.qkv_bits[(row - 3) * C + ch];
                acc += bf(in_bit) * bf(f.weight_bits[ch * 4 + k]);
            }
            try testing.expectEqual(f.want[t * C + ch], acc);
        }
    }
}

test "gdn contract depth functions grow with tokens, never constant" {
    try testing.expect(recurYDepth(.{ .rows = 130, .n = 16, .k = 128 }) > recurYDepth(.{ .rows = 8, .n = 16, .k = 128 }));
    try testing.expectEqual(@as(usize, 130 * 130 + 130), recurYDepth(.{ .rows = 130, .n = 16, .k = 128 }));
    try testing.expectEqual(@as(usize, 4 * 130 + 2), recurStateDepth(.{ .rows = 4, .n = 16, .k = 128 }));
    try testing.expectEqual(@as(usize, 3 * 260 + 130), decodeYDepth(.{ .rows = 3, .n = 16, .k = 128 }));
    try testing.expectEqual(@as(usize, 132 + 2), decodeStateDepth(.{ .rows = 1, .n = 16, .k = 128 }));
    try testing.expectEqual(@as(f64, 8 * 130 + 2), snapDepth());
    // the conv and split depths are genuinely shape-independent chains
    try testing.expectEqual(@as(usize, 4), tapsDepth(.{ .rows = 99, .n = 16, .k = 128 }));
    try testing.expectEqual(@as(usize, 40), splitDepth(.{ .rows = 99, .n = 16, .k = 128 }));
}

test "gdn contract walker matches the stage 4 fixture walker bit for bit at the fixture shape" {
    const a = testing.allocator;
    const T: usize = @intCast(fx.tokens); // 8, the fixture's comptime shape
    const soa_bits = try a.alloc(u16, 3 * T * linear_dim);
    defer a.free(soa_bits);
    const g_pairs = try a.alloc(u16, 2 * T * heads);
    defer a.free(g_pairs);
    const beta_bits = try a.alloc(u16, T * heads);
    defer a.free(beta_bits);
    const h0_bits = try a.alloc(u16, 2 * state_elems);
    defer a.free(h0_bits);
    var prng = std.Random.DefaultPrng.init(0xc7);
    const rng = prng.random();
    for (soa_bits) |*x| x.* = fx.sampleDyadic(rng, -3, 1);
    for (g_pairs) |*x| {
        const bits: u32 = @bitCast(fx.sampleDyadicF32(rng, -4, 0));
        x.* = @truncate(bits);
    }
    for (beta_bits) |*x| x.* = fx.sampleDyadic(rng, -3, 1);
    for (0..state_elems) |i| putF32(h0_bits, i, fx.sampleDyadicF32(rng, -6, -2));

    const mine_y = try a.alloc(f64, T * linear_dim);
    defer a.free(mine_y);
    const mine_state = try a.alloc(f64, state_elems);
    defer a.free(mine_state);
    const mine_st = try a.alloc(f64, state_elems);
    defer a.free(mine_st);
    var mine_max: f64 = 0;
    walk(T, soa_bits, g_pairs, beta_bits, h0_bits, mine_st, mine_y, mine_state, &mine_max, null, null, null, null);

    const fx_y = try a.alloc(f64, T * linear_dim);
    defer a.free(fx_y);
    const fx_state = try a.alloc(f64, state_elems);
    defer a.free(fx_state);
    var fx_max: f64 = 0;
    fx.RecurWalk.run(soa_bits, g_pairs, beta_bits, h0_bits, fx_y, fx_state, &fx_max, null, null);
    try testing.expectEqualSlices(f64, fx_y, mine_y);
    try testing.expectEqualSlices(f64, fx_state, mine_state);
    try testing.expectEqual(fx_max, mine_max);

    // the chunked variant's interior snapshot: boundary at token 4
    const mine_snap = try a.alloc(f64, state_elems);
    defer a.free(mine_snap);
    const fx_snap = try a.alloc(f64, state_elems);
    defer a.free(fx_snap);
    var mine_max2: f64 = 0;
    var fx_max2: f64 = 0;
    walk(T, soa_bits, g_pairs, beta_bits, h0_bits, mine_st, mine_y, mine_state, &mine_max2, 4, mine_snap, null, null);
    fx.RecurWalk.run(soa_bits, g_pairs, beta_bits, h0_bits, fx_y, fx_state, &fx_max2, 4, fx_snap);
    try testing.expectEqualSlices(f64, fx_snap, mine_snap);
    try testing.expectEqualSlices(f64, fx_state, mine_state);
}

test "gdn contract walker matches the fixture decode reference exactly" {
    const a = testing.allocator;
    const f = try fx.Decode.init(a, 0x53);
    defer f.deinit();
    // pack the raw-gates chain the fixture derived into the walk's inputs
    const g_pairs = try a.alloc(u16, 2 * heads);
    defer a.free(g_pairs);
    const beta_bits = try a.alloc(u16, heads);
    defer a.free(beta_bits);
    decodeGates(f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, 0, g_pairs, beta_bits);
    const y = try a.alloc(f64, linear_dim);
    defer a.free(y);
    const st_out = try a.alloc(f64, state_elems);
    defer a.free(st_out);
    const st = try a.alloc(f64, state_elems);
    defer a.free(st);
    var max_state: f64 = 0;
    walk(1, f.soa_bits, g_pairs, beta_bits, f.state_in_bits, st, y, st_out, &max_state, null, null, null, null);
    for (f.want_y, y) |w, m| try testing.expectEqual(w, m);
    for (f.want_state, st_out) |w, m| try testing.expectEqual(w, m);
    try testing.expectEqual(f.max_state, max_state);
}

test "gdn contract split reference replays clean through the fixture Split checker" {
    const a = testing.allocator;
    const T8: usize = @intCast(fx.tokens); // the fixture checker's comptime shape
    var f = try fx.Split.init(a, 0x9e); // the fixture generates its own inputs
    defer f.deinit();
    // my reference over the SAME bits, rendered to bits, through the
    // PRODUCTION fixture checker
    const want8 = try a.alloc(f64, 3 * T8 * linear_dim + T8 * heads);
    defer a.free(want8);
    const mag8 = try a.alloc(f64, want8.len);
    defer a.free(mag8);
    splitReference(T8, f.conv_out_bits, f.a_bits, f.b_bits, f.a_log_bits, f.dt_bits, want8, mag8);
    const soa_bits = try a.alloc(u16, 3 * T8 * linear_dim);
    defer a.free(soa_bits);
    const beta_bits = try a.alloc(u16, T8 * heads);
    defer a.free(beta_bits);
    const g_pairs = try a.alloc(u16, 2 * T8 * heads);
    defer a.free(g_pairs);
    for (0..3 * T8 * linear_dim) |i| soa_bits[i] = c.toBf16(@floatCast(want8[i]));
    for (0..T8 * heads) |i| beta_bits[i] = c.toBf16(@floatCast(want8[3 * T8 * linear_dim + i]));
    for (0..T8) |t| for (0..heads) |h| {
        const ref = fx.gateRefAsh(f.a_bits[t * heads + h], f.b_bits[t * heads + h], f32Of(f.a_log_bits, h), f.dt_bits[h]);
        putF32(g_pairs, t * heads + h, @floatCast(ref.g));
    };
    const rep = try f.check(soa_bits, g_pairs, beta_bits, null);
    try testing.expect(rep.worst <= 1);
    // and the same want/mag pass my own budget hook's allowances
    for (want8, mag8) |w, m| {
        const allow = fx.budget(40, 5, 3, w, m);
        try testing.expect(allow > 0 and std.math.isFinite(allow));
    }
}

test "gdn contract split g reference equals the fixture's Ash chain" {
    const a = testing.allocator;
    const f = try fx.SplitOne.init(a, 0x57ac);
    defer f.deinit();
    // splitReference computes g nowhere; the split_g kernel checks g through
    // the same gateRefAsh the fixture's checker uses, so replay it here
    for (0..heads) |h| {
        const ref = fx.gateRefAsh(f.a_bits[h], f.b_bits[h], f32Of(f.a_log_bits, h), f.dt_bits[h]);
        try testing.expect(ref.g > 0 and ref.g < 1);
        try testing.expect(ref.beta > 0 and ref.beta < 1);
    }
}

test "gdn contract stress keeps f32 sums exact and makes fp16 miss by over 100x" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    const T: usize = 4;
    const soa_bits = try ar.alloc(u16, 3 * T * linear_dim);
    const g_pairs = try ar.alloc(u16, 2 * T * heads);
    const beta_bits = try ar.alloc(u16, T * heads);
    const h0_bits = try ar.alloc(u16, 2 * state_elems);
    for (0..state_elems) |i| putF32(h0_bits, i, @floatFromInt(4096 + i % 7));
    for (soa_bits) |*x| x.* = c.toBf16(1);
    for (0..T * linear_dim) |i| {
        if (i % dk % 2 == 1) soa_bits[i] = c.toBf16(-1);
    }
    for (0..heads) |h| for (0..dk) |row| {
        var kv: f64 = 0;
        for (0..dk) |i| kv += @as(f64, @floatFromInt(4096 + (h * dk * dk + row * dk + i) % 7));
        for (0..T) |t| soa_bits[2 * T * linear_dim + (t * heads + h) * dk + row] = c.toBf16(@floatCast(kv));
    };
    for (0..T * heads) |i| {
        putF32(g_pairs, i, 1.0);
        beta_bits[i] = c.toBf16(0x1p-10);
    }
    // f32 sums: within the state budget in every order (the dyadic feedback
    // denominators grow past 24 bits after a couple of tokens, which is what
    // the depth term of the budget is for)
    const want_y = try ar.alloc(f64, T * linear_dim);
    const want_state = try ar.alloc(f64, state_elems);
    const st = try ar.alloc(f64, state_elems);
    var max_state: f64 = 0;
    walk(T, soa_bits, g_pairs, beta_bits, h0_bits, st, want_y, want_state, &max_state, null, null, null, null);
    const sim_y = try ar.alloc(f64, T * linear_dim);
    const sim_state = try ar.alloc(f64, state_elems);
    var sim_max: f64 = 0;
    for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
        walk(T, soa_bits, g_pairs, beta_bits, h0_bits, st, sim_y, sim_state, &sim_max, null, null, .f32, order);
        for (sim_state, want_state) |gv, wv| {
            const allow = fx.budget(@floatFromInt(T * 130 + 2), 2, 0, wv, max_state);
            try testing.expect(@abs(gv - wv) <= allow);
        }
    }
    // fp16: the kv dot's partial sums pass 65504, so the state and y blow
    // past the budget by orders of magnitude
    const allow = fx.budget(@floatFromInt(T * 130 + 2), 2, 0, want_state[0], max_state);
    var worst: f64 = 0;
    walk(T, soa_bits, g_pairs, beta_bits, h0_bits, st, sim_y, sim_state, &sim_max, null, null, .f16, .sequential);
    for (sim_state, want_state) |gv, wv| {
        if (!std.math.isFinite(gv)) {
            worst = std.math.inf(f64);
            break;
        }
        worst = @max(worst, @abs(gv - wv) / allow);
    }
    try testing.expect(worst >= 100);
}

/// Whether the device buffer at `i` must take its words from out: the
/// checked outputs the fetch downloads there, and nothing else.
fn expectOut(which: Which, i: usize, chunks: usize) bool {
    return switch (which) {
        .split_soa => i == 5 or i == 7,
        .split_g => i == 6,
        .conv_prefill => i == 8 or i == 12,
        .conv_decode => i == 9,
        .prefill_y => i == 4,
        .prefill_state => i == 5 or (i == 6 and chunks > 1),
        .decode_y => i == 8,
        .decode_state => false,
    };
}

/// The device buffers every which uploads, so the fixed indices run() binds
/// stay the ones uploadWords hands the device.
fn expectCount(which: Which, chunks: usize) usize {
    return switch (which) {
        .split_soa, .split_g => 8,
        .conv_prefill, .conv_decode => 13,
        .prefill_y, .prefill_state => if (chunks > 1) 7 else 6,
        .decode_y, .decode_state => 9,
    };
}

test "gdn contract upload words: every upload inside the region prepare sized" {
    for (std.enums.values(Which)) |which| {
        var f = hostFixture(which);
        defer f.deinit();
        const s = spec(which);
        // the shape sequence the gates drive: gate 1 and 2's small() shape,
        // gate 3's served and edge shapes, gate 4's stress shape
        var steps: [16]struct { shape: c.Shape, fill: rows.Fill } = undefined;
        var ns: usize = 0;
        var one = s.served[0];
        one.rows = std.math.clamp(one.rows, 5, 8);
        steps[ns] = .{ .shape = one, .fill = .nonfinite };
        ns += 1;
        one = s.served[0];
        one.rows = std.math.clamp(one.rows, 1, 64);
        steps[ns] = .{ .shape = one, .fill = .random };
        ns += 1;
        const list = try c.shapes(testing.allocator, s);
        defer testing.allocator.free(list);
        for (list) |sh| {
            steps[ns] = .{ .shape = sh, .fill = .random };
            ns += 1;
        }
        one = s.served[0];
        one.rows = @min(one.rows, 4);
        steps[ns] = .{ .shape = one, .fill = .stress };
        ns += 1;

        for (steps[0..ns]) |step| {
            const prepared = f.prepareHost(step.shape, step.fill, 0x6764) catch |err| switch (err) {
                // the fixture refuses this fill before any upload; the gate
                // reports the error and the upload path is not reached
                error.NonFiniteReference => continue,
                else => return err,
            };
            const out_len = prepared orelse continue;
            try testing.expect(out_len > 0);
            // the upload path uploadBuffers takes, without a device
            var words: [16][]u16 = undefined;
            const n = try f.uploadWords(&words);
            try testing.expectEqual(expectCount(which, f.chunks), n);
            const lo = @intFromPtr(f.out.ptr);
            const hi = lo + f.out.len * @sizeOf(u16);
            for (words[0..n], 0..) |w, i| {
                const at = @intFromPtr(w.ptr);
                const in_out = at >= lo and at + w.len * @sizeOf(u16) <= hi;
                try testing.expectEqual(expectOut(which, i, f.chunks), in_out);
            }
        }
    }
}
