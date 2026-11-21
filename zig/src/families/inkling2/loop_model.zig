//! f64 Mini: f32 sums/combine ascend (A12), bf16 experts/acts (A1/A2), f32 F/G (A7), postdot scale (A14); casts round.
const std = @import("std");
const bf = @import("bf16_round");

pub const hidden_size = 2304;
pub const gate_rank = 64;
pub const physical_layers = 24;
pub const core_start = 6;
pub const core_end = 18;
pub const core_layers = core_end - core_start;
pub const num_passes = 3;
pub const visit_count = 48;
pub const message_roles = 4;
pub const rms_eps = 1e-6;

pub const ShapeError = error{BadShape};
pub const VisitError = error{BadVisit};
pub const DriverError = error{ BadShape, BadVisit, BadDriverState, BadSequence };

fn narrow32(v: f64) f32 {
    return @floatCast(v);
}

fn widen32(v: f32) f64 {
    return @floatCast(v);
}

/// One round to nearest-even bf16; the returned f64 is exactly that bf16 value.
pub fn bf16(v: f64) f64 {
    return bf.bf16(v);
}

/// rho(x), with an f32 ordered sum of squares and the specified f32 mean/rsqrt/multiply.
pub fn rms(x: []const f64, eps: f64, out: []f64) ShapeError!void {
    if (x.len == 0 or out.len != x.len or !std.math.isFinite(eps) or eps < 0) return error.BadShape;
    var squares: f32 = 0;
    for (x) |v| {
        const a = narrow32(v);
        squares += a * a;
    }
    const mean = squares / @as(f32, @floatFromInt(x.len));
    const inverse: f32 = 1 / @sqrt(mean + narrow32(eps));
    for (x, out) |v, *o| o.* = widen32(narrow32(v) * inverse);
}

/// N_w(x): no bf16 rounding between rho and the weight product, one bf16 rounding at the end.
pub fn weightedRms(x: []const f64, weight: []const f64, eps: f64, out: []f64, scratch: []f64) ShapeError!void {
    if (weight.len != x.len or scratch.len != x.len) return error.BadShape;
    try rms(x, eps, scratch);
    for (scratch, weight, out) |v, w, *o| o.* = bf16(widen32(narrow32(w) * narrow32(v)));
}

/// binary32(1/sqrt(H)); in particular Mini's value is binary32(1/48), not a double scale.
pub fn residualScale(hidden: usize) ShapeError!f32 {
    if (hidden == 0) return error.BadShape;
    return narrow32(1 / @sqrt(@as(f64, @floatFromInt(hidden))));
}

fn normalizeSources(states: []const f64, partial: []const f64, eps: f64, normalized: []f64) ShapeError!usize {
    if (partial.len == 0 or states.len % partial.len != 0) return error.BadShape;
    const h = partial.len;
    const count = states.len / h + 1;
    if (normalized.len != count * h) return error.BadShape;
    for (0..count - 1) |i| try rms(states[i * h ..][0..h], eps, normalized[i * h ..][0..h]);
    try rms(partial, eps, normalized[(count - 1) * h ..][0..h]);
    return count;
}

fn retrieve(normalized: []const f64, query: []const f64, out: []f64, weights: []f64) ShapeError!void {
    if (query.len == 0 or out.len != query.len or normalized.len % query.len != 0) return error.BadShape;
    const h = query.len;
    const count = normalized.len / h;
    if (count == 0 or weights.len != count) return error.BadShape;
    const scale = try residualScale(h);
    var top: f32 = -std.math.inf(f32);
    for (0..count) |i| {
        var dot: f32 = 0;
        for (query, normalized[i * h ..][0..h]) |q, v| dot += narrow32(q) * narrow32(v);
        const logit = dot * scale;
        weights[i] = widen32(logit);
        top = @max(top, logit);
    }
    var denominator: f32 = 0;
    for (weights) |*w| {
        const d = narrow32(w.* - widen32(top));
        const e = @exp(d);
        w.* = widen32(e);
        denominator += e;
    }
    @memset(out, 0);
    for (0..count) |i| {
        const weight = narrow32(weights[i]) / denominator;
        for (out, normalized[i * h ..][0..h]) |*o, v| {
            const a = narrow32(o.*);
            o.* = widen32(a + weight * narrow32(v));
        }
    }
}

/// AR_Q(S,p): states[S][H], queries/out[Q][H] row-major; empty S okay, partial last, normalize before attention.
pub fn attentionResidual(gpa: std.mem.Allocator, states: []const f64, partial: []const f64, queries: []const f64, query_rows: usize, eps: f64, out: []f64) !void {
    if (partial.len == 0 or query_rows == 0 or states.len % partial.len != 0) return error.BadShape;
    const h = partial.len;
    if (queries.len != query_rows * h or out.len != queries.len) return error.BadShape;
    const count = states.len / h + 1;
    const normalized = try gpa.alloc(f64, count * h);
    defer gpa.free(normalized);
    const weights = try gpa.alloc(f64, count);
    defer gpa.free(weights);
    _ = try normalizeSources(states, partial, eps, normalized);
    for (0..query_rows) |q| try retrieve(normalized, queries[q * h ..][0..h], out[q * h ..][0..h], weights);
}

pub const MessageRole = enum(u2) { q = 0, k = 1, v = 2, mlp = 3 };

/// The previous pass's B_12 and P_18. No embedding, prelude, or current-pass state is included.
pub const MessageBank = struct {
    block: []const f64,
    partial: []const f64,
};

/// Both checkpoint tensors in their stored row-major layouts: 24 rows by H.
pub const MessageQueries = struct {
    q_mlp: []const f64,
    kv: []const f64,

    /// Core j uses row 2j for q/k and row 2j+1 for mlp/v, in their respective tensors.
    pub fn row(q: MessageQueries, core: usize, role: MessageRole, hidden: usize) ShapeError![]const f64 {
        if (core >= core_layers or hidden == 0 or q.q_mlp.len != 2 * core_layers * hidden or q.kv.len != 2 * core_layers * hidden) return error.BadShape;
        const odd: usize = @intFromBool(role == .mlp or role == .v);
        const row_index = 2 * core + odd;
        const source = if (role == .q or role == .mlp) q.q_mlp else q.kv;
        return source[row_index * hidden ..][0..hidden];
    }
};

fn bankNormalized(gpa: std.mem.Allocator, bank: MessageBank, eps: f64) ![]f64 {
    if (bank.block.len == 0 or bank.partial.len != bank.block.len) return error.BadShape;
    const normalized = try gpa.alloc(f64, 2 * bank.block.len);
    errdefer gpa.free(normalized);
    try rms(bank.block, eps, normalized[0..bank.block.len]);
    try rms(bank.partial, eps, normalized[bank.block.len..]);
    return normalized;
}

/// One of the 48 cross-pass messages. Its bank values and output are binary32 values held in f64.
pub fn crossPassMessage(gpa: std.mem.Allocator, bank: MessageBank, query: []const f64, eps: f64, out: []f64) !void {
    if (query.len != bank.block.len or out.len != query.len) return error.BadShape;
    const normalized = try bankNormalized(gpa, bank, eps);
    defer gpa.free(normalized);
    var weights: [2]f64 = undefined;
    try retrieve(normalized, query, out, &weights);
}

/// All messages in [core layer][q,k,v,mlp][H] order. Queries are shared by passes two and three.
pub fn crossPassMessages(gpa: std.mem.Allocator, bank: MessageBank, queries: MessageQueries, eps: f64, out: []f64) !void {
    const h = bank.block.len;
    if (h == 0 or bank.partial.len != h or out.len != core_layers * message_roles * h) return error.BadShape;
    const normalized = try bankNormalized(gpa, bank, eps);
    defer gpa.free(normalized);
    var weights: [2]f64 = undefined;
    const roles = [_]MessageRole{ .q, .k, .v, .mlp };
    for (0..core_layers) |j| for (roles, 0..) |role, r| {
        const query = try queries.row(j, role, h);
        const at = (j * message_roles + r) * h;
        try retrieve(normalized, query, out[at..][0..h], &weights);
    };
}

/// Mini's rank-64 controller, represented with smaller ranks too for inexpensive host probes.
pub const GateWeights = struct {
    down: []const f64, // [rank][2H], with the h half before the message half
    forget: []const f64, // [H][rank]
    gain: []const f64, // [H][rank]
    rank: usize,
};

fn gateDot(row: []const f64, values: []const f64) f32 {
    var sum: f32 = 0;
    for (row, values) |w, v| sum += narrow32(w) * narrow32(v);
    return sum;
}

/// G(h,m) = (1-f)*h + g*m. The joined norm and bottleneck are bf16; F/G stay f32 (A7).
pub fn messageGate(gpa: std.mem.Allocator, h: []const f64, m: []const f64, w: GateWeights, eps: f64, out: []f64) !void {
    const width = h.len;
    if (width == 0 or m.len != width or out.len != width or w.rank == 0) return error.BadShape;
    if (w.down.len != w.rank * 2 * width or w.forget.len != width * w.rank or w.gain.len != width * w.rank) return error.BadShape;
    const joined = try gpa.alloc(f64, 2 * width);
    defer gpa.free(joined);
    const zeta = try gpa.alloc(f64, w.rank);
    defer gpa.free(zeta);
    try rms(h, eps, joined[0..width]);
    try rms(m, eps, joined[width..]);
    for (joined) |*v| v.* = bf16(v.*);
    for (0..w.rank) |r| zeta[r] = bf16(widen32(gateDot(w.down[r * 2 * width ..][0 .. 2 * width], joined)));
    for (0..width) |i| {
        const forget = std.math.tanh(gateDot(w.forget[i * w.rank ..][0..w.rank], zeta));
        const gain = std.math.tanh(gateDot(w.gain[i * w.rank ..][0..w.rank], zeta));
        const keep: f32 = 1 - forget;
        const left = keep * narrow32(h[i]);
        const right = gain * narrow32(m[i]);
        out[i] = widen32(left + right);
    }
}

pub const AttentionKind = enum { sliding, global };

/// One row of the fixed 48-visit schedule. cache_slot is visit numbering; reference_slot is HF's.
pub const Visit = struct {
    index: u8,
    layer: u8,
    pass: u8, // zero outside the core, one-based inside it
    cache_slot: u8,
    reference_slot: u8,
    kind: AttentionKind,
    boundary: bool,
    has_messages: bool,
    ar_queries: u8,
    attention_sources: u8,
    mlp_sources: u8,
};

pub fn visit(index: usize) VisitError!Visit {
    if (index >= visit_count) return error.BadVisit;
    const layer: u8 = if (index < core_start)
        @intCast(index)
    else if (index < 42)
        @intCast(core_start + (index - core_start) % core_layers)
    else
        @intCast(index - 24);
    const pass: u8 = if (index >= 18 and index < 42) @intCast(1 + (index - core_start) / core_layers) else if (index >= core_start and index < 18) 1 else 0;
    const reference: u8 = if (pass >= 2) @intCast(24 + (@as(usize, pass) - 2) * core_layers + layer - core_start) else layer;
    const global = (@as(usize, layer) + 1) % 6 == 0;
    const boundary = layer % 6 == 0;
    const blocks: u8 = (layer + 5) / 6;
    return .{
        .index = @intCast(index),
        .layer = layer,
        .pass = pass,
        .cache_slot = @intCast(index),
        .reference_slot = reference,
        .kind = if (global) .global else .sliding,
        .boundary = boundary,
        .has_messages = pass >= 2,
        .ar_queries = if (layer == 0) 0 else if (global) 3 else 1,
        .attention_sources = if (layer == 0) 0 else blocks + 1,
        .mlp_sources = blocks + 1 + @intFromBool(boundary),
    };
}

pub const Sources = struct {
    blocks: []const f64, // [completed block][H]
    partial: []const f64, // p before attention, or unrounded t before the MLP
};

/// Residual/block only; caller attention/conv/MoE. Visit: begin->attentionSources->afterAttention->mlpSources->finish.
pub const Driver = struct {
    gpa: std.mem.Allocator,
    hidden: usize,
    storage: []f64, // blocks[4H], prelude snapshot[2H], bank[2H], p[H], t[H]
    block_count: usize,
    next: usize,
    phase: enum { ready, attention, mlp },
    active: Visit,
    have_snapshot: bool,

    pub fn init(gpa: std.mem.Allocator, embedding: []const f64) !Driver {
        if (embedding.len == 0) return error.BadShape;
        const h = embedding.len;
        const storage = try gpa.alloc(f64, 10 * h);
        @memset(storage, 0);
        for (embedding, storage[8 * h .. 9 * h]) |v, *p| p.* = bf16(v);
        return .{ .gpa = gpa, .hidden = h, .storage = storage, .block_count = 0, .next = 0, .phase = .ready, .active = undefined, .have_snapshot = false };
    }

    pub fn deinit(d: *Driver) void {
        d.gpa.free(d.storage);
        d.* = undefined;
    }

    fn blocks(d: *Driver) []f64 {
        return d.storage[0 .. d.block_count * d.hidden];
    }

    fn partial(d: *Driver) []f64 {
        return d.storage[8 * d.hidden .. 9 * d.hidden];
    }

    fn work(d: *Driver) []f64 {
        return d.storage[9 * d.hidden .. 10 * d.hidden];
    }

    /// Applies the pass reset before visits 18 and 30, after saving that previous pass's two bank states.
    pub fn begin(d: *Driver, index: usize) DriverError!Visit {
        if (index >= visit_count) return error.BadVisit;
        if (d.phase != .ready or d.next != index) return error.BadSequence;
        if (index == core_start) {
            if (d.block_count != 1) return error.BadDriverState;
            @memcpy(d.storage[4 * d.hidden .. 5 * d.hidden], d.partial());
            @memcpy(d.storage[5 * d.hidden .. 6 * d.hidden], d.blocks());
            d.have_snapshot = true;
        }
        if (index == 18 or index == 30) {
            if (!d.have_snapshot or d.block_count != 3) return error.BadDriverState;
            @memcpy(d.storage[6 * d.hidden .. 7 * d.hidden], d.storage[2 * d.hidden .. 3 * d.hidden]);
            @memcpy(d.storage[7 * d.hidden .. 8 * d.hidden], d.partial());
            @memcpy(d.partial(), d.storage[4 * d.hidden .. 5 * d.hidden]);
            @memcpy(d.storage[0..d.hidden], d.storage[5 * d.hidden .. 6 * d.hidden]);
            d.block_count = 1;
        }
        d.active = visit(index) catch return error.BadVisit;
        d.phase = .attention;
        return d.active;
    }

    pub fn attentionSources(d: *Driver) DriverError!Sources {
        if (d.phase != .attention) return error.BadSequence;
        return .{ .blocks = d.blocks(), .partial = d.partial() };
    }

    /// Saves p as a completed boundary block, then forms f32 t without rounding it to bf16.
    pub fn afterAttention(d: *Driver, attention: []const f64) DriverError!void {
        if (d.phase != .attention) return error.BadSequence;
        if (attention.len != d.hidden) return error.BadShape;
        if (d.active.boundary) {
            if (d.block_count == 4) return error.BadDriverState;
            @memcpy(d.storage[d.block_count * d.hidden ..][0..d.hidden], d.partial());
            d.block_count += 1;
            for (d.work(), attention) |*t, a| t.* = widen32(narrow32(a));
        } else {
            for (d.work(), d.partial(), attention) |*t, p, a| t.* = widen32(narrow32(p) + narrow32(a));
        }
        d.phase = .mlp;
    }

    /// AR's second call sees the extended block list and t, not bf16(t).
    pub fn mlpSources(d: *Driver) DriverError!Sources {
        if (d.phase != .mlp) return error.BadSequence;
        return .{ .blocks = d.blocks(), .partial = d.work() };
    }

    /// p_out = bf16(t + f32(post-MLP-convolution output)); the residual itself is never gated.
    pub fn finish(d: *Driver, mlp: []const f64) DriverError!void {
        if (d.phase != .mlp) return error.BadSequence;
        if (mlp.len != d.hidden) return error.BadShape;
        for (d.partial(), d.work(), mlp) |*p, t, o| p.* = bf16(widen32(narrow32(t) + narrow32(o)));
        d.phase = .ready;
        d.next += 1;
    }

    /// Raw B_12 and P_18 of the immediately previous pass; valid until the next pass reset.
    pub fn messageBank(d: *Driver) DriverError!MessageBank {
        if (d.phase == .ready or !d.active.has_messages) return error.BadDriverState;
        return .{ .block = d.storage[6 * d.hidden .. 7 * d.hidden], .partial = d.storage[7 * d.hidden .. 8 * d.hidden] };
    }

    /// The five final AR sources after visit 47: four completed blocks and P_24.
    pub fn finalSources(d: *Driver) DriverError!Sources {
        if (d.phase != .ready or d.next != visit_count or d.block_count != 4) return error.BadDriverState;
        return .{ .blocks = d.blocks(), .partial = d.partial() };
    }
};

pub fn e2m1(code: u8) f64 {
    const table = [_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const magnitude = table[code & 7];
    return std.math.copysign(magnitude, if (code & 8 != 0) @as(f64, -1) else @as(f64, 1));
}

/// OCP E4M3FN, including its subnormals and sole NaN encodings.
pub fn e4m3(code: u8) f64 {
    const negative = code & 0x80 != 0;
    const exponent: u8 = (code >> 3) & 0xf;
    const mantissa: u8 = code & 7;
    if (exponent == 15 and mantissa == 7) return std.math.nan(f64);
    const magnitude = if (exponent == 0)
        std.math.ldexp(@as(f64, @floatFromInt(mantissa)), -9)
    else
        std.math.ldexp(1 + @as(f64, @floatFromInt(mantissa)) / 8, @as(i32, exponent) - 7);
    return if (negative) -magnitude else magnitude;
}

/// A2(a): both source f32 scalars are cast to bf16 before one binary32 division.
pub fn nvfp4OuterFactor(amax: f64, denominator: f64) f64 {
    const a = narrow32(bf16(amax));
    const d = narrow32(bf16(denominator));
    return widen32(a / d);
}

fn nvWeight(code: u8, scale: u8, factor: f32) f64 {
    const inner = narrow32(e2m1(code) * e4m3(scale)); // the E2M1*E4M3 product is exact
    return bf16(widen32(inner * factor));
}

pub fn nvfp4ReferenceWeight(code: u8, scale: u8, amax: f64, denominator: f64) f64 {
    return nvWeight(code, scale, narrow32(nvfp4OuterFactor(amax, denominator)));
}

/// Decode one packed NVFP4 row: low nibble is the even input, one E4M3 scale per 16 inputs.
pub fn nvfp4ReferenceRow(codes: []const u8, scales: []const u8, amax: f64, denominator: f64, out: []f64) ShapeError!void {
    if (out.len == 0 or out.len % 16 != 0 or codes.len != out.len / 2 or scales.len != out.len / 16) return error.BadShape;
    const factor = narrow32(nvfp4OuterFactor(amax, denominator));
    for (out, 0..) |*v, c| {
        const packed_byte = codes[c / 2];
        const code = if (c % 2 == 0) packed_byte & 0xf else packed_byte >> 4;
        v.* = nvWeight(code, scales[c / 16], factor);
    }
}

pub fn mxfp8Scale(scale: u8) f64 {
    return std.math.ldexp(@as(f64, 1), @as(i32, scale) - 127);
}

pub fn mxfp8ReferenceWeight(code: u8, scale: u8) f64 {
    return bf16(e4m3(code) * mxfp8Scale(scale));
}

/// Decode one MXFP8 row using the stored per-row exponent for each group of 32 input columns (A4).
pub fn mxfp8ReferenceRow(codes: []const u8, scales: []const u8, out: []f64) ShapeError!void {
    if (out.len == 0 or out.len % 32 != 0 or codes.len != out.len or scales.len != out.len / 32) return error.BadShape;
    for (out, codes, 0..) |*v, code, c| v.* = mxfp8ReferenceWeight(code, scales[c / 32]);
}

/// A bf16 reference GEMM output, with ascending f32 accumulation and no activation quantization.
pub fn expertProjection(weights: []const f64, x: []const f64, rows: usize, out: []f64) ShapeError!void {
    if (rows == 0 or x.len == 0 or out.len != rows or weights.len != rows * x.len) return error.BadShape;
    for (0..rows) |r| {
        var sum: f32 = 0;
        for (weights[r * x.len ..][0..x.len], x) |w, v| sum += narrow32(w) * narrow32(v);
        out[r] = bf16(widen32(sum));
    }
}

fn softcap32(x: f32, cap: f32) f32 {
    return cap * std.math.tanh(x / cap);
}

/// bf16(softcap4(g) * sigmoid(g) * softcap25(u)), evaluated left to right in f32.
pub fn expertActivation(gate: f64, up: f64) f64 {
    const g = narrow32(gate);
    const sigmoid: f32 = 1 / (1 + @exp(-g));
    const act = softcap32(g, 4) * sigmoid * softcap32(narrow32(up), 25);
    return bf16(widen32(act));
}

/// Combine f32 from zero by ascending id, not route order (A12); expert_outputs [route][H], ids/route_weights [route].
pub fn combineExperts(expert_count: usize, ids: []const u32, route_weights: []const f64, expert_outputs: []const f64, hidden: usize, out: []f64) ShapeError!void {
    if (expert_count == 0 or hidden == 0 or ids.len != route_weights.len or expert_outputs.len != ids.len * hidden or out.len != hidden) return error.BadShape;
    @memset(out, 0);
    for (0..expert_count) |e| {
        var found = false;
        for (ids, 0..) |id, r| if (id == e) {
            if (found) return error.BadShape;
            found = true;
            const weight = narrow32(route_weights[r]);
            for (out, expert_outputs[r * hidden ..][0..hidden]) |*sum, value| {
                const term = weight * narrow32(value);
                sum.* = widen32(narrow32(sum.*) + term);
            }
        } else if (id >= expert_count) return error.BadShape;
    }
    for (out) |*v| v.* = bf16(v.*);
}
