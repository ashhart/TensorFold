//! The GLM head's draw rules: types the family and the host references share, Metal-free (gpu_full's contract).
const std = @import("std");
const lanes = @import("lanes");

// A sampled row's keyed draw: gpu_full.zig's contract, one rule a row; the Metal shader mirrors this layout.
pub const Rule = extern struct {
    seed_lo: u32,
    seed_hi: u32,
    position: u32, // this row's absolute token position: the noise is keyed by (seed, position, id)
    top_k: u32, // 0: off (the whole vocabulary races)
    inv_t: f32,
    top_p: f32,
    near: f32 = 20.0,
    min_log: f32,
    vocab: u32,

    /// A row's absolute position no pass row ever has: the shader draws that row greedily (argmax).
    pub const greedy_position: u32 = 0xFFFF_FFFF;

    /// The greedy row's rule: the shader's argmax over the row's `vocab` bf16 logits, glm_argmax's tie rule.
    pub fn greedy(vocab: u32) Rule {
        return .{ .seed_lo = 0, .seed_hi = 0, .position = greedy_position, .top_k = 0, .inv_t = 1.0, .top_p = 1.0, .near = 20.0, .min_log = 0.0, .vocab = vocab };
    }

    /// Whether this rule draws greedily (the host's and the shader's shared marker).
    pub fn isGreedy(r: Rule) bool {
        return r.position == greedy_position;
    }
};

comptime {
    // The Metal shader reads the rules as u32/f32 words behind a 16-byte header: the layout is the contract.
    if (@sizeOf(Rule) != 36) @compileError("GlmSampleRule: the Metal payload is 36 bytes a rule");
    if (@offsetOf(Rule, "seed_lo") != 0 or @offsetOf(Rule, "seed_hi") != 4 or @offsetOf(Rule, "position") != 8 or
        @offsetOf(Rule, "top_k") != 12 or @offsetOf(Rule, "inv_t") != 16 or @offsetOf(Rule, "top_p") != 20 or
        @offsetOf(Rule, "near") != 24 or @offsetOf(Rule, "min_log") != 28 or @offsetOf(Rule, "vocab") != 32)
        @compileError("GlmSampleRule: the field order is the Metal ABI's (seed, position, top_k, filters, vocab)");
}

/// A pass's draw rules, one a row in row order; rows past `n` draw greedily (argmax, lowest id).
pub const Draws = struct {
    rules: [max_rows]Rule = undefined, // the lane core's widest shared pass
    n: usize = 0,

    /// Row `r`'s draw: its rule, or greedy when `r` is past the filled rules.
    pub fn at(d: *const Draws, r: usize) ?Rule {
        return if (r < d.n) d.rules[r] else null;
    }

    /// Row `r` draws sampled: its rule is filled and is not the greedy marker.
    pub fn sampled(d: *const Draws, r: usize) bool {
        return r < d.n and !d.rules[r].isGreedy();
    }
};

/// The lane core's max_rows (state.zig's 16), mirrored here so this file stays Metal-free.
pub const max_rows: usize = 16;

pub const Header = extern struct { n: u32, pad: [3]u32 = .{ 0, 0, 0 } };

/// The exact buffer-1 payload shared by production dispatch and the synthetic device gate.
pub const Payload = extern struct {
    header: Header,
    rules: [max_rows]Rule,

    pub fn init(draws: ?*const Draws, rows: usize, vocab: u32) Payload {
        std.debug.assert(rows > 0 and rows <= max_rows and vocab > 0);
        var p: Payload = .{ .header = .{ .n = @intCast(rows) }, .rules = @splat(Rule.greedy(vocab)) };
        if (draws) |d| for (0..rows) |r| if (d.at(r)) |rule| {
            if (rule.vocab != vocab) std.debug.panic("glm head: draw vocabulary {d} differs from head vocabulary {d}", .{ rule.vocab, vocab });
            p.rules[r] = rule;
        };
        return p;
    }

    pub fn sampled(p: *const Payload) bool {
        for (p.rules[0..p.header.n]) |rule| if (!rule.isGreedy()) return true;
        return false;
    }
};

comptime {
    if (@sizeOf(Header) != 16 or @offsetOf(Payload, "rules") != 16 or @sizeOf(Payload) != 592)
        @compileError("GlmDraws: 16-byte header followed by sixteen 36-byte rules");
}

/// A prompt head projects one output row, keyed at the full prompt length even after cache restoration.
pub fn promptDraws(s: ?lanes.Sampling, position: u32, vocab: u32) Draws {
    var d: Draws = .{};
    if (s) |p| if (p.temperature > 0) {
        d.rules[0] = ruleOf(p, position, vocab);
        d.n = 1;
    };
    return d;
}

/// Populate every shared row from its stream, including explicit greedy rows.
pub fn windowDraws(windows: []const lanes.backend.Window, vocab: u32) Draws {
    var d: Draws = .{};
    for (windows) |w| {
        std.debug.assert(w.positions.len == w.rows() and d.n + w.rows() <= max_rows);
        for (w.positions) |pos| {
            d.rules[d.n] = if (w.stream.sampling) |p|
                if (p.temperature > 0) ruleOf(p, @intCast(pos), vocab) else Rule.greedy(vocab)
            else
                Rule.greedy(vocab);
            d.n += 1;
        }
    }
    return d;
}

/// The keyed rule `s`'s head draws a row with at `position` (typed: minLog lives on lanes.Sampling).
pub fn ruleOf(s: lanes.Sampling, position: u32, vocab: u32) Rule {
    return .{
        .seed_lo = @truncate(s.seed),
        .seed_hi = @truncate(s.seed >> 32),
        .position = position,
        .top_k = s.top_k,
        .inv_t = @floatCast(1.0 / @max(s.temperature, 1e-6)),
        .top_p = @floatCast(s.top_p),
        .near = 20.0,
        .min_log = @floatCast(s.minLog()),
        .vocab = vocab,
    };
}

/// The standalone pass's rules at `position`: `n` rules, positions advancing by one.
pub fn rulesFor(p: lanes.Sampling, position: u32, vocab: u32, n: u32) Draws {
    var d: Draws = .{};
    const count = @min(n, max_rows);
    for (0..count) |r| d.rules[r] = ruleOf(p, position + @as(u32, @intCast(r)), vocab);
    d.n = count;
    return d;
}

pub const SamplingError = error{ BadSampling, MissingSamplingField };

/// Parse GLM_SAMPLING=seed,temperature,top_k,top_p,min_p; zero temperature is greedy and zero top_k is off.
pub fn parseSampling(text: []const u8) !lanes.Sampling {
    var it = std.mem.splitScalar(u8, text, ',');
    const seed_field = it.next() orelse return error.MissingSamplingField;
    const seed = std.fmt.parseInt(u64, seed_field, 10) catch return error.BadSampling;
    const temperature: f64 = if (it.next()) |v| std.fmt.parseFloat(f64, v) catch return error.BadSampling else 1.0;
    const top_k: u32 = if (it.next()) |v| std.fmt.parseInt(u32, v, 10) catch return error.BadSampling else 0;
    const top_p: f64 = if (it.next()) |v| std.fmt.parseFloat(f64, v) catch return error.BadSampling else 1.0;
    const min_p: f64 = if (it.next()) |v| std.fmt.parseFloat(f64, v) catch return error.BadSampling else 0.0;
    if (it.next() != null) return error.BadSampling; // a sixth field: not a silently rounded value
    if (!std.math.isFinite(temperature) or temperature < 0) return error.BadSampling;
    if (!std.math.isFinite(top_p) or top_p <= 0 or top_p > 1) return error.BadSampling;
    if (!std.math.isFinite(min_p) or min_p < 0 or min_p >= 1) return error.BadSampling;
    return .{ .seed = seed, .temperature = temperature, .top_k = top_k, .top_p = top_p, .min_p = min_p };
}
