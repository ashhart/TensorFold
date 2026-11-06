//! Mixed-TP rule 4: divergence and greedy agreement between a mixed run and one box.
const std = @import("std");
const dump = @import("tp_check_dump.zig");
const Mode = dump.Mode;
const Logits = dump.Logits;
const TokenHash = dump.TokenHash;
const Dump = dump.Dump;
/// Where greedy decoding of one prompt first chose a different token.
pub const Split = struct { prompt: u32, position: u64 };

pub const AgreeReport = struct {
    mode: Mode,
    /// Prompts in the solo dump, and (prompt, position) pairs present in both dumps.
    prompts: usize = 0,
    positions: usize = 0,
    /// Solo positions with no mixed record (greedy: only before a prompt's first split or gap); any is a bad run.
    missing: usize = 0,
    /// Greedy only: prompts ended at a gap rather than a split (owned).
    gaps: []Split = &.{},
    /// Positions with the same chosen token; greedy counts only before each prompt's first split, where prefixes match.
    greedy_same: usize = 0,
    /// Greedy only: each prompt's first split, in prompt order (owned; free with `deinit`).
    splits: []Split = &.{},
    /// Teacher-forced only: whether the mixed run's `fed_hash` is the hash of the solo run's chosen tokens.
    fed_ok: ?bool = null,
    /// Positions in the divergence figures: all when fed this run's tokens; greedy, each prompt up to its first split.
    compared: usize = 0,
    top1_same: usize = 0,
    max_abs: f64 = 0,
    mean_abs: f64 = 0,
    /// KL(solo || mixed) per position over the softmax, in nats: mean and worst.
    kl_mean: f64 = 0,
    kl_max: f64 = 0,
    /// Positions left out of the divergence figures because the vocab sizes differ or a logit is not finite.
    skipped: usize = 0,

    pub fn deinit(r: *AgreeReport, a: std.mem.Allocator) void {
        a.free(r.splits);
        a.free(r.gaps);
    }
};

fn argmax(l: Logits) usize {
    var best: usize = 0;
    for (1..l.vocab()) |i| {
        if (l.at(i) > l.at(best)) best = i;
    }
    return best;
}

/// log-sum-exp over the logits, in f64.
fn logSumExp(l: Logits) f64 {
    var m: f64 = -std.math.inf(f64);
    for (0..l.vocab()) |i| m = @max(m, @as(f64, l.at(i)));
    var s: f64 = 0;
    for (0..l.vocab()) |i| s += @exp(@as(f64, l.at(i)) - m);
    return m + @log(s);
}

fn finite(l: Logits) bool {
    for (0..l.vocab()) |i| {
        if (!std.math.isFinite(l.at(i))) return false;
    }
    return true;
}

const Key = struct { prompt: u32, position: u64 };

fn byKey(logits: []const Logits, x: u32, y: u32) bool {
    const p = logits[x];
    const q = logits[y];
    return if (p.prompt != q.prompt) p.prompt < q.prompt else p.position < q.position;
}

/// The dump's logits records as indices in (prompt, position) order (owned).
fn ordered(a: std.mem.Allocator, d: Dump) ![]u32 {
    const order = try a.alloc(u32, d.logits.len);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, d.logits, byKey);
    return order;
}

/// `TokenHash` over a dump's chosen tokens in (prompt, position) order: the `fed_hash` a run fed from it carries.
pub fn chosenHash(a: std.mem.Allocator, d: Dump) !u64 {
    const order = try ordered(a, d);
    defer a.free(order);
    var h: TokenHash = .{};
    for (order) |i| h.add(d.logits[i].prompt, d.logits[i].position, d.logits[i].chosen);
    return h.h;
}

/// Rule 4: `mixed` vs `solo` matched by (prompt, position); greedy: agreement and splits; fed: every position.
pub fn checkAgree(a: std.mem.Allocator, mixed: Dump, solo: Dump) !AgreeReport {
    var r: AgreeReport = .{ .mode = mixed.header.mode };
    errdefer r.deinit(a);
    var at: std.AutoHashMapUnmanaged(Key, u32) = .empty;
    defer at.deinit(a);
    for (mixed.logits, 0..) |m, i| try at.put(a, .{ .prompt = m.prompt, .position = m.position }, @intCast(i));
    const forced = mixed.header.mode == .teacher_forced;
    if (forced) r.fed_ok = mixed.header.fed_hash == try chosenHash(a, solo);
    const order = try ordered(a, solo);
    defer a.free(order);
    var splits: std.ArrayList(Split) = .empty;
    defer splits.deinit(a);
    var gaps: std.ArrayList(Split) = .empty;
    defer gaps.deinit(a);
    var abs_sum: f64 = 0;
    var abs_count: f64 = 0;
    var kl_sum: f64 = 0;
    var prompt: ?u32 = null;
    // greedy: set at a prompt's first split or gap; after it the runs may decode different prefixes
    var stopped = false;
    for (order) |si| {
        const s = solo.logits[si];
        if (prompt != s.prompt) {
            prompt = s.prompt;
            stopped = false;
            r.prompts += 1;
        }
        const mi = at.get(.{ .prompt = s.prompt, .position = s.position }) orelse {
            // greedy: past a split the runs decode different text, so a shorter mixed run is normal there
            if (forced) {
                r.missing += 1;
            } else if (!stopped) {
                r.missing += 1;
                stopped = true;
                try gaps.append(a, .{ .prompt = s.prompt, .position = s.position });
            }
            continue;
        };
        const m = mixed.logits[mi];
        r.positions += 1;
        if (!forced and stopped) continue;
        const same = m.chosen == s.chosen;
        if (same) r.greedy_same += 1;
        if (!forced and !same) {
            stopped = true;
            try splits.append(a, .{ .prompt = s.prompt, .position = s.position });
        }
        if (forced and r.fed_ok != true) continue;
        if (m.vocab() != s.vocab() or !finite(m) or !finite(s)) {
            r.skipped += 1;
            continue;
        }
        r.compared += 1;
        if (argmax(m) == argmax(s)) r.top1_same += 1;
        const lm = logSumExp(m);
        const ls = logSumExp(s);
        var kl: f64 = 0;
        for (0..s.vocab()) |i| {
            const x: f64 = s.at(i);
            const y: f64 = m.at(i);
            const d = @abs(x - y);
            r.max_abs = @max(r.max_abs, d);
            abs_sum += d;
            abs_count += 1;
            const lp = x - ls;
            kl += @exp(lp) * (lp - (y - lm));
        }
        kl_sum += kl;
        r.kl_max = @max(r.kl_max, kl);
    }
    if (abs_count > 0) r.mean_abs = abs_sum / abs_count;
    if (r.compared > 0) r.kl_mean = kl_sum / @as(f64, @floatFromInt(r.compared));
    r.splits = try splits.toOwnedSlice(a);
    r.gaps = try gaps.toOwnedSlice(a);
    return r;
}
