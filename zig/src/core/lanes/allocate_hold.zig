//! A shared round's drafts priced by their head steps or a timed hold grid, over allocate.HeadCost.
const std = @import("std");
const hold_price = @import("hold_price.zig");
const alloc = @import("allocate.zig");
const Allocator = std.mem.Allocator;
const Table = @import("table.zig").Table;

/// Prices growing blocks in stream order; batched blocks pad to each physical pass's deepest, excluding free nodes.
const GridPass = struct {
    costs: []const hold_price.HoldCost,
    contexts: []const hold_price.ContextCost,
    debts: []const u32,
    batched: bool,
    free: u32,
    rows: u32,
    block_rows: u32,
    counts: []const u32, // the allocation so far, which prices a split batch exactly
    charged: []const bool,
    blocks: usize = 0, // batched: streams with a node past the free one
    deepest: i64 = 0, // and the deepest of them past it
    context_rows: u32 = 0,
    ms: f64 = 0.0, // an empty pass costs nothing

    /// A stream's count grew to `count`: the pass's new price.
    fn grow(p: *GridPass, index: usize, count: u32) f64 {
        const depth: i64 = count - p.free;
        if (depth <= 0) return p.ms;
        if (p.batched) {
            p.blocks += @intFromBool(depth == 1);
            p.deepest = @max(p.deepest, depth);
            if (p.debts.len > 0) p.context_rows = @max(p.context_rows, p.debts[index]);
            p.ms = if (p.rows == 0 or @as(i64, @intCast(p.blocks)) * @max(p.deepest, p.block_rows) <= p.rows)
                hold_price.holdMs(p.costs, p.blocks, p.deepest) + hold_price.contextMs(p.contexts, p.blocks, p.context_rows)
            else
                hold_price.countsMs(p.costs, p.contexts, p.rows, p.block_rows, p.counts, p.charged, p.free, p.debts);
        } else {
            p.ms += hold_price.holdMs(p.costs, 1, depth) - (if (depth > 1) hold_price.holdMs(p.costs, 1, depth - 1) else 0.0);
            if (depth == 1 and p.debts.len > 0) p.ms += hold_price.contextMs(p.contexts, 1, p.debts[index]);
        }
        return p.ms;
    }
};

/// allocate, each count also priced by the head steps that draft it.
pub fn allocateDrafts(gpa: Allocator, fixed: []const u32, probs: []const []const f64, costs: *const Table, overhead_ms: f64, max_rows: u64, head: alloc.HeadCost) ![]u32 {
    const counts = try gpa.alloc(u32, probs.len);
    defer gpa.free(counts);
    const best_counts = try gpa.alloc(u32, probs.len);
    @memset(best_counts, 0);
    var fixed_rows: u64 = 0;
    var live: u64 = 0;
    var fixed_host_ms: f64 = 0;
    for (fixed, 0..) |f, i| {
        fixed_rows += f;
        live += @intFromBool(f > 0);
        if (head.host_row_ms.len > 0) {
            fixed_host_ms += @as(f64, @floatFromInt(f)) * head.host_row_ms[i];
        }
    }
    const grid = head.hold_costs.len > 0;
    const timed = !costs.empty();
    const mixed_host = timed and head.host_row_ms.len > 0 and mixed: {
        var quote: ?f64 = null;
        for (probs, 0..) |p, s| {
            if (p.len == 0) continue;
            if (quote) |first| {
                if (head.host_row_ms[s] != first) break :mixed true;
            } else quote = head.host_row_ms[s];
        }
        break :mixed false;
    };
    const context_order = timed and head.hold_context_costs.len > 0 and head.hold_context_rows.len > 0 and context: {
        for (probs, 0..) |p, s| {
            if (p.len > @intFromBool(head.free_first) and head.hold_context_rows[s] > 0 and
                (head.charged.len == 0 or head.charged[s])) break :context true;
        }
        break :context false;
    };
    const mixed = mixed_host or context_order;
    const least = @min(head.least_rows, max_rows);
    var best = if (fixed_rows >= least) rate(costs, timed, fixed_rows, @floatFromInt(fixed_rows), (head.plain_ms orelse overhead_ms) + fixed_host_ms + rowTerm(costs, timed, head, fixed_rows, live)) else -std.math.inf(f64);
    while (true) {
        const incumbent = best;
        // Unknown/nonfinite rates cannot price marginal utility (in particular, infinity * a free row).
        const price_order = mixed and incumbent >= 0 and std.math.isFinite(incumbent);
        @memset(counts, 0);
        var rows = fixed_rows;
        var host_ms = fixed_host_ms;
        var expected: f64 = @floatFromInt(rows);
        var drafts: u64 = 0;
        var deepest: u32 = 0;
        var firsts: u64 = 0;
        var pass: GridPass = .{ .costs = head.hold_costs, .contexts = head.hold_context_costs, .debts = head.hold_context_rows, .batched = head.batched, .free = @intFromBool(head.free_first), .rows = head.hold_rows, .block_rows = head.hold_block_rows, .counts = counts, .charged = head.charged };
        var head_ms: f64 = 0.0; // the nodes counted so far: the head's price
        while (rows < max_rows) {
            // Solo first-block context quotes order candidates; the complete physical group decides funding.
            var pick: ?usize = null;
            var pick_p: f64 = 0.0;
            var pick_score: f64 = 0.0;
            for (probs, 0..) |p, s| {
                if (counts[s] >= p.len) continue;
                const v = p[counts[s]];
                const score = if (price_order) ordered: {
                    var ms = if (head.host_row_ms.len > 0) head.host_row_ms[s] else 0;
                    if (context_order and counts[s] == @intFromBool(head.free_first) and
                        (head.charged.len == 0 or head.charged[s]))
                        ms += hold_price.contextMs(head.hold_context_costs, 1, head.hold_context_rows[s]);
                    break :ordered if (std.math.isFinite(ms)) v - incumbent * ms else -std.math.inf(f64);
                } else v;
                if (pick == null or score > pick_score) {
                    pick = s;
                    pick_p = v;
                    pick_score = score;
                }
            }
            const s = pick orelse break;
            counts[s] += 1;
            rows += 1;
            expected += pick_p;
            if (head.host_row_ms.len > 0) host_ms += head.host_row_ms[s];
            const neural = if (head.heads) |mask| mask[s] else true;
            if (neural and (head.charged.len == 0 or head.charged[s])) {
                firsts += @intFromBool(counts[s] == 1);
                drafts += 1;
                deepest = @max(deepest, counts[s]);
                if (grid) head_ms = pass.grow(s, counts[s]);
            }
            if (!grid) {
                const steps: f64 = @floatFromInt(if (head.batched) deepest -| @intFromBool(head.free_first) else drafts - (if (head.free_first) firsts else 0));
                const extra_steps: f64 = @floatFromInt(if (head.batched) deepest -| 1 else drafts - firsts);
                head_ms = head.step_ms * steps + head.extra_step_ms * extra_steps + (if (drafts > 0) head.block_ms else 0);
            }
            if (rows < least) continue;
            const now = rate(costs, timed, rows, expected, overhead_ms + host_ms + head_ms + rowTerm(costs, timed, head, rows, live));
            if (now > best or (!timed and head.step_ms == 0 and !grid)) {
                best = now;
                @memcpy(best_counts, counts);
            }
        }
        // Strict improvement over finite prefix orders terminates without claiming a global grid search.
        if (!mixed or !(best > incumbent) or !std.math.isFinite(best)) break;
    }
    // a measuring round whose width is out of reach takes every node on offer
    if (best == -std.math.inf(f64)) @memcpy(best_counts, counts);
    return best_counts;
}

/// head.row_ms for the rows past the streams' first; a fall below the table stops at the streams' plain width.
pub fn rowTerm(costs: *const Table, timed: bool, head: alloc.HeadCost, rows: u64, live: u64) f64 {
    const term = head.row_ms * @as(f64, @floatFromInt(rows -| live));
    if (term >= 0 or !timed) return term;
    const at = costs.get(@intCast(rows)) orelse return term;
    const plain = costs.get(@intCast(@max(live, 1))) orelse return term;
    return @max(term, plain - at);
}

fn rate(costs: *const Table, timed: bool, total: u64, expected: f64, overhead_ms: f64) f64 {
    var cost: f64 = 1.0;
    if (timed) cost = costs.get(@intCast(total)) orelse return -1.0;
    // A negative total cannot price a round; a zero one is an infinitely cheap baseline no draft improves on.
    if (cost + overhead_ms < 0) return -1;
    if (cost + overhead_ms == 0) return std.math.inf(f64);
    return expected / (cost + overhead_ms);
}
