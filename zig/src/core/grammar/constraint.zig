//! One reply's grammar: `cut` drops rejected drafts, `fill` writes allowed bits, `advance` follows chosen tokens.

const std = @import("std");
const engine = @import("engine.zig");

pub const Error = engine.Error || error{ Rejected, Empty, RowsChanged };

/// A chain window as the grammar keeps it: rows [0, keep) of the window run; rows [first, keep) are constrained.
pub const Cut = struct {
    keep: u32,
    first: u32,
    stop: bool = false, // row `keep` is the grammar's stop token: it and every later row are cut

    pub fn rows(c: Cut) u32 {
        return c.keep - c.first;
    }
};

pub const Constraint = struct {
    m: engine.Matcher,
    /// tokens the grammar never takes (special tokens other than the stop tokens): a draft of one is cut unasked
    never: *const std.DynamicBitSetUnmanaged,
    think_end: u32,
    active: bool, // false: unconstrained until `think_end` is chosen
    words: u32,

    /// A fresh reply over `compiled`; `active` false: the grammar starts after `think_end` is chosen.
    pub fn init(compiled: engine.Compiled, never: *const std.DynamicBitSetUnmanaged, think_end: u32, active: bool, words: u32) engine.Error!Constraint {
        return .{ .m = try compiled.matcher(), .never = never, .think_end = think_end, .active = active, .words = words };
    }

    pub fn deinit(c: *Constraint) void {
        c.m.deinit();
    }

    /// The grammar has taken its stop token: the reply is complete (nothing after it is constrained).
    pub fn finished(c: *const Constraint) bool {
        return c.active and c.m.terminated();
    }

    fn never_(c: *const Constraint, t: u32) bool {
        return t < c.never.bit_length and c.never.isSet(t);
    }

    /// Window rows (row 0: the pending token, already followed) an accepted path can use; the matcher ends unchanged.
    pub fn cut(c: *Constraint, tokens: []const u32) Error!Cut {
        const n: u32 = @intCast(tokens.len);
        if (c.finished()) return .{ .keep = n, .first = n };
        var active = c.active;
        var first: u32 = if (active) 0 else n;
        var keep: u32 = 1;
        var accepted: u32 = 0;
        var stop = false;
        defer if (accepted > 0) c.m.rollback(accepted) catch {};
        for (tokens[1..], 1..) |t, r| {
            if (active) {
                if (c.never_(t) or !try c.m.accept(t)) break; // the parent row's masked draw could never choose it
                accepted += 1;
                if (c.m.terminated()) { // its stop token: nothing is verified after it
                    stop = true;
                    break;
                }
            } else {
                active = t == c.think_end; // the grammar starts at the row after </think>
                if (active) first = @intCast(r);
            }
            keep += 1;
        }
        if (first > keep) first = keep;
        return .{ .keep = keep, .first = first, .stop = stop };
    }

    /// Each constrained row r's allowed bits at `bits[(r - k.first) * words ..]`; error.Empty when a row allows none.
    pub fn fill(c: *Constraint, tokens: []const u32, k: Cut, bits: []u32) Error!void {
        if (k.rows() == 0) return;
        if (bits.len < @as(usize, k.rows()) * c.words) return error.RowsChanged;
        var active = c.active;
        var accepted: u32 = 0;
        var j: u32 = 0;
        defer if (accepted > 0) c.m.rollback(accepted) catch {};
        for (tokens[0..k.keep], 0..) |t, r| {
            if (r > 0) {
                if (active) {
                    if (!try c.m.accept(t)) return error.Rejected;
                    accepted += 1;
                } else active = t == c.think_end;
            }
            if (!active) continue;
            if (r != k.first + j) return error.RowsChanged;
            const row = bits[@as(usize, j) * c.words ..][0..c.words];
            try c.m.fill(row);
            if (std.mem.allEqual(u32, row, 0)) return error.Empty;
            j += 1;
        }
        if (j != k.rows()) return error.RowsChanged;
    }

    /// Follow chosen tokens (each chosen under this grammar's mask); after the stop token nothing follows.
    pub fn advance(c: *Constraint, tokens: []const u32) Error!void {
        for (tokens) |t| {
            if (!c.active) {
                c.active = t == c.think_end;
                continue;
            }
            if (c.m.terminated()) return;
            if (!try c.m.accept(t)) return error.Rejected;
        }
    }
};

/// False when `prompt` ends inside an open think block (its last `open` after its last `end`).
pub fn thinkActive(prompt: []const u32, open: ?u32, end: ?u32) bool {
    var i = prompt.len;
    while (i > 0) {
        i -= 1;
        if (end != null and prompt[i] == end.?) return true;
        if (open != null and prompt[i] == open.?) return false;
    }
    return true;
}
