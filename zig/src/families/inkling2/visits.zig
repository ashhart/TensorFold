//! The visit plan: the order a forward runs its layers in, with Mini's loop passes, cache slots, messages and taps.
const std = @import("std");
const config = @import("config.zig");

/// The four inputs of a core layer that blend a message, in the order of the reference's messages and gate kernels.
pub const Stream = enum(u2) { q, k, v, mlp };

/// Which cross-pass table holds a stream's query, and its row there.
pub const QueryRow = struct { kv: bool, row: u32 };

/// Rows per core layer j: queries 2j (q) and 2j+1 (MLP), kv_queries 2j (k) and 2j+1 (v), as vLLM's loader stacks them.
pub fn queryRow(core: u32, s: Stream) QueryRow {
    return switch (s) {
        .q => .{ .kv = false, .row = 2 * core },
        .k => .{ .kv = true, .row = 2 * core },
        .v => .{ .kv = true, .row = 2 * core + 1 },
        .mlp => .{ .kv = false, .row = 2 * core + 1 },
    };
}

pub const Visit = struct {
    layer: u32,
    pass: u32, // 0 for the prelude, the first pass over the core and the coda
    slot: u32, // KV and conv cache slot: the layer itself in pass 0, then after the model's layers pass by pass
    core: ?u32, // layer - core start on a core layer: its cross-pass query rows and its message gates
    messages: bool, // the q, k, v and MLP inputs blend a message from the previous pass's bank
    restart: bool, // hidden and block states go back to the prelude's before this visit
    saves_bank: bool, // the visit's output closes a pass that a later pass reads messages from
};

pub const Plan = struct {
    visits: [config.max_visits]Visit,
    n: u32,
    prelude_blocks: u32, // finished blocks when the core starts, as every pass restarts with
    bank_blocks: [config.max_layers]u32, // core layers whose input joins a pass's bank, in order
    bank_n: u32, // the bank: bank_n block states (inputs of bank_blocks) then the pass's output

    pub fn at(p: *const Plan, v: u32) *const Visit {
        return &p.visits[v];
    }

    /// States a message attends over: the bank's blocks plus the previous pass's output.
    pub fn bankSources(p: *const Plan) u32 {
        return p.bank_n + 1;
    }

    /// The visit whose input drafter tap `tap` reads, or null for the tap after the final norm (tap == n).
    pub fn tapVisit(p: *const Plan, tap: u32) error{NoSuchVisit}!?u32 {
        if (tap > p.n) return error.NoSuchVisit;
        return if (tap == p.n) null else tap;
    }

    /// Cache slots the forward keeps per stream: one per visit.
    pub fn slots(p: *const Plan) u32 {
        return p.n;
    }
};

/// The visit plan of a parsed config; Small's is its layers in order, each in pass 0.
pub fn build(c: *const config.Text) Plan {
    const p = of(c.layers, c.block, c.loop);
    std.debug.assert(p.n == c.visitCount());
    return p;
}

/// The plan of `layers` layers in attention-residual blocks of `block`, with or without a loop.
pub fn of(layers: u32, block: u32, loop: ?config.Loop) Plan {
    var p = std.mem.zeroes(Plan);
    const l = loop orelse config.Loop{ .core_start = layers, .core_end = layers, .passes = 1, .gate_rank = 0 };
    p.prelude_blocks = (l.core_start + block - 1) / block;
    // the first pass appends a block at each boundary in the core; the bank keeps those after the opening one
    for (l.core_start..l.core_end) |k| if (k > l.core_start and k % block == 0) {
        p.bank_blocks[p.bank_n] = @intCast(k);
        p.bank_n += 1;
    };
    for (0..l.core_start) |i| add(&p, @intCast(i), 0, l, layers);
    for (0..l.passes) |pass| for (l.core_start..l.core_end) |i| add(&p, @intCast(i), @intCast(pass), l, layers);
    for (l.core_end..layers) |i| add(&p, @intCast(i), 0, l, layers);
    return p;
}

/// Whether a visit to `layer` in `pass` writes the bank a later pass reads (its input, as one of bank_blocks).
pub fn savesBlock(p: *const Plan, v: *const Visit, passes: u32) bool {
    if (v.core == null or v.pass + 1 >= passes) return false;
    for (p.bank_blocks[0..p.bank_n]) |b| if (b == v.layer) return true;
    return false;
}

/// The bank row a block-saving visit writes: its place among bank_blocks.
pub fn bankIndex(p: *const Plan, layer: u32) u32 {
    for (p.bank_blocks[0..p.bank_n], 0..) |b, i| if (b == layer) return @intCast(i);
    unreachable;
}

/// Layers in the loop's core.
pub fn coreLen(l: config.Loop) u32 {
    return l.core_end - l.core_start;
}

fn add(p: *Plan, i: u32, pass: u32, l: config.Loop, layers: u32) void {
    const in_core = i >= l.core_start and i < l.core_end;
    p.visits[p.n] = .{
        .layer = i,
        .pass = pass,
        // the reference's cache_offset: pass p > 0 of core layer i uses layers + (p - 1) * core + (i - start)
        .slot = if (pass == 0) i else layers + (pass - 1) * coreLen(l) + (i - l.core_start),
        .core = if (in_core) i - l.core_start else null,
        .messages = in_core and pass > 0,
        .restart = in_core and pass > 0 and i == l.core_start,
        .saves_bank = in_core and i + 1 == l.core_end and pass + 1 < l.passes,
    };
    p.n += 1;
}
