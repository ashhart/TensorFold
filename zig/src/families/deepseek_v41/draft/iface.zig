//! The lanes backend's contract with the model: `Target` (verify windows) and `Pass` (DSpark over its own rings).
const std = @import("std");
const lanes = @import("lanes");
const dspark = @import("dspark.zig");

/// One slot's verify rows: row 0 the pending token at `start`; row r attends cache + `parents`, keyed at `draws[r]`.
pub const Segment = struct {
    slot: u32,
    start: u64,
    tokens: []const u32,
    parents: ?[]const i32, // each row's parent row (row 0: -1); null: a chain
    draws: []const u64,
    sampling: ?lanes.Sampling,
};

/// A slot's last rows' taps (bf16 [rows, 3 x hidden] at `address`); `map`: tree window row -> taps row (branches.zig).
pub const Taps = struct { address: u64, rows: u32, stride: u32, map: ?[]const u32 = null };

pub const Target = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Run the prompt into the slot's state; the keyed choice after its last row (drawn at `draw`).
        prefill: *const fn (ptr: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!u32,
        /// One forward over every segment into `choices[i][r]`; rows stay until `keep` (never kept: all rows stay).
        window: *const fn (ptr: *anyopaque, segments: []const Segment, choices: [][]u32) anyerror!void,
        /// Keep the listed rows of the slot's last window (a parent chain); pool, SWA, carries roll back.
        keep: *const fn (ptr: *anyopaque, slot: u32, path: []const u32) anyerror!void,
        /// The taps of the slot's last window (or its prompt, after prefill).
        taps: *const fn (ptr: *anyopaque, slot: u32) Taps,
        release: *const fn (ptr: *anyopaque, slot: u32) void,
        release_checked: ?*const fn (*anyopaque, u32) anyerror!void = null,
        available: ?*const fn (*anyopaque, u32) bool = null,
        /// Before `prefill`: the request's max new tokens, for page-based pool admission. null: nothing to do.
        admit: ?*const fn (ptr: *anyopaque, slot: u32, max_new: u64) void = null,
        /// Prompt in pieces between rounds (kv/sched.zig): `begin`, `piece`s, `finish`. null: whole prompts.
        begin: ?*const fn (ptr: *anyopaque, slot: u32, ids: []const u32) anyerror!Begun = null,
        piece: ?*const fn (ptr: *anyopaque, slot: u32, ids: []const u32, start: u64, end: u64, save: bool) anyerror!void = null,
        finish: ?*const fn (ptr: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!u32 = null,
        /// TF_DSV41_TAIL_JOIN: `finish` short of the last token, so a round's shared window runs it. null: `finish`.
        tail: ?*const fn (ptr: *anyopaque, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) anyerror!?u32 = null,
        /// A round's pieces of several slots (TF_DSV41_PIECE_RUNS) in one forward where possible. null: `piece` each.
        pieces: ?*const fn (ptr: *anyopaque, list: []const Piece) anyerror!void = null,
        /// A round's finished prompts' CED decoder replays together (TF_DSV41_REPLAY_RUNS). null: nothing.
        replays: ?*const fn (ptr: *anyopaque, list: []const Final) anyerror!void = null,
        /// Engram reads ahead of the next window (TF_DSV41_ENGRAM_WARM); reads only, window bits are unaffected.
        warm: ?*const fn (ptr: *anyopaque, asks: []const Warm) void = null,
    };

    /// A slot's whole prompt whose rows but the last are in (its `finish` / `tail` follows).
    pub const Final = struct { slot: u32, ids: []const u32 };

    /// A slot's prompt `ids` (the whole prompt) rows [start, end) of a round; `save`: its snapshot right after them.
    pub const Piece = struct { slot: u32, ids: []const u32, start: u64, end: u64, save: bool };

    pub fn warm(t: Target, asks: []const Warm) void {
        if (t.vtable.warm) |f| f(t.ptr, asks);
    }

    pub const Begun = struct { at: u64 = 0, damaged: bool = false };

    pub fn admit(t: Target, slot: u32, max_new: u64) void {
        if (t.vtable.admit) |f| f(t.ptr, slot, max_new);
    }

    pub fn pieces(t: Target) bool {
        return t.vtable.begin != null and t.vtable.piece != null and t.vtable.finish != null;
    }
    pub fn begin(t: Target, slot: u32, ids: []const u32) !Begun {
        return (t.vtable.begin orelse return error.NoPieces)(t.ptr, slot, ids);
    }
    pub fn piece(t: Target, slot: u32, ids: []const u32, start: u64, end: u64, save: bool) !void {
        return (t.vtable.piece orelse return error.NoPieces)(t.ptr, slot, ids, start, end, save);
    }
    pub fn finish(t: Target, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) !u32 {
        return (t.vtable.finish orelse return error.NoPieces)(t.ptr, slot, ids, sampling, draw);
    }
    pub fn runReplays(t: Target, list: []const Final) !void {
        if (t.vtable.replays) |f| return f(t.ptr, list);
    }
    pub fn runPieces(t: Target, list: []const Piece) !void {
        if (t.vtable.pieces) |f| return f(t.ptr, list);
        for (list) |p| try t.piece(p.slot, p.ids, p.start, p.end, p.save);
    }
    pub fn tail(t: Target, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) !?u32 {
        const f = t.vtable.tail orelse return try t.finish(slot, ids, sampling, draw);
        return f(t.ptr, slot, ids, sampling, draw);
    }

    pub fn prefill(t: Target, slot: u32, ids: []const u32, sampling: ?lanes.Sampling, draw: u64) !u32 {
        return t.vtable.prefill(t.ptr, slot, ids, sampling, draw);
    }
    pub fn window(t: Target, segments: []const Segment, choices: [][]u32) !void {
        return t.vtable.window(t.ptr, segments, choices);
    }
    pub fn keep(t: Target, slot: u32, path: []const u32) !void {
        return t.vtable.keep(t.ptr, slot, path);
    }
    pub fn taps(t: Target, slot: u32) Taps {
        return t.vtable.taps(t.ptr, slot);
    }
    pub fn releaseChecked(t: Target, slot: u32) !void {
        if (t.vtable.release_checked) |checked| return checked(t.ptr, slot);
        t.vtable.release(t.ptr, slot);
    }
    pub fn available(t: Target, slot: u32) bool {
        return if (t.vtable.available) |available_| available_(t.ptr, slot) else true;
    }
    pub fn release(t: Target, slot: u32) void {
        t.vtable.release(t.ptr, slot);
    }
};

/// One slot of a DSpark pass: its anchor (the pending token) at P and the keyed draft parameters.
pub const Ask = struct { slot: u32, anchor: u32, start: u64, params: dspark.Params };

/// One slot's pass results: `block` drafts and confidences; for trees also candidates and pre-Markov base logits.
pub const Proposal = struct { drafts: []u32, conf: []f32, cand: ?[]i32 = null, base: ?[]f32 = null };

pub const Pass = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Committed rows into the slot's context rings: `taps` rows `rows[i]` land at positions start + i.
        ingest: *const fn (ptr: *anyopaque, slot: u32, start: u64, taps: Taps, rows: []const u32) anyerror!void,
        /// One pass over every ask (one launch a block over all slots); results into `out`.
        propose: *const fn (ptr: *anyopaque, asks: []const Ask, out: []Proposal) anyerror!void,
        /// A new request on the slot: its context restarts at position 0.
        reset: *const fn (ptr: *anyopaque, slot: u32) void,
        /// The Markov / confidence heads on the host, for trees (null: no trees).
        markov: *const fn (ptr: *anyopaque) ?dspark.Markov,
        /// Start a pass `collect` finishes (TF_DSV41_SPEC_DRAFT); only collective-free work between. Null: none.
        begin: ?*const fn (ptr: *anyopaque, asks: []const Ask) anyerror!void = null,
        /// The begun pass's results, in its asks' order (as `propose` gives them).
        collect: ?*const fn (ptr: *anyopaque, out: []Proposal) anyerror!void = null,
        /// Before a speculated pass's verify window: start from its device pick (`sampled`: keyed). Null: none.
        arm: ?*const fn (ptr: *anyopaque, slot: u32, start: u64, rows: u32, sampled: bool) void = null,
    };

    pub fn ingest(p: Pass, slot: u32, start: u64, taps: Taps, rows: []const u32) !void {
        return p.vtable.ingest(p.ptr, slot, start, taps, rows);
    }
    pub fn propose(p: Pass, asks: []const Ask, out: []Proposal) !void {
        return p.vtable.propose(p.ptr, asks, out);
    }
    pub fn reset(p: Pass, slot: u32) void {
        p.vtable.reset(p.ptr, slot);
    }
    pub fn markov(p: Pass) ?dspark.Markov {
        return p.vtable.markov(p.ptr);
    }
};

/// Which committed rows an ingest writes: only the last `window` (older ones are never read).
pub fn ingestSpan(n: usize, window: usize) struct { skip: usize, count: usize } {
    const skip = n -| window;
    return .{ .skip = skip, .count = n - skip };
}

/// An Engram warm ask (Target.warm).
pub const Warm = struct { slot: u32, start: u64, ids: []const u32 };
