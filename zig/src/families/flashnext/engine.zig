//! Flash Next served from the replay engine: one reply at a time, its prompt through prompt chunks (the MTP head's...
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const lanes = @import("lanes");
const fz = @import("replay.zig");
const config = @import("config.zig");
const segments = @import("../../core/segments.zig");
const snapshot = @import("snapshot.zig");
const CallLog = @import("call_log.zig").CallLog;
const tp_settings = @import("tp_settings.zig");
const follow_mod = @import("follow.zig");
const marks_mod = @import("marks.zig");
const pack_io = @import("pack_io.zig");
const pack = @import("pack.zig");
const pack_cache = @import("pack_cache.zig");
const load_admission = @import("load_admission.zig");
const think = @import("think.zig");
const sample = @import("sample.zig");
pub const Think = think.Think;
pub const quantizationProblem = load_admission.problemFor;
const Io = std.Io;
const Allocator = std.mem.Allocator;

const D = fz.D;
const WIDE = fz.WIDE;
const LAYERS = fz.LAYERS;
const VOCAB = fz.VOCAB;
const PLE_TAIL = fz.PLE_TAIL;
const GROUPS = fz.GROUPS;
const MAXR = fz.MAXR;
const CS_ROW = fz.CS_ROW;
const SO_ROW = fz.SO_ROW;
const TOP = fz.TOP;
const PMAX = fz.PMAX;
const Buf = fz.Buf;
const Run = fz.Run;
const Model = fz.Model;
const Layer = fz.Layer;
const Prompt = fz.Prompt;
const GSelect = fz.GSelect;
const Select = fz.Select;
const Tmp = fz.Tmp;
const Slot = fz.Slot;
const DepthRule = fz.DepthRule;
const CostDepth = @import("drafts_cost.zig").CostDepth;
const hcOf = fz.hcOf;
const laneOf = fz.laneOf;
const i32Buf = fz.i32Buf;
const f32Buf = fz.f32Buf;
const jsonInt = fz.jsonInt;

/// Rounds the token ring holds (fz_accept writes round & 511).
pub const RING = 512;
/// Positions a reply keeps free past its last token: two rounds in flight.
pub const MARGIN = 2 * MAXR;
/// An MoE layer's expert projections in the engine's order: gate, up, shared gate, shared up, down, shared down.
const expert_projs = [_][]const u8{ "switch_mlp.gate_proj", "switch_mlp.up_proj", "shared_expert.gate_proj", "shared_expert.up_proj", "switch_mlp.down_proj", "shared_expert.down_proj" };
/// Bytes a conversation's next kept state usually adds (about 4k tokens of reply and tool output): the pool readies...
pub const NEXT_TURN = 4096 * 13 * 2304;

/// Hash the shared token prefix through its lookahead token.
pub fn keyOf(tokens: []const u32) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens));
}
/// Prompt chunks run as two staggered segments once each gets this many rows (core/segments.zig).
pub const SEG_MIN = 4096;
/// Speed-up mode splits a prompt chunk across the two Macs once each gets this many rows.
pub const PAIR_MIN: u32 = 256;

pub const Reason = enum { stop, length, cancelled };

pub const Result = struct {
    reason: Reason,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    min_rows: u32 = 0,
    copy_rounds: u64 = 0,
    copy_accepted: u64 = 0,
    copy_by_len: [9][2]u64 = @splat(.{ 0, 0 }), // copied rounds and drafts landed by match length
};

/// What a reply reports while it runs (called on the engine's thread).
pub const Out = struct {
    ctx: *anyopaque,
    /// The prompt is in the cache.
    prefilled: *const fn (ctx: *anyopaque) void,
    /// Tokens committed, in order; true ends the reply as a stop (a stop string matched).
    tokens: *const fn (ctx: *anyopaque, toks: []const u32) bool,
    /// Checked between chunks and rounds.
    cancelled: *const fn (ctx: *anyopaque) bool,
    /// The prompt pass stands at one of generateFrom's marks (a chunk end): the prompt cache keeps its state here.
    marked: ?*const fn (ctx: *anyopaque, at: usize) void = null,
    /// A sampled reply (a temperature above 0): each token drawn on the GPU by tf_sample_full; null: greedy.
    sampling: ?lanes.Sampling = null,
};

pub const Engine = struct {
    /// A mark a prompt call ran past: its DeltaNet states in m.marks' slot, and the n-gram tail rows and history there.
    pub const Passed = struct { at: usize, slot: usize, tail: Buf, hist: [2]i64 };

    gpa: Allocator,
    arena_state: std.heap.ArenaAllocator,
    r: *Run,
    m: *Model,
    pr: *Prompt,
    pr2: *Prompt, // the second staggered segment's buffers, selection and queue
    segments: bool = true, // prompt chunks as two staggered segments (false: one serial chunk at a time)
    g_cs: mtl.Buffer,
    g_so: mtl.Buffer,
    o_cs: mtl.Buffer,
    o_so: mtl.Buffer,
    cins: [2]Buf,
    ring: mtl.Buffer,
    hist: mtl.Buffer, // the token history the copy drafts search
    copy: bool = true, // copy drafts from the history when a long enough match exists
    tree: ?*@import("tree.zig").Tree = null, // FZ_TREE=a,b: rounds as a root, a head chain of a and a branch of b
    copy_min: u32 = 3,
    copy_long: u32 = 6, // shorter matches copy only when the head's first draft agrees
    mark_taps: bool = true, // a prompt call writes the DeltaNet states at marks inside it (false: calls end at marks)
    pair_min: u32 = PAIR_MIN, // speed-up mode's pair chunks: rows each Mac takes at least (FZ_PAIR_MIN; rank 1 takes rank 0's)
    call_log: bool = false, // FZ_CALL_LOG: one line a prompt pass with each call's rows, wall and GPU time
    snap_pool: snapshot.Pool = .{}, // kept states' buffers, reused and readied ahead
    handoff_ns: i96 = 0, // speed-up rank 0: the last request's handoff to rank 1 and its resume answer
    passed: ?Passed = null, // the mark a prompt call ran past, while the prompt cache keeps the state there
    peer_drops: std.ArrayList(u64) = .empty, // speed-up rank 0: kept states rank 1 drops with the next request
    peer_kept: std.AutoHashMapUnmanaged(u64, *snapshot.State) = .empty, // speed-up rank 1: its halves of rank 0's kept states
    peer_budget: u64 = 0, // speed-up rank 1: the budget its kept states' buffers and free ones stay inside
    peer_held: u64 = 0, // speed-up rank 1: its kept states' buffers
    identity_hash: u64 = 0, // the checkpoint's identity text (pack_io.zig) hashed: --learn keys learned states by it
    // while copies land worse than the head's drafts, only 8-token matches the head agrees with are copied
    wids: Buf,
    draw_at: Buf, // a sampled window's draw positions (sample.zig)
    rows_w: [MAXR + 1]Buf,
    mdims_w: [MAXR + 1]Buf,
    host: struct { rows: Buf, mdims: Buf, pos8: Buf, nk8: Buf, kvmeta: Buf, slots: [2 * MAXR]Slot },
    gpu_slots: [2 * MAXR]Slot,
    tsel: GSelect,
    msel: GSelect,

    /// Load `model_dir` from a dump directory, or from the checked-in kernels when `dump_dir` is null.
    pub fn load(gpa: Allocator, io: std.Io, model_dir: []const u8, dump_dir: ?[]const u8) !*Engine {
        return loadWith(gpa, io, model_dir, dump_dir, null);
    }

    /// `load`, in speed-up mode when `speed_up` (or TF_FLASHNEXT_TP) names this Mac's settings: rank, MCDMA library...
    pub const loadWith = @import("engine_load.zig").loadWith;

    pub const loadConfigured = @import("engine_load.zig").loadConfigured;

    /// The GPU-side rounds' buffers (engine_rounds.zig).
    pub fn rounds(e: *Engine) !void {
        return @import("engine_rounds.zig").rounds(e);
    }

    /// One DeltaNet state a layer in the engine's own rounds, kept rows replayed (engine_rounds.zig).
    pub fn keepStates(e: *Engine) !void {
        return @import("engine_rounds.zig").keepStates(e);
    }

    /// Host-driven calls (prompt chunks, the head's first drafts) read the host's metadata buffers.
    pub fn hostMode(e: *Engine) void {
        const m = e.m;
        e.r.gpu_round = false;
        e.r.gsel = null;
        m.mtp.gsel = null;
        m.t.rows, m.t.mdims, m.t.pos8, m.t.nk8, m.t.kvmeta = .{ e.host.rows, e.host.mdims, e.host.pos8, e.host.nk8, e.host.kvmeta };
        m.mtp.slots = e.host.slots;
        m.ple.cin = e.cins[0];
    }

    /// GPU-side rounds read positions and widths the previous round's verdict wrote into the arena.
    pub fn gpuMode(e: *Engine) void {
        const m = e.m;
        const r = e.r;
        r.gpu_round = true;
        m.t.pos8 = .{ .b = r.ar.b, .off = fz.AR_POS * 4 };
        m.t.nk8 = .{ .b = r.ar.b, .off = fz.AR_NK * 4 };
        m.t.kvmeta = .{ .b = r.ar.b, .off = fz.AR_KV * 4 };
        m.mtp.slots = e.gpu_slots;
    }

    pub fn statesCopy(e: *Engine, states: bool) void { // the kept row of every DeltaNet layer's window output into its state
        const r = e.r;
        if (states) r.copyKept(.{ .b = e.o_so }, .{ .b = e.g_so }, SO_ROW / 4, SO_ROW / 4, MAXR * SO_ROW / 4, SO_ROW / 4, 36, -1);
        r.copyKept(.{ .b = e.o_cs }, .{ .b = e.g_cs }, CS_ROW / 4, CS_ROW / 4, MAXR * CS_ROW / 4, CS_ROW / 4, 36, -1);
    }

    /// A prompt call of up to `left` rows: one chunk, as two staggered segments once each gets SEG_MIN rows.
    pub fn chunkCall(e: *const Engine, left: usize) segments.Call {
        const step = fz.chunkRows(e.m.pos, e.pr.step); // past the trained window a chunk scores more blocks a row
        return if (e.segments) segments.next(left, step, SEG_MIN) else .{ .rows = @min(step, left), .parts = 1 };
    }

    pub fn isEos(eos: []const u32, tok: u32) bool {
        return std.mem.indexOfScalar(u32, eos, tok) != null;
    }

    /// One reply, sampled when out.sampling is set. `depth`: a round's drafts (0: one token a round); null: the rule.
    pub const generate = @import("engine_generate.zig").generate;

    /// `generate` after snapshot.restore put `from` tokens in the model, cut at marks for out.marked, under `budget`.
    pub const generateFrom = @import("engine_generate.zig").generateFrom;

    /// Where `decode` left a stream it handed over: its state as a host round leaves it, `cache` rows in the cache.
    pub const Handoff = @import("engine_generate.zig").Handoff;

    /// GPU rounds after prefill; cancellation yields through handoff when supplied.
    pub const decode = @import("engine_generate.zig").decode;

    /// Hands over at cache rows after the queued round, restoring the n-gram tail.
    pub const handOver = @import("engine_generate.zig").handOver;

    /// Speed-up mode: whether to stop at a step both ranks take (rank 0 decides, rank 1 follows); alone, `quit`.
    pub const agree = @import("engine_generate.zig").agree;

    /// A round's stop decision and the next round's drafts: rank 0's for both Macs (each Mac's own timings differ).
    pub const agreeRound = @import("engine_generate.zig").agreeRound;

    /// Speed-up mode's rank 1: it runs rank 0's requests (follow) and serves none of its own.
    pub fn followsPeer(e: *const Engine) bool {
        return if (e.r.tp) |tp| tp.rank == 1 else false;
    }

    /// Served speed-up mode, rank 1: end `follow`'s wait for rank 0's next request (before deinit).
    pub fn stopFollowing(e: *Engine) void {
        if (e.r.tp) |tp| tp.quitting.store(true, .release);
    }

    /// Served speed-up mode, rank 1: rank 0's requests in step until its empty request (follow.zig).
    pub fn follow(e: *Engine) !void {
        return follow_mod.run(e);
    }

    /// Rank 1: rank 0's next request, its reply into `out` (null: hashed and logged, this Mac's states kept); null...
    pub fn followWith(e: *Engine, out: ?Out) !?Result {
        return follow_mod.one(e, out);
    }

    pub const Quiet = struct {
        fn prefilled(_: *anyopaque) void {}
        fn tokens(_: *anyopaque, _: []const u32) bool {
            return false;
        }
        fn cancelled(_: *anyopaque) bool {
            return false;
        }
    };

    /// A short reply on fixed tokens: the kernels' first launches and the weights' first reads happen here, not in...
    pub fn warm(e: *Engine) !void {
        if (e.r.tp) |tp| if (tp.rank == 1) {
            _ = try follow_mod.one(e, null);
            return;
        };
        var toks: [96]u32 = undefined;
        for (&toks, 0..) |*t, i| t.* = @intCast(1000 + i);
        var dummy: u8 = 0;
        _ = try e.generate(&toks, 24, &.{}, null, .{ .ctx = &dummy, .prefilled = Quiet.prefilled, .tokens = Quiet.tokens, .cancelled = Quiet.cancelled });
    }

    pub fn deinit(e: *Engine) void {
        if (e.r.tp) |tp| { // rank 1's follow ends on rank 0's empty request; then the link closes, before the memory
            if (tp.rank == 0) {
                const none: [12]u32 = @splat(0);
                tp.sendRequest(std.mem.asBytes(&none), &.{}) catch {};
            }
        }
        const gpa = e.gpa;
        var kept = e.peer_kept.valueIterator();
        while (kept.next()) |st| snapshot.drop(gpa, st.*);
        e.peer_kept.deinit(gpa);
        e.snap_pool.deinit();
        e.peer_drops.deinit(gpa);
        e.r.deinit();
        e.r.device.deinit();
        e.arena_state.deinit();
        gpa.destroy(e);
    }
};

const PackBuilder = @import("engine_load.zig").PackBuilder;

test "load sets no DeltaNet state view before rounds() points them at the round buffers" {
    const src = @embedFile("engine_load.zig");
    const at = std.mem.indexOf(u8, src, "pub fn loadConfigured(").?;
    const body = src[at..std.mem.indexOfPos(u8, src, at, "try e.rounds();").?];
    try std.testing.expect(std.mem.indexOf(u8, body, ".cs[") == null and std.mem.indexOf(u8, body, ".so[") == null);
}
