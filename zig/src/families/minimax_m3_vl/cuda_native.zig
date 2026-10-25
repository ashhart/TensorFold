//! MiniMax-M3.1-Flash's native CUDA family: the openWith API (Options/open/capacity, refusals in words),
//! the pack's real admission — per-module bit-width discovery from the trellis headers, the whole weights
//! plan, the served shapes — and the exact serve boundary this stage stops at (the MSA device forward).

const std = @import("std");
const cuda = @import("cuda");
pub const lane_api = @import("lanes"); // the boundary tests reach Stream/Feed/Backend through the family
const lanes = lane_api;
const config = @import("config.zig");
const exl3 = @import("exl3.zig");
const weights = @import("weights.zig");
pub const weights_mod = weights; // the harness's fixture builder rides this surface
/// The MSA indexer's CPU contracts (block-max-pool scores, the fixed-width topk with the boost
/// overrides): part of this family's API, the reference its CUDA kernels are graded against.
pub const indexer = @import("sparse.zig");
pub const sparse_cuda = @import("cuda_sparse.zig"); // the MSA indexer's CUDA launches
pub const serve_cuda = @import("cuda_serve.zig"); // the family's own glue (the swigluoai rows)
pub const serve = @import("serve.zig"); // the host-graded forward: the device path's contract
pub const rank_plan = @import("shard.zig"); // the TP2 rank placement: this stage receipts the per-rank bytes
pub const graph = @import("graph.zig"); // the launch graph this stage attaches (TP1 through open())

pub const model_type = "minimax_m3_vl";
/// The checkpoints this family reads: Mia-AiLab's MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.06-bpw and its line.
pub const formats: []const []const u8 = &.{"exl3"};
/// The server serves this stage's requests as fixed greedy: temperature at any value (or absent)
/// serves, its other sampling fields refuse (SamplingNotAttached) — the keyed sampler is a later pass.
pub const temperature_greedy = true;
/// The pack's own window (max_position_embeddings 1,048,576) when a config.json names none.
pub const default_context: i64 = 1048576;
pub const max_segments: u32 = 1;
/// The prompt chunk: the prefill's rows, at most one sparse block's granularity of 128 per position.
pub const prompt_rows: u32 = 4096;
/// `explain` words this family's open refusals too (the admission's failures and the serve boundary).
pub const explains_open = true;

/// The server's reading of the room left after the weights; null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };

/// TP1 only: no `rank` field, so --tp refuses by name (`refusals.family`); no vision field yet, --vision
/// refuses too (the tower's tensors sit in the checkpoint and are stage-3's placement work).
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 8, lanes_fixed: bool = false, room: ?Room = null, layer_cap: usize = 0 };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

/// What the native server drives; the fields the server's startup path reads after a family opens.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    learn_bytes: usize = 0, // no learner: the streams keep all the room
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    target: *anyopaque,
};

/// The registry's entry: a declared comptime TYPE (the engines' openWith surface, aliased to this
/// module's declarations — glm's pattern), not a struct value: `openWith(F)` resolves F.max_segments,
/// F.Options, F.open through it, and the capabilities table reads F.model_type/F.formats.
const me = @This();
pub const native = struct {
    pub const model_type = me.model_type;
    pub const formats = me.formats;
    pub const default_context = me.default_context;
    pub const max_segments = me.max_segments;
    pub const prompt_rows = me.prompt_rows;
    pub const explains_open = me.explains_open;
    pub const Options = me.Options;
    pub const Loaded = me.Loaded;
    pub const open = me.open;
    pub const explain = me.explain;
    pub const weightShare = me.weightShare; // the server's admission reads this rank's split share
};

/// Slots for `context` plus one whole block's reserve, a multiple of the 128-token key block.
pub const block_tokens: usize = 128; // the MSA key block (sparse_block_size): the pool's granularity

pub fn capacity(context: usize) usize {
    return std.mem.alignForward(usize, context + block_tokens, block_tokens);
}

/// The server's admission reads this rank's checkpoint share (glm's `weightShare` shape): the
/// receipted placement's exact plan bytes when the env names this rank, the whole pack without
/// the env (TP1 owns everything).
pub fn weightShare(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64) u64 {
    var ad = admitDir(a, io, dir) catch return total;
    defer ad.deinit();
    const p = rank_plan.place() catch return total; // no env: TP1, the whole pack
    return @min(rank_plan.bytesOf(&ad.load, p.world, p.rank), total);
}

/// The admission's result: the parsed shapes, the trellis headers' discovered widths and the whole weights plan.
pub const Admitted = struct {
    c: config.Config,
    groups: exl3.Plan,
    load: weights.Load,

    pub fn deinit(a: *Admitted) void {
        a.load.deinit();
        a.groups.deinit();
    }

    /// The checkpoint's weights bytes by the plan's own count (trellis groups + passthrough rows).
    pub fn bytes(a: *const Admitted) u64 {
        return a.load.bytes();
    }
};

/// `admit` over a checkpoint folder: the config parse plus the admission that reads the checkpoint itself.
pub fn admitDir(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Admitted {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    const c = try config.parse(gpa, text, null);
    var groups = try exl3.discover(gpa, io, dir, &c);
    errdefer groups.deinit();
    var load = try weights.plan(gpa, io, dir, &c, &groups);
    errdefer load.deinit();
    return .{ .c = c, .groups = groups, .load = load };
}

/// The admission the engines' config path runs (config text only): `qc_json` carries a standalone
/// quantization_config.json when given.
pub fn admit(gpa: std.mem.Allocator, config_json: []const u8, qc_json: ?[]const u8) !config.Config {
    return config.parse(gpa, config_json, qc_json);
}

/// The serve boundary this build stops at, said exactly: admission passed, the family's forward
/// is built and graded on the host, the device kernels are built and graded on the GPU — the
/// full serving engine (weights upload to the device buffers, the per-layer launch graph, the
/// lanes attach) is the remaining stage.
pub const serve_boundary = "minimax_m3_vl passed CUDA admission (every module's EXL3 widths discovered " ++
    "from its trellis headers, the whole weights plan and the served shapes checked), and the family's " ++
    "forward is graded on the host against an independent f64 reference — and this build's device kernels " ++
    "are graded on the GPU (the GB10 device parity receipts: tf_m3_scores/topk/decode/decode_combine and " ++
    "tf_m3_swigluoai) and its TP2 weight plumbing is receipted over the real pack on two GB10 nodes " ++
    "(the two ranks' checkpoint bytes exactly tile the plan's measured 166928433556) — and this build " ++
    "attaches the launch graph: graph.Engine runs the per-layer device forward per row (embedding → " ++
    "q/k/v through the mainline EXL3 gemms with per-head qk norm + partial rope 1e7/64/0.5, the indexer's " ++
    "side plane, the block-sparse selection + attention, dense MLP layers 0-2, the sigmoid+bias router " ++
    "top-4-of-128 with the shared experts through tf_m3_swigluoai, final norm, lm_head) with serve.zig's " ++
    "own published normRow/ropeRow/route/f16r as the host glue, and open() returns the Loaded handle over " ++
    "it — the fixture smoke (minimax-m3-smoke) and the token smoke (minimax-m3-token, the Stage-5 rank " ++
    "recipe: rank 0 layers 0-29 + embed + final norm + lm_head, rank 1 layers 30-59, NCCL handoffs) grade " ++
    "it — but this stage still refuses loudly rather than fake anything it does not have: draft rows and " ++
    "drafted verify windows refuse (error.LanesNotAttached: no draft head — the draft/spec-decode surface " ++
    "is unimplemented), a sampled draw spec that names more than temperature and grammar masks refuse (error.SamplingNotAttached: the keyed " ++
    "sampler is not wired for this family's lanes; temperature alone serves as fixed greedy), a second stream refuses while one is live " ++
    "(error.LanesNotAttached: one stream at a time), tree keeps refuse (prefix rollbacks only), TP2 booted " ++
    "through open() now ATTACHES since Stage-11 — the TF_MINIMAXM3_RANK/TF_MINIMAXM3_WORLD env server " ++
    "boot serves when the place is the receipted world-2 split: rank 0 returns the Loaded the HTTP " ++
    "server drives on this lane surface, rank 1 mirrors rank 0's rows through the per-row control wire " ++
    "and never serves (glm's rank pattern, additively) — a place outside the receipted split still " ++
    "refuses (error.TpServingNotAttached), a world-2 boot without NCCL_COMM_ID refuses (error.NoCommId) — and this " ++
    "build's SERVING lane surface (the Loaded handle's vtable) serves a stream through prefill/first/" ++
    "queue/read/verify/keep/release with TP1 through open() and the split-pack TP2 receipt through " ++
    "openTP2 (tf-cuda-test minimax-m3-lane-tp2 <rank> <host:port> <pack>, rank 1 first over the NCCL id, " ++
    "the Stage-9 receipts) — and contexts wider than the smoke's 4096-key pool refuse (error.ContextTooWide)";

/// The serve boundary this build's refusals draw (see serve_boundary): the lane surface serves
/// one greedy stream; drafts, sampled draws, second streams, tree keeps and the TP2 server boot
/// stay loud refusals, each with its own words.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.ServeNotAttached => serve_boundary,
        error.LanesNotAttached => "minimax_m3_vl's lane surface serves one greedy stream through the attached graph (prefill/first/queue/read/verify/keep/release — the boundary tests grade a short prompt token-in token-out); the calls that refuse here are the ones this stage does not have: draft rows and drafted verify windows (no draft head — the draft/spec-decode surface is unimplemented), a second stream while one is live, a verify batch wider than one window, and tree keeps",
        error.SamplingNotAttached => "minimax_m3_vl's lane surface serves greedy draws only in this stage: the server takes temperature as fixed greedy (any value or absent); a Sampling spec that names more than temperature (top_k/top_p/min_p/seed) or a grammar mask refuses here rather than serve an ungraded draw — the keyed sampler is a later pass",
        error.TpServingNotAttached => "minimax_m3_vl's TF_MINIMAXM3_RANK/TF_MINIMAXM3_WORLD env server boot attaches as the receipted world-2 split (rank 0 serves the lane surface, rank 1 mirrors rank 0's rows; NCCL_COMM_ID names rank 0's id host:port). What still refuses: an env place outside that split (one env var alone, a non-numeric rank/world, a rank at or past its world, a world other than 2), while TP2 serving receipts live at `tf-cuda-test minimax-m3-token <rank> <host:port> <pack>` (the Stage-6/7 token receipts) and openTP2's lane surface — `tf-cuda-test minimax-m3-lane-tp2 <rank> <host:port> <pack>` (the Stage-9 receipts: the same lane vtable answering over the split pack, rank 1 first)",
        error.NoCommId => "minimax_m3_vl's world-2 server boot meets its rank over NCCL: set NCCL_COMM_ID to rank 0's address:port on every rank (the receipted dance serves rank 0's fresh UniqueId there; rank 1 connects first)",
        error.ContextTooWide => "the launch graph's decode pool this stage is 4096 keys: give a context at or under it (the wide paged prompt path is a later stage)",
        error.LayerCapTP2 => "the launch graph's TP1-only layer cap refuses under TP2: each rank owns its exact half (the cap serves the Stage-8 grade's reduced real-pack stack)",
        error.LayerCapTooWide => "the layer cap exceeds the rank's owned layers",
        error.CapturedKernels => "TENSORFOLD_CUDA_KERNELS points at a captured set: minimax_m3_vl serves with its own glued kernels only; unset it",
        error.NotMiniMaxM3VL => "this checkpoint's shapes are not MiniMax-M3.1-Flash's (or its flags are): the family reads the M3.1 VL MoE pack",
        error.MissingTensor, error.UnexpectedTensor => "the checkpoint misses a tensor the family reads, or one has a different dtype or shape: this pack is not the one admission checked",
        error.MissingTrellis, error.SurplusTrellis => "an EXL3 trellis group needed by the plan is missing, or one is admitted that no module reads: the checkpoint is incomplete or from another pack",
        error.ForeignTrellis, error.NotMul1, error.HeadBits => "a quantized tensor falls outside this pack's EXL3 contract (mul1 codebook, the head's 8 halves): not Mia-AiLab's M3.1-Flash line",
        else => null,
    };
}

/// The attached graph's own state in the Loaded handle: the engine plus THE one greedy stream's
/// lane state. The lane surface serves one stream at a time (Loaded.facts.max_streams), greedy
/// draws only (a `Sampling` spec refuses), and everything it cannot do honestly it refuses by
/// name — the refusals stay part of the surface (drafts, trees, second streams), not decoration.
const Attached = struct {
    e: *graph.Engine,

    /// The stream the lane surface is serving (null: none). A second stream's calls refuse.
    live: ?*lanes.Stream = null,
    /// World 2's NCCL hold (the env server boot): the library + communicator the engine's
    /// collectives ride, freed on the surface's close (the server's clean stop).
    hold: ?*NcclHold = null,
    /// Drawn tokens waiting for their `read`: the handle -> token slab (eager greedy draws —
    /// the engine produces the logits row synchronously, so the draw is on the host already).
    pending: [pending_cap]Draw = undefined,
    pending_n: usize = 0,
    next_handle: u64 = 0,

    /// How many drawn tokens may sit unread at once (the one-token round's working set; a
    /// queue burst wider than this refuses rather than silently forgetting a draw).
    const pending_cap: usize = 8;
    const Draw = struct { handle: u64 = 0, pos: u64 = 0, tok: u32 = 0 };

    fn live_ok(a: *Attached, s: *lanes.Stream) !void {
        if (a.live != s) return error.LanesNotAttached;
    }
    /// The greedy draw the fork's contract makes from the logits row: first-max argmax. A
    /// non-finite lane in the row means the row is not servable: refuse.
    fn drawGreedy(a: *Attached, pos: u64) !u64 {
        if (a.pending_n >= pending_cap) return error.LaneDrawsFull;
        var bad = false;
        for (a.e.logits) |v| {
            if (std.math.isNan(v) or std.math.isInf(v)) bad = true;
        }
        if (bad) return error.RowNotFinite;
        var best: usize = 0;
        var bv = -std.math.inf(f32);
        for (a.e.logits, 0..) |v, i| {
            if (v > bv) {
                bv = v;
                best = i;
            }
        }
        a.next_handle += 1;
        a.pending[a.pending_n] = .{ .handle = a.next_handle, .pos = pos, .tok = @intCast(best) };
        a.pending_n += 1;
        return a.next_handle;
    }
    fn pendingOf(a: *Attached, handle: u64) !u32 {
        for (a.pending[0..a.pending_n]) |d| {
            if (d.handle == handle) return d.tok;
        }
        return error.LaneHandleUnknown;
    }
    fn pendingTake(a: *Attached, handle: u64) !u32 {
        var tok: u32 = 0;
        for (a.pending[0..a.pending_n], 0..) |d, i| {
            if (d.handle == handle) {
                tok = d.tok;
                a.pending[i] = a.pending[a.pending_n - 1];
                a.pending_n -= 1;
                return tok;
            }
        }
        return error.LaneHandleUnknown;
    }
};

// ------------------------------------------------------------ the lane rows (the serving half)
//
// The contract this surface serves (each row's own words in the error paths):
//   prefill: the stream's whole prompt through the attached graph, rows keyed at the engine's
//            own cache length (a stream prefills once per cache life; a reuse restore never
//            happened on this surface, so `s.cached` must be 0).
//   first:   the prompt's last row's greedy draw; the caller's `position` must be the engine's
//            cache length (the row the drawn token occupies when it is fed back).
//   queue:   feed one token (a host value or this surface's own handle) and draw the greedy
//            continuation; the fed row keys at the engine's own next cache length — the host's
//            window `position` is advisory (it repeats across plain rounds, glm's pattern).
//   read:    a drawn token's value (the draw is already on the host; nothing waits).
//   verify:  ONE one-row window: feed the pending token, draw the greedy continuation. Drafted
//            windows (held rows, tree tokens), grammar masks and multi-window batches refuse —
//            the draft/spec-decode surface stays unimplemented for this family.
//   keep:    ONE window whose path is the kept prefix 0..kept-1: the engine's cache length rolls
//            back (stale cache rows past the kept length are never read — every row's forward
//            keys at the engine's own length and rewrites its row first). Tree paths refuse.
//   release: the stream leaves the rounds: the engine's cache length resets to 0 and the draws
//            drop. The pool's rows are positional and rewritten per row, so nothing leaks.
// Draft rows (`draft`), sampled draws (a non-null `Sampling`), and every second-stream call
// refuse by name below — honestly, with the exact reason this build does not do them.

/// The lane surface's row under world 2: rank 0 leads (control word + row), rank 1 (the
/// harness's receipted lane-tp2 surface) mirrors through the control wire — its fed value
/// stays meaningless. World 1 keeps the plain `put`.
fn surfacePut(a: *Attached, tok: u32) !void {
    if (a.e.world == 2 and a.e.rank == 0) return a.e.leaderRow(tok);
    if (a.e.world == 2) return a.e.mirrorStep();
    return a.e.put(tok);
}

fn laneSelf(ptr: *anyopaque) *Attached {
    return @ptrCast(@alignCast(ptr));
}

fn laneRequireGreedy(s: *lanes.Stream) !void {
    if (s.sampling != null) return error.SamplingNotAttached;
    if (s.drafts) return error.LanesNotAttached;
}

/// The stream's whole prompt through the graph, one row at a time (the engine keys every row
/// at its own cache length; a resumed/edited prompt whose lengths disagree with the graph's
/// would silently mis-rope — refused rather than served).
fn laneServePrefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
    const a = laneSelf(ptr);
    if (a.live != null and a.live != s) return error.LanesNotAttached;
    if (a.e.len != 0) return error.StreamAlreadyPrefilled; // release() first: one prompt per cache life
    try laneRequireGreedy(s);
    if (s.prompt_len == 0) return error.EmptyPrompt;
    if (s.cached != 0) return error.PromptCacheMismatch; // this surface never restored a prefix
    a.live = s;
    const prompt = s.prompt();
    if (prompt.len != s.prompt_len) return error.PromptCacheMismatch;
    if (a.e.len + prompt.len > graph.keys_cap)
        return error.ContextTooWide;
    for (prompt) |tok| try surfacePut(a, tok);
    s.fill_at.store(@intCast(s.prompt_len), .monotonic);
}

fn laneServeFirst(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
    const a = laneSelf(ptr);
    try a.live_ok(s);
    if (a.e.len == 0) return error.EmptyPrompt;
    if (position != a.e.len) return error.RowKeying; // the draw keys at the next row's position
    return a.drawGreedy(position);
}

fn laneServeQueue(ptr: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, position: u64) anyerror!u64 {
    _ = position; // the host's window positions are advisory and repeat across plain rounds (glm's
    // pattern); the rows are positional and sequential, so this surface keys its own cache length
    const a = laneSelf(ptr);
    try a.live_ok(s);
    const tok: u32 = switch (feed) {
        .value => |v| v,
        .handle => |h| try a.pendingOf(h), // this surface's own drawn token, still unread
    };
    if (a.e.len == 0) return error.EmptyPrompt; // the fed row keys after the prompt's rows
    if (a.e.len + 1 > graph.keys_cap) return error.ContextTooWide;
    try surfacePut(a, tok); // the fed row lands at the engine's next cache length
    return a.drawGreedy(a.e.len);
}

fn laneServeRead(ptr: *anyopaque, handle: u64) anyerror!u32 {
    return laneSelf(ptr).pendingTake(handle);
}

fn laneServeVerify(ptr: *anyopaque, windows: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
    const a = laneSelf(ptr);
    if (windows.len != 1) return error.LanesNotAttached; // one stream at a time: no batching
    const w = windows[0];
    if (w.held > 0 or w.tokens.len > 0) return error.LanesNotAttached; // no draft head: drafted verify refuses
    if (w.masks.len > 0) return error.SamplingNotAttached; // structured output masks refuse
    try a.live_ok(w.stream);
    // the host's advisory window positions repeat across plain rounds (glm's pattern); the
    // one-row undrafted window keys at this surface's own cache length, not the host's copy
    if (out.len < windows.len) return error.LanesNotAttached;
    const h = try laneServeQueue(ptr, w.stream, .{ .value = w.pending }, w.positions[0]);
    out[0].sampled[0] = try a.pendingTake(h);
}

fn laneServeKeep(ptr: *anyopaque, windows: []const lanes.backend.Window, paths: []const []const u32) anyerror!void {
    const a = laneSelf(ptr);
    if (windows.len != 1) return error.LanesNotAttached; // one stream at a time
    if (paths.len != 1) return error.LaneHandleUnknown;
    const path = paths[0];
    if (path.len == 0) return error.RowKeying; // an ambiguous rollback refuses
    for (path, 0..) |row, i| {
        if (row != i) return error.LanesNotAttached; // tree paths refuse: prefix only
    }
    if (path.len > a.e.len) return error.RowKeying;
    a.e.len = @intCast(path.len); // the cache rolls back; stale rows rewrite before any read
    a.e.leaderKeep(a.e.len) catch |err| {
        // world 2: the mirror must hear the rollback or it keys stale rows. A failed control
        // exchange here leaves the NCCL wire broken for every later row; surface it now.
        std.log.err("minimax_m3_vl: the keep's control exchange failed ({s}); the TP2 mirror will desync", .{@errorName(err)});
        return err;
    };
}

fn laneServeRelease(ptr: *anyopaque, s: *lanes.Stream) void {
    laneServeFree(ptr, s);
}

fn laneServeFree(ptr: *anyopaque, s: *lanes.Stream) void {
    const a = laneSelf(ptr);
    if (a.live == s) {
        a.live = null;
        a.pending_n = 0;
        a.e.len = 0; // the pool's rows are positional: the next prefill rewrites from row 0
        if (a.e.world == 2 and a.e.rank == 0) a.e.leaderReset() catch {
            // world 2: the mirror must hear the reset. The NCCL wire is broken if this fails
            // (every later exchange fails too); the release itself must stay non-throwing.
        };
        s.fill_at.store(0, .monotonic);
    }
}

// The window `positions` repeat across the host's plain rounds; `laneServeQueue` keys its own
// rows, so nothing compares against the host's copy here.

// The draft rows stay loud refusals: this family has no draft head, and the receipt words say
// exactly what refuses and what serves.
const lane_refuse_draft = struct {
    fn refuse(_: *anyopaque, _: []const lanes.backend.DraftRequest) anyerror!void {
        return error.LanesNotAttached;
    }
}.refuse;

const lane_vtable = lanes.backend.Backend.VTable{
    .prefill = laneServePrefill,
    .first = laneServeFirst,
    .queue = laneServeQueue,
    .read = laneServeRead,
    .verify = laneServeVerify,
    .keep = laneServeKeep,
    .draft = lane_refuse_draft,
    .release = laneServeRelease,
};

fn closeGraph(ptr: *anyopaque) void {
    const a: *Attached = @ptrCast(@alignCast(ptr));
    const g = a.e.gpa;
    a.e.deinit();
    g.destroy(a.e);
    if (a.hold) |h| { // the env boot's NCCL: the communicator aborts, the library closes
        _ = h.lib.api.ncclCommAbort(h.comm);
        h.lib.close();
        g.destroy(h);
    }
    g.destroy(a);
}

/// The env boot's NCCL lifetime: the library and communicator one surface close releases.
const NcclHold = struct { lib: cuda.nccl.Library, comm: cuda.nccl.Comm };

/// The TF_MINIMAXM3_RANK/WORLD env SERVER boot (Stage-11, glm's rank pattern mirrored over the
/// receipted rank-1-first id dance): rank 0 returns the Loaded the server drives on the ONE
/// greedy stream's lane surface; rank 1 never returns — it mirrors rank 0's rows through the
/// control wire until rank 0's tp_watch line closes (a clean stop ends it too) or NCCL breaks.
/// A place outside the receipted world-2 split refuses by exact name, before any device work.
fn openServerRank(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options) !Loaded {
    const p = try placeOfBoot(std.c.getenv("TF_MINIMAXM3_RANK"), std.c.getenv("TF_MINIMAXM3_WORLD"));
    const addr = std.mem.span(std.c.getenv("NCCL_COMM_ID") orelse return error.NoCommId);
    const cap = capacity(o.context);
    if (cap > graph.keys_cap) return error.ContextTooWide; // the smoke pool's window, as ever
    try cuda.tp_watch.start(io, p.rank, p.world); // the ranks' lines before NCCL: a dying rank stops every rank
    // NCCL's cuMem host buffers wore out GB10's driver pool under load (glm's receipted defaults)
    if (std.c.getenv("NCCL_RUNTIME_CONNECT") == null) _ = c_setenv("NCCL_RUNTIME_CONNECT", "0");
    if (std.c.getenv("NCCL_CUMEM_HOST_ENABLE") == null) _ = c_setenv("NCCL_CUMEM_HOST_ENABLE", "0");
    if (std.c.getenv("NCCL_CUMEM_ENABLE") == null) _ = c_setenv("NCCL_CUMEM_ENABLE", "0");
    std.log.info("minimax_m3_vl server rank {d} of world {d}: NCCL_COMM_ID {s} (the receipted rank-1-first id dance)", .{ p.rank, p.world, addr });
    const hold = try gpa.create(NcclHold);
    errdefer gpa.destroy(hold);
    hold.lib = try cuda.nccl.Library.open();
    errdefer hold.lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    const id_bytes = std.mem.asBytes(&id);
    if (p.rank == 0) {
        try hold.lib.check(hold.lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        serveCommId(io, addr, id_bytes) catch |err| {
            cuda.tp_watch.fail(@errorName(err)); // rank 1 stops with this rank
            return err;
        };
    } else {
        receiveCommId(io, addr, id_bytes) catch |err| {
            cuda.tp_watch.fail(@errorName(err));
            return err;
        };
    }
    try hold.lib.check(hold.lib.api.ncclCommInitRank(&hold.comm, @intCast(p.world), id, @intCast(p.rank)), "ncclCommInitRank");
    const ep = try gpa.create(graph.Engine);
    errdefer gpa.destroy(ep);
    ep.* = graph.Engine.build(gpa, io, ctx.d, dir, p.rank, p.world, .{ .lib = &hold.lib, .comm = hold.comm }) catch |err| {
        cuda.tp_watch.fail(@errorName(err));
        return err;
    };
    errdefer ep.deinit();
    if (p.rank != 0) {
        std.log.info("minimax_m3_vl server rank 1: mirroring rank 0's rows (its own cache length rides the control wire)", .{});
        while (true) ep.mirrorStep() catch |err| {
            std.log.err("minimax_m3_vl server rank 1: the row mirror stopped ({s})", .{@errorName(err)});
            cuda.tp_watch.fail(@errorName(err)); // rank 0 stops with this rank's error
            std.c._exit(1); // a stream may hold kernels waiting on a gone rank: a normal exit would join them
        };
    }
    const own = try gpa.create(Attached);
    errdefer gpa.destroy(own);
    own.* = .{ .e = ep, .live = null, .hold = hold, .pending = undefined, .pending_n = 0, .next_handle = 0 };
    return Loaded{
        .backend = .{ .ptr = own, .vtable = &lane_vtable },
        .facts = .{ .exact_width = 1, .drafts = 0, .max_streams = 1 },
        .rows = 1,
        .stream_bytes = 0,
        .free_streams = 1,
        .ctx = own,
        .deinit = closeGraph,
        .lone = null,
        .target = ep,
    };
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
fn c_setenv(name: [*:0]const u8, value: [*:0]const u8) c_int {
    return setenv(name, value, 0);
}

// The receipted id dance (the lane-tp2 harness's serveId/receiveId, family-owned here): rank 0
// serves the freshly minted UniqueId's bytes on NCCL_COMM_ID's exact host:port, rank 1 connects
// (retrying until rank 0's bind, up to one minute) and reads them. Caps bound both waits, so a
// rank whose peer never comes refuses instead of hanging the boot.

fn serveCommId(io: std.Io, addr_text: []const u8, bytes: []const u8) !void {
    const at = std.mem.lastIndexOfScalar(u8, addr_text, ':') orelse return error.BadAddress;
    const ip = try std.Io.net.IpAddress.parseIp4(addr_text[0..at], try std.fmt.parseInt(u16, addr_text[at + 1 ..], 10));
    var storage: std.Io.Threaded.PosixAddress = undefined;
    const len = std.Io.Threaded.addressToPosix(&ip, &storage);
    const rc = std.posix.system.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    if (std.posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    const fd: std.posix.socket_t = @intCast(rc);
    defer _ = std.posix.system.close(fd);
    const one: c_int = 1;
    try std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.REUSEADDR, std.mem.asBytes(&one));
    switch (std.posix.errno(std.posix.system.bind(fd, &storage.any, len))) {
        .SUCCESS => {},
        .ADDRINUSE => return error.AddressInUse,
        else => return error.BindFailed,
    }
    if (std.posix.errno(std.posix.system.listen(fd, 1)) != .SUCCESS) return error.ListenFailed;
    var waited: u64 = 0;
    while (true) {
        var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        const ready = std.posix.poll(&fds, 500) catch return error.AcceptFailed;
        if (ready > 0) break;
        waited += 500;
        if (waited > 70_000) return error.PeerTimeout;
    }
    var peer: std.Io.Threaded.PosixAddress = undefined;
    var peer_len: std.posix.socklen_t = @sizeOf(std.Io.Threaded.PosixAddress);
    const rc2 = std.posix.system.accept(fd, &peer.any, &peer_len);
    if (std.posix.errno(rc2) != .SUCCESS) return error.AcceptFailed;
    const cfd: std.posix.socket_t = @intCast(rc2);
    defer _ = std.posix.system.close(cfd);
    var off: usize = 0;
    while (off < bytes.len) {
        const n = std.posix.system.write(cfd, bytes.ptr + off, bytes.len - off);
        if (std.posix.errno(n) != .SUCCESS) return error.WriteFailed;
        off += @intCast(n);
    }
    _ = io;
}

fn receiveCommId(io: std.Io, addr_text: []const u8, bytes: []u8) !void {
    const at = std.mem.lastIndexOfScalar(u8, addr_text, ':') orelse return error.BadAddress;
    const ip = try std.Io.net.IpAddress.parseIp4(addr_text[0..at], try std.fmt.parseInt(u16, addr_text[at + 1 ..], 10));
    var fd: std.posix.socket_t = -1;
    defer {
        if (fd >= 0) _ = std.posix.system.close(fd);
    }
    var connect_ms: u64 = 0; // rank 1 boots first: retry the connect until rank 0 binds (the receipted ~6 s gap)
    while (fd < 0) {
        var storage: std.Io.Threaded.PosixAddress = undefined;
        const len = std.Io.Threaded.addressToPosix(&ip, &storage);
        const rc = std.posix.system.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
        if (std.posix.errno(rc) != .SUCCESS) return error.SocketFailed;
        const c: std.posix.socket_t = @intCast(rc);
        if (std.posix.errno(std.posix.system.connect(c, &storage.any, len)) == .SUCCESS) {
            fd = c;
        } else {
            _ = std.posix.system.close(c);
            if (connect_ms >= 70_000) return error.PeerTimeout;
            std.Io.sleep(io, .fromMilliseconds(200), .awake) catch {};
            connect_ms += 200;
        }
    }
    var off: usize = 0;
    var read_waited: u64 = 0;
    while (off < bytes.len) {
        var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        const ready = std.posix.poll(&fds, 500) catch return error.ReadFailed;
        if (ready == 0) {
            read_waited += 500;
            if (read_waited > 60_000) return error.PeerTimeout;
            continue;
        }
        const n = std.posix.system.read(fd, bytes.ptr + off, bytes.len - off);
        switch (std.posix.errno(n)) {
            .SUCCESS => {
                if (n == 0) return error.PeerClosed;
                off += @intCast(n);
            },
            .INTR => {},
            else => return error.ReadFailed,
        }
    }
}

/// The checkpoint, the shared kernels and every stream's caches — this stage: admission, then the
/// attached launch graph serving ONE greedy stream through the lane surface (TP1 through open();
/// TP2 through the harness command), with drafts/sampled draws/second streams/TP2-open refusing by name.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.CapturedKernels;
    if (!cuda.kernels.available) return error.ServeNotAttached; // a build without the glued kernels: the graph needs the device
    // the ranks: TP1 unchanged; the TF_MINIMAXM3_RANK/WORLD env SERVER boot now attaches here
    // (glm's pattern, additively: rank 0 returns the Loaded the server serves on its lane
    // surface, rank 1 mirrors rank 0's rows through the control wire and never serves). A
    // place outside the receipted world-2 split still refuses, by exact name — before any
    // device work.
    const env_rank = std.c.getenv("TF_MINIMAXM3_RANK");
    const env_world = std.c.getenv("TF_MINIMAXM3_WORLD");
    if (env_rank != null or env_world != null) {
        _ = try placeOfBoot(env_rank, env_world); // the exact-name refusals fire before anything device
        if (kernels != null) return error.CapturedKernels;
        if (!cuda.kernels.available) return error.ServeNotAttached;
        return openServerRank(gpa, io, ctx, dir, o);
    }
    if (kernels != null) return error.CapturedKernels;
    if (!cuda.kernels.available) return error.ServeNotAttached;
    return openLane(gpa, io, ctx, dir, o);
}

/// The env boot's place: both vars, numeric, inside the receipted world-2 split — or refuse
/// by exact name (the old honest refusal stays live for any place that is not the receipted
/// split; world 1 boots through open() WITHOUT the env).
fn placeOfBoot(rank_env: ?[*:0]const u8, world_env: ?[*:0]const u8) !rank_plan.Place {
    const r = rank_env orelse return error.TpServingNotAttached;
    const w = world_env orelse return error.TpServingNotAttached;
    const p = rank_plan.checked(.{
        .rank = std.fmt.parseInt(u32, std.mem.span(r), 10) catch return error.TpServingNotAttached,
        .world = std.fmt.parseInt(u32, std.mem.span(w), 10) catch return error.TpServingNotAttached,
    }) catch return error.TpServingNotAttached;
    if (p.world != rank_plan.max_world) return error.TpServingNotAttached;
    return p;
}

/// TP1's own admission + attached graph body (the pre-Stage-11 open() body, byte-equal).
fn openLane(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options) !Loaded {
    var a = try admitDir(gpa, io, dir);
    defer a.deinit();
    if (o.context < 1) return error.ServeNotAttached; // below one block of context there is no row to serve
    // the ranks: TP1 (rank 0 of world 1); the env server boot routes through openServerRank above
    const rank: u32 = 0;
    const world: u32 = 1;
    const cap = capacity(o.context);
    if (cap > graph.keys_cap) return error.ContextTooWide; // the smoke pool's window: a wider prefill path is a later stage
    const own = try gpa.create(Attached);
    errdefer gpa.destroy(own);
    const ep = try gpa.create(graph.Engine);
    errdefer gpa.destroy(ep);
    ep.* = try graph.Engine.build(gpa, io, ctx.d, dir, rank, world, .{ .layer_cap = o.layer_cap });
    own.* = .{ .e = ep, .live = null, .pending = undefined, .pending_n = 0, .next_handle = 0 };
    return Loaded{
        .backend = .{ .ptr = own, .vtable = &lane_vtable },
        .facts = .{ .exact_width = 1, .drafts = 0, .max_streams = 1 },
        .rows = 1,
        .stream_bytes = 0,
        .free_streams = 1,
        .ctx = own,
        .deinit = closeGraph,
        .lone = null,
        .target = ep,
    };
}

/// The TP2 lane-surface boot (Stage-9): the harness's rank plumbing — the caller builds the NCCL
/// library and the rank's communicator first (rank 1 first over the id, exactly the token-harness
/// recipe), and this entry wraps open()'s own admission + the SAME lane vtable over the rank's
/// half of the split pack. open() itself keeps refusing the TF_MINIMAXM3_RANK/WORLD env boot; the
/// rank boot here is explicit, receipted, and serves only through the lane surface like TP1 does.
/// Rank 0 argues the greedy draws (its `logits` row is the only real one); rank 1's surface keeps
/// the row lockstep (every `put` is a lockstep row; the fed token's value is rank 0's business).
pub const RankBoot = struct { lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, rank: u32, world: u32 };

pub fn openTP2(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, o: Options, boot: RankBoot) !Loaded {
    if (boot.world != 2 or boot.rank >= boot.world) return error.NoRank; // the receipted split: exactly two ranks
    if (!cuda.kernels.available) return error.ServeNotAttached; // a build without the glued kernels: the graph needs the device
    var a = try admitDir(gpa, io, dir);
    defer a.deinit();
    if (o.context < 1) return error.ServeNotAttached;
    if (o.layer_cap != 0) return error.LayerCapTP2; // under TP2 each rank owns its exact half
    const cap = capacity(o.context);
    if (cap > graph.keys_cap) return error.ContextTooWide; // the smoke pool's window: a wider prefill path is a later stage
    const own = try gpa.create(Attached);
    errdefer gpa.destroy(own);
    const ep = try gpa.create(graph.Engine);
    errdefer gpa.destroy(ep);
    ep.* = try graph.Engine.build(gpa, io, ctx.d, dir, boot.rank, boot.world, .{ .lib = boot.lib, .comm = boot.comm });
    own.* = .{ .e = ep, .live = null, .pending = undefined, .pending_n = 0, .next_handle = 0 };
    return Loaded{
        .backend = .{ .ptr = own, .vtable = &lane_vtable },
        .facts = .{ .exact_width = 1, .drafts = 0, .max_streams = 1 },
        .rows = 1,
        .stream_bytes = 0,
        .free_streams = 1,
        .ctx = own,
        .deinit = closeGraph,
        .lone = null,
        .target = ep,
    };
}

test "the family dispatches over the staged pack's model_type, formats and admission" {
    const gpa = std.testing.allocator;
    try std.testing.expectEqualStrings("minimax_m3_vl", model_type);
    try std.testing.expectEqual(@as(usize, 1), formats.len);
    try std.testing.expectEqualStrings("minimax_m3_vl", native.model_type); // the registry reads these two fields
    try std.testing.expectEqualStrings("exl3", native.formats[0]);
    try std.testing.expectEqual(@as(usize, registry_index), 7); // in native/cuda.zig's engine-dispatch registry
    const c = try admit(gpa, @embedFile("test_pack_config.json"), null);
    try std.testing.expect(c.exl3 and config.whole(c)); // 3.06 bpw, mul1, mixed widths, the 128-column vocab cut
}

/// The openWith registry's tuple position (native/cuda.zig's `registry`): kept asserted so both move together.
pub const registry_index = 7;

test "admission runs the whole checkpoint and open refuses exactly at the serve boundary" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try weights.checkpointFixture(gpa, io, tmp.dir, false);
    const path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(path);
    var a = try admitDir(gpa, io, path);
    defer a.deinit();
    try std.testing.expect(a.c.exl3); // the inline quantization_config named EXL3
    try std.testing.expectEqual(config.Codebook.mul1, a.c.qc_codebook);
    try std.testing.expectEqual(@as(usize, 27), a.groups.groups.len); // the discovered trellis groups
    try std.testing.expectEqual(@as(usize, 47), a.load.items.len); // the whole plan, in load order
    try std.testing.expect(a.bytes() > 0);
    // no serving without a context row (or without the glued kernels on a host build); a captured
    // kernel set also refuses — both paths stay before any device work, so the test is device-free
    try std.testing.expectError(error.ServeNotAttached, open(gpa, io, undefined, path, null, .{ .context = 0, .drafts = false }));
    try std.testing.expectError(error.CapturedKernels, open(gpa, io, undefined, path, "captured", .{ .context = 4096, .drafts = false }));
    const words = explain(null, error.ServeNotAttached).?;
    try std.testing.expectStringStartsWith(words, "minimax_m3_vl passed CUDA admission");
    try std.testing.expect(std.mem.indexOf(u8, words, "graded on the host") != null); // the host forward is real
    try std.testing.expect(std.mem.indexOf(u8, words, "graded on the GPU") != null); // the device kernels are too
    try std.testing.expect(std.mem.indexOf(u8, words, "tf_m3_swigluoai") != null); // the parity receipts name them
    try std.testing.expect(std.mem.indexOf(u8, words, "normRow/ropeRow/route/f16r") != null); // the glue is serve.zig's own fns
    try std.testing.expect(std.mem.indexOf(u8, words, "attaches the launch graph") != null); // open() carries it
    try std.testing.expect(std.mem.indexOf(u8, words, "minimax-m3-token") != null); // the token receipts' command
    try std.testing.expect(std.mem.indexOf(u8, words, "SERVING lane surface") != null); // Stage-8: the lanes serve
    try std.testing.expect(std.mem.indexOf(u8, words, "prefill/first/") != null); // the served rows are named
    try std.testing.expect(std.mem.indexOf(u8, words, "LanesNotAttached") != null); // drafts+second streams still refuse
    try std.testing.expect(std.mem.indexOf(u8, words, "SamplingNotAttached") != null); // sampled draws refuse
    try std.testing.expect(temperature_greedy); // the server serves temperature as fixed greedy
    try std.testing.expect(std.mem.indexOf(u8, words, "TpServingNotAttached") != null); // and the TP2 server boot
    try std.testing.expect(std.mem.indexOf(u8, words, "ContextTooWide") != null); // and the pool's window
    try std.testing.expect(explain(null, error.LanesNotAttached) != null);
    try std.testing.expect(explain(null, error.SamplingNotAttached) != null);
    try std.testing.expect(explain(null, error.TpServingNotAttached) != null);
    try std.testing.expect(explain(null, error.ContextTooWide) != null);
    try std.testing.expect(explain(null, error.LayerCapTP2) != null);
    try std.testing.expect(explain(null, error.LayerCapTooWide) != null);
    try std.testing.expect(explain(null, error.BadConfig) == null); // not this family's wording
}

test "the TP2 lane boot refuses a bad rank shape before admitting anything (device-free)" {
    const gpa = std.testing.allocator;
    // the NoRank shape check fires before admission or any device work: no CUDA context touched
    try std.testing.expectError(error.NoRank, openTP2(gpa, undefined, undefined, "nowhere", .{ .context = 128, .drafts = false }, .{ .lib = undefined, .comm = null, .rank = 2, .world = 2 }));
    try std.testing.expectError(error.NoRank, openTP2(gpa, undefined, undefined, "nowhere", .{ .context = 128, .drafts = false }, .{ .lib = undefined, .comm = null, .rank = 0, .world = 1 }));
    try std.testing.expect(explain(null, error.TpServingNotAttached) != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.TpServingNotAttached).?, "minimax-m3-lane-tp2") != null);
}

extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "the env server boot attaches only as the receipted world-2 split (device-free refusals)" {
    // placeOfBoot reads env only: nothing device, nothing io. The receipted places validate;
    // every place outside the split refuses TpServingNotAttached (the exact-name law holds).
    try std.testing.expectEqual(rank_plan.Place{ .rank = 0, .world = 2 }, try placeOfBoot("0", "2"));
    try std.testing.expectEqual(rank_plan.Place{ .rank = 1, .world = 2 }, try placeOfBoot("1", "2"));
    try std.testing.expectError(error.TpServingNotAttached, placeOfBoot(null, "2")); // one env var alone
    try std.testing.expectError(error.TpServingNotAttached, placeOfBoot("0", null));
    try std.testing.expectError(error.TpServingNotAttached, placeOfBoot("x", "2")); // non-numeric
    try std.testing.expectError(error.TpServingNotAttached, placeOfBoot("2", "2")); // a rank at its world
    try std.testing.expectError(error.TpServingNotAttached, placeOfBoot("0", "3")); // not the receipted split
    try std.testing.expectError(error.TpServingNotAttached, placeOfBoot("0", "1")); // world 1 boots WITHOUT the env
    try std.testing.expect(explain(null, error.NoCommId) != null); // world 2 but no NCCL id: its own exact name
}

test {
    _ = @import("serve.zig"); // the family's forward and its f64-reference grade run with the suite
    _ = @import("config.zig");
    _ = @import("weights.zig");
    _ = @import("exl3.zig");
    _ = @import("sparse.zig");
    _ = @import("cuda_sparse.zig");
    _ = @import("cuda_serve.zig");
    _ = @import("shard.zig"); // the TP2 rank split's tests ride the same module
    _ = @import("graph.zig"); // the launch graph and its helpers compile with the suite
}
