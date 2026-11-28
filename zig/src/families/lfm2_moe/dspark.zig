//! LiquidAI's DSpark drafter on LFM2's taps: a stream's context keys, a block of rows, a Markov chain of drafts.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const c = @import("config.zig");
const dc = @import("dspark_config.zig");
const dw = @import("dspark_weights.zig");
const block = @import("dspark_block.zig");
const st = @import("state.zig");
const snapshot = @import("snapshot.zig");
const Model = @import("model.zig").Model;
const backend = @import("backend.zig");
const stage = @import("dspark_stage.zig");
const admission = @import("dspark_admission.zig");
const scratch = @import("dspark_scratch.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Block rows one hold pass takes (the decode twin's rows), and context rows one absorb pass takes.
pub const pass_rows = 32;
pub const absorb_rows = 2048;
/// Owned prompt and verified rows, deferred until a hold, snapshot or stage overflow.
pub const stage_rows = absorb_rows;
/// Staged rows a hold can gather to share fc reads, in calls of at most 32 rows.
pub const gather_rows = 128;
/// Context calibration block counts; clean holds cover every legal count up to this maximum.
const hold_grid = [_]u32{ 1, 2, 4, 8, 16 };
const context_row_levels = std.math.log2_int(usize, stage_rows) + 1;
const kw = c.kv_heads * c.head_dim;

/// `bits`: 0 runs the bf16 matrices as stored, 8 quantizes them at load.
pub const Options = struct { capacity: usize, bits: u8 = 0 };

/// One stream's context keys and values a layer ([kv head][cap][head dim], as the target's attention reads them).
pub const Lane = struct {
    mem: mtl.Buffer, // each layer's keys then values, then the staged rows (shared storage)
    gpu_base: u64, // `mem`'s base address for this allocation, never part of a snapshot
    cap: usize, // positions each holds: the context and one block past it
    stage_at: usize, // the staged rows' byte offset in `mem`
    end: u64 = 0, // context rows absorbed: positions [0, end)
    staged: u32 = 0, // tapped rows [end, end + staged), not absorbed yet
    held: [16]u32 = @splat(0),
    held_len: u32 = 0,
    conf: [16]f32 = @splat(0), // each held draft's chance given the drafts before it landed (the confidence head)
    conf_len: u32 = 0, // the drafts `conf` covers: the last settled hold's depth
    slot: ?u32 = null, // its block's place in the last signalled hold, whose picks still hold its drafts

    /// Layer `layer`'s keys, or its values when `values`, in `mem`.
    pub fn kv(l: *const Lane, layer: usize, values: bool) block.Ref {
        return .{ .buf = l.mem, .off = (2 * layer + @intFromBool(values)) * l.cap * kw * 2 };
    }
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    target: *Model,
    config: dc.Config,
    w: dw.Weights,
    options: Options,
    s: scratch.Scratch,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    lanes: std.AutoHashMapUnmanaged(*Stream, *Lane) = .empty,
    /// One stream's clean CPU-encode + GPU ms at full depth, and that over the block (the depth rule's cost a draft).
    hold_ms: f64 = 0,
    step_ms: f64 = 0,
    holds: [pass_rows]lanes.hold_price.HoldCost = undefined, // block * floor(pass_rows / block) bounds the clean grid
    holds_len: usize = 0,
    contexts: [hold_grid.len * context_row_levels]lanes.hold_price.ContextCost = undefined,
    contexts_len: usize = 0,
    target_single: ?*const lanes.table.Table = null,
    target_shared: ?*const lanes.table.Table = null,
    head_history: admission.History = .{},
    hold_spent_ms: f64 = 0, // GPU ms of every buffer waited on or settled; drained at round boundaries, never waits to read it
    encode_cpu_ms: f64 = 0, // native precommit CPU spans, drained with hold_spent_ms without waiting
    event: mtl.Event = undefined,
    signalled: u64 = 0,
    pending: ?mtl.CommandBuffer = null,
    pass: [pass_rows]*Lane = undefined, // the signalled hold's lanes and depths, in its blocks' order
    depths: [pass_rows]u32 = undefined,
    pass_len: usize = 0,
    pass_context_rows: u32 = 0, // largest actual staged debt, captured before the signalled hold clears it
    pass_encode_cpu_ms: f64 = 0, // pending pass's CPU for EMA only; already charged to encode_cpu_ms at commit

    pub fn init(gpa: std.mem.Allocator, io: std.Io, target: *Model, dir: []const u8, options: Options) !*Drafter {
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .io = io, .target = target, .config = try dc.Config.read(gpa, io, dir), .w = undefined, .options = options, .s = undefined };
        d.event = try target.device.event();
        errdefer d.event.deinit();
        if (options.bits == 0) try target.kernels.loadBf16(target.device);
        d.w = try dw.load(gpa, target.device, io, dir, &d.config, options.bits);
        errdefer d.w.deinit(gpa);
        try d.prepare();
        return d;
    }

    fn prepare(d: *Drafter) !void {
        errdefer {
            for (d.owned.items) |b| b.deinit();
            d.owned.deinit(d.gpa);
        }
        try scratch.prepare(d);
        try d.warm();
    }

    pub fn deinit(d: *Drafter) void {
        d.settle() catch {};
        d.event.deinit();
        var it = d.lanes.valueIterator();
        while (it.next()) |l| d.dropLane(l.*);
        d.lanes.deinit(d.gpa);
        for (d.owned.items) |b| b.deinit();
        d.owned.deinit(d.gpa);
        d.w.deinit(d.gpa);
        d.gpa.destroy(d);
    }

    /// Query rows the attention partials hold at once: a whole hold pass's, so its blocks attend side by side.
    pub fn partRows(d: *const Drafter) usize {
        return @max(pass_rows, d.config.block);
    }

    /// The drafter's buffers that every round reads (weights and scratch), for the server's residency set.
    pub fn buffers(d: *const Drafter, gpa: std.mem.Allocator) ![]mtl.Buffer {
        return std.mem.concat(gpa, mtl.Buffer, &.{ d.w.buffers(), d.owned.items });
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .ptr = d, .hold_timing = .{ .ptr = d, .take = block.takeHoldMs }, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .enroll = enrollFn, .observe_head = observeHeadFn, .open = openFn, .absorb = stage.absorb, .hold = holdFn, .held = heldFn, .release = releaseFn, .chances = chancesFn } };
    }

    /// Borrow stable bare-target costs, never head-inclusive EMA; unbound diagnostics keep their tracked route.
    pub fn useTargetCosts(d: *Drafter, single: *const lanes.table.Table, shared_costs: *const lanes.table.Table) void {
        d.target_single = single;
        d.target_shared = shared_costs;
    }

    fn sampled(s: *const Stream) bool {
        return if (s.sampling) |settings| !(settings.temperature <= 0) else false;
    }

    fn enrollFn(ptr: *anyopaque, s: *Stream, cohort: dr.Enrollment) bool {
        const d = cast(ptr);
        const recent = d.head_history.recent(sampled(s));
        if (d.target_single == null and d.target_shared == null) {
            s.head_prior = recent.rates[0..d.config.block];
            return true; // Direct diagnostics have no serving calibration bound.
        }
        const depth: u32 = @intCast(@min(d.config.block, @max(0, s.draftRoom() - 1)));
        var rates = recent.rates;
        var heads: usize = 1;
        var debt = s.prompt_len;
        // Eligible peers fund any subset, priced conservatively at their weakest conditionals and deepest debt.
        var peers = d.lanes.iterator();
        while (peers.next()) |entry| {
            const peer = entry.key_ptr.*;
            if (peer == s or !peer.headDrafts() or peer.finished or peer.paused or peer.draftRoom() <= depth) continue;
            heads += 1;
            debt = @max(debt, entry.value_ptr.*.staged);
            const other = d.head_history.recent(sampled(peer));
            for (&rates, other.rates) |*rate, p| rate.* = @min(rate.*, p);
        }
        const tracked = recent.enroll(.{
            .target = if (cohort.streams == 1) d.target_single else d.target_shared,
            .holds = d.holds[0..d.holds_len],
            .contexts = d.contexts[0..d.contexts_len],
            .streams = cohort.streams,
            .heads = @min(heads, cohort.heads),
            .target_rows = if (cohort.streams == 1) st.window_rows else st.batch_rows,
            .pass_rows = pass_rows,
            .block_rows = d.config.block,
            .depth = depth,
            .context_rows = debt,
            .context_chunk = stage_rows,
        }, rates[0..depth]);
        s.head_prior = if (tracked) recent.rates[0..d.config.block] else &.{};
        return tracked;
    }

    fn observeHeadFn(ptr: *anyopaque, s: *Stream, proposed: u32, accepted: u32) void {
        // The engine calls this only for an actual neural chain, excluding copies/grafts/grammar.
        if (!s.headDrafts()) return;
        cast(ptr).head_history.recent(sampled(s)).observe(proposed, accepted);
    }

    /// Whether holds compute the confidence head: the verify rows follow its chances, else the core's depth rule.
    pub fn confident(d: *const Drafter) bool {
        return d.w.confidence != null;
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(ptr: *anyopaque) []const u32 {
        const d = cast(ptr);
        return d.config.taps[0..d.config.tap_count];
    }

    /// Acceptance a depth until a stream has its own: the card's 5.16 tokens a step at block 9 is 0.84 a depth.
    const prior = admission.prior;

    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        var f: dr.Facts = .{ .depth = d.config.block, .step_ms = d.step_ms, .prior = prior[0..d.config.block], .batched = true, .stream_bytes = d.laneBytes(d.options.capacity), .copy_first = true };
        f.chances = d.confident();
        f.hold_costs = d.holds[0..d.holds_len];
        f.hold_context_costs = d.contexts[0..d.contexts_len];
        f.hold_rows = pass_rows;
        f.hold_block_rows = d.config.block;
        return f;
    }

    /// A held chain's chances: the confidence head's running product, or for a fixed window, ones that fall by depth.
    fn chancesFn(ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const d = cast(ptr);
        const l = d.lanes.get(s) orelse return false;
        if (!d.confident()) return false;
        if (d.pending != null and l.slot != null and l.conf_len > 0) {
            // the hold still runs: the lane's last settled confidences price it, so the host does not wait on the GPU
            if (out.len > d.depths[l.slot.?]) return error.DraftHeldFewer;
            var reach: f64 = 1;
            for (out, 0..) |*o, j| {
                reach *= l.conf[@min(j, l.conf_len - 1)];
                o.* = reach;
            }
            return true;
        }
        try d.settle();
        if (out.len > l.held_len) return error.DraftHeldFewer;
        var reach: f64 = 1;
        for (out, l.conf[0..out.len]) |*o, p| {
            reach *= p;
            o.* = reach;
        }
        return true;
    }

    /// A stream's lane at `capacity` positions: its context keys and values a layer, and the tapped rows it owns.
    pub fn laneBytes(d: *const Drafter, capacity: usize) usize {
        const cap = capacity + d.config.block;
        return @intCast(d.keptBytes(cap) + d.stageBytes(cap));
    }

    /// Bytes of the tapped rows a lane of `cap` positions owns: stage_rows, never more rows than positions.
    fn stageBytes(d: *const Drafter, cap: usize) usize {
        return @min(stage_rows, cap) * d.tapBytes();
    }

    /// A tapped row's bytes: every tap's hidden row, bf16.
    pub fn tapBytes(d: *const Drafter) usize {
        return d.config.tap_count * c.hidden * 2;
    }

    /// The lane's staged rows as an absorb piece (null: none staged).
    pub fn stagedPiece(l: *Lane) ?block.Piece {
        if (l.staged == 0) return null;
        return .{ .lane = l, .src = .{ .buf = l.mem, .off = l.stage_at }, .rows = l.staged, .p0 = l.end };
    }

    /// The lane's staged rows absorbed now, in a command buffer of their own.
    pub fn flush(d: *Drafter, s: *Stream, l: *Lane) !void {
        defer s.hold_context_rows = l.staged;
        const pc = stagedPiece(l) orelse return;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try block.absorbNow(d, &.{pc});
        l.end += l.staged;
        l.staged = 0;
    }

    fn openFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.lanes.contains(s)) return error.DraftStreamOpen;
        const want = s.prompt_len + s.max_new + 2 * d.config.block;
        const l = try d.newLane(@min(d.options.capacity, (want + 63) / 64 * 64));
        errdefer d.dropLane(l);
        try d.lanes.put(d.gpa, s, l);
        s.hold_context_rows = 0;
    }

    /// A lane in one buffer of laneBytes(capacity).
    pub fn newLane(d: *Drafter, capacity: usize) !*Lane {
        const l = try d.gpa.create(Lane);
        errdefer d.gpa.destroy(l);
        const at: usize = @intCast(d.keptBytes(capacity + d.config.block));
        const mem = try d.target.device.buffer(d.laneBytes(capacity), shared);
        l.* = .{ .mem = mem, .gpu_base = mem.gpuAddress(), .cap = capacity + d.config.block, .stage_at = at };
        return l;
    }

    pub fn dropLane(d: *Drafter, l: *Lane) void {
        l.mem.deinit();
        d.gpa.destroy(l);
    }

    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const d = cast(ptr);
        d.unpass() catch {};
        if (d.lanes.fetchRemove(s)) |e| d.dropLane(e.value);
        s.hold_context_rows = 0;
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(ptr);
        try d.settle();
        for (streams, out) |s, o| {
            const l = d.lanes.get(s) orelse return error.DraftStreamUnknown;
            if (o.len > l.held_len) return error.DraftHeldFewer;
            @memcpy(o, l.held[0..o.len]);
        }
    }

    /// Bytes a lane's context at `at` takes beside a kept prompt state.
    pub fn keptBytes(d: *const Drafter, at: usize) u64 {
        return 2 * @as(u64, d.config.layers) * at * kw * 2;
    }

    /// Stream `s`'s context copied into `snap` when its lane stands at the snapshot's position (a prompt chunk's end).
    pub fn keepLane(d: *Drafter, s: *Stream, snap: *snapshot.Snap) !void {
        const l = d.lanes.get(s) orelse return;
        try d.settle();
        // A final anchor is not absorbed yet: declining must leave its stage slot intact.
        if (l.end + l.staged != snap.at) return;
        try d.flush(s, l);
        const bytes = try d.gpa.alloc(u8, @intCast(d.keptBytes(snap.at)));
        copyLane(d, l, snap.at, bytes, true);
        snap.draft = bytes;
    }

    /// The lane takes the context kept with the state its stream's prompt pass resumed at `start`.
    pub fn resumeLane(d: *Drafter, s: *Stream, l: *Lane, start: u64) !void {
        const saved = s.reuse.saved orelse return error.DraftContextGap;
        const snap: *const snapshot.Snap = @ptrCast(@alignCast(saved));
        const bytes = snap.draft orelse return error.DraftContextGap;
        if (s.cached != start or snap.at != start or l.end != 0 or l.staged != 0 or start + d.config.block > l.cap) return error.DraftContextGap;
        copyLane(d, l, snap.at, bytes, false);
        l.end = start;
        s.hold_context_rows = 0;
    }

    /// Rows [0, at) of every layer's keys and values and `bytes` (`out`: the lane's into bytes), kv head by kv head.
    fn copyLane(d: *const Drafter, l: *const Lane, at: usize, bytes: []u8, out: bool) void {
        const n = at * c.head_dim * 2;
        var off: usize = 0;
        for (0..d.config.layers) |i| for ([_]block.Ref{ l.kv(i, false), l.kv(i, true) }) |r| for (0..c.kv_heads) |h| {
            const live = (r.buf.contents() + r.off + h * l.cap * c.head_dim * 2)[0..n];
            if (out) @memcpy(bytes[off..][0..n], live) else @memcpy(live, bytes[off..][0..n]);
            off += n;
        };
    }

    /// Each stream's chain of `depth` drafts after `pending` at `position`, its block's rows in a pass of pass_rows.
    fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        try d.unpass();
        var at: usize = 0;
        while (at < items.len) {
            // Every retained prefix still runs the configured bidirectional query block.
            const end = @min(items.len, at + pass_rows / d.config.block);
            var blocks: [pass_rows]block.Block = undefined;
            const group = items[at..end];
            at = end;
            for (group, blocks[0..group.len]) |it, *b| {
                const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
                if (it.depth == 0 or it.depth > d.config.block or it.position == 0) return error.DraftDepth;
                const anchor = it.position - 1;
                if (l.end + l.staged != anchor or anchor + d.config.block > l.cap) return error.DraftContextGap;
                b.* = .{ .lane = l, .pending = it.pending, .anchor = anchor, .depth = it.depth };
            }
            defer for (group, blocks[0..group.len]) |it, b| {
                it.stream.hold_context_rows = b.lane.staged;
            };
            if (group.len == items.len) { // one pass: signalled, not waited on
                d.signalled += 1;
                d.pending = try block.hold(d, blocks[0..group.len], .{ .event = d.event, .value = d.signalled });
                for (blocks[0..group.len], 0..) |b, bi| {
                    d.pass[bi] = b.lane;
                    d.depths[bi] = b.depth;
                    b.lane.slot = @intCast(bi);
                    b.lane.held_len = 0;
                }
                d.pass_len = group.len;
                continue;
            }
            _ = try block.hold(d, blocks[0..group.len], null);
        }
    }

    /// The signalled hold waited on, its picks read into its lanes' held chains (nothing pending: nothing done).
    pub fn settle(d: *Drafter) !void {
        const cb = d.pending orelse return;
        d.pending = null;
        const context_rows = d.pass_context_rows;
        d.pass_context_rows = 0;
        const encode_ms = d.pass_encode_cpu_ms;
        d.pass_encode_cpu_ms = 0;
        defer cb.release();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("lfm2 dspark: a hold failed on the GPU: {s}", .{msg});
            return error.DraftGpuFailed;
        }
        const ms = cb.gpuSeconds() * 1000; // the whole buffer, its lanes' staged rows absorbed too
        const timed = ms > 0 and std.math.isFinite(ms);
        if (timed) {
            d.hold_spent_ms += ms;
            d.priceHold(d.pass_len, std.mem.max(u32, d.depths[0..d.pass_len]), context_rows, encode_ms + ms);
        }
        // Releasing a peer must not discard completed picks when only the timing receipt is unavailable.
        try block.readPicks(d, d.pass[0..d.pass_len], d.depths[0..d.pass_len]);
        if (!timed) return error.DraftTimingUnavailable;
    }

    /// Reprices the clean grid with CPU-encode + GPU ms minus the matching calibrated context estimate.
    pub fn priceHold(d: *Drafter, streams: usize, depth: u32, context_rows: u32, ms: f64) void {
        const timed = d.holds[0..d.holds_len];
        if (timed.len == 0 or streams == 0 or !(ms > 0) or !std.math.isFinite(ms)) return;
        const clean_ms = ms - lanes.hold_price.contextMs(d.contexts[0..d.contexts_len], streams, context_rows);
        if (!(clean_ms > 0) or !std.math.isFinite(clean_ms)) return;
        const priced = lanes.hold_price.holdMs(timed, streams, depth);
        if (!(priced > 0) or !std.math.isFinite(priced)) return;
        const step = 1 + 0.1 * (clean_ms / priced - 1);
        if (!std.math.isFinite(step)) return;
        var lo: u32 = 0;
        var hi: u32 = std.math.maxInt(u32);
        for (timed) |h| {
            if (h.streams <= streams) lo = @max(lo, h.streams);
            if (h.streams >= streams) hi = @min(hi, h.streams);
        }
        for (timed) |*h| if (h.streams >= lo and h.streams <= hi) {
            h.ms *= step;
        };
    }

    /// The signalled hold settled and its lanes' picks forgotten (a lane may be dropped, or a hold may overwrite).
    fn unpass(d: *Drafter) !void {
        defer {
            for (d.pass[0..d.pass_len]) |l| l.slot = null;
            d.pass_len = 0;
        }
        try d.settle();
    }

    /// Where stream `s`'s drafts from the last signalled hold lie in the picks, for a verify to copy on the GPU.
    pub fn device(ptr: *anyopaque, s: *Stream) ?backend.Device {
        const d = cast(ptr);
        const l = d.lanes.get(s) orelse return null;
        const slot = l.slot orelse return null;
        return .{ .src = d.s.picks, .off = slot * 8, .stride = pass_rows * 2, .wait = .{ .event = d.event, .value = d.signalled } };
    }

    /// The target's verify reading the held drafts on the device, or as host tokens where it cannot.
    pub fn heldSource(d: *Drafter) backend.HeldSource {
        return .{ .ptr = d, .get = device, .host = hostHeld };
    }

    /// The target's forwards capturing taps straight into the lanes' staged rows.
    pub fn stageSource(d: *Drafter) backend.StageSource {
        return stage.source(d);
    }

    fn hostHeld(ptr: *anyopaque, s: *Stream, out: []u32) anyerror!void {
        return heldFn(ptr, &.{s}, &.{out});
    }

    fn mibOf(bufs: []const mtl.Buffer) f64 {
        var n: usize = 0;
        for (bufs) |b| n += b.length();
        return @as(f64, @floatFromInt(n)) / (1024.0 * 1024.0);
    }

    /// Temporary calibration lanes own only K/V; all context pieces borrow one raw-tap fixture.
    fn warmLane(d: *Drafter) !*Lane {
        const l = try d.gpa.create(Lane);
        errdefer d.gpa.destroy(l);
        const cap = stage_rows + d.config.block;
        const bytes: usize = @intCast(d.keptBytes(cap));
        const mem = try d.target.device.buffer(bytes, shared);
        l.* = .{ .mem = mem, .gpu_base = mem.gpuAddress(), .cap = cap, .stage_at = bytes };
        return l;
    }

    /// Empty-context holds and context-only passes, best CPU-encode + GPU sum of three after each shape's warm pass.
    fn warm(d: *Drafter) !void {
        defer {
            // Every startup buffer, including a failed calibration, is no round's work.
            d.hold_spent_ms = 0;
            d.encode_cpu_ms = 0;
        }
        var made: [hold_grid[hold_grid.len - 1]]*Lane = undefined;
        var n: usize = 0;
        defer for (made[0..n]) |l| d.dropLane(l);
        while (n < made.len) : (n += 1) made[n] = try d.warmLane();
        const fixture_bytes = stage_rows * d.tapBytes();
        const fixture = try d.target.device.buffer(fixture_bytes, shared);
        defer fixture.deinit();
        // Deterministic bf16 magnitudes in [0.5, 1), alternating signs; every tap is finite and nonzero.
        for (fixture.slice(u16, fixture_bytes / 2), 0..) |*v, i| {
            v.* = 0x3f00 | @as(u16, @intCast(i & 127)) | @as(u16, @intCast((i & 128) << 8));
        }
        var blocks: [hold_grid[hold_grid.len - 1]]block.Block = undefined;
        for (made, &blocks) |lane, *b| b.* = .{ .lane = lane, .pending = 1, .anchor = 0, .depth = 1 };
        // Publish only after both grids are complete, so calibration holds cannot reprice the clean grid.
        var timed: usize = 0;
        for (1..@min(made.len, pass_rows / d.config.block) + 1) |streams| for (1..d.config.block + 1) |depth| {
            for (blocks[0..streams]) |*b| b.depth = @intCast(depth);
            var ms_best = std.math.inf(f64);
            for (0..4) |i| {
                _ = block.takeHoldMs(d);
                _ = try block.hold(d, blocks[0..streams], null);
                const ms = block.takeHoldMs(d);
                if (i > 0) {
                    if (!(ms > 0) or !std.math.isFinite(ms)) return error.DraftTimingUnavailable;
                    ms_best = @min(ms_best, ms);
                }
            }
            d.holds[timed] = .{ .streams = @intCast(streams), .depth = @intCast(depth), .ms = ms_best };
            timed += 1;
        };
        var pieces: [hold_grid[hold_grid.len - 1]]block.Piece = undefined;
        for (made, &pieces) |lane, *pc| pc.* = .{ .lane = lane, .src = .{ .buf = fixture }, .rows = 1, .p0 = 0 };
        var context_timed: usize = 0;
        for (hold_grid) |streams| {
            var rows: u32 = 1;
            while (rows <= stage_rows) : (rows *= 2) {
                for (pieces[0..streams]) |*pc| pc.rows = rows;
                var ms_best = std.math.inf(f64);
                for (0..4) |i| {
                    const ms = try block.contextPass(d, pieces[0..streams]);
                    if (i > 0) {
                        if (!(ms > 0) or !std.math.isFinite(ms)) return error.DraftTimingUnavailable;
                        ms_best = @min(ms_best, ms);
                    }
                }
                d.contexts[context_timed] = .{ .streams = streams, .rows = rows, .ms = ms_best };
                context_timed += 1;
            }
        }
        d.holds_len = timed;
        d.contexts_len = context_timed;
        d.hold_ms = d.holds[d.config.block - 1].ms; // the first grid count is one stream, depths 1..block
        d.step_ms = d.hold_ms / @as(f64, @floatFromInt(d.config.block));
        std.log.info("lfm2 dspark: a 1-draft clean hold takes {d:.2} CPU-encode + GPU ms, a {d}-draft hold {d:.2} CPU-encode + GPU ms; {d} context quotes", .{ d.holds[0].ms, d.config.block, d.hold_ms, d.contexts_len });
        std.log.info("lfm2 dspark: weights {d:.1} MiB, scratch {d:.1} MiB, a stream {d} KiB a position and up to {d} KiB staged", .{ mibOf(d.w.buffers()), mibOf(d.owned.items), d.keptBytes(1) / 1024, d.stageBytes(d.options.capacity + d.config.block) / 1024 });
    }
};

test "lfm2 dspark: scratch and warm allocation failures release buffers before their list" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var target: Model = undefined;
    target.device = device;
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .resize_fail_index = 0 });
    var d: Drafter = .{
        .gpa = failing.allocator(),
        .io = undefined,
        .target = &target,
        .config = try dc.Config.parse(std.testing.allocator, dc.stock),
        .w = undefined,
        .options = .{ .capacity = 64 },
        .s = undefined,
    };
    const warm_fail_index = blk: {
        defer {
            for (d.owned.items) |b| b.deinit();
            d.owned.deinit(d.gpa);
        }
        try scratch.prepare(&d);
        break :blk failing.alloc_index;
    };
    for ([_]usize{ 1, warm_fail_index }) |fail_index| {
        failing = std.testing.FailingAllocator.init(std.testing.allocator, .{
            .fail_index = fail_index,
            .resize_fail_index = 0,
        });
        d.owned = .empty;
        try std.testing.expectError(error.OutOfMemory, d.prepare());
        try std.testing.expect(failing.has_induced_failure);
        try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
    }
}
