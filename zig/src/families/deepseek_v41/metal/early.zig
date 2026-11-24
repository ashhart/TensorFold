//! Drafts behind the verify: multi-slot and filtered (greedy, raced, min_p, top_k, nucleus) GPU picks and accept.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const dspark = @import("dspark.zig");
const plan_ = @import("dspark_plan.zig");
const pipelines = @import("pipelines.zig");
const picks_ = @import("early_picks.zig");
const mkv = @import("markov.zig");
const round_prof = @import("round_prof.zig");

const iface = host.iface;
const attn = pipelines.attn;
const R = pipelines.max_rows;
const Plan = plan_.Plan;
const slot_plan_ints = 68;
pub const raced = picks_.raced;
pub const supported = picks_.supported;

pub const Stats = struct {
    /// chains queued behind a window (`keyed`: sampled windows); passes the lanes took, and those dropped
    queued: u64 = 0,
    keyed: u64 = 0,
    used: u64 = 0,
    dropped: u64 = 0,
    /// chains whose accepted count a test swapped (`skew_every`)
    skewed: u64 = 0,
    /// multi-slot chains queued (2+ slots in one behind-the-verify chain)
    multi: u64 = 0,
};

/// A window about to be verified whose slot may draft behind it (Pass.arm).
const Arm = struct { slot: u32, start: u64, rows: u32 };

/// One slot of a queued chain behind a verify window.
const ChainSlot = struct {
    slot: u32,
    start: u64,
    rows: u32,
    r0: u32,
    taps: mtl.Buffer,
    /// the window's first rows the lanes asked ingested: the chain's own ingest when its picks accepted as many
    deferred: ?u32 = null,
    /// the pass ask the lanes began for this slot
    ask: ?dspark.Ask = null,
};

/// A chain queued behind a window: its slots, its command buffer, and what the lanes asked of it since.
const Chain = struct {
    slots: [R]ChainSlot = undefined,
    n_slots: usize = 0,
    /// maps caller's `begin(asks)` index k -> `slots[idx]`
    ask_order: [R]u8 = @splat(0),
    begun: bool = false,
    /// its pass's chain runs on the GPU (dspark.Pass.drawsOnGpu for all slots)
    gpu: bool = false,
    /// whether this chain needs top-K GPU Markov filtering (greedy/raced stay on shared-w2 full-vocab)
    use_topk: bool = false,
    /// the drafts its pass makes per slot
    block: u32 = 0,
    cb: mtl.CommandBuffer = undefined,
    profile_start: u64 = 0,
};

pub const Early = struct {
    a: std.mem.Allocator,
    x: *dspark.Pass,
    picks: picks_.Picks,
    /// tokens [R] u32, picks [R] u32, results [2*R] u32, Row [R], Slot [R]
    state: mtl.Buffer,
    /// compact per-(slot, acc) plans: [R][R][slot_plan_ints] i32 (68 KiB)
    plans: mtl.Buffer,
    armed: [R]Arm = undefined,
    n_armed: usize = 0,
    chain: ?Chain = null,
    stats: Stats = .{},
    /// tests: every this many chains (0: none) the GPU swaps slot 0's acc 0 and 1, so the chain is dropped
    skew_every: u32 = 0,

    const tokens_at = 0;
    const picks_at = 4 * R;
    const result_at = 8 * R;
    const rows_at = 16 * R;
    const slots_at = rows_at + R * @sizeOf(picks_.Row);
    const state_bytes = slots_at + R * @sizeOf(picks_.Slot);
    const plans_bytes = @as(usize, R) * R * slot_plan_ints * 4;

    pub fn init(a: std.mem.Allocator, x: *dspark.Pass) !*Early {
        var picks = try picks_.Picks.init(x.device);
        errdefer picks.deinit();
        const state = try x.device.buffer(state_bytes, mtl.ResourceOptions.shared);
        errdefer state.deinit();
        @memset(state.contents()[0..state_bytes], 0);
        const plans = try x.device.buffer(plans_bytes, mtl.ResourceOptions.shared);
        errdefer plans.deinit();
        @memset(plans.contents()[0..plans_bytes], 0);
        if (x.resident) |set| {
            set.add(state);
            set.add(plans);
            set.commit();
        }
        const e = try a.create(Early);
        e.* = .{ .a = a, .x = x, .picks = picks, .state = state, .plans = plans };
        return e;
    }

    pub fn deinit(e: *Early) void {
        e.drain() catch {};
        e.picks.deinit();
        e.state.deinit();
        e.plans.deinit();
        e.a.destroy(e);
    }

    // the window ---------------------------------------------------------------------------------------------------

    /// Before a window: `slot`'s rows from `start` may draft behind it (sampling read at `queue`).
    pub fn arm(e: *Early, slot: u32, start: u64, rows: u32, sampled: bool) void {
        _ = sampled;
        if (e.n_armed < R) {
            e.armed[e.n_armed] = .{ .slot = slot, .start = start, .rows = rows };
            e.n_armed += 1;
        }
    }

    /// Right after a window's command buffers: queue picks, accept, ingest and next draft pass.
    pub fn queue(e: *Early, group: []const iface.Segment, logits: mtl.Buffer, logits_off: usize, taps: []const mtl.Buffer) void {
        const n_armed = e.n_armed;
        defer e.n_armed = 0;
        const x = e.x;
        const c = x.c;
        const G = group.len;
        if (G == 0 or G != n_armed or G > x.group() or taps.len != G or e.chain != null) return;
        var B: u32 = c.dspark_block;
        var total_rows: u32 = 0;
        var any_keyed = false;
        var any_topk = false;
        var gpu = x.mk != null;
        var ch: Chain = .{ .n_slots = G };
        for (group, 0..) |g, j| {
            const n: u32 = @intCast(g.tokens.len);
            var matched = false;
            for (e.armed[0..n_armed]) |ar| if (ar.slot == g.slot and ar.start == g.start and ar.rows == n) {
                matched = true;
                break;
            };
            if (!matched or g.slot >= x.runs.len) return;
            for (group[0..j]) |prev| if (prev.slot == g.slot) return;
            if (!picks_.supported(g.sampling, c.vocab)) return;
            if (g.draws.len != n or n == 0 or n > c.dspark_block + 1 or total_rows + n > R) return;
            B = @min(B, x.blockOf(g.slot));
            const keyed = if (g.sampling) |s| s.temperature > 0 else false;
            any_keyed = any_keyed or keyed;
            const p = host.dspark.Params.of(g.sampling, 0);
            gpu = gpu and x.drawsOnGpu(p);
            any_topk = any_topk or x.needsTopK(p);
            ch.slots[j] = .{ .slot = g.slot, .start = g.start, .rows = n, .r0 = total_rows, .taps = taps[j] };
            total_rows += n;
        }
        if (G * B > R) return;
        for (group) |g| if (g.start + g.tokens.len + B > x.tables.positions) return;
        ch.block = B;
        ch.gpu = gpu;
        ch.use_topk = gpu and x.mk != null and x.mk.?.topk > 0 and any_topk;
        const skew = e.skew_every > 0 and ch.slots[0].rows > 1 and (e.stats.queued + 1) % e.skew_every == 0;
        e.stage(group, &ch, B, skew);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const started = if (x.prof) |p| p.clock() else 0;
        const cb = x.queue.commandBuffer();
        e.encode(cb, logits, logits_off, &ch, total_rows) catch return;
        cb.commit();
        ch.cb = cb.retain();
        ch.profile_start = started;
        e.chain = ch;
        e.stats.skewed += @intFromBool(skew);
        e.stats.keyed += @intFromBool(any_keyed);
        e.stats.multi += @intFromBool(G > 1);
        e.stats.queued += 1;
    }

    fn stage(e: *Early, group: []const iface.Segment, ch: *const Chain, B: u32, skew: bool) void {
        const x = e.x;
        const c = x.c;
        const G = group.len;
        const ids = e.state.slice(u32, R);
        const row_specs: []picks_.Row = @alignCast(std.mem.bytesAsSlice(picks_.Row, e.state.contents()[rows_at..][0 .. R * @sizeOf(picks_.Row)]));
        const slot_descs: []picks_.Slot = @alignCast(std.mem.bytesAsSlice(picks_.Slot, e.state.contents()[slots_at..][0 .. R * @sizeOf(picks_.Slot)]));
        const all_plans = e.plans.slice(i32, @as(usize, R) * R * slot_plan_ints);
        const block: []u32 = @alignCast(std.mem.bytesAsSlice(u32, x.b.bytes(x.b.ids, 4 * G * B)));
        for (group, 0..) |g, j| {
            const cs = ch.slots[j];
            const n = cs.rows;
            for (0..n) |r| {
                ids[cs.r0 + r] = g.tokens[r];
                row_specs[cs.r0 + r] = picks_.Row.of(g.sampling, g.draws[r]);
            }
            const pos_base = g.start + 2;
            slot_descs[j] = .{ .r0 = cs.r0, .n = n, .block = B, .skew = @intFromBool(skew and j == 0), .pos_lo = @truncate(pos_base), .pos_hi = @truncate(pos_base >> 32) };
            if (ch.gpu) x.mk.?.writeAsk(j, host.dspark.Params.of(g.sampling, 0));
            for (0..n) |acc| {
                const m = all_plans[(j * R + acc) * slot_plan_ints ..][0..slot_plan_ints];
                for (0..n) |r| {
                    m[r] = @intCast(g.start + r);
                    const dest = if (r <= acc) plan_.ringRow(c, g.slot, g.start + r) else plan_.blockRow(c, g.slot, @intCast(r - acc - 1));
                    m[R + r] = @intCast(dest);
                }
                const start_next = g.start + acc + 1;
                var run = x.runs[g.slot];
                run.land(g.start, acc + 1);
                const cx = run.context(c.window, start_next);
                for (0..B) |i| {
                    m[2 * R + i] = @intCast(start_next + i);
                    m[3 * R + i] = @intCast(plan_.blockRow(c, g.slot, @intCast(i)));
                }
                const rec: attn.DsparkAsk = .{ .base = g.slot * plan_.poolRows(c), .first_mod = @intCast(cx.first % c.window), .ctx_count = cx.count, .start_pos = @intCast(start_next) };
                @memcpy(m[4 * R ..][0..4], @as(*const [4]i32, @ptrCast(&rec)));
            }
            host.dspark.blockIds(.{ .block = B, .noise = c.dspark_noise_token }, 0, block[j * B ..][0..B]);
        }
    }

    fn encode(e: *Early, cb: mtl.CommandBuffer, logits: mtl.Buffer, logits_off: usize, ch: *const Chain, total_rows: u32) !void {
        const x = e.x;
        const G: u32 = @intCast(ch.n_slots);
        if (x.hasQueued()) {
            for (ch.slots[0..ch.n_slots]) |cs| x.undo.encodeSettleAndSave(cb, cs.slot, cs.start, cs.start, cs.rows);
        } else {
            for (ch.slots[0..ch.n_slots]) |cs| {
                x.undo.settle(cs.slot, cs.start);
                x.undo.save(cs.slot, cs.start, cs.rows);
            }
        }
        const ce = cb.compute(.concurrent);
        e.picks.encodeRows(ce, .{ .buf = logits, .off = logits_off }, .{ .buf = e.state, .off = picks_at }, .{ .buf = e.state, .off = rows_at }, total_rows, x.c.vocab);
        ce.barrier();
        ce.setPipeline(e.picks.accept_multi);
        ce.setValue([5]u32{ G, @intFromBool(ch.gpu), @sizeOf(mkv.Ask), @offsetOf(mkv.Ask, "pos0"), Plan.asks }, 0);
        ce.setBuffer(e.state, slots_at, 1);
        ce.setBuffer(e.state, tokens_at, 2);
        ce.setBuffer(e.state, picks_at, 3);
        ce.setBuffer(e.plans, 0, 4);
        ce.setBuffer(x.plan.buf, 0, 5);
        ce.setBuffer(x.b.ids.buf, x.b.ids.off, 6);
        ce.setBuffer(e.state, result_at, 7);
        if (ch.gpu) ce.setBuffer(x.mk.?.o.buf, mkv.Out.asks_at, 8) else ce.setBuffer(e.state, result_at, 8);
        ce.dispatchGroups(mtl.Size.of(G, 1, 1), mtl.Size.of(64, 1, 1));
        ce.barrier();
        for (ch.slots[0..ch.n_slots], 0..) |cs, j| {
            if (j > 0) ce.barrier();
            x.encodeIngestAt(ce, .{ .buf = cs.taps, .rows = cs.rows }, cs.r0, cs.rows) catch |err| {
                ce.end();
                return err;
            };
        }
        ce.barrier();
        const l: dspark.Launch = .{ .rows = G * ch.block, .n = ch.block, .asks = &.{}, .gpu = ch.gpu, .use_topk = ch.use_topk };
        const done = x.encodeAll(ce, &l);
        ce.end();
        try done;
    }

    // the lanes' asks ----------------------------------------------------------------------------------------------

    /// An ingest of one of the queued chain's slots' window rows 0 .. m - 1 from its taps: left to the chain (true).
    pub fn covers(e: *Early, slot: u32, start: u64, taps: mtl.Buffer, off: usize, rows: []const u32) bool {
        const ch = &(e.chain orelse return false);
        for (ch.slots[0..ch.n_slots]) |*cs| {
            if (cs.slot != slot) continue;
            if (cs.start != start or cs.deferred != null or taps.id != cs.taps.id or off != 0) return false;
            if (rows.len == 0 or rows.len > cs.rows) return false;
            for (rows, 0..) |r, i| if (r != i) return false;
            cs.deferred = @intCast(rows.len);
            return true;
        }
        return false;
    }

    /// The lanes begin the pass after the deferred rows: matches our queued chain's slots 1-to-1 (true), else not ours.
    pub fn begin(e: *Early, asks: []const iface.Ask) bool {
        const ch = &(e.chain orelse return false);
        if (asks.len != ch.n_slots or ch.begun) return false;
        var seen: [R]bool = @splat(false);
        for (asks, 0..) |k, ai| {
            var found: ?usize = null;
            for (ch.slots[0..ch.n_slots], 0..) |cs, idx| {
                if (cs.slot != k.slot) continue;
                if (seen[idx]) return false;
                const m = cs.deferred orelse return false;
                if (k.start != cs.start + m) return false;
                found = idx;
                break;
            }
            const idx = found orelse return false;
            seen[idx] = true;
            ch.ask_order[ai] = @intCast(idx);
        }
        for (asks, 0..) |k, ai| {
            const idx = ch.ask_order[ai];
            ch.slots[idx].ask = .{ .slot = k.slot, .anchor = k.anchor, .start = k.start, .params = k.params };
        }
        ch.begun = true;
        return true;
    }

    /// The begun pass's results: the chain's when every slot's picks matched the host, else exact host fallback.
    pub fn collect(e: *Early, out: []dspark.Proposal) !void {
        const ch = e.chain orelse return error.NothingBegun;
        if (!ch.begun or out.len != ch.n_slots) return error.NothingBegun;
        e.chain = null;
        var got: [R]Got = undefined;
        try e.wait(ch, got[0..ch.n_slots]);
        var all_ok = true;
        for (ch.slots[0..ch.n_slots], got[0..ch.n_slots]) |cs, g| {
            if (g.acc + 1 != cs.deferred.? or g.bonus != cs.ask.?.anchor) {
                all_ok = false;
                break;
            }
        }
        if (all_ok) {
            var asks_grp: [R]dspark.Ask = undefined;
            var props_grp: [R]dspark.Proposal = undefined;
            for (ch.slots[0..ch.n_slots]) |cs| e.keepSlot(cs, cs.deferred.?);
            for (0..ch.n_slots) |k| {
                const idx = ch.ask_order[k];
                asks_grp[idx] = ch.slots[idx].ask.?;
                props_grp[idx] = out[k];
            }
            const l: dspark.Launch = .{ .rows = @as(u32, @intCast(ch.n_slots)) * ch.block, .n = ch.block, .asks = asks_grp[0..ch.n_slots], .gpu = ch.gpu, .use_topk = ch.use_topk };
            try e.x.finish(&l, props_grp[0..ch.n_slots]);
            e.stats.used += 1;
            return;
        }
        e.stats.dropped += 1;
        for (ch.slots[0..ch.n_slots], got[0..ch.n_slots]) |cs, g| {
            const m = cs.deferred.?;
            if (g.acc + 1 == m) e.keepSlot(cs, m) else try e.replaceSlot(cs, m);
        }
        var caller_asks: [R]dspark.Ask = undefined;
        for (0..ch.n_slots) |k| caller_asks[k] = ch.slots[ch.ask_order[k]].ask.?;
        try e.x.propose(caller_asks[0..ch.n_slots], out);
    }

    /// The queued chain finished and settled: deferred ingests kept or redone, other ring rows put back.
    pub fn drain(e: *Early) !void {
        const ch = e.chain orelse return;
        e.chain = null;
        var got: [R]Got = undefined;
        try e.wait(ch, got[0..ch.n_slots]);
        if (ch.begun) e.stats.dropped += 1;
        for (ch.slots[0..ch.n_slots], got[0..ch.n_slots]) |cs, g| {
            const m = cs.deferred orelse {
                try e.x.restoreUndo(cs.slot, cs.start);
                continue;
            };
            if (g.acc + 1 == m) e.keepSlot(cs, m) else try e.replaceSlot(cs, m);
        }
    }

    const Got = struct { acc: u32, bonus: u32 };

    fn wait(e: *Early, ch: Chain, out: []Got) !void {
        defer ch.cb.release();
        ch.cb.wait();
        if (e.x.prof) |p| {
            p.event(.chain_wall, p.clock() -| ch.profile_start);
            p.event(.chain_gpu, round_prof.secondsNs(ch.cb.gpuSeconds()));
        }
        if (ch.cb.failure()) |msg| {
            std.debug.print("dsv41 early drafts: command buffer failed: {s}\n", .{msg});
            return error.CommandBufferFailed;
        }
        const r = e.state.slice(u32, result_at / 4 + 2 * R)[result_at / 4 ..][0 .. 2 * R];
        for (ch.slots[0..ch.n_slots], out, 0..) |cs, *g, j| g.* = .{ .acc = @min(r[2 * j], cs.rows - 1), .bonus = r[2 * j + 1] };
    }

    /// The slot's chain ingest matches the host's: its ring rows landed, kept in case a later commit is shorter.
    fn keepSlot(e: *Early, cs: ChainSlot, m: u32) void {
        e.x.runs[cs.slot].land(cs.start, m);
        e.x.undo.shrink(cs.slot, m);
    }

    /// The slot's chain picks differed from the host's (`m`): its rows put back, the host's ingested.
    fn replaceSlot(e: *Early, cs: ChainSlot, m: u32) !void {
        try e.x.restoreUndo(cs.slot, cs.start);
        var rows: [R]u32 = undefined;
        for (rows[0..m], 0..) |*r, i| r.* = @intCast(i);
        try e.x.ingest(cs.slot, cs.start, .{ .buf = cs.taps, .rows = cs.rows }, rows[0..m]);
    }

    pub fn describe(e: *const Early, buf: []u8) []const u8 {
        const s = e.stats;
        return std.fmt.bufPrint(buf, "drafts behind the verify: {d} queued ({d} keyed, {d} multi), {d} used, {d} dropped; {d} drafter ring rows put back", .{ s.queued, s.keyed, s.multi, s.used, s.dropped, e.x.undo.restored }) catch buf[0..0];
    }
};
