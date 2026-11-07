//! MTP drafts for the 27B on Metal: slot 0's head absorbs each row beside the token after it, then chains its picks.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const be = tf.lanes.backend;
const Ref = q.projection.Ref;
const Tree = q.dflash.selector.Tree;
const Round = q.round_plan.Round;
const lb = @import("qwen27_lanes.zig");

/// Drafts a chain holds at most: a verify window's rows less the pending token.
pub const max_drafts = 15;
const max_rows = q.gpu_frame.max_rows;
/// Hazard-tracked: a pass's staging, its carry and its saved row move between blits and kernels in one buffer.
const opts = mtl.ResourceOptions.shared;

pub const Mtp = struct {
    gpa: std.mem.Allocator,
    runner: *q.decode_round.Runner,
    scratch: q.mtp.Scratch,
    stage: mtl.Buffer, // a head pass's hidden rows: the carry, then the tap rows whose next token is known
    index: mtl.Buffer, // the tap rows the stage gathers
    carry_row: mtl.Buffer, // the last target row whose next token is not known yet
    out_row: mtl.Buffer, // the head's last row, the chain's first input
    controls: tf.lanes.mtp_policy.Controls,
    len: u32 = 0, // head cache rows
    carry: bool = false,
    ready: bool = false, // the head stands at the target's offset with its first pick
    first: q.mtp.Pick = .{ .token = 0, .chance = 0 },
    sampled: [max_rows]u32 = undefined, // slot 0's draws in the last verify, by window row
    window: ?u32 = null, // slot 0's window in the last verify

    /// Attach to a runner whose model holds the head: its taps keep the last layer's rows (before the final norm).
    pub fn init(gpa: std.mem.Allocator, runner: *q.decode_round.Runner, controls: tf.lanes.mtp_policy.Controls) !*Mtp {
        const m = runner.model;
        if (m.weights.mtp == null) return error.NoMtpHead;
        const row = m.config.hidden * 2;
        const cap = m.frame.capacity;
        var scratch = try q.mtp.Scratch.init(m.device, cap + 1, m.config.hidden, m.config.vocab, 1);
        errdefer scratch.deinit();
        const stage = try m.device.buffer((cap + 1) * row, opts);
        errdefer stage.deinit();
        const index = try m.device.buffer(cap * 4, opts);
        errdefer index.deinit();
        const carry_row = try m.device.buffer(row, opts);
        errdefer carry_row.deinit();
        const out_row = try m.device.buffer(row, opts);
        errdefer out_row.deinit();
        const x = try gpa.create(Mtp);
        x.* = .{ .gpa = gpa, .runner = runner, .scratch = scratch, .stage = stage, .index = index, .carry_row = carry_row, .out_row = out_row, .controls = controls };
        runner.capture_taps = true;
        runner.taps.ids[0] = m.config.layers - 1;
        runner.taps.count = 1;
        return x;
    }

    pub fn deinit(x: *Mtp) void {
        x.scratch.deinit();
        for ([_]mtl.Buffer{ x.stage, x.index, x.carry_row, x.out_row }) |b| b.deinit();
        x.gpa.destroy(x);
    }

    /// A target pass at `start` fed `fed0` first; its tap `rows` pair with `nexts`, and a last row without one waits.
    pub fn absorb(x: *Mtp, start: u32, fed0: u32, rows: []const u32, nexts: []const u32) !void {
        const r = x.runner;
        const m = r.model;
        const row = m.config.hidden * 2;
        if (rows.len != nexts.len and rows.len != nexts.len + 1) return error.BadMtpRows;
        // a state the head did not follow (a resume, a promoted stream): its count starts over at the target's
        if (x.len + @intFromBool(x.carry) != start) {
            x.len = start;
            x.carry = false;
        }
        x.ready = false;
        const c: usize = @intFromBool(x.carry);
        const n = c + nexts.len;
        if (n > max_rows) return error.BadMtpRows;
        var ids: [max_rows]u32 = undefined;
        if (c == 1) ids[0] = fed0;
        @memcpy(ids[c..n], nexts);
        const waits = rows.len > nexts.len;
        {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = m.queue.commandBuffer();
            if (nexts.len > 0) {
                @memcpy(x.index.slice(u32, nexts.len), rows[0..nexts.len]);
                const e = cb.compute(.serial);
                errdefer e.end(); // an encoder left open aborts the process when it is released
                try m.glue.tapRows(e, .{ .buffer = r.taps.buffer }, .{ .buffer = x.index }, .{ .buffer = x.stage, .offset = c * row }, .{ .rows = @intCast(nexts.len), .width = @intCast(m.config.hidden), .capacity = r.taps.capacity, .planes = 1 });
                e.end();
            }
            if (c == 1) {
                const bl = cb.blit();
                bl.copy(x.carry_row, 0, x.stage, 0, row);
                bl.end();
            }
            if (waits) {
                const bl = cb.blit();
                bl.copy(r.taps.buffer, @as(usize, rows[rows.len - 1]) * row, x.carry_row, 0, row);
                bl.end();
            }
            cb.commit();
            cb.wait();
            if (q.decode_round.gpuFailed(cb, "the MTP head's rows")) return error.RoundGpuFailure;
        }
        x.carry = waits;
        if (n == 0) return;
        var round = try Round.init(x.gpa, &.{.{ .slot = 0, .start = x.len, .capacity = r.capacity, .ids = ids[0..n] }}, @intCast(m.config.conv_kernel), r.slots, @intCast(m.shard.global.vocab));
        defer round.deinit();
        // logits only when the head then stands at the target's offset: a row still waiting means no draft yet
        const out = try r.headPass(&round, &x.scratch, .{ .buffer = x.stage }, if (waits) .none else .last);
        x.len += @intCast(n);
        if (waits) return;
        try x.save(out, n - 1);
        x.first = try q.mtp.pick(x.scratch.logits.slice(u16, m.config.vocab));
        x.ready = true;
    }

    /// One chained step's ms at load, the fastest of a few (its host pick included); slot 0's row 0 is rewritten.
    pub fn time(x: *Mtp) !f64 {
        const r = x.runner;
        const m = r.model;
        var best: f64 = std.math.inf(f64);
        for (0..4) |_| {
            var round = try Round.init(x.gpa, &.{.{ .slot = 0, .start = 0, .capacity = r.capacity, .ids = &.{0} }}, @intCast(m.config.conv_kernel), r.slots, @intCast(m.shard.global.vocab));
            defer round.deinit();
            const t0 = mtl.clock.seconds();
            _ = try r.headPass(&round, &x.scratch, .{ .buffer = x.stage }, .last);
            _ = try q.mtp.pick(x.scratch.logits.slice(u16, m.config.vocab));
            best = @min(best, (mtl.clock.seconds() - t0) * 1000);
        }
        x.len = 0;
        x.carry = false;
        x.ready = false;
        return best;
    }

    fn save(x: *Mtp, out: Ref, at: usize) !void {
        const row = x.runner.model.config.hidden * 2;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = x.runner.model.queue.commandBuffer();
        const bl = cb.blit();
        bl.copy(out.buffer, out.offset + at * row, x.out_row, 0, row);
        bl.end();
        cb.commit();
        cb.wait();
        if (q.decode_round.gpuFailed(cb, "the MTP head's row")) return error.RoundGpuFailure;
    }

    /// The chain from the head's first pick: `most` drafts at most, cut where the cumulative chance passes the floor.
    pub fn chain(x: *Mtp, most: usize) !?Tree {
        if (!x.ready or most == 0) return null;
        const r = x.runner;
        const m = r.model;
        const room = r.capacity - x.len;
        const limit = x.controls.limit(@min(most, max_drafts, room -| 1));
        if (limit == 0) return null;
        var tokens: [max_drafts]u32 = undefined;
        var chances: [max_drafts]f64 = undefined;
        tokens[0] = x.first.token;
        chances[0] = x.first.chance;
        var hidden = Ref{ .buffer = x.out_row };
        var n: usize = 1;
        while (n < limit) : (n += 1) {
            // a later draft's floor: the chain stops before running the head for a draft it would not keep
            if (x.controls.floor) |floor| if (!(chances[n - 1] >= floor)) break;
            var round = try Round.init(x.gpa, &.{.{ .slot = 0, .start = x.len + @as(u32, @intCast(n - 1)), .capacity = r.capacity, .ids = tokens[n - 1 .. n] }}, @intCast(m.config.conv_kernel), r.slots, @intCast(m.shard.global.vocab));
            defer round.deinit();
            hidden = try r.headPass(&round, &x.scratch, hidden, .last);
            const p = try q.mtp.pick(x.scratch.logits.slice(u16, m.config.vocab));
            tokens[n] = p.token;
            chances[n] = chances[n - 1] * p.chance;
        }
        const held = x.controls.prefix(chances[0..n]);
        const t = Tree{ .gpa = x.gpa, .tokens = try x.gpa.dupe(u32, tokens[0..held]), .parents = try x.gpa.alloc(i32, held), .scores = try x.gpa.alloc(f64, held) };
        for (t.parents, t.scores, 0..) |*p, *s, i| {
            p.* = @as(i32, @intCast(i)) - 1;
            s.* = @log(@max(chances[i], 1e-30));
        }
        return t;
    }
};

/// --mtp-drafts and --mtp-confidence on a Mac: Qwen3.8-27B's MTP head (`target`), opt-in, exclusive with DFlash2.
pub fn refusal(drafts: ?u32, confidence: ?f64, drafter: bool, target: bool) ?[]const u8 {
    if (drafts == null and confidence == null) return null;
    if (!target) return "--mtp-drafts and --mtp-confidence draft with Qwen3.8-27B's MTP head on a Mac; serve this checkpoint without them";
    if (confidence) |c| if (!(c >= 0 and c <= 1)) return "--mtp-confidence must be from 0 to 1";
    const n = drafts orelse return "--mtp-confidence stops the MTP head's chains, which --mtp-drafts N turns on: serve with both or neither";
    if (n > max_drafts) return std.fmt.comptimePrint("--mtp-drafts: Qwen3.8-27B on a Mac drafts at most {d} a round", .{max_drafts});
    if (n > 0 and drafter) return "--mtp-drafts drafts with the MTP head, and --drafter with DFlash2: serve with one of them";
    return null;
}

/// Slot 0's verified rows into the head after the keep: each kept row beside the token drawn at it.
pub fn kept(b: *lb.Metal, paths: []const []const u32) !void {
    const x = b.mtp orelse return;
    const i = x.window orelse return;
    x.window = null;
    const round = &(b.round orelse return);
    const first = round.firsts[i];
    const path = paths[i];
    var rows: [max_rows]u32 = undefined;
    var nexts: [max_rows]u32 = undefined;
    for (path, 0..) |row, j| {
        rows[j] = first + row;
        nexts[j] = x.sampled[row];
    }
    try x.absorb(round.windows[i].start, round.ids[first], rows[0..path.len], nexts[0..path.len]);
}

/// Slot 0's next chain: after a prompt, the head first takes its last row beside the first token.
pub fn draft(b: *lb.Metal, requests: []const be.DraftRequest) !void {
    const x = b.mtp.?;
    // a lone round whose rows were all kept gets no keep call: it settles here, and its rows reach the head
    if (b.count == 1) try b.settleRound();
    for (requests) |r| {
        const slot = try b.slotFor(r.stream);
        b.dropTreeOf(slot);
        if (slot != 0 or r.depth == 0 or b.round != null) continue; // a shared round's rows reach the head at its keep
        if (r.rows == null) try x.absorb(b.runner.offsets[0], try b.value(r.first orelse return error.NoFirstToken), &.{}, &.{});
        if (x.len != b.runner.offsets[0]) continue;
        b.trees[0] = try x.chain(r.depth);
    }
}

/// Slot 0 alone drafts, up to the head's cap.
pub fn capacity(b: *const lb.Metal, slot: u32) u32 {
    const x = b.mtp orelse return 0;
    if (slot != 0) return 0;
    for (b.owners[1..b.runner.slots]) |o| if (o != null) return 0;
    return @intCast(x.controls.limit(@min(max_drafts, b.window - 1)));
}
