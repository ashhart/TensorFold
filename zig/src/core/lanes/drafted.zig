//! A target backend with an external drafter in front; deinit releases no stream, so callers release first.
const std = @import("std");
const Allocator = std.mem.Allocator;
const be = @import("backend.zig");
const Model = @import("config.zig").Model;
const Stream = @import("stream.zig").Stream;
const Held = @import("stream.zig").Held;
const dr = @import("drafter.zig");

/// An admitted request; only an opened head owns drafter context.
const Lane = struct {
    opened: bool = false,
    rows: bool = false, // the drafter has absorbed the row before the next draft (it can draft)
    filler: u32 = 0, // drafts the next verify takes as fillers instead (a prompt restored whole: no row to draft from)
    tree_first: ?u32 = null, // the first token of the grown tree read back this round, until a verify carries it
    offered: u32 = 0, // the grown tree's nodes the last verify carried, told to the drafter with its kept rows
    stepped: ?u64 = null, // a prompt pass going a step at a time: the prompt rows the drafter has been given
    head_held: bool = false, // a real hold, not restored-prompt fillers, supplied the current head
};

pub const Drafted = struct {
    gpa: Allocator,
    target: be.Backend,
    drafter: dr.Drafter,
    lanes: std.AutoHashMapUnmanaged(*Stream, Lane) = .empty, // all admissions, including target-only
    heads: usize = 0,
    translated: bool = false, // the last verify replaced device drafts with host tokens
    // scratch kept between rounds, so a steady round allocates nothing
    windows: std.ArrayList(be.Window) = .empty, // the last verify's windows, held drafts swapped for host tokens
    tokens: std.ArrayList(u32) = .empty, // their drafts, back to back
    streams: std.ArrayList(*Stream) = .empty,
    outs: std.ArrayList([]u32) = .empty,
    absorbs: std.ArrayList(dr.Absorb) = .empty,
    holds: std.ArrayList(dr.Hold) = .empty,
    firsts: std.ArrayList(u32) = .empty,
    /// The target reads held drafts from the drafter itself (Window.held), so windows without fillers pass as they are.
    held_on_target: bool = false,
    first_rows: bool = false,
    prepares: bool = false, // the drafter owns context; facts() sets it before backend() picks the prepared tables

    /// The wrapper, with the target told the drafter's taps before any forward.
    pub fn init(gpa: Allocator, target: be.Backend, drafter: dr.Drafter) !Drafted {
        if (target.vtable.prepare_features) |prep| try prep(target.ptr, drafter.taps());
        return .{ .gpa = gpa, .target = target, .drafter = drafter };
    }

    pub fn deinit(x: *Drafted) void {
        x.lanes.deinit(x.gpa);
        x.windows.deinit(x.gpa);
        x.tokens.deinit(x.gpa);
        x.streams.deinit(x.gpa);
        x.outs.deinit(x.gpa);
        x.absorbs.deinit(x.gpa);
        x.holds.deinit(x.gpa);
        x.firsts.deinit(x.gpa);
    }

    const legacy_vtable: be.Backend.VTable = .{ .prefill = prefill, .first = first, .first_masked = firstMasked, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .held = held, .probabilities = probabilities, .tree = tree, .observe_head = observeHead, .release = release };
    const planned_vtable: be.Backend.VTable = blk: {
        var v = legacy_vtable;
        v.proposal_draft = draft;
        v.clear_proposals = clearProposals;
        break :blk v;
    };
    /// The same over a target that steps its prompt passes and reports their fill: the host interleaves them.
    const stepped_legacy: be.Backend.VTable = blk: {
        var v = legacy_vtable;
        v.prefill_step = prefillStep;
        v.reports_fill = true;
        break :blk v;
    };
    const stepped_planned: be.Backend.VTable = blk: {
        var v = planned_vtable;
        v.prefill_step = prefillStep;
        v.reports_fill = true;
        break :blk v;
    };
    /// The target's policy through the wrapper; it never calls the drafter, which a target-only test may leave inert.
    pub fn backend(x: *Drafted) be.Backend {
        const planned = if (x.drafter.proposal_head) |service| service.mode != .legacy and x.drafter.vtable.hold_planned != null and x.drafter.vtable.clear_proposals != null else false;
        return .{ .proposal_head = x.drafter.proposal_head, .ptr = x, .draft_timing = x.drafter.hold_timing, .round_observer = x.target.round_observer, .first_before_draft = x.target.first_before_draft, .keep_full_lone = x.target.keep_full_lone, .penalties = x.target.penalties, .chunk_invariant = x.target.chunk_invariant, .warm_images = x.target.warm_images, .vtable = vtableFor(planned, x.target.vtable.prefill_step != null and x.target.vtable.reports_fill and x.target.vtable.features != null, x.target.vtable.cancel_prefill != null, x.prepares, x.first_rows) };
    }

    const cancellable: [4]be.Backend.VTable = blk: {
        var tables = [_]be.Backend.VTable{ legacy_vtable, planned_vtable, stepped_legacy, stepped_planned };
        for (&tables) |*v| v.cancel_prefill = cancelPrefill;
        break :blk tables;
    };

    const prepared: [8]be.Backend.VTable = blk: {
        var tables = [_]be.Backend.VTable{ legacy_vtable, planned_vtable, stepped_legacy, stepped_planned } ++ cancellable;
        for (&tables) |*v| v.prepare_draft = prepareDraft;
        break :blk tables;
    };

    const row_tables: [8]be.Backend.VTable = blk: {
        var tables = [_]be.Backend.VTable{ legacy_vtable, planned_vtable, stepped_legacy, stepped_planned } ++ cancellable;
        for (&tables) |*v| v.first_row = firstRow;
        break :blk tables;
    };

    const row_prepared: [8]be.Backend.VTable = blk: {
        var tables = prepared;
        for (&tables) |*v| v.first_row = firstRow;
        break :blk tables;
    };

    fn vtableFor(planned: bool, steps: bool, cancels: bool, prepares: bool, rows: bool) *const be.Backend.VTable {
        const at = @as(usize, @intFromBool(cancels)) * 4 +
            @as(usize, @intFromBool(steps)) * 2 + @intFromBool(planned);
        if (prepares) return if (rows) &row_prepared[at] else &prepared[at];
        if (rows) return &row_tables[at];
        if (cancels) return &cancellable[@as(usize, @intFromBool(steps)) * 2 + @intFromBool(planned)];
        if (steps) return if (planned) &stepped_planned else &stepped_legacy;
        return if (planned) &planned_vtable else &legacy_vtable;
    }

    fn cancelPrefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        errdefer x.dropStream(s);
        try x.target.vtable.cancel_prefill.?(x.target.ptr, s);
    }

    /// The target's facts with the drafter's depth, step cost, prior, guard, batching and copy match; chains late.
    pub fn facts(x: *Drafted, target: Model) Model {
        const d = x.drafter.facts();
        x.prepares = d.hold_context_costs.len > 0; // the prepared tables stage this context before a draft
        var m = target;
        m.mtp = true;
        m.speculate = true;
        m.speculate_early = false;
        m.allocate_shared_drafts = true;
        m.drafts = @min(d.depth, if (target.exact_width > 1) target.exact_width - 1 else 0);
        m.mtp_step_ms = d.step_ms;
        m.fixed_draft_block = d.fixed_block;
        m.draft_prior = d.prior;
        m.plain_guard = d.plain_guard;
        m.draft_streams = d.batched;
        m.draft_probabilities = false;
        m.head_trees = d.trees;
        m.copy_first = d.copy_first;
        m.hold_costs = d.hold_costs;
        m.hold_context_costs = d.hold_context_costs;
        m.hold_rows = d.hold_rows;
        m.hold_block_rows = d.hold_block_rows;
        if (d.chances and x.drafter.vtable.chances != null) {
            // a chain's drafts priced by the drafter's own chances; the next hold as deep as the grant and its slack
            m.draft_probabilities = d.confidence_floor == null;
            m.draft_confidence = d.confidence_floor;
            m.granted_depth = d.confidence_floor == null;
            m.draft_slack = d.slack;
        }
        if (d.tree_nodes > 0) {
            // the drafter grows its own trees, best first, and prices their nodes
            m.drafts = @min(d.tree_nodes, if (target.exact_width > 1) target.exact_width - 1 else 0);
            m.draft_probabilities = x.drafter.vtable.chances != null;
            m.head_trees = false;
        }
        m.limit_shared_drafts = (target.limit_shared_drafts or d.limit_shared_drafts) and
            !m.draft_probabilities and m.draft_confidence == null;
        if (d.copy_match > 0) m.copy_match = d.copy_match;
        if (d.window_costs.len > 0) m.window_costs = d.window_costs;
        if (d.lane_costs.len > 0) m.lane_costs = d.lane_costs;
        return m;
    }

    /// Device bytes an admitted stream takes: the target's own and the drafter's for the same stream.
    pub fn streamBytes(x: *const Drafted, target: usize) !usize {
        return std.math.add(usize, target, x.drafter.facts().stream_bytes) catch error.StreamSizeOverflow;
    }

    fn clearProposals(ptr: *anyopaque, s: *Stream) void {
        const x = self(ptr);
        if (x.openedLane(s)) |lane| {
            lane.filler = 0;
            lane.tree_first = null;
            x.drafter.vtable.clear_proposals.?(x.drafter.ptr, s);
        }
    }

    fn self(ptr: *anyopaque) *Drafted {
        return @ptrCast(@alignCast(ptr));
    }

    fn openedLane(x: *Drafted, s: *Stream) ?*Lane {
        const lane = x.lanes.getPtr(s) orelse return null;
        return if (lane.opened) lane else null;
    }

    /// Admission owns the route across prompt chunks and same-backend handovers, until actual release.
    fn enroll(x: *Drafted, s: *Stream) !void {
        if (x.lanes.contains(s)) return;
        try x.lanes.ensureUnusedCapacity(x.gpa, 1);
        // Disabled drafts never own head context; enabled legacy providers keep their feature-prefill path.
        s.head_context = if (!s.drafts)
            false
        else if (x.drafter.vtable.enroll != null)
            x.drafter.enroll(s, .{ .streams = @max(x.lanes.count() + 1, s.prefill_cohort), .heads = x.heads + 1 })
        else
            null;
        if (s.head_context == false) s.head_prior = &.{};
        x.lanes.putAssumeCapacity(s, .{});
        x.heads += @intFromBool(s.head_context != false);
    }

    fn observeHead(ptr: *anyopaque, s: *Stream, proposed: u32, accepted: u32) void {
        const x = self(ptr);
        const lane = x.openedLane(s) orelse return;
        if (!lane.head_held) return;
        lane.head_held = false;
        if (x.target.vtable.observe_head) |observe| observe(x.target.ptr, s, proposed, accepted);
        if (x.drafter.vtable.observe_head) |observe| observe(x.drafter.ptr, s, proposed, accepted);
    }

    /// The target's prompt pass, then the drafter absorbs its rows but the last; a later failure releases both.
    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = self(ptr);
        try x.enroll(s);
        errdefer x.dropStream(s);
        if (s.head_context != false) {
            // The target's prompt pass captures taps into the drafter's stage, so the lane must exist first.
            try x.drafter.open(s);
            x.lanes.getPtr(s).?.opened = true;
            if (x.target.vtable.prefill_features) |streamed| // chunk by chunk: no whole prompt's taps held
                return streamed(x.target.ptr, s, x.drafter.taps(), .{ .ptr = x, .take = take });
        }
        try x.target.prefill(s);
        if (s.head_context == false) return;
        const ids = s.prompt();
        const from: usize = s.cached;
        if (ids.len < from + 2) return;
        const n: u32 = @intCast(ids.len - 1 - from);
        const f = try x.target.features(s, x.drafter.taps(), from, n);
        try x.drafter.absorb(&.{.{ .stream = s, .features = f, .start = from, .follow = ids[from + 1 ..] }});
    }

    /// A step of the target's prompt pass, then the drafter absorbs its rows (as `take`); it opens at the first step.
    fn prefillStep(ptr: *anyopaque, s: *Stream, max_rows: u32) anyerror!bool {
        const x = self(ptr);
        const step = x.target.vtable.prefill_step.?;
        try x.enroll(s);
        errdefer x.dropStream(s);
        if (s.head_context == false) return step(x.target.ptr, s, max_rows);
        // The first step's target forward captures into the drafter's stage, so the lane must exist first.
        if (!x.lanes.getPtr(s).?.opened) {
            try x.drafter.open(s);
            x.lanes.getPtr(s).?.opened = true;
        }
        const done = try step(x.target.ptr, s, max_rows);
        const lane = x.lanes.getPtr(s).?;
        const from = lane.stepped orelse s.cached; // a resumed pass starts at its kept state
        const to: u64 = if (done) s.prompt().len else s.fill_at.load(.monotonic);
        lane.stepped = if (done) null else to;
        if (to > from) try take(x, s, try x.target.features(s, x.drafter.taps(), from, @intCast(to - from)), from);
        return done;
    }

    /// A prompt chunk's rows but the prompt's last (absorbed with the first token, as `prefill` leaves it).
    fn take(ptr: *anyopaque, s: *Stream, f: be.Features, start: u64) anyerror!void {
        const x = self(ptr);
        const ids = s.prompt();
        const last = ids.len - 1;
        if (start >= last) return;
        var rows = f;
        rows.rows = @intCast(@min(f.rows, last - start));
        try x.drafter.absorb(&.{.{ .stream = s, .features = rows, .start = start, .follow = ids[@intCast(start + 1)..][0..rows.rows] }});
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        return self(ptr).target.first(s, position);
    }

    fn firstRow(ptr: *anyopaque, s: *Stream) anyerror!@import("logprob.zig").Row {
        return self(ptr).target.firstRow(s);
    }

    fn firstMasked(ptr: *anyopaque, s: *Stream, position: u64, mask: []const u32) anyerror!u64 {
        const t = self(ptr).target;
        const masked = t.vtable.first_masked orelse return error.StructuresUnsupported;
        return masked(t.ptr, s, position, mask);
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        return self(ptr).target.queue(s, feed, position);
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).target.read(handle);
    }

    /// Held drafts reach the target as host tokens, all streams read back in one call (fillers: the pending token).
    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const x = self(ptr);
        x.translated = false;
        // a window carrying the stream's grown tree (allocation may keep only its first nodes): how many it offers
        if (x.heads > 0) for (windows) |w| if (x.openedLane(w.stream)) |lane| {
            const head = lane.tree_first orelse continue;
            lane.offered = if (w.held == 0 and w.tokens.len > 0 and w.tokens[0] == head) @intCast(w.tokens.len) else 0;
            lane.tree_first = null;
        };
        var total: usize = 0;
        var through = x.held_on_target;
        for (windows) |w| {
            if (w.held > 0 and w.tokens.len != 0) return error.MixedDrafts;
            total += w.held;
            if (through and w.held > 0 and (x.openedLane(w.stream) orelse return error.NotDrafting).filler > 0) through = false;
        }
        if (total == 0 or through) return x.target.verify(windows, out);
        x.windows.clearRetainingCapacity();
        try x.windows.appendSlice(x.gpa, windows);
        x.translated = true;
        try x.tokens.resize(x.gpa, total);
        x.streams.clearRetainingCapacity();
        x.outs.clearRetainingCapacity();
        var at: usize = 0;
        for (windows, x.windows.items) |w, *v| {
            if (w.held == 0) continue;
            const lane = x.openedLane(w.stream) orelse return error.NotDrafting;
            const slot = x.tokens.items[at..][0..w.held];
            at += w.held;
            v.tokens = slot;
            v.held = 0;
            if (lane.filler > 0) {
                @memset(slot, w.pending);
                lane.filler = 0;
                continue;
            }
            try x.streams.append(x.gpa, w.stream);
            try x.outs.append(x.gpa, slot);
        }
        if (x.streams.items.len > 0) try x.drafter.held(x.streams.items, x.outs.items);
        return x.target.verify(x.windows.items, out);
    }

    /// One stream's held drafts as host tokens for a grammar to walk (fillers: the pending token, kept for the verify).
    fn held(ptr: *anyopaque, s: *Stream, out: []u32) anyerror!void {
        const x = self(ptr);
        const lane = x.openedLane(s) orelse return error.NotDrafting;
        if (lane.filler > 0) {
            @memset(out, s.pending.?);
            return;
        }
        try x.drafter.held(&.{s}, &.{out});
    }

    /// The target keeps from the windows it verified (host tokens): the round loop's, position for position.
    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const x = self(ptr);
        if (!x.translated) return x.target.keep(windows, paths);
        if (windows.len != x.windows.items.len) return error.WindowMismatch;
        for (windows, x.windows.items) |w, v| if (w.stream != v.stream) return error.WindowMismatch;
        return x.target.keep(x.windows.items, paths);
    }

    /// Stage each unprepared request once and absorb its rows, retaining pending tokens for the subsequent hold.
    fn prepareDraft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const x = self(ptr);
        try x.stage(requests);
        if (x.absorbs.items.len > 0) try x.drafter.absorb(x.absorbs.items);
    }

    /// Each unprepared request's rows to absorb, and its pending token in `firsts`.
    fn stage(x: *Drafted, requests: []const be.DraftRequest) !void {
        x.absorbs.clearRetainingCapacity();
        try x.firsts.resize(x.gpa, requests.len);
        for (requests, x.firsts.items) |r, *tok| {
            if (r.prepared) {
                if (r.rows) |rows| tok.* = r.follow[rows.len - 1];
                continue;
            }
            if (r.lanes != null and !x.drafter.facts().trees) return error.TreesNotBuilt;
            const lane = x.openedLane(r.stream) orelse return error.NotDrafting;
            lane.filler = 0; // a new request supersedes fillers a copy round left unused
            var pending: u32 = undefined;
            if (r.rows) |rows| {
                // a tree's kept path names the window's rows out of order: the features of all up to its last
                var prefix = true;
                for (rows, 0..) |row, i| prefix = prefix and row == i;
                const n: u32 = @intCast(if (prefix) rows.len else rows[rows.len - 1] + 1);
                const f = try x.target.features(r.stream, x.drafter.taps(), r.start, n);
                try x.absorbs.append(x.gpa, .{ .stream = r.stream, .features = f, .start = r.start, .follow = r.follow[0..rows.len], .rows = if (prefix) null else rows, .prompt = false, .propose = r.depth > 0, .offered = lane.offered });
                lane.offered = 0;
                pending = r.follow[rows.len - 1];
                lane.rows = true;
            } else {
                const feed = r.first orelse return error.NoFirstToken;
                tok.* = switch (feed) {
                    .handle => |h| try x.target.read(h),
                    .value => |v| v,
                };
                pending = tok.*;
                const last = r.stream.prompt_len - 1;
                if (r.stream.cached <= last) { // the pass computed the prompt's last row (not restored whole)
                    // right after this stream's own prompt pass: a target keeping one pass's taps refuses another's
                    const f = try x.target.features(r.stream, x.drafter.taps(), last, 1);
                    try x.absorbs.append(x.gpa, .{ .stream = r.stream, .features = f, .start = last, .follow = @as(*const [1]u32, tok), .propose = r.depth > 0 });
                    lane.rows = true;
                }
            }
            tok.* = pending;
        }
    }

    /// Prepared requests staged before funding; others stage, then absorb and hold (fused if the drafter can).
    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const x = self(ptr);
        try x.stage(requests);
        x.holds.clearRetainingCapacity();
        for (requests, x.firsts.items) |r, pending| {
            if (r.lanes != null and !x.drafter.facts().trees) return error.TreesNotBuilt;
            const lane = x.openedLane(r.stream) orelse return error.NotDrafting;
            lane.head_held = false;
            if (r.depth == 0) continue;
            if (!lane.rows) { // nothing to draft from yet: fillers this round, the drafter from the next
                lane.filler = r.depth;
                continue;
            }
            try x.holds.append(x.gpa, .{ .stream = r.stream, .pending = pending, .position = r.position, .depth = r.depth, .plan = r.plan, .lanes = r.lanes, .tokens = r.stream.context.items, .sampling = r.stream.sampling });
        }
        const fused = x.drafter.vtable.draft_round != null and (x.drafter.proposal_head == null or x.drafter.proposal_head.?.mode == .legacy);
        if (fused and (x.absorbs.items.len > 0 or x.holds.items.len > 0)) {
            try x.drafter.vtable.draft_round.?(x.drafter.ptr, x.absorbs.items, x.holds.items);
        } else {
            if (x.absorbs.items.len > 0) try x.drafter.absorb(x.absorbs.items);
            if (x.holds.items.len > 0) try x.drafter.hold(x.holds.items);
        }
        for (x.holds.items) |h| x.openedLane(h.stream).?.head_held = true;
    }

    /// A drafter-grown tree's node chances (false: none, so the rule's per-depth rates price it).
    fn probabilities(ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
        const x = self(ptr);
        if (x.openedLane(s) == null) return false;
        return x.drafter.chances(s, out);
    }

    /// A drafter-grown tree as host tokens and parents; null: the drafter holds a chain or nothing grown.
    fn tree(ptr: *anyopaque, s: *Stream, gpa: Allocator) anyerror!?Held {
        const x = self(ptr);
        const lane = x.openedLane(s) orelse return null;
        if (lane.filler > 0) return null;
        const grown = try x.drafter.tree(s, gpa) orelse return null;
        lane.tree_first = if (grown.tokens) |t| (if (t.len > 0) t[0] else null) else null;
        return grown;
    }

    /// A stream another driver decoded on the drafter's own state (a lone driver): the next draft absorbs and holds.
    pub fn adopt(x: *Drafted, s: *Stream) !void {
        try x.enroll(s);
        if (s.head_context == false) return;
        const lane = x.lanes.getPtr(s).?;
        if (!lane.opened) {
            try x.drafter.open(s);
            lane.opened = true;
        }
        lane.* = .{ .opened = true, .rows = true };
    }

    fn dropStream(x: *Drafted, s: *Stream) void {
        if (x.lanes.fetchRemove(s)) |entry| {
            x.heads -= @intFromBool(s.head_context != false);
            if (entry.value.opened) x.drafter.release(s);
            s.head_context = null;
            s.head_prior = &.{};
        }
        x.target.release(s);
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        self(ptr).dropStream(s);
    }
};

test "draft_round fuses absorb and hold" {
    const gpa = std.testing.allocator;
    const Mock = struct {
        rounds: u32 = 0,
        absorbs: u32 = 0,
        holds: u32 = 0,
        rows: [4]u32 = .{ 1, 2, 3, 4 },
        fn cast(ptr: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ptr));
        }
        fn taps(_: *anyopaque) []const u32 {
            return &.{0};
        }
        fn facts(_: *anyopaque) dr.Facts {
            return .{ .depth = 2, .step_ms = 0.5 };
        }
        fn open(_: *anyopaque, _: *Stream) anyerror!void {}
        fn absorb(ptr: *anyopaque, _: []const dr.Absorb) anyerror!void {
            cast(ptr).absorbs += 1;
        }
        fn hold(ptr: *anyopaque, _: []const dr.Hold) anyerror!void {
            cast(ptr).holds += 1;
        }
        fn draftRound(ptr: *anyopaque, absorbs: []const dr.Absorb, holds: []const dr.Hold) anyerror!void {
            const m = cast(ptr);
            m.rounds += 1;
            try std.testing.expectEqual(@as(usize, 1), absorbs.len);
            try std.testing.expectEqual(@as(usize, 1), holds.len);
        }
        fn held(_: *anyopaque, _: []const *Stream, _: []const []u32) anyerror!void {}
        fn release(_: *anyopaque, _: *Stream) void {}
        fn prefill(_: *anyopaque, _: *Stream) anyerror!void {}
        fn prefillFeatures(ptr: *anyopaque, s: *Stream, _: []const u32, sink: be.FeatureSink) anyerror!void {
            const m = cast(ptr);
            try sink.take(sink.ptr, s, .{ .buffer = @intFromPtr(&m.rows), .rows = 4, .row_bytes = 4, .space = .host, .dtype = .u32 }, 0);
        }
        fn features(ptr: *anyopaque, _: *Stream, _: []const u32, _: u64, n: u32) anyerror!be.Features {
            return .{ .buffer = @intFromPtr(&cast(ptr).rows), .rows = n, .row_bytes = 4, .space = .host, .dtype = .u32 };
        }
        fn first(_: *anyopaque, _: *Stream, _: u64) anyerror!u64 {
            return 0;
        }
        fn queue(_: *anyopaque, _: *Stream, _: be.Feed, _: u64) anyerror!u64 {
            return 0;
        }
        fn read(_: *anyopaque, _: u64) anyerror!u32 {
            return 7;
        }
        fn verify(_: *anyopaque, _: []const be.Window, _: []be.Verified) anyerror!void {}
        fn keep(_: *anyopaque, _: []const be.Window, _: []const []const u32) anyerror!void {}
        fn draftFn(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {}
    };
    var mock: Mock = .{};
    const target: be.Backend = .{ .ptr = &mock, .vtable = &.{ .prefill = Mock.prefill, .prefill_features = Mock.prefillFeatures, .features = Mock.features, .first = Mock.first, .queue = Mock.queue, .read = Mock.read, .verify = Mock.verify, .keep = Mock.keep, .draft = Mock.draftFn, .release = Mock.release } };
    const drafter: dr.Drafter = .{ .ptr = &mock, .vtable = &.{ .taps = Mock.taps, .facts = Mock.facts, .open = Mock.open, .absorb = Mock.absorb, .hold = Mock.hold, .draft_round = Mock.draftRound, .held = Mock.held, .release = Mock.release } };
    var x = try Drafted.init(gpa, target, drafter);
    defer x.deinit();
    var s = try Stream.init(gpa, .{ .id = "dr", .prompt = &.{ 10, 20, 30, 40 }, .max_new = 4 });
    defer s.deinit(gpa);
    try x.backend().prefill(&s);
    defer x.backend().release(&s);
    try std.testing.expectEqual(@as(u32, 1), mock.absorbs);
    try x.backend().draft(&.{.{ .stream = &s, .follow = &.{}, .first = .{ .value = 50 }, .rows = null, .start = 4, .position = 5, .depth = 2 }});
    try std.testing.expectEqual(@as(u32, 1), mock.rounds);
    try std.testing.expectEqual(@as(u32, 1), mock.absorbs);
    try std.testing.expectEqual(@as(u32, 0), mock.holds);
}
