//! The XPU Qwen27 DFlash2 drafter behind the core Drafter VTable: lanes.drafted.Drafted puts it in front of the lanes.
const std = @import("std");
const lanes = @import("lanes");
const dr = lanes.drafter;
const Stream = lanes.Stream;
const Held = lanes.stream.Held;
const Allocator = std.mem.Allocator;

pub const TREE_MIN = 5; // drafts from which a stream's window is a DFlash2 tree (below: its chain)

/// The round loop's facts through Drafted as the lane target gave them alone (its own rule shares a round's rows).
pub fn facts(wrap: *const lanes.drafted.Drafted, target: lanes.Model) lanes.Model {
    var m = wrap.facts(target);
    m.allocate_shared_drafts = target.allocate_shared_drafts;
    return m;
}

/// `M`: the drafter's module (Drafter, Block, Absorb, TOPK, WIN); `H`: the lane target it reads slots and taps from.
pub fn Adapter(comptime M: type, comptime H: type) type {
    return struct {
        const Self = @This();
        d: *M.Drafter,
        host: *H,
        chain: [H.max_slots][16]u32 = undefined, // each slot's drafts for its next window (a chain or a tree)
        chain_par: [H.max_slots][16]i32 = undefined, // and each one's parent (-1: the pending token)
        chain_n: [H.max_slots]u32 = @splat(0),

        pub fn drafter(a: *Self) dr.Drafter {
            return .{ .kind = "DFlash2", .ptr = a, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn, .tree = treeFn } };
        }

        fn cast(ptr: *anyopaque) *Self {
            return @ptrCast(@alignCast(ptr));
        }

        fn slot(a: *const Self, s: *const Stream) !u32 {
            return a.host.slotOf(s) orelse error.UnknownStream;
        }

        fn tapsFn(_: *anyopaque) []const u32 {
            return &H.tap_layers;
        }

        /// The lane target's own drafting facts before the move: a block's drafts, every stream in one batch.
        fn factsFn(ptr: *anyopaque) dr.Facts {
            const a = cast(ptr);
            return .{ .depth = a.host.block - 1, .step_ms = H.draft_step_ms, .plain_guard = false, .batched = true };
        }

        /// The slot's context is the target's: its prompt pass resets it or restores a kept one.
        fn openFn(_: *anyopaque, _: *Stream) anyerror!void {}

        fn releaseFn(ptr: *anyopaque, s: *Stream) void {
            const a = cast(ptr);
            if (a.host.slotOf(s)) |sl| a.chain_n[sl] = 0;
        }

        /// Prompt rows a call at a time as the prompt pass made them, kept rows of a round in one batched pass.
        fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
            const a = cast(ptr);
            var multi: [16]M.Absorb = undefined;
            var n: usize = 0;
            for (items) |it| {
                const sl = try a.slot(it.stream);
                a.chain_n[sl] = 0;
                const f = it.features;
                if (f.space != .device or f.dtype != .bf16 or f.row_bytes != H.tap_row) return error.DraftFeaturesUnsupported;
                const taps: *anyopaque = @ptrFromInt(f.buffer + f.offset);
                if (it.prompt) {
                    try a.prompt(sl, it.stream, it.start, f.rows, taps);
                    continue;
                }
                if (n == multi.len) return error.TooManyBlocks;
                multi[n] = .{ .slot = sl, .taps = taps, .n = if (it.rows) |r| @intCast(r.len) else f.rows, .rows = it.rows };
                n += 1;
            }
            try a.d.absorbMulti(multi[0..n]);
        }

        /// Prompt rows [from, from + rows); the last prompt row goes with its chunk, rows further than WIN are skipped.
        fn prompt(a: *Self, sl: u32, s: *const Stream, from: u64, rows: u32, taps: *anyopaque) !void {
            const d = a.d;
            try d.use(sl);
            const len: u64 = s.prompt_len;
            var n: u64 = rows;
            if (from + n <= d.end()) return; // the prompt's last row, taken with its chunk
            if (from != d.end()) return error.DraftContextGap;
            if (from + n + 1 == len) if (a.host.promptTail(sl)) |t| {
                if (t.start <= from and t.start + t.rows == len) n += 1; // that chunk's taps hold the last row too
            };
            const k = @min(n, (len -| M.WIN) -| from);
            d.skip(@intCast(k));
            if (k < n) try d.addTaps(@ptrFromInt(@intFromPtr(taps) + k * H.tap_row), @intCast(n - k));
        }

        /// Every stream's lattice in one drafter pass, then its chain (few drafts) or tree, keyed if sampled.
        fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
            const a = cast(ptr);
            const d = a.d;
            const block = a.host.block;
            var blocks: [16]M.Block = undefined;
            var asked: [16]u32 = undefined;
            var holds: [16]*const dr.Hold = undefined;
            var ni: usize = 0;
            var depth: u32 = 0;
            for (items) |*it| {
                const sl = try a.slot(it.stream);
                a.chain_n[sl] = 0;
                if (it.depth == 0 or ni == blocks.len) continue;
                blocks[ni] = .{ .slot = sl, .pending = it.pending };
                asked[ni] = @min(it.depth, a.host.nodes);
                holds[ni] = it;
                depth = @max(depth, @min(asked[ni], block - 1));
                ni += 1;
            }
            if (ni == 0) return;
            for (items) |it| if (it.depth > 0) {
                try d.use(try a.slot(it.stream));
                if (d.end() + 1 != it.position) return error.BadDraftPosition;
            };
            try d.latticeMulti(blocks[0..ni], depth);
            for (blocks[0..ni], asked[0..ni], holds[0..ni], 0..) |b, n, h, i| {
                d.take(i);
                var noise: [16][M.TOPK]f64 = undefined;
                const sp = h.sampling;
                if (sp) |x| for (0..depth) |dd| for (0..M.TOPK) |j| {
                    noise[dd][j] = -@log(-@log(lanes.sampling.uniform(x.seed, h.position + dd, d.cand[dd][j])));
                };
                if (n < TREE_MIN) { // few drafts: a chain goes deeper than a tree of siblings
                    d.chain(b.pending, n, a.chain[b.slot][0..n]);
                    for (a.chain_par[b.slot][0..n], 0..) |*x, j| x.* = @as(i32, @intCast(j)) - 1;
                    a.chain_n[b.slot] = n;
                    continue;
                }
                a.chain_n[b.slot] = d.tree(b.pending, n, if (sp != null) &noise else null, if (sp) |x| x.temperature else 1.0, a.chain[b.slot][0..n], a.chain_par[b.slot][0..n]);
            }
        }

        fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
            const a = cast(ptr);
            for (streams, out) |s, o| {
                const sl = try a.slot(s);
                if (o.len > a.chain_n[sl]) return error.DraftHeldFewer;
                @memcpy(o, a.chain[sl][0..o.len]);
            }
        }

        /// A slot's held drafts as host tokens and parents (taken: the next window spends them).
        fn treeFn(ptr: *anyopaque, s: *Stream, gpa: Allocator) anyerror!?Held {
            const a = cast(ptr);
            const sl = a.host.slotOf(s) orelse return null;
            const n = a.chain_n[sl];
            a.chain_n[sl] = 0;
            const tokens = try gpa.dupe(u32, a.chain[sl][0..n]);
            errdefer gpa.free(tokens);
            const parents = try gpa.dupe(i32, a.chain_par[sl][0..n]);
            return .{ .count = n, .tokens = tokens, .parents = parents };
        }
    };
}
