//! Living Weights' trainer for Flash Next on CUDA: sequences through the batched pass, losses, gradients to the nudges.
const std = @import("std");
const cuda = @import("cuda");
const living = @import("living");
const prompt = @import("cuda_prompt.zig");
const wt = @import("cuda_prompt_wt.zig");
const gdn = @import("cuda_gdn.zig");
const fb = @import("cuda_fn_back.zig");
const buf = @import("cuda_train_buf.zig");
const spans_ = @import("cuda_batch_train.zig");

const batch = prompt.batch;
const train_mode = living.train_mode;
const lesson_route = living.lesson_route;
const Count = living.pocket_clock.Count;

pub const Mode = train_mode.Mode;
pub const Result = train_mode.Result;
pub const Seq = train_mode.Seq;
pub const Span = train_mode.Span;
pub const Answer = train_mode.Answer;
pub const Turn = lesson_route.Turn;

const layers_n = wt.layers_n;
const dims = wt.dims;
const wide = wt.wide;
const streams = wt.streams;
const vocab = wt.vocab;
const slots = wt.slots;
const top_k = slots - 1;
const moe_w = wt.moe_w;
const experts_n = wt.experts_n;
const q_heads = wt.q_heads;
const kv_heads = wt.kv_heads;
const head_dim = wt.head_dim;
const out_k = wt.out_k;

pub const max_rows = buf.max_rows;
pub const pack_rows = buf.pack_rows;
pub const max_seqs = buf.max_seqs;
pub const keep_low = buf.keep_low;
pub const memory = buf.memory;
const Mapped = buf.Mapped;

fn up(n: usize, by: usize) usize {
    return (n + by - 1) / by * by;
}

/// The trainer over the engine `e`, its glue launched through `Tri` (our own kernels: a captured set keeps no up rows).
pub fn Trainer(comptime Tri: type) type {
    return struct {
        const Self = @This();

        gpa: std.mem.Allocator,
        e: *prompt.Engine,
        tri: Tri,
        seq: prompt.Seq,
        k: fb.Kernels,
        gk: cuda.gdn_train.Kernels,
        b: buf.Buffers = undefined,
        // host-mapped
        host: Mapped = undefined, // targets u32, weights f32, stats [2]f32, scales f32: [pack_rows] each
        inputs: Mapped = undefined, // bf16 [layers, pack_rows, D]: each layer's MoE input as the last keys or target pass read it
        gates: Mapped = undefined, // f32 [layers, pack_rows, slots]: each layer's routing weights, the shared gate last
        delta: Mapped = undefined, // f32 [max_seqs, 3, layers, D]: each sequence's nudges after each layer it names
        push: Mapped = undefined, // f32 [max_seqs, 3, layers, D]: the loss's gradient at each nudged row after each of them
        resid: [dims]u16 = undefined,
        // the learner's view
        spans: [max_seqs]Span = undefined,
        answers: [max_seqs]Answer = undefined,
        seqs: [max_seqs]Seq = undefined,
        nseq: usize = 0,
        rows: usize = 0,
        pack: usize = pack_rows - max_seqs, // rows a pass may hold: room left for the rows between sequences
        count: Count = .{},
        prefixes: struct {
            pub const most = 0; // no prompt prefix is run once and kept: every pass runs its sequences whole
        } = .{},
        nudge: train_mode.Nudge = .{},
        /// A check's loss pass stops at the scores (`scores`), before their gradient replaces them.
        scores_only: bool = false,
        from: [living.family.max_layers]u32 = @splat(std.math.maxInt(u32)),

        pub fn init(gpa: std.mem.Allocator, io: std.Io, e: *prompt.Engine, tri: Tri) !*Self {
            const t = try gpa.create(Self);
            errdefer gpa.destroy(t);
            t.* = .{ .gpa = gpa, .e = e, .tri = tri, .seq = undefined, .k = undefined, .gk = undefined };
            const d = e.driver;
            t.k = try fb.Kernels.load(d);
            errdefer t.k.deinit();
            t.gk = try cuda.gdn_train.Kernels.load(d);
            errdefer t.gk.deinit();
            t.seq = try e.newSeq(max_rows);
            errdefer t.seq.deinit(gpa);
            _ = try e.reserve(pack_rows);
            const t0 = std.Io.Clock.awake.now(io);
            t.b = try buf.Buffers.init(gpa, io, e);
            errdefer t.b.deinit();
            t.host = try Mapped.init(d, pack_rows * 20);
            errdefer t.host.deinit();
            t.inputs = try Mapped.init(d, layers_n * pack_rows * dims * 2);
            errdefer t.inputs.deinit();
            t.gates = try Mapped.init(d, layers_n * pack_rows * slots * 4);
            errdefer t.gates.deinit();
            t.delta = try Mapped.init(d, max_seqs * 3 * layers_n * dims * 4);
            errdefer t.delta.deinit();
            t.push = try Mapped.init(d, max_seqs * 3 * layers_n * dims * 4);
            errdefer t.push.deinit();
            try e.stream.synchronize();
            const t1 = std.Io.Clock.awake.now(io);
            std.log.info("slide: the trainer's MLX copies and buffers made in {d:.1} s, {d:.2} GiB", .{ seconds(t0, t1), @as(f64, @floatFromInt(t.b.bytes())) / (1 << 30) });
            return t;
        }

        fn seconds(a: std.Io.Timestamp, b: std.Io.Timestamp) f64 {
            return @as(f64, @floatFromInt(b.toNanoseconds() - a.toNanoseconds())) / 1e9;
        }

        pub fn drain(t: *Self) void {
            @import("qwen_vision").completion.drain(t.e.driver, t.e.stream.*);
        }

        pub fn deinit(t: *Self) void {
            const gpa = t.gpa;
            t.drain();
            inline for (.{ "host", "inputs", "gates", "delta", "push" }) |f| @field(t, f).deinit();
            t.b.deinit();
            t.seq.deinit(gpa);
            t.gk.deinit();
            t.k.deinit();
            gpa.destroy(t);
        }

        fn ops(t: *const Self) fb.Ops {
            return .{ .k = &t.k, .s = t.e.stream.* };
        }

        fn gops(t: *const Self) cuda.gdn_train.Ops {
            return .{ .k = &t.gk, .s = t.e.stream.* };
        }

        // ---- the learner's hooks ----------------------------------------------------------------------------------

        fn targets(t: *const Self) []u32 {
            return t.host.slice(u32, pack_rows);
        }

        fn weights(t: *const Self) []f32 {
            return t.host.slice(f32, 2 * pack_rows)[pack_rows..];
        }

        fn stats(t: *const Self) [][2]f32 {
            return t.host.slice([2]f32, 2 * pack_rows)[pack_rows..];
        }

        fn scales(t: *const Self) []f32 {
            return t.host.slice(f32, 5 * pack_rows)[4 * pack_rows ..];
        }

        /// Layer l's MoE input (what the shared expert and the pocket read) at the pass's `row`, bf16, as last read.
        pub fn normed(t: *const Self, l: usize, row: usize) []const u16 {
            return t.inputs.slice(u16, layers_n * pack_rows * dims)[(l * pack_rows + row) * dims ..][0..dims];
        }

        /// The shared expert's gate (sigmoid of its router logit) at layer l, the pass's `row`.
        pub fn gateAt(t: *const Self, l: usize, row: usize) f32 {
            return t.gates.slice(f32, layers_n * pack_rows * slots)[(l * pack_rows + row) * slots + slots - 1];
        }

        /// The streams after layer l - 1's write-back at `row`, in its MoE output's units (sum inj h / sum inj^2).
        pub fn residual(t: *Self, l: usize, row: usize) []const u16 {
            @memset(&t.resid, 0);
            if (l == 0 or l > layers_n or l - 1 < keep_low) return &t.resid;
            const kp = t.b.full;
            const h_at = if (l < layers_n) kp.layers[l].attn_hc.h else kp.mixer.h;
            if (h_at == 0) return &t.resid;
            var h: [wide]u16 = undefined;
            var inj: [streams]u16 = undefined;
            const d = t.e.driver;
            t.e.stream.synchronize() catch return &t.resid;
            d.check(d.api.cuMemcpyDtoH_v2(&h, h_at + row * wide * 2, wide * 2), "cuMemcpyDtoH") catch return &t.resid;
            d.check(d.api.cuMemcpyDtoH_v2(&inj, kp.layers[l - 1].mlp_hc.inject + row * streams * 2, streams * 2), "cuMemcpyDtoH") catch return &t.resid;
            var a: [streams]f32 = undefined;
            var norm: f32 = 0;
            for (&a, inj) |*x, b| {
                x.* = wt.promote(b);
                norm += x.* * x.*;
            }
            if (norm == 0) return &t.resid;
            for (0..dims) |i| {
                var v: f32 = 0;
                for (0..streams) |s| v += a[s] * wt.promote(h[s * dims + i]);
                t.resid[i] = wt.toBf16(v / norm);
            }
            return &t.resid;
        }

        fn vec(s: usize, i: usize, l: usize) usize {
            return ((s * 3 + i) * layers_n + l) * dims;
        }

        /// The loss's gradient sequence s's target pass read at its key i's row after layer l (dL/d the MoE output).
        pub fn pushAt(t: *const Self, s: usize, i: usize, l: usize) []const f32 {
            return t.push.slice(f32, max_seqs * 3 * layers_n * dims)[vec(s, i, l)..][0..dims];
        }

        /// Sequence s's nudge for key i after layer l, which its target pass adds to that layer's MoE output.
        pub fn deltaAt(t: *const Self, s: usize, i: usize, l: usize) []f32 {
            return t.delta.slice(f32, max_seqs * 3 * layers_n * dims)[vec(s, i, l)..][0..dims];
        }

        /// Whether a pass may hold several sequences: always (each runs from its own start).
        pub fn packs(_: *const Self) bool {
            return true;
        }

        /// No prompt prefix is kept (prefixes.most is 0): nothing to make room for.
        pub fn prefixSlots(_: *Self, n: usize) !void {
            if (n > 0) return error.TooManyPrefixes;
        }

        pub fn step(t: *Self, ids: []const u32, start: usize, mode: Mode, turn: lesson_route.Turn) !Result {
            var out: [1]Result = undefined;
            try t.pass(&.{.{ .ids = ids, .start = start, .decide = turn.decide, .nudge = t.nudge }}, mode, turn, &out);
            return out[0];
        }

        // ---- the pass ---------------------------------------------------------------------------------------------

        /// `seqs` side by side in a pass: keys (MoE inputs, gates), loss (each answer's), target (and dL/d the nudges).
        pub fn pass(t: *Self, seqs: []const Seq, mode: Mode, turn: lesson_route.Turn, out: []Result) !void {
            _ = turn;
            if (seqs.len == 0 or seqs.len > max_seqs or out.len < seqs.len) return error.BadPass;
            switch (mode) {
                .keys, .loss, .target => {},
                .prime => {
                    for (out[0..seqs.len]) |*r| r.* = .{ .loss = 0, .recalled = false };
                    return;
                },
                else => return error.RoutedUnsupported,
            }
            const e = t.e;
            const d = e.driver;
            const stream = e.stream.*;
            var rows: usize = 0;
            for (seqs, t.spans[0..seqs.len]) |sq, *sp| {
                if (sq.ids.len < 2 or sq.start < 1 or sq.start >= sq.ids.len or sq.ids.len - 1 > max_rows) return error.BadExample;
                if (sq.skip != 0) return error.BadPrefix;
                rows = up(rows, 2);
                sp.* = .{ .at = rows, .rows = sq.ids.len - 1 };
                rows += sp.rows;
            }
            if (rows > pack_rows) return error.PassTooLong;
            const spans = t.spans[0..seqs.len];
            @memcpy(t.seqs[0..seqs.len], seqs);
            t.nseq = seqs.len;
            t.rows = rows;
            t.count = .{ .passes = t.count.passes + 1, .rows = t.count.rows + rows };
            try stream.synchronize(); // the host rows below are memory the last pass's kernels may still read
            const ids = try t.gpa.alloc(u32, rows);
            defer t.gpa.free(ids);
            @memset(ids, @intCast(e.eos));
            @memset(t.scales()[0..rows], 1);
            @memset(t.targets()[0..rows], 0);
            @memset(t.weights()[0..rows], 0);
            _ = train_mode.answers(seqs, spans, t.answers[0..seqs.len]);
            for (seqs, spans, t.answers[0..seqs.len]) |sq, sp, a| {
                @memcpy(ids[sp.at..][0..sp.rows], sq.ids[0..sp.rows]);
                @memset(t.scales()[sp.at..][0..sp.rows], sq.scale);
                const share = 1 / @as(f32, @floatFromInt(a.rows));
                for (0..a.rows) |r| {
                    t.targets()[a.first + r] = sq.ids[sq.start + r];
                    t.weights()[a.first + r] = share;
                }
            }
            var low: usize = layers_n;
            if (mode == .target) {
                var mask: u64 = 0;
                for (seqs) |sq| mask |= sq.nudge.layers[0] | sq.nudge.layers[1] | sq.nudge.layers[2];
                if (mask != 0) low = @ctz(mask);
                if (low < keep_low) return error.NudgeBelowKept;
            }
            // the forward
            var keep = t.b.light;
            if (mode == .target) {
                // from the lowest nudge up; with none, every layer a host's residual reads
                keep = t.b.full;
                for (0..if (low == layers_n) keep_low else low) |l| keep.layers[l] = t.b.light.layers[l];
            }
            // the pockets' layers and the nudged ones go through the fp32 sum; the others write back as serving does
            var adds = e.pockets.mask();
            for (seqs) |sq| adds |= sq.nudge.layers[0] | sq.nudge.layers[1] | sq.nudge.layers[2];
            const s = &e.batch_s.?;
            var p: batch.Pass(Tri) = .{
                .gpa = t.gpa,
                .io = e.io,
                .stream = stream,
                .tri = t.tri,
                .k = &e.batch_k,
                .s = s,
                .m = e.model(),
                // a sequence of at most 512 rows never reaches the block indexer (past 2,048 keys): no block scores
                .seq = t.seq.view(0),
                .hooks = .{ .logits = if (mode == .keys) .none else .all, .logits_out = t.b.logits, .keep = &keep, .moe_add = .{ .ctx = t, .add = moeAdd, .layers = adds } },
            };
            var list: [max_seqs]batch.Span = undefined;
            for (spans, list[0..seqs.len]) |sp, *x| x.* = .{ .at = sp.at, .rows = sp.rows };
            try spans_.spans(Tri, &p, ids, list[0..seqs.len], e.eos);
            const o = t.ops();
            if (mode == .keys or mode == .target) {
                // each layer's MoE inputs and routing weights where the host reads them
                for (0..layers_n) |l| {
                    try d.check(d.api.cuMemcpyDtoDAsync_v2(t.inputs.dev + l * pack_rows * dims * 2, keep.layers[l].moe_x, rows * dims * 2, stream.handle), "cuMemcpyDtoDAsync");
                    try d.check(d.api.cuMemcpyDtoDAsync_v2(t.gates.dev + l * pack_rows * slots * 4, keep.layers[l].moe_wts, rows * slots * 4, stream.handle), "cuMemcpyDtoDAsync");
                }
            }
            if (mode == .keys or t.scores_only) {
                try stream.synchronize();
                for (out[0..seqs.len]) |*r| r.* = .{ .loss = 0, .recalled = false };
                return;
            }
            // every row's softmax: the answer rows' (p - onehot) / their count, every other row's zero
            const hd = t.host.dev;
            try o.softmax(t.b.logits, hd, hd + pack_rows * 4, hd + pack_rows * 8, rows, vocab);
            if (mode == .target and low < layers_n) try t.backward(&keep, rows, low);
            try stream.synchronize();
            for (out[0..seqs.len], t.answers[0..seqs.len]) |*r, a| {
                r.* = .{ .loss = 0, .recalled = true };
                for (t.stats()[a.first..][0..a.rows], t.weights()[a.first..][0..a.rows]) |st, w| {
                    r.loss += st[0] * w;
                    r.recalled = r.recalled and st[1] > 0.5;
                }
            }
        }

        /// Sequence `sq`'s rows' scores into b.logits (bf16 [rows, vocab]) as a loss pass makes them, before softmax.
        pub fn scores(t: *Self, sq: Seq) !void {
            t.scores_only = true;
            defer t.scores_only = false;
            var r: [1]Result = undefined;
            try t.pass(&.{sq}, .loss, .{ .decide = 0, .bits = .empty, .whole = true }, &r);
        }

        /// The pass's adds: each layer's pocket (the open fact's neurons scaled by each row's sequence), then nudges.
        fn moeAdd(ctx: *anyopaque, layer: usize, x: u64, ids: u64, wts: u64, out: u64, rows: usize) anyerror!void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            const o = t.ops();
            if (t.e.pocketOf(layer)) |pk0| {
                var pk = pk0;
                pk.scale = t.host.dev + 4 * pack_rows * 4;
                pk.from = t.from[layer];
                try o.pocketRows(pk, .{ .x = x, .ids = ids, .wts = wts }, out, true, rows, dims, top_k);
            }
            for (t.seqs[0..t.nseq], t.spans[0..t.nseq], 0..) |sq, sp, si| for (sq.nudge.layers, sq.nudge.rows, 0..) |mask, row, i| {
                if (mask >> @intCast(layer) & 1 == 0) continue;
                try o.axpy(out + sp.row(row) * dims * 4, t.delta.dev + vec(si, i, layer) * 4, dims);
            };
        }

        fn copyPush(t: *Self, l: usize) !void {
            const d = t.e.driver;
            for (t.seqs[0..t.nseq], t.spans[0..t.nseq], 0..) |sq, sp, si| for (sq.nudge.layers, sq.nudge.rows, 0..) |mask, row, i| {
                if (mask >> @intCast(l) & 1 == 0) continue;
                try d.check(d.api.cuMemcpyDtoDAsync_v2(t.push.dev + vec(si, i, l) * 4, t.b.dbranch + sp.row(row) * dims * 4, dims * 4, t.e.stream.handle), "cuMemcpyDtoDAsync");
            };
        }

        /// The loss back from the head to layer `low`'s MoE output, its nudged rows' gradients into `push` on the way.
        fn backward(t: *Self, keep: *const batch.Keep, rows: usize, low: usize) !void {
            const o = t.ops();
            const e = t.e;
            // the head (its bf16 rows' gradient in place) back to the final mixer's output, then its read-out
            try o.headBack(t.b.logits, t.b.head, t.b.dmix, rows, t.b.parts);
            try o.zero(t.b.g, rows * wide);
            try o.hcReadoutBack(t.b.mixer, hcKept(keep.mixer), t.b.dmix, 0, t.b.g, t.b.hs, rows);
            var l = layers_n;
            while (l > low) {
                l -= 1;
                const kp = keep.layers[l];
                const w = t.b.w[l];
                // the MoE write-back h' = h + moe inject: its output's gradient, and its inject gates'
                try o.hcBranchBack(w.mlp_hc, t.b.g, kp.mlp_hc.inject, kp.moe_branch, t.b.dbranch, t.b.dinj, rows);
                try t.copyPush(l);
                if (l == low) return;
                try o.zero(t.b.dmix, rows * dims);
                var pk = e.pocketOf(l);
                if (pk) |*x| {
                    x.scale = t.host.dev + 4 * pack_rows * 4;
                    x.from = t.from[l];
                }
                const lw = &e.layers[l];
                try o.moeBackPacked(.{ .up = lw.experts_up.ptr, .down = lw.experts_down.ptr, .routed = experts_n, .width = moe_w, .dim = dims }, lw.router.ptr, .{ .x = kp.moe_x, .ids = kp.moe_ids, .wts = kp.moe_wts }, pk, t.b.dbranch, t.b.dmix, t.b.ms, rows, top_k);
                try o.hcReadoutBack(w.mlp_hc, hcKept(kp.mlp_hc), t.b.dmix, t.b.dinj, t.b.g, t.b.hs, rows);
                // the mixer's write-back, the mixer one sequence at a time, its read-out
                try o.hcBranchBack(w.attn_hc, t.b.g, kp.attn_hc.inject, kp.mix_branch, t.b.dbranch, t.b.dinj, rows);
                try o.zero(t.b.dmix, rows * dims);
                for (t.spans[0..t.nseq]) |sp| {
                    const gy = t.b.dbranch + sp.at * dims * 4;
                    const dx = t.b.dmix + sp.at * dims * 4;
                    if (w.gdn) |g| {
                        try t.gops().backward(g, t.b.gs, kp.proj + sp.at * gdn.proj_width * 2, null, null, gy, dx, sp.rows, false);
                    } else {
                        var at = w.attn.?;
                        at.inv = e.batch_s.?.inv; // RoPE's frequencies in the pass's scratch, wherever it now sits
                        const kept: fb.Ops.AttnKept = .{ .pq = kp.pq + sp.at * q_heads * 2 * head_dim * 2, .pk = kp.pk + sp.at * kv_heads * head_dim * 2, .pv = kp.pv + sp.at * kv_heads * head_dim * 2, .gated = kp.gated + sp.at * out_k * 2, .pos0 = 0 };
                        try o.attnBack(at, kept, gy, dx, t.b.as, sp.rows, false);
                    }
                }
                try o.hcReadoutBack(w.attn_hc, hcKept(kp.attn_hc), t.b.dmix, t.b.dinj, t.b.g, t.b.hs, rows);
            }
        }

        fn hcKept(k: batch.HcKeep) fb.Ops.HcKept {
            return .{ .h = k.h, .normed = k.normed, .v = k.v, .up = k.up, .inject = k.inject };
        }
    };
}
