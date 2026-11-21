//! Kolibri-1's published EAGLE-3 drafter: core's release checks, cuda/eagle3.zig's passes, the lane core's Drafted.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const be = @import("backend.zig");
const kern = @import("kernels.zig");
const wts = @import("weights.zig");
const row_costs = @import("row_costs.zig");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const release = core.drafter_release;
const published = core.drafter_published;
const E = cuda.eagle3;

/// The one checkpoint of model type kolibri1: the target a published release names.
pub const target_repo = "Aleph-Alpha/Kolibri-1";
const chain = release.most_chain;
/// Stream counts a hold pass is timed at (the lane core interpolates between them and extends past the widest).
const hold_streams = [_]u32{ 1, 2, 4, 8 };

/// The target the server loaded, as a release's adapter compares it.
pub fn adapter(c: @import("config.zig").Config) release.Adapter {
    return .{ .layers = @intCast(c.layers), .hidden = @intCast(c.hidden), .vocab = @intCast(c.vocab), .embed_scale = 1.0, .tied = false, .softcap = null, .head_bias = false };
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

const Lane = struct { slot: u32, pos: u64 = 0, token: u32 = 0, rows: bool = false, held: [chain]u32 = undefined, held_len: u32 = 0 };

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    kb: *be.Kolibri,
    act: cuda.act_rows.ActRows,
    e: E.Eagle3,
    vocab: []u32, // draft index -> target id
    taps: [8]u32 = undefined,
    n_taps: usize,
    depth: u32,
    step_ms: f64 = 0,
    costs: [row_costs.widths.len]lanes.config.Cost = undefined, // the target's round ms by rows (timed at reserve)
    holds: [2 * hold_streams.len]lanes.hold_price.HoldCost = undefined, // a hold pass's ms by streams and depth
    n_holds: usize = 0,
    timed: bool = false,
    lanes: std.AutoHashMapUnmanaged(*Stream, Lane) = .empty,
    free: []bool = &.{}, // drafter slot i is free (the last slot times a step and is never a stream's)
    layers: []E.Layer,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,

    /// The release in `dir` over `kb`'s model: refused by name (logged) if not served or past `left` device bytes.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, kb: *be.Kolibri, k: *const kern.Kernels, dir: []const u8, left: ?u64) !*Drafter {
        var folder = try std.Io.Dir.cwd().openDir(io, dir, .{});
        defer folder.close(io);
        var why: release.Why = .{};
        var r = published.Release.open(gpa, io, folder, .{ .repo = target_repo, .adapter = adapter(kb.m.c) }, &why) catch |e| {
            if (why.len > 0) std.log.err("--drafter {s}: {s}", .{ dir, why.text() });
            return e;
        };
        defer r.deinit(gpa);
        const c = &r.manifest.config;
        if (c.architecture != .eagle3) {
            std.log.err("--drafter {s}: a {s} release; Kolibri-1 on CUDA serves eagle3 releases", .{ dir, @tagName(c.architecture) });
            return error.DrafterArchitectureNotServed;
        }
        _ = admit(c, r.vocab.len, kb.m.slots, left) catch |e| {
            std.log.err("--drafter {s}: needs {d:.2} GiB of device memory for {d} stream{s} and a timing slot; the budget leaves {d:.2} GiB", .{ dir, gib(need(c, r.vocab.len, kb.m.slots)), kb.m.slots, if (kb.m.slots == 1) "" else "s", gib(left.?) });
            return e;
        };
        const x = try gpa.create(Drafter);
        errdefer gpa.destroy(x);
        x.* = .{ .gpa = gpa, .io = io, .kb = kb, .act = undefined, .e = undefined, .vocab = try gpa.dupe(u32, r.vocab), .n_taps = c.taps.ids.len, .depth = c.serving.depth, .layers = &.{} };
        errdefer gpa.free(x.vocab);
        for (c.taps.ids, 0..) |id, i| x.taps[i] = id - 1; // the format's "after id layers" is layer id - 1's output
        x.act = try cuda.act_rows.ActRows.load(kb.d);
        errdefer x.act.unload();
        errdefer x.freeWeights();
        const w = try x.upload(dir, c, r.names, k);
        const ops: E.Ops = .{ .dense = &k.dense, .norms = &k.norms, .qk = &k.qk, .win = &k.win, .embed = &k.embed, .act = &x.act, .argmax = k.argmax_fn, .to_bf16 = k.to_bf16_fn };
        x.e = try E.Eagle3.init(gpa, kb.d, kb.s, dimsOf(c, x.vocab.len), w, ops);
        return x;
    }

    /// The passes' shapes for release config `c` and a draft vocabulary of `vocab` rows.
    fn dimsOf(c: *const release.Config, vocab: usize) E.Dims {
        const s = c.drafter;
        return .{ .hidden = s.hidden, .heads = s.heads, .head_dim = s.head_dim, .ffn = s.ffn, .layers = s.layers, .taps = c.taps.ids.len, .vocab = vocab, .window = s.window, .chain = chain, .eps = @floatCast(s.eps) };
    }

    /// Device bytes `upload` puts, from the shapes `core.drafter_layout` checked (projections bf16, norms fp32).
    pub fn weightBytes(c: *const release.Config, vocab: usize) usize {
        const s = c.drafter;
        const d: usize = s.hidden;
        const att = @as(usize, s.heads) * s.head_dim;
        const t: usize = c.target.hidden;
        const layer = [_]usize{ 3 * att * d * 2, d * att * 2, 2 * s.ffn * d * 2, d * s.ffn * 2, d * 4, d * 4 };
        const rest = [_]usize{ vocab * 4, vocab * d * 2, d * c.taps.ids.len * t * 2, d * (t + d) * 2, d * 4, s.head_dim / 2 * 4 };
        var sum: usize = 0;
        for (layer) |b| sum += s.layers * @max(b, 256);
        for (rest) |b| sum += @max(b, 256);
        return sum;
    }

    /// Device bytes the drafter takes beside `slots` streams: weights, scratch, each slot's caches and a timing slot's.
    pub fn need(c: *const release.Config, vocab: usize, slots: usize) usize {
        const dm = dimsOf(c, vocab);
        return weightBytes(c, vocab) + dm.scratchBytes() + (slots + 1) * dm.streamBytes();
    }

    /// `need` when `left` (null: unread) holds it, else DrafterPastMemory; from the config alone, nothing allocated.
    pub fn admit(c: *const release.Config, vocab: usize, slots: usize, left: ?u64) !usize {
        const bytes = need(c, vocab, slots);
        if (left) |l| if (l < bytes) return error.DrafterPastMemory;
        return bytes;
    }

    fn put(x: *Drafter, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.alloc(x.kb.d, @max(bytes.len, 256));
        errdefer {
            var y = b;
            y.free();
        }
        try b.upload(0, bytes);
        try x.buffers.append(x.gpa, b);
        return b.ptr;
    }

    fn freeWeights(x: *Drafter) void {
        for (x.buffers.items) |*b| b.free();
        x.buffers.deinit(x.gpa);
        x.gpa.free(x.layers);
    }

    /// The release's tensors on the device: projections bf16 (q|k|v and gate|up stacked), norms fp32, the head's rows.
    fn upload(x: *Drafter, dir: []const u8, c: *const release.Config, names: core.drafter_layout.Names, k: *const kern.Kernels) !E.Weights {
        var arena: std.heap.ArenaAllocator = .init(x.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var f = try core.safetensors.File.open(x.gpa, x.io, try std.fs.path.join(a, &.{ dir, published.weights }));
        defer f.close(x.io);
        const L = struct {
            fn get(file: *const core.safetensors.File, al: std.mem.Allocator, n: core.drafter_layout.Names, name: []const u8) !core.safetensors.Tensor {
                return file.get(try n.of(al, name)) orelse error.DrafterTensorMissing;
            }
        };
        const bf = struct {
            fn rows(al: std.mem.Allocator, ts: []const core.safetensors.Tensor) ![]const u8 {
                var out: std.ArrayList(u8) = .empty;
                for (ts) |t| {
                    if (t.dtype != .bf16) return error.DrafterWeightNotBf16; // the published releases store bf16
                    try out.appendSlice(al, t.bytes);
                }
                return out.items;
            }
            fn norm(al: std.mem.Allocator, t: core.safetensors.Tensor) ![]const u8 {
                const n = t.shape[0];
                const v = try al.alloc(f32, n);
                for (v, 0..) |*o, i| o.* = switch (t.dtype) {
                    .f32 => @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little)),
                    .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16),
                    else => return error.DrafterNormDtype,
                };
                return std.mem.sliceAsBytes(v);
            }
        };
        x.layers = try x.gpa.alloc(E.Layer, c.drafter.layers);
        for (x.layers, 0..) |*l, i| {
            var g: [7]core.safetensors.Tensor = undefined;
            for (&g, [_][]const u8{ "q", "k", "v", "o", "gate", "up", "down" }) |*t, p| t.* = try L.get(&f, a, names, try std.fmt.allocPrint(a, "layers.{d}.{s}.weight", .{ i, p }));
            l.* = .{ .qkv = try x.put(try bf.rows(a, g[0..3])), .o = try x.put(try bf.rows(a, g[3..4])), .gate_up = try x.put(try bf.rows(a, g[4..6])), .down = try x.put(try bf.rows(a, g[6..7])), .n1 = try x.put(try bf.norm(a, try L.get(&f, a, names, try std.fmt.allocPrint(a, "layers.{d}.n1.w", .{i})))), .n2 = try x.put(try bf.norm(a, try L.get(&f, a, names, try std.fmt.allocPrint(a, "layers.{d}.n2.w", .{i})))) };
        }
        const inv = try a.alloc(f32, c.drafter.head_dim / 2);
        core.rope_freq.inverseFrequencies(c.drafter.theta, c.drafter.head_dim, inv);
        // the head's rows for the draft vocabulary, gathered on the device from the target's head
        const ids = try a.alloc(i32, x.vocab.len);
        for (ids, x.vocab) |*o, v| o.* = @intCast(v);
        const id_buf = try x.put(std.mem.sliceAsBytes(ids));
        const slice = try cuda.DeviceBuffer.alloc(x.kb.d, x.vocab.len * c.drafter.hidden * 2);
        x.buffers.append(x.gpa, slice) catch |e| {
            var y = slice;
            y.free();
            return e;
        };
        try k.embed.bf16(x.kb.s, id_buf, x.kb.m.w.head, slice.ptr, x.vocab.len, c.drafter.hidden);
        try x.kb.s.synchronize();
        return .{
            .fuse = try x.put(try bf.rows(a, &.{try L.get(&f, a, names, "fuse.weight")})),
            .fc = try x.put(try bf.rows(a, &.{try L.get(&f, a, names, "fc.weight")})),
            .layers = x.layers,
            .out_norm = try x.put(try bf.norm(a, try L.get(&f, a, names, "out_norm.w"))),
            .inv = try x.put(std.mem.sliceAsBytes(inv)),
            .embed = x.kb.m.w.embed,
            .head = slice.ptr,
        };
    }

    pub fn deinit(x: *Drafter) void {
        x.kb.s.synchronize() catch {};
        x.e.deinit();
        x.act.unload();
        x.freeWeights();
        x.lanes.deinit(x.gpa);
        x.gpa.free(x.free);
        x.gpa.free(x.vocab);
        x.gpa.destroy(x);
    }

    /// Caches for `streams` streams and one more that times a step, then the step's and hold pass's costs.
    pub fn reserve(x: *Drafter, streams: u32) !void {
        if (streams + 1 <= x.free.len) return;
        if (x.lanes.count() != 0) return error.DraftReserveWithStreams;
        try x.e.reserve(streams + 1);
        x.gpa.free(x.free);
        x.free = try x.gpa.alloc(bool, streams + 1);
        @memset(x.free, true);
        x.free[streams] = false;
        x.step_ms = try x.time(streams);
        x.n_holds = try x.timeHolds(streams); // on the streams' slots: zeroed after, before any stream opens
        for (x.e.caches.items) |b| try x.kb.d.check(x.kb.d.api.cuMemsetD8Async(b.ptr, 0, b.len, x.kb.s.handle), "cuMemsetD8Async");
        try row_costs.time(x.kb, x.io, &x.costs); // without them the depth rule cannot trade drafts against rows
        x.timed = true;
    }

    /// One more draft's ms: chains of `depth` and of 1 on the spare slot, the best of three each.
    fn time(x: *Drafter, spare: u32) !f64 {
        var out: [1][chain]u32 = undefined;
        var outs = [_][]u32{&out[0]};
        var best: [2]f64 = @splat(std.math.inf(f64));
        const steps = [2]usize{ 1, @max(x.depth, 2) };
        for (0..4) |round| for (steps, 0..) |n, i| {
            const t0 = std.Io.Clock.awake.now(x.io);
            try x.e.chain(&.{.{ .slot = spare, .pos = 64, .token = 0 }}, n, x.vocab, &outs);
            const ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(x.io)).toNanoseconds())) / 1e6;
            if (round > 0) best[i] = @min(best[i], ms);
        };
        return @max(best[1] - best[0], 0) / @as(f64, @floatFromInt(steps[1] - 1));
    }

    /// A hold pass of n streams at depth 1 and at the serving depth, best of three, on slots no stream holds yet.
    fn timeHolds(x: *Drafter, streams: u32) !usize {
        var out: [hold_streams[hold_streams.len - 1]][chain]u32 = undefined;
        var outs: [out.len][]u32 = undefined;
        var items: [out.len]E.Row = undefined;
        for (&outs, &out, &items, 0..) |*o, *b, *it, slot| {
            o.* = b;
            it.* = .{ .slot = @intCast(slot), .pos = 64, .token = 0 };
        }
        var n: usize = 0;
        for (hold_streams) |k| {
            if (k > streams) break;
            for ([2]u32{ 1, @max(x.depth, 2) }) |depth| {
                var best = std.math.inf(f64);
                for (0..4) |round| {
                    const t0 = std.Io.Clock.awake.now(x.io);
                    try x.e.chain(items[0..k], depth, x.vocab, outs[0..k]);
                    const ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(x.io)).toNanoseconds())) / 1e6;
                    if (round > 0) best = @min(best, ms);
                }
                x.holds[n] = .{ .streams = k, .depth = depth, .ms = best };
                n += 1;
            }
        }
        if (n == 0) return 0;
        std.log.info("Kolibri drafter hold pass: 1 stream {d:.2}/{d:.2} ms, {d} streams {d:.2}/{d:.2} ms (depth 1/{d})", .{ x.holds[0].ms, x.holds[1].ms, x.holds[n - 1].streams, x.holds[n - 2].ms, x.holds[n - 1].ms, x.holds[1].depth });
        return n;
    }

    pub fn drafter(x: *Drafter) dr.Drafter {
        return .{ .kind = "EAGLE-3", .ptr = x, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    fn tapsFn(ptr: *anyopaque) []const u32 {
        const x = cast(ptr);
        return x.taps[0..x.n_taps];
    }

    /// Chains of the release's serving depth, every stream's step in one pass, capped by the rule in shared rounds.
    fn factsFn(ptr: *anyopaque) dr.Facts {
        const x = cast(ptr);
        const costs: []const lanes.config.Cost = if (x.timed) &x.costs else &.{};
        const holds: []const lanes.hold_price.HoldCost = if (x.timed) x.holds[0..x.n_holds] else &.{};
        return .{ .limit_shared_drafts = true, .depth = x.depth, .step_ms = x.step_ms, .batched = true, .stream_bytes = x.e.dims.streamBytes(), .window_costs = costs, .lane_costs = costs, .hold_costs = holds };
    }

    fn openFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const x = cast(ptr);
        if (x.lanes.contains(s)) return error.DraftStreamOpen;
        const slot = std.mem.indexOfScalar(bool, x.free, true) orelse return error.NoDraftSlot;
        try x.lanes.put(x.gpa, s, .{ .slot = @intCast(slot) });
        x.free[slot] = false;
    }

    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const x = cast(ptr);
        if (x.lanes.fetchRemove(s)) |e| x.free[e.value.slot] = true; // one CUDA stream: later writes queue behind
    }

    /// Each item's kept rows into its stream's ring (its last row's output the next chain's start).
    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const x = cast(ptr);
        var rows: std.ArrayList(E.Row) = .empty;
        defer rows.deinit(x.gpa);
        var parts: std.ArrayList(E.Eagle3.Part) = .empty;
        defer parts.deinit(x.gpa);
        const width = x.n_taps * x.e.dims.hidden * 2;
        for (items) |it| {
            const f = it.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != width or it.rows != null) return error.DraftFeaturesUnsupported;
            if (it.follow.len < f.rows) return error.DraftFollowShort;
            if (!x.lanes.contains(it.stream)) return error.DraftStreamUnknown;
        }
        for (items) |it| { // rows of every stream first: `parts` slices them once they no longer move
            const l = x.lanes.getPtr(it.stream).?;
            for (0..it.features.rows) |i| try rows.append(x.gpa, .{ .slot = l.slot, .pos = @intCast(it.start + i), .token = it.follow[i] });
        }
        var at: usize = 0;
        for (items) |it| {
            const f = it.features;
            if (f.rows == 0) continue;
            try parts.append(x.gpa, .{ .rows = rows.items[at..][0..f.rows], .taps = f.buffer + f.offset });
            at += f.rows;
            const l = x.lanes.getPtr(it.stream).?;
            l.pos = it.start + f.rows - 1;
            l.token = it.follow[f.rows - 1];
            l.rows = true;
            l.held_len = 0;
        }
        try x.e.absorbMany(parts.items); // one pass for every stream's rows: the drafter's weights read once
    }

    /// Every stream's chain in one pass a step; the drafts read back as target ids.
    fn holdFn(ptr: *anyopaque, holds: []const dr.Hold) anyerror!void {
        const x = cast(ptr);
        if (holds.len > E.pass_rows) return error.DraftTooManyStreams;
        var items: [E.pass_rows]E.Row = undefined;
        var picks: [E.pass_rows][chain]u32 = undefined;
        var outs: [E.pass_rows][]u32 = undefined;
        var steps: usize = 0;
        for (holds, 0..) |h, i| {
            const l = x.lanes.getPtr(h.stream) orelse return error.DraftStreamUnknown;
            if (h.depth == 0 or h.depth > chain) return error.DraftDepth;
            if (h.lanes != null) return error.DraftTreesUnsupported;
            if (!l.rows or l.token != h.pending) return error.DraftContextGap; // it drafts after its last absorbed row
            items[i] = .{ .slot = l.slot, .pos = @intCast(l.pos), .token = l.token };
            outs[i] = &picks[i];
            steps = @max(steps, h.depth);
        }
        try x.e.chain(items[0..holds.len], steps, x.vocab, outs[0..holds.len]);
        for (holds, 0..) |h, i| {
            const l = x.lanes.getPtr(h.stream).?;
            for (l.held[0..h.depth], picks[i][0..h.depth]) |*t, p| t.* = x.vocab[p];
            l.held_len = h.depth;
        }
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const x = cast(ptr);
        for (streams, out) |s, o| {
            const l = x.lanes.get(s) orelse return error.DraftStreamUnknown;
            if (o.len > l.held_len) return error.DraftHeldFewer;
            @memcpy(o, l.held[0..o.len]);
        }
    }
};

test "the loaded Kolibri-1 is the adapter its published drafters were trained against" {
    const config = @import("config.zig");
    var why: config.Why = .{};
    const c = try config.parse(std.testing.allocator, config.pinned, &why);
    const a = adapter(c);
    try std.testing.expectEqual(@as(u32, 50), a.layers);
    try std.testing.expectEqual(@as(u32, 2560), a.hidden);
    try std.testing.expectEqual(@as(u32, 128000), a.vocab);
}

test "the drafter's device bytes come from its config alone; one byte short refuses by name, nothing allocated" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const config = @import("config.zig");
    var why: config.Why = .{};
    const kc = try config.parse(arena.allocator(), config.pinned, &why);
    // the published EAGLE-3 release's config.json (josch15366/Kolibri-1-EAGLE3-drafter), training counts left out
    const text =
        \\{"architecture": "kolibri1-eagle3-drafter", "target": "Aleph-Alpha/Kolibri-1",
        \\ "drafter": {"hidden": 2560, "heads": 20, "head_dim": 128, "ffn": 4096, "eps": 1e-06, "theta": 10000.0,
        \\  "layers": 1, "taps": 3}, "taps": [44, 47, 49], "draft_vocab": "draft_vocab.json", "draft_vocab_size": 32768}
    ;
    var rwhy: release.Why = .{};
    const got = try published.parse(arena.allocator(), text, .{ .repo = target_repo, .adapter = adapter(kc) }, &rwhy);
    const c = &got.config;
    const v = 32768;
    // per layer q|k|v, o, gate|up, down, 2 norms; ids, head rows, fuse, fc, out_norm, RoPE (each listed in weightBytes)
    try std.testing.expectEqual(@as(usize, 348813568), Drafter.weightBytes(c, v));
    const stream: usize = 2 * 2368 * 2560 * 2 + 2560 * 2; // a slot's keys, values and last output
    const one = Drafter.need(c, v, 1);
    try std.testing.expectEqual(348813568 + 70528256 + 2 * stream, one); // scratch 70528256: a pass of 256 rows
    try std.testing.expectError(error.DrafterPastMemory, Drafter.admit(c, v, 1, one - 1));
    try std.testing.expectEqual(one, try Drafter.admit(c, v, 1, one));
    try std.testing.expectEqual(one + 7 * stream, try Drafter.admit(c, v, 8, null)); // linear caches: 8 at open
}
