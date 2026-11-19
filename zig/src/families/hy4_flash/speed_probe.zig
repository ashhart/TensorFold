//! tf-hy4-speed: greedy Hy4-flash decode on the native Metal graph, for speed A/B and per-stage GPU profiles.
//! Contexts stay under 2048 slots, so sparse layers attend every earlier key and no indexer pass is needed.
//!
//!   tf-hy4-speed MODEL_DIR GEN [--profile STEPS] [--long TOKENS] [--chunk ROWS]
//!
//! Prints each prompt's generated IDs and their FNV-1a hash (the exactness check between builds), decode ms a token
//! (wall and GPU), and with --profile the GPU time of every barrier-delimited stage, summed per token by its pipelines.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const target = @import("metal_model.zig");
const ihc = @import("ihc.zig");
const engram = @import("engram.zig");
const dfly = @import("metal_draft.zig");

/// Cache slots (TF_HY4_CAPACITY, a multiple of 512; past 2,048 sparse layers attend every key unless TF_HY4_INDEX).
var capacity: u32 = 2048;
const index = @import("metal_index.zig");
const key_width = 576; // BF16 words in a latent KV slot: 512 latent + 64 rope
const window: u32 = 512;
const bos: u32 = 120000;
const max_rows = 512; // a prompt chunk's rows (one router window is 16)

const prompts = [_][]const u8{
    "The capital of France is",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "def merge_sort(values):\n    \"\"\"Sort a list of numbers.\"\"\"\n",
    "Explain why the sky is blue in three sentences.",
};

const long_text =
    "The history of the printing press begins long before Gutenberg. Carved wooden blocks were inked and pressed " ++
    "onto paper in East Asia for centuries, and movable type made of clay and later metal appeared well before it " ++
    "reached Europe. What changed in the fifteenth century was the combination of an oil-based ink, a hand mould " ++
    "that cast uniform letters quickly, and a screw press adapted from wine making. Together they made books cheap " ++
    "enough that ideas could travel faster than the authorities who wanted to control them. ";

const Name = struct { id: usize, name: []const u8 };

fn collect(comptime T: type, v: *const T, comptime path: []const u8, list: *std.ArrayList(Name), a: std.mem.Allocator) !void {
    if (T == mtl.Pipeline) return list.append(a, .{ .id = @intFromPtr(v.id), .name = path });
    switch (@typeInfo(T)) {
        .@"struct" => |s| inline for (s.field_names, s.field_types, s.field_attrs) |f, F, attrs| {
            if (!attrs.@"comptime") try collect(F, &@field(v.*, f), path ++ "." ++ f, list, a);
        },
        .optional => |o| if (v.*) |*x| try collect(o.child, x, path, list, a),
        .array => |ar| for (v) |*x| try collect(ar.child, x, path ++ "[]", list, a),
        .@"union" => |u| if (u.tag_type != null) switch (v.*) {
            inline else => |*x, tag| try collect(@TypeOf(x.*), x, path ++ "." ++ @tagName(tag), list, a),
        },
        else => {},
    }
}

fn nameOf(names: []const Name, id: usize) []const u8 {
    for (names) |n| if (n.id == id) return n.name;
    return "?";
}

fn bf16(word: u16) f32 {
    return @bitCast(@as(u32, word) << 16);
}

fn argmax(row: []const u16) !u32 {
    var best: u32 = 0;
    var top: f32 = -std.math.inf(f32);
    for (row, 0..) |w, i| {
        const v = bf16(w);
        if (std.math.isNan(v)) return error.NanLogit;
        if (v > top) {
            top = v;
            best = @intCast(i);
        }
    }
    return best;
}

const Stage = struct { label: []const u8, ns: f64 = 0, count: usize = 0 };
const Pass = struct { next: u32, gpu: f64 };

/// DFly2 drafting and greedy lane verification. The draft writes its block's KV into the target layers it shares,
/// at positions the verify pass rewrites before any read, so rejection needs no rollback.
const Lanes = struct {
    draft: dfly.Draft,
    frame: dfly.Frame,
    ids: mtl.Buffer,
    previous: mtl.Buffer,
    tokens: mtl.Buffer,
    positions: mtl.Buffer,
    writes: mtl.Buffer,
    keys: mtl.Buffer,
    counts: mtl.Buffer,
    caches: []target.LayerCache,
    mask: u32,
    rounds: usize = 0,
    kept: usize = 0,
    proposed: usize = 0,
    draft_gpu: f64 = 0,
    verify_gpu: f64 = 0,
    profile_rounds: u32 = 0,
    stage_wall: f64 = 0,
    copy_rounds: usize = 0,

    fn init(a: std.mem.Allocator, run: *Run) !Lanes {
        var draft = try dfly.Draft.open(a, run.device, &run.model.host);
        errdefer draft.deinit();
        const frame = try dfly.Frame.init(a, run.device, run.model, &draft, 1, capacity);
        const opts = mtl.ResourceOptions.shared;
        const w = draft.width;
        const out: Lanes = .{
            .draft = draft,
            .frame = frame,
            .ids = try run.device.buffer(w * 4, opts),
            .previous = try run.device.buffer(4, opts),
            .tokens = try run.device.buffer(16 * 4, opts),
            .positions = try run.device.buffer(w * 3 * 4, opts),
            .writes = try run.device.buffer(w * 4, opts),
            .keys = try run.device.buffer(@as(usize, w) * capacity * 4, opts),
            .counts = try run.device.buffer(w * 4, opts),
            .caches = try a.alloc(target.LayerCache, draft.blocks.len),
            .mask = run.model.host.config.value.dfly_mask_token_id,
        };
        for (out.caches, draft.blocks) |*c, block| c.* = .{ .rows = .{
            .positions = .{ .buffer = out.positions },
            .write_slots = .{ .buffer = out.writes },
            .retained = .{ .buffer = out.keys },
            .counts = .{ .buffer = out.counts },
            .cache = .{ .buffer = run.caches[block.target_kv] },
            .capacity = capacity,
        }, .key_stride = capacity };
        return out;
    }

    fn submit(run: *Run, cb: mtl.CommandBuffer) !f64 {
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.debug.print("draft command buffer failed: {s}\n", .{why});
            return error.Hy4ProbeCommandFailed;
        }
        _ = run;
        return cb.gpuSeconds();
    }

    /// Encode and commit the DFly2 chain for `count` tokens after `bonus` at position `p`, from the target's final
    /// hidden row `source`: the block pass, then per column the correction, readout and a GPU argmax whose token the
    /// next column reads, so one command buffer and no host trips. Read the tokens after `land`.
    fn launch(self: *Lanes, run: *Run, p: u32, bonus: u32, source: u32, count: u32) !struct { cb: mtl.CommandBuffer, segs: usize, c0: mtl.counters.ClockSample } {
        const w = self.draft.width;
        const ids = self.ids.slice(u32, w);
        @memset(ids, self.mask);
        ids[0] = bonus;
        const pos3 = self.positions.slice(i32, w * 3);
        const wr = self.writes.slice(i32, w);
        const stride: u32 = ((p + w + window - 1) / window) * window;
        const keys = self.keys.slice(u32, @as(usize, w) * capacity);
        const counts = self.counts.slice(u32, w);
        for (0..w) |i| {
            const at = p + @as(u32, @intCast(i));
            pos3[i * 3 ..][0..3].* = @splat(@intCast(at));
            wr[i] = @intCast(at);
            for (0..at + 1) |j| keys[i * stride + j] = @intCast(j);
            counts[i] = at + 1;
        }
        for (self.caches) |*c| c.key_stride = stride;
        const h = self.draft.hidden;
        const hidden = run.frame.common.normalized;
        const anchor: target.Ref = .{ .buffer = hidden.buffer, .offset = hidden.offset + @as(usize, source) * h * 2 };
        const rows = run.model.common.rows orelse return error.Hy4ProbeNeedsRowKernels;
        const cb = run.queue.commandBuffer().retain();
        const profile = self.profile_rounds > 0;
        var split = if (profile) try mtl.Splitter.timed(cb, run.pass.?, run.samples.?, run.segments) else mtl.Splitter.plain(cb, run.segments);
        const c0 = if (profile) mtl.counters.sampleClocks(run.device) else mtl.counters.ClockSample{};
        const e = split.encoder();
        try self.draft.encode(run.model, e, &self.frame, .{ .buffer = self.ids }, anchor, 1, self.caches);
        e.barrier();
        for (0..count) |c| {
            const previous: target.Ref = if (c == 0) .{ .buffer = self.ids } else .{ .buffer = self.tokens, .offset = (c - 1) * 4 };
            const logits = try self.draft.next(run.model, e, &self.frame, previous, @intCast(c + 1));
            e.barrier();
            rows.pick(e, logits, .{ .buffer = self.tokens }, @intCast(c), 1, run.vocab);
            e.barrier();
        }
        const segs = split.finish();
        cb.commit();
        return .{ .cb = cb, .segs = segs, .c0 = c0 };
    }

    fn land(self: *Lanes, run: *Run, l: anytype, count: u32, out: []u32) !void {
        l.cb.wait();
        defer l.cb.release();
        if (l.cb.failure()) |why| {
            std.debug.print("draft command buffer failed: {s}\n", .{why});
            return error.Hy4ProbeCommandFailed;
        }
        self.draft_gpu += l.cb.gpuSeconds();
        if (self.profile_rounds > 0) try run.account(l.segs, l.c0, mtl.counters.sampleClocks(run.device));
        @memcpy(out[0..count], self.tokens.slice(u32, count));
    }
};

const Run = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    model: *target.Model,
    frame: *target.Frame,
    positions: mtl.Buffer,
    writes: mtl.Buffer,
    slide_keys: mtl.Buffer,
    slide_counts: mtl.Buffer,
    sparse_keys: mtl.Buffer,
    sparse_counts: mtl.Buffer,
    /// Each row's key range [lo, hi) for sliding and for sparse layers, and each 32-key tile's first slot (slots
    /// are positions here), for the tensor units' flash kernel.
    slide_ranges: mtl.Buffer,
    sparse_ranges: mtl.Buffer,
    tiles: mtl.Buffer,
    caches: []mtl.Buffer,
    lanes: []target.LayerCache,
    sliding: []bool,
    /// TF_HY4_INDEX: the compressed indexer's caches (sparse layers), so contexts past 2,048 recall groups.
    indexed: bool = false,
    picks: mtl.Buffer,
    all_logits: bool = false,
    pad: u32,
    vocab: u32,
    // profile
    names: []const Name,
    pass: ?mtl.counters.ComputePass = null,
    samples: ?mtl.counters.SampleBuffer = null,
    segments: []mtl.Segment,
    stages: std.StringArrayHashMapUnmanaged(Stage) = .empty,
    profiled: usize = 0,

    /// One forward over `tokens` at positions start.. with full history `history` (tokens before start).
    fn step(self: *Run, history: []const u32, start: u32, tokens: []const u32, profile: bool) !Pass {
        return self.stepRows(history, start, tokens, profile, null);
    }

    /// Engram windows and the frame's ids for `tokens` at positions start.. (history holds the tokens before).
    fn stage(self: *Run, history: []const u32, start: u32, tokens: []const u32) !void {
        const n: u32 = @intCast(tokens.len);
        var windows: [max_rows][engram.order]u32 = undefined;
        var media: [max_rows]u32 = @splat(0);
        for (tokens, 0..) |t, r| {
            const pos = start + @as(u32, @intCast(r));
            windows[r][0] = t;
            for (1..engram.order) |lag| {
                windows[r][lag] = if (lag <= r) tokens[r - lag] else if (lag <= pos) history[pos - lag] else self.pad;
            }
        }
        try self.frame.stage(self.model, tokens, windows[0..n], media[0..n]);
    }

    /// Positions, write slots and every layer's key list for `n` rows at start..
    fn layout(self: *Run, start: u32, n: u32) void {
        const pos3 = self.positions.slice(i32, max_rows * 3);
        const wr = self.writes.slice(i32, max_rows);
        const last_end = start + n;
        const sparse_stride: u32 = ((last_end + window - 1) / window) * window;
        const sk = self.slide_keys.slice(u32, max_rows * window);
        const sc = self.slide_counts.slice(u32, max_rows);
        const pk = self.sparse_keys.slice(u32, max_rows * capacity);
        const pc = self.sparse_counts.slice(u32, max_rows);
        for (0..n) |r| {
            const pos = start + @as(u32, @intCast(r));
            const end = pos + 1;
            pos3[r * 3 ..][0..3].* = @splat(@intCast(pos));
            wr[r] = @intCast(pos);
            const count = @min(end, window);
            for (0..count) |j| sk[r * window + j] = end - count + @as(u32, @intCast(j));
            sc[r] = count;
            for (0..end) |j| pk[r * sparse_stride + j] = @intCast(j);
            pc[r] = end;
            self.slide_ranges.slice([2]u32, max_rows)[r] = .{ end - count, end };
            self.sparse_ranges.slice([2]u32, max_rows)[r] = .{ 0, end };
        }
        const lists = self.indexed and last_end - 1 >= index.keys_a_row;
        if (self.indexed) self.frame.index.stage(start, n) catch unreachable;
        for (self.lanes, self.sliding) |*lane, slide| {
            lane.key_stride = if (slide) window else if (lists) index.keys_a_row else sparse_stride;
            lane.rows.live_keys = if (slide) @min(last_end, window) else last_end;
        }
    }

    /// The target pass over the staged rows with every row's greedy token into `picks`, encoded, not committed.
    fn encodePicks(self: *Run, n: u32) !mtl.CommandBuffer {
        const rows = self.model.common.rows orelse return error.Hy4ProbeNeedsRowKernels;
        const cb = self.queue.commandBuffer().retain();
        var split = mtl.Splitter.plain(cb, self.segments);
        const e = split.encoder();
        const logits = try self.model.encode(e, self.frame, self.lanes);
        e.barrier();
        rows.pick(e, logits, .{ .buffer = self.picks }, 0, n, self.vocab);
        _ = split.finish();
        return cb;
    }

    /// `step`, with every row's greedy token in `each` when given.
    fn stepRows(self: *Run, history: []const u32, start: u32, tokens: []const u32, profile: bool, each: ?[]u32) !Pass {
        const n: u32 = @intCast(tokens.len);
        try self.stage(history, start, tokens);
        self.layout(start, n);
        // prompt chunks (no per-row picks asked) read out only their last row
        self.frame.last_logits = each == null and n > 1 and !self.all_logits;
        defer self.frame.last_logits = false;

        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = self.queue.commandBuffer();
        var split = if (profile)
            try mtl.Splitter.timed(cb, self.pass.?, self.samples.?, self.segments)
        else
            mtl.Splitter.plain(cb, self.segments);
        const c0 = if (profile) mtl.counters.sampleClocks(self.device) else mtl.counters.ClockSample{};
        const logits = try self.model.encode(split.encoder(), self.frame, self.lanes);
        const segs = split.finish();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.debug.print("command buffer failed: {s}\n", .{why});
            return error.Hy4ProbeCommandFailed;
        }
        if (profile) try self.account(segs, c0, mtl.counters.sampleClocks(self.device));
        const out_rows: usize = if (self.frame.last_logits) 1 else n;
        const all = logits.buffer.slice(u16, out_rows * self.vocab);
        if (each) |out| for (0..n) |r| {
            out[r] = try argmax(all[r * self.vocab ..][0..self.vocab]);
        };
        return .{ .next = try argmax(all[(out_rows - 1) * self.vocab ..][0..self.vocab]), .gpu = cb.gpuSeconds() };
    }

    fn account(self: *Run, segs: usize, c0: mtl.counters.ClockSample, c1: mtl.counters.ClockSample) !void {
        const resolved = try self.samples.?.resolveCompleted(2 * segs);
        defer resolved.deinit();
        const v = resolved.values();
        const ratio = @as(f64, @floatFromInt(c1.cpu - c0.cpu)) / @as(f64, @floatFromInt(c1.gpu - c0.gpu));
        for (self.segments[0..segs], 0..) |seg, i| {
            const t0 = v[2 * i].timestamp;
            const t1 = v[2 * i + 1].timestamp;
            if (t0 == mtl.counters.error_value or t1 == mtl.counters.error_value or t1 < t0) continue;
            var label: std.ArrayList(u8) = .empty;
            for (seg.pipelines[0..seg.named], 0..) |id, k| {
                if (k != 0) try label.appendSlice(self.a, " + ");
                try label.appendSlice(self.a, nameOf(self.names, id));
            }
            if (seg.dispatches > seg.named) try label.print(self.a, " (+{d} dispatches)", .{seg.dispatches - seg.named});
            const key = try label.toOwnedSlice(self.a);
            const got = try self.stages.getOrPut(self.a, key);
            if (!got.found_existing) got.value_ptr.* = .{ .label = key } else self.a.free(key);
            got.value_ptr.ns += @as(f64, @floatFromInt(t1 - t0)) * ratio;
            got.value_ptr.count += 1;
        }
        self.profiled += 1;
    }
};

/// Copy drafts: the tokens that followed the latest earlier match of the context's last 3 to 8 tokens (the longest
/// match wins), up to `out.len`. Returns how many it found; 0 when nothing earlier matches 3 tokens.
fn copyDraft(context: []const u32, out: []u32) usize {
    const n = context.len;
    if (n < 4 or out.len == 0) return 0;
    var best_len: usize = 0;
    var best_end: usize = 0;
    var end: usize = n - 1;
    while (end > 0) {
        end -= 1;
        var m: usize = 0;
        while (m < 8 and m <= end and context[end - m] == context[n - 1 - m]) m += 1;
        if (m > best_len and end + 1 < n) {
            best_len = m;
            best_end = end;
            if (m == 8) break;
        }
    }
    if (best_len < 3) return 0;
    const take = @min(out.len, n - 1 - best_end);
    @memcpy(out[0..take], context[best_end + 1 ..][0..take]);
    return take;
}

/// TF_HY4_FLASH=slide or sparse limits the flash kernel to one kind of layer (a diagnostic).
fn bf(word: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, word) << 16)));
}

/// TF_HY4_ACHECK: one 1-row pass at `at`, then the last layer's attention output against an fp64 reference from the
/// same absorbed query, cache rows and sink (softmax over its key range plus the sink, P V over the 512 latent columns).
fn attentionCheck(run: *Run, hist: []const u32, at: u32) !void {
    run.all_logits = false;
    _ = try run.step(hist, at, hist[at .. at + 1], false);
    const layer = run.model.blocks.len - 1;
    const block = &run.model.blocks[layer];
    const heads = block.attention.heads;
    const s = &run.frame.common;
    const q = s.absorbed.buffer.slice(u16, (s.absorbed.offset / 2) + heads * 576)[s.absorbed.offset / 2 ..];
    const out = s.latent.buffer.slice(u16, (s.latent.offset / 2) + heads * 512)[s.latent.offset / 2 ..];
    const cache = run.caches[layer].slice(u16, capacity * 576);
    const sinks = block.attention.sinks.slice(f32, heads);
    const range = (if (run.sliding[layer]) run.slide_ranges else run.sparse_ranges).slice([2]u32, max_rows)[0];
    const keys = range[1] - range[0];
    const sc = try run.a.alloc(f64, keys);
    defer run.a.free(sc);
    var err2: f64 = 0;
    var ref2: f64 = 0;
    var round2: f64 = 0;
    var worst: f64 = 0;
    for (0..heads) |h| {
        var top: f64 = sinks[h];
        for (0..keys) |j| {
            const k = cache[(range[0] + j) * 576 ..][0..576];
            var dot: f64 = 0;
            for (0..576) |d| dot += bf(q[h * 576 + d]) * bf(k[d]);
            sc[j] = dot * 0.0625;
            top = @max(top, sc[j]);
        }
        var denom: f64 = @exp(@as(f64, sinks[h]) - top);
        for (sc) |*v| {
            v.* = @exp(v.* - top);
            denom += v.*;
        }
        for (0..512) |c| {
            var acc: f64 = 0;
            for (0..keys) |j| acc += sc[j] * bf(cache[(range[0] + j) * 576 + c]);
            const ref = acc / denom;
            const got = bf(out[h * 512 + c]);
            const rounded: f64 = @floatCast(@as(f32, @floatCast(ref)));
            const r16 = bf(@truncate(@as(u32, @bitCast(@as(f32, @floatCast(rounded)))) >> 16));
            err2 += (got - ref) * (got - ref);
            round2 += (r16 - ref) * (r16 - ref);
            ref2 += ref * ref;
            worst = @max(worst, @abs(got - ref));
        }
    }
    std.debug.print("ACHECK layer {d} ({s}, {d} heads, keys [{d}, {d})): relative rms error {e:.3} (bf16 truncation of the reference {e:.3}), max |error| {e:.3}\n", .{ layer, if (run.sliding[layer]) "sliding" else "sparse", heads, range[0], range[1], @sqrt(err2 / ref2), @sqrt(round2 / ref2), worst });
}

/// TF_HY4_ICHECK (with TF_HY4_INDEX): after a pass, the last sparse layer's indexer against fp64 from the GPU's own
/// inputs: the last row's token-key norm, sampled group keys pooled from the cached token keys (and RoPE'd), and
/// the last row's 511 recalled groups against an fp64 ranking of the GPU's query, group keys and head weights.
fn indexCheck(run: *Run) !void {
    var layer: usize = run.model.blocks.len;
    while (layer > 0) {
        layer -= 1;
        if (!run.sliding[layer]) break;
    }
    const iw = run.model.blocks[layer].attention.index orelse return;
    const c = run.lanes[layer].index orelse return;
    const sc = &run.frame.index;
    const rows = sc.rows;
    const pos = sc.first + rows - 1;
    const eps: f64 = run.model.host.config.value.text_config.rms_norm_eps;
    const tk = c.token_keys.buffer.slice(u16, @as(usize, capacity) * 128);
    const gk = c.group_keys.buffer.slice(u16, @as(usize, capacity / 4) * 128);
    // the token key norm of the last row
    {
        const x = sc.keys.buffer.slice(u16, @as(usize, rows) * 128)[(rows - 1) * 128 ..][0..128];
        const gain = iw.token_norm.slice(u16, 128);
        var ss: f64 = 0;
        for (x) |v| ss += @as(f64, bf(v)) * bf(v);
        const inv = 1 / @sqrt(ss / 128 + eps);
        var worst: f64 = 0;
        var off: usize = 0;
        for (0..128) |d| {
            const want = bf(gain[d]) * bf(rne(bf(x[d]) * inv));
            const got = bf(tk[pos * 128 + d]);
            worst = @max(worst, @abs(got - want));
            if (rne(want) != tk[pos * 128 + d]) off += 1;
        }
        std.debug.print("ICHECK token key at {d}: {d}/128 off the rounded fp64, worst {e:.3}\n", .{ pos, off, worst });
    }
    // group keys: pooled in fp64 from the cached token keys, normalized, rotated at the group's first slot
    const vw = iw.values.slice(f32, 128 * 128);
    const gw = iw.gates.slice(f32, 128 * 128);
    const bias = iw.bias.slice(f32, 4 * 128);
    const nw = iw.norm.slice(f32, 128);
    const p = run.model.blocks[layer].attention.rope;
    const closed = pos / 4;
    var worst_g: f64 = 0;
    var off_g: usize = 0;
    var top_g: f64 = 0;
    const samples = [_]u32{ 0, 1, closed / 3, closed / 2, closed -| 2, closed -| 1 };
    for (samples) |g| {
        var pooled: [128]f64 = undefined;
        var v: [4][128]f64 = undefined;
        var z: [4][128]f64 = undefined;
        for (0..4) |slot| for (0..128) |o| {
            var sv: f64 = 0;
            var sg: f64 = 0;
            for (0..128) |k| {
                const xk: f64 = bf(tk[(@as(usize, g) * 4 + slot) * 128 + k]);
                sv += @as(f64, vw[o * 128 + k]) * xk;
                sg += @as(f64, gw[o * 128 + k]) * xk;
            }
            v[slot][o] = sv;
            z[slot][o] = sg + bias[slot * 128 + o];
        };
        var ss: f64 = 0;
        for (0..128) |o| {
            var m: f64 = -std.math.inf(f64);
            for (0..4) |slot| m = @max(m, z[slot][o]);
            var den: f64 = 0;
            for (0..4) |slot| den += @exp(z[slot][o] - m);
            var acc: f64 = 0;
            for (0..4) |slot| acc += v[slot][o] * @exp(z[slot][o] - m) / den;
            pooled[o] = acc;
            ss += acc * acc;
        }
        const inv = 1 / @sqrt(ss / 128 + eps);
        var key: [128]f64 = undefined;
        for (0..128) |o| key[o] = bf(rne(nw[o] * pooled[o] * inv));
        for (0..32) |i| {
            const phase: f64 = @as(f64, @floatFromInt(g * 4)) * p.frequencies[i];
            const cs: f64 = bf(rne(@cos(phase)));
            const sn: f64 = bf(rne(@sin(phase)));
            const a0 = key[64 + 2 * i];
            const b0 = key[64 + 2 * i + 1];
            key[64 + 2 * i] = a0 * cs - b0 * sn;
            key[64 + 2 * i + 1] = b0 * cs + a0 * sn;
        }
        for (0..128) |o| {
            const got: f64 = bf(gk[@as(usize, g) * 128 + o]);
            worst_g = @max(worst_g, @abs(got - key[o]));
            top_g = @max(top_g, @abs(key[o]));
            if (rne(key[o]) != gk[@as(usize, g) * 128 + o]) off_g += 1;
        }
    }
    std.debug.print("ICHECK group keys ({d} groups): {d}/{d} off the rounded fp64, worst {e:.3} of max |k| {d:.3}\n", .{ samples.len, off_g, samples.len * 128, worst_g, top_g });
    if (closed < index.budget_groups) {
        std.debug.print("ICHECK recall: row at {d} keeps every group\n", .{pos});
        return;
    }
    // the last row's ranking
    const q = sc.query.buffer.slice(u16, @as(usize, rows) * 2048)[(rows - 1) * 2048 ..][0..2048];
    const hw = sc.head_weights.buffer.slice(u16, @as(usize, rows) * 16)[(rows - 1) * 16 ..][0..16];
    const scores = try run.a.alloc(f64, closed);
    defer run.a.free(scores);
    for (scores, 0..) |*sco, g| {
        var acc: f64 = 0;
        for (0..16) |h| {
            var dot: f64 = 0;
            for (0..128) |d| dot += @as(f64, bf(q[h * 128 + d])) * bf(gk[g * 128 + d]);
            acc += @as(f64, bf(hw[h])) * @as(f64, @import("index_recall.zig").head_scale) * @max(dot, 0);
        }
        sco.* = acc;
    }
    const order = try run.a.alloc(u32, closed);
    defer run.a.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, scores, struct {
        fn more(sv: []f64, x: u32, y: u32) bool {
            return sv[x] > sv[y];
        }
    }.more);
    const keep = index.budget_groups - 1;
    const want = try run.a.alloc(bool, closed);
    defer run.a.free(want);
    @memset(want, false);
    for (order[0..keep]) |g| want[g] = true;
    const threshold = scores[order[keep - 1]];
    const list = c.tokens.buffer.slice(i32, @as(usize, rows) * index.keys_a_row)[(rows - 1) * index.keys_a_row ..][0..index.keys_a_row];
    const count = c.counts.buffer.slice(i32, rows)[rows - 1];
    var missing: usize = 0;
    var worst_gap: f64 = 0;
    for (0..keep) |i| {
        const g: usize = @intCast(@divTrunc(list[i * 4], 4));
        if (g >= closed or !want[g]) {
            missing += 1;
            if (g < closed) worst_gap = @max(worst_gap, (threshold - scores[g]) / @max(@abs(threshold), 1e-30));
        }
    }
    var own_ok = true;
    for (keep * 4..@as(usize, @intCast(count))) |i| own_ok = own_ok and list[i] == @as(i32, @intCast(closed * 4 + (i - keep * 4)));
    std.debug.print("ICHECK recall at {d}: {d} slots, {d}/{d} groups outside the fp64 top {d} (worst relative gap {e:.2} at threshold {d:.4}), own group {s}\n", .{ pos, count, missing, keep, keep, worst_gap, threshold, if (own_ok) "ok" else "WRONG" });
}

fn rne(v: f64) u16 {
    const f: f32 = @floatCast(v);
    const b: u32 = @bitCast(f);
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

fn flashFor(slide: bool) bool {
    const v = std.mem.span(std.c.getenv("TF_HY4_FLASH") orelse return true);
    return if (std.mem.eql(u8, v, "slide")) slide else if (std.mem.eql(u8, v, "sparse")) !slide else true;
}

fn fnv(ids: []const u32) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (ids) |id| for (std.mem.asBytes(&id)) |b| {
        h ^= b;
        h *%= 0x100000001b3;
    };
    return h;
}

pub fn main(init: std.process.Init) !void {
    const a = std.heap.page_allocator;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: tf-hy4-speed MODEL_DIR GEN [--profile STEPS] [--long TOKENS] [--chunk ROWS]\n", .{});
        return error.Usage;
    }
    const dir = args[1];
    const gen = try std.fmt.parseInt(u32, args[2], 10);
    if (std.c.getenv("TF_HY4_CAPACITY")) |v| capacity = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2048;
    if (capacity < 2048 or capacity % 512 != 0) return error.BadCapacity;
    var profile_steps: u32 = 0;
    var long_tokens: u32 = 0;
    var chunk: u32 = 16;
    var rows_time = false;
    var lane_rows: u32 = 0;
    var warm_engram = false;
    var text_path: ?[]const u8 = null;
    var nll = false;
    const nll_dump = std.c.getenv("TF_HY4_NLL_DUMP") != null;
    var copies = false;
    const adapt: ?[2]u32 = if (std.c.getenv("TF_HY4_ADAPT")) |v| blk: {
        var it = std.mem.splitScalar(u8, std.mem.span(v), ',');
        const lo = std.fmt.parseInt(u32, it.next() orelse "2", 10) catch 2;
        const hi = std.fmt.parseInt(u32, it.next() orelse "6", 10) catch 6;
        break :blk .{ @max(2, lo), @min(6, hi) };
    } else null;
    const profile_prompt: usize = if (std.c.getenv("TF_HY4_PROFILE_PROMPT")) |v| (std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0) else 0;
    var i: usize = 3;
    if (args.len > 3 and std.mem.eql(u8, args[3], "--rows-time")) {
        rows_time = true;
        i = 4;
    }
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--profile")) profile_steps = try std.fmt.parseInt(u32, args[i + 1], 10) else if (std.mem.eql(u8, args[i], "--long")) long_tokens = try std.fmt.parseInt(u32, args[i + 1], 10) else if (std.mem.eql(u8, args[i], "--chunk")) chunk = try std.fmt.parseInt(u32, args[i + 1], 10) else if (std.mem.eql(u8, args[i], "--lanes")) lane_rows = try std.fmt.parseInt(u32, args[i + 1], 10) else if (std.mem.eql(u8, args[i], "--text")) {
            text_path = args[i + 1];
        } else if (std.mem.eql(u8, args[i], "--nll")) {
            nll = std.mem.eql(u8, args[i + 1], "1");
        } else if (std.mem.eql(u8, args[i], "--copies")) {
            copies = std.mem.eql(u8, args[i + 1], "1");
        } else if (std.mem.eql(u8, args[i], "--warm-engram")) {
            warm_engram = std.mem.eql(u8, args[i + 1], "1");
        } else return error.Usage;
    }
    if (chunk == 0 or chunk > max_rows) return error.BadChunk;

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    std.debug.print("device {s}, tensor units {}\n", .{ device.name(), device.tensorUnits() });

    const tok_path = try std.fs.path.join(a, &.{ dir, "tokenizer.json" });
    var tok = try core.tokenizer.loadTokenizer(io, a, tok_path);
    defer tok.deinit();

    const t_load = mtl.clock.seconds();
    var model = try target.Model.open(a, io, device, dir);
    defer model.deinit();
    const t = model.host.config.value.text_config;
    if (warm_engram) {
        // Read every Engram table page once, so lookups find their rows in RAM (a resident-table server).
        const tw = mtl.clock.seconds();
        var bytes: usize = 0;
        var sink: u8 = 0;
        for (model.blocks) |*b| if (b.memory) |*m| {
            const tab = m.table.affine;
            for ([_][]const u8{ tab.words.bytes, tab.scales.bytes, tab.biases.bytes }) |region| {
                var at: usize = 0;
                while (at < region.len) : (at += 16384) sink +%= @as(*const volatile u8, &region[at]).*;
                bytes += region.len;
            }
        };
        std.debug.print("warmed {d:.1} GB of Engram tables in {d:.1}s ({d})\n", .{ @as(f64, @floatFromInt(bytes)) / 1e9, mtl.clock.seconds() - tw, sink });
    }
    var frame = try target.Frame.init(a, device, &model, max_rows, capacity);
    defer frame.deinit();
    std.debug.print("{d}-bit model loaded in {d:.3}s, device allocations {d:.3} GB\n", .{ model.host.bits, mtl.clock.seconds() - t_load, @as(f64, @floatFromInt(device.allocated())) / 1e9 });

    const opts = mtl.ResourceOptions.shared;
    var run: Run = .{
        .a = a,
        .device = device,
        .queue = queue,
        .model = &model,
        .frame = &frame,
        .positions = try device.buffer(max_rows * 3 * 4, opts),
        .writes = try device.buffer(max_rows * 4, opts),
        .slide_keys = try device.buffer(max_rows * window * 4, opts),
        .slide_counts = try device.buffer(max_rows * 4, opts),
        .sparse_keys = try device.buffer(max_rows * capacity * 4, opts),
        .sparse_counts = try device.buffer(max_rows * 4, opts),
        .slide_ranges = try device.buffer(max_rows * 8, opts),
        .sparse_ranges = try device.buffer(max_rows * 8, opts),
        .tiles = try device.buffer(capacity / 32 * 4, opts),
        .caches = try a.alloc(mtl.Buffer, model.blocks.len),
        .lanes = try a.alloc(target.LayerCache, model.blocks.len),
        .sliding = try a.alloc(bool, model.blocks.len),
        .picks = try device.buffer(max_rows * 4, opts),
        .pad = t.engram_pad_id,
        .vocab = t.vocab_size,
        .names = &.{},
        .segments = try a.alloc(mtl.Segment, mtl.max_segments),
    };
    for (run.caches, run.lanes, run.sliding, 0..) |*c, *lane, *slide, l| {
        c.* = try device.buffer(@as(usize, capacity) * key_width * 2, opts);
        slide.* = t.layer_types[l] == .sliding_attention;
        lane.* = .{ .rows = .{
            .positions = .{ .buffer = run.positions },
            .write_slots = .{ .buffer = run.writes },
            .retained = .{ .buffer = if (slide.*) run.slide_keys else run.sparse_keys },
            .counts = .{ .buffer = if (slide.*) run.slide_counts else run.sparse_counts },
            .cache = .{ .buffer = c.* },
            .capacity = capacity,
            .ranges = if (std.c.getenv("TF_HY4_NO_FLASH") == null and flashFor(slide.*)) .{ .ranges = .{ .buffer = if (slide.*) run.slide_ranges else run.sparse_ranges }, .tiles = .{ .buffer = run.tiles } } else null,
        }, .key_stride = if (slide.*) window else capacity };
    }
    for (run.tiles.slice(u32, capacity / 32), 0..) |*t_, i_| t_.* = @intCast(32 * i_);
    if (std.c.getenv("TF_HY4_INDEX") != null) {
        // one score row a pass row, shared by the sparse layers; each sparse layer's own token and group keys and lists
        const scores = try device.buffer(@as(usize, max_rows) * (capacity / 4) * 4, mtl.ResourceOptions.private);
        for (run.lanes, run.sliding) |*lane, slide| {
            if (slide) continue;
            lane.index = .{
                .token_keys = .{ .buffer = try device.buffer(index.Cache.tokenBytes(capacity), opts) },
                .group_keys = .{ .buffer = try device.buffer(index.Cache.groupBytes(capacity), opts) },
                .capacity = capacity,
                .scores = .{ .buffer = scores },
                .score_groups = capacity / 4,
                .tokens = .{ .buffer = try device.buffer(@as(usize, max_rows) * index.keys_a_row * 4, opts) },
                .counts = .{ .buffer = try device.buffer(max_rows * 4, opts) },
            };
        }
        run.indexed = true;
        std.debug.print("indexer on: {d} slots\n", .{capacity});
    }
    var skip_list: []const usize = &.{};
    if (std.c.getenv("TF_HY4_SKIP")) |list| {
        // timing only (--rows-time): drop dispatches whose pipeline name contains any comma-separated part, from the
        // timed passes on (after a real warm-up), always with the router's select dropped too, so the picks stay
        // the warm-up's real ones and every variant reads the same experts
        var names: std.ArrayList(Name) = .empty;
        try collect(@TypeOf(model.common), &model.common, "common", &names, a);
        try collect(@TypeOf(model.routed), &model.routed, "routed", &names, a);
        try collect(@TypeOf(model.memory), &model.memory, "engram", &names, a);
        var ids: std.ArrayList(usize) = .empty;
        var it = std.mem.splitScalar(u8, std.mem.span(list), ',');
        while (it.next()) |part| for (names.items) |n| if (part.len > 0 and std.mem.indexOf(u8, n.name, part) != null) {
            try ids.append(a, n.id);
            std.debug.print("skip: {s}\n", .{n.name});
        };
        for (names.items) |n| if (std.mem.indexOf(u8, n.name, "routed.select") != null) try ids.append(a, n.id);
        skip_list = ids.items;
    }
    if (profile_steps > 0) {
        var names: std.ArrayList(Name) = .empty;
        try collect(@TypeOf(model.common), &model.common, "common", &names, a);
        try collect(@TypeOf(model.routed), &model.routed, "routed", &names, a);
        try collect(@TypeOf(model.memory), &model.memory, "engram", &names, a);
        try collect(mtl.Pipeline, &model.head.lookup, "head.embed", &names, a);
        run.names = names.items;
        run.pass = try mtl.counters.ComputePass.init();
        run.samples = try mtl.counters.SampleBuffer.init(device, 2 * mtl.max_segments);
        std.debug.print("profile: {d} named pipelines, counter mode {s}\n", .{ names.items.len, @tagName(mtl.counters.query(device).mode()) });
    }

    // Build the prompt list: the short set, and optionally a long prompt near `long_tokens` tokens.
    var lists: std.ArrayList([]u32) = .empty;
    for (prompts) |p| {
        const ids = try tok.encode(a, p);
        const with = try a.alloc(u32, ids.len + 1);
        with[0] = bos;
        @memcpy(with[1..], ids);
        try lists.append(a, with);
    }
    if (long_tokens > 0) {
        var text: std.ArrayList(u8) = .empty;
        var ids = try tok.encode(a, "");
        while (ids.len < long_tokens) {
            try text.appendSlice(a, long_text);
            ids = try tok.encode(a, text.items);
        }
        try text.appendSlice(a, "\nSummarize the passage above in one paragraph.\n");
        ids = try tok.encode(a, text.items);
        const with = try a.alloc(u32, ids.len + 1);
        with[0] = bos;
        @memcpy(with[1..], ids);
        try lists.append(a, with);
    }

    if (text_path) |path| {
        // one prompt from a file: its first 1,900 tokens after BOS
        const body = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
        const ids = try tok.encode(a, body);
        const n = @min(ids.len, if (std.c.getenv("TF_HY4_TEXT_TOKENS")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 1900 else 1900);
        const with = try a.alloc(u32, n + 1);
        with[0] = bos;
        @memcpy(with[1..], ids[0..n]);
        lists.items.len = 0;
        try lists.append(a, with);
    }
    // Lanes: a first, untimed copy of prompt 0 warms the draft and multi-row pipelines.
    const warm_lanes = lane_rows > 1;
    if (warm_lanes) try lists.insert(a, 0, lists.items[0]);
    var total_wall: f64 = 0;
    var total_gpu: f64 = 0;
    var total_tokens: usize = 0;
    var all_ids: std.ArrayList(u32) = .empty;
    if (std.c.getenv("TF_HY4_KCHECK")) |_| {
        // every dense kernel against an fp64 reference on real weights and random rows
        const dense = @import("metal_dense.zig");
        const Case = struct { name: []const u8, w: dense.Linear };
        const b1 = &model.blocks[1];
        const b2 = &model.blocks[2];
        var cases: std.ArrayList(Case) = .empty;
        try cases.append(a, .{ .name = "o (sparse layer)", .w = b1.attention.o });
        try cases.append(a, .{ .name = "o (sliding layer)", .w = b2.attention.o });
        try cases.append(a, .{ .name = "latent down", .w = b1.feed.routed.down });
        try cases.append(a, .{ .name = "latent up", .w = b1.feed.routed.up });
        try cases.append(a, .{ .name = "shared gate", .w = b1.feed.routed.shared_gate });
        try cases.append(a, .{ .name = "shared down", .w = b1.feed.routed.shared_down });
        try cases.append(a, .{ .name = "lm_head", .w = model.head.vocab });
        var prng = std.Random.DefaultPrng.init(7);
        const rnd = prng.random();
        for (cases.items) |c| {
            const plain = switch (c.w) {
                .portable => |w| w,
                .tensor => |tw| tw.plain orelse return error.KcheckNeedsPlain,
            };
            const n: usize = plain.n;
            const k: usize = plain.k;
            for ([_]u32{ 1, 4 }) |rows| {
                const xb = try device.buffer(rows * k * 2, opts);
                const xs = xb.slice(u16, rows * k);
                for (xs) |*v| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatCast(rnd.floatNorm(f64))))) >> 16);
                // fp64 reference
                const ref = try a.alloc(f64, rows * n);
                const words = plain.w.slice(u32, plain.w.length() / 4);
                const scales = plain.scales.slice(u16, plain.scales.length() / 2);
                const biases = plain.biases.slice(u16, plain.biases.length() / 2);
                const per_row = k * plain.bits / 32;
                const groups = k / plain.group;
                for (0..rows) |r| for (0..n) |o| {
                    var acc: f64 = 0;
                    for (0..k) |col| {
                        const bitpos = col * plain.bits;
                        const word = words[o * per_row + bitpos / 32];
                        const q: f64 = @floatFromInt((word >> @intCast(bitpos % 32)) & ((@as(u32, 1) << @intCast(plain.bits)) - 1));
                        const g = col / plain.group;
                        const wv = bf16(scales[o * groups + g]) * q + bf16(biases[o * groups + g]);
                        acc += @as(f64, bf16(xs[r * k + col])) * wv;
                    }
                    ref[r * n + o] = acc;
                };
                var worst: [3]f64 = @splat(0);
                var scale_max: f64 = 0;
                for (ref) |v| scale_max = @max(scale_max, @abs(v));
                for ([_]?u32{ 8, 0, 0 }, 0..) |override, kind| {
                    const yb = try device.buffer(rows * n * 2, opts);
                    const p2 = mtl.objc.Pool.push();
                    defer p2.pop();
                    const cb = queue.commandBuffer();
                    var split = mtl.Splitter.plain(cb, run.segments);
                    const e = split.encoder();
                    dense.decode_rows_override = override;
                    switch (kind) {
                        0 => try model.common.project(e, plain, .{ .buffer = xb }, .{ .buffer = yb }, rows),
                        1 => try model.common.project(e, plain, .{ .buffer = xb }, .{ .buffer = yb }, rows),
                        else => if (c.w == .tensor) try model.common.linear(e, c.w, .{ .buffer = xb }, .{ .buffer = yb }, rows),
                    }
                    dense.decode_rows_override = null;
                    _ = split.finish();
                    cb.commit();
                    cb.wait();
                    if (kind == 2 and c.w != .tensor) continue;
                    const ys = yb.slice(u16, rows * n);
                    for (0..rows * n) |idx| worst[kind] = @max(worst[kind], @abs(@as(f64, bf16(ys[idx])) - ref[idx]));
                }
                std.debug.print("KCHECK {s:18} n {d:6} k {d:5} bits {d} rows {d}: max |y| {d:.4}; max error: decode GEMV {d:.5}, row kernels {d:.5}, tensor lanes {d:.5}\n", .{ c.name, n, k, plain.bits, rows, scale_max, worst[0], worst[1], worst[2] });
            }
        }
        return;
    }
    if (std.c.getenv("TF_HY4_BOS_TOP")) |_| {
        // the first token's logits after BOS alone, against the MLX reference's top five
        var hist: [1]u32 = .{bos};
        _ = try run.step(&hist, 0, hist[0..1], false);
        const row = frame.logits.buffer.slice(u16, run.vocab);
        var top: [5]u32 = @splat(0);
        var used: [5]bool = @splat(false);
        _ = &used;
        for (0..5) |k| {
            var best: f32 = -std.math.inf(f32);
            var id: u32 = 0;
            for (row, 0..) |w, t_| {
                const v = bf16(w);
                var seen = false;
                for (top[0..k]) |p_| seen = seen or p_ == t_;
                if (!seen and v > best) {
                    best = v;
                    id = @intCast(t_);
                }
            }
            top[k] = id;
            std.debug.print("BOS TOP {d}: id {d} logit {d:.4}\n", .{ k, id, bf16(row[id]) });
        }
        return;
    }
    if (std.c.getenv("TF_HY4_BARRIER_TEST")) |_| {
        // the cost of a dispatch and of a barrier on this GPU: 1400 tiny argmax launches over 64 values
        const rows = model.common.rows orelse return error.Hy4ProbeNeedsRowKernels;
        for ([_]bool{ false, true }) |barriers| {
            var best: f64 = 1e9;
            for (0..5) |_| {
                const p2 = mtl.objc.Pool.push();
                defer p2.pop();
                const cb = queue.commandBuffer();
                var split = mtl.Splitter.plain(cb, run.segments);
                const e = split.encoder();
                for (0..1400) |_| {
                    rows.pick(e, frame.logits, .{ .buffer = run.picks }, 0, 1, 64);
                    if (barriers) e.barrier();
                }
                _ = split.finish();
                cb.commit();
                cb.wait();
                best = @min(best, cb.gpuSeconds());
            }
            std.debug.print("BARRIER TEST: 1400 tiny dispatches {s}: {d:.3} ms GPU ({d:.2} us each)\n", .{ if (barriers) "each followed by a barrier" else "with no barriers", 1000 * best, 1e6 * best / 1400.0 });
        }
        return;
    }
    var lanes: ?Lanes = if (lane_rows > 1) try Lanes.init(a, &run) else null;
    if (lanes) |*l| {
        if (lane_rows > l.draft.width) return error.Hy4ProbeTooManyLaneRows;
        std.debug.print("lanes: DFly2 {d} layers, block {d}, verifying {d} rows a round\n", .{ l.draft.blocks.len, l.draft.width, lane_rows });
        if (run.pass != null) {
            var more: std.ArrayList(Name) = .empty;
            try more.appendSlice(a, run.names);
            try collect(mtl.Pipeline, &l.draft.fuse, "draft.fuse", &more, a);
            try collect(mtl.Pipeline, &l.draft.residual, "draft.residual", &more, a);
            run.names = more.items;
        }
    }
    // warm the pipelines and caches: a short untimed decode
    {
        var hist: [64]u32 = undefined;
        hist[0] = bos;
        var next = (try run.step(&hist, 0, hist[0..1], false)).next;
        for (1..8) |p| {
            hist[p] = next;
            next = (try run.step(&hist, @intCast(p), hist[p .. p + 1], false)).next;
        }
    }
    if (rows_time) {
        // One prompt, `gen` plain tokens, then target passes over R real rows at a fixed position (rewritten each time).
        // With --long or --text: that long prompt instead of the short set's second, and passes that fit the cache.
        const prompt = if (long_tokens > 0 or text_path != null) lists.items[lists.items.len - 1] else lists.items[1];
        const hist = try a.alloc(u32, prompt.len + gen + max_rows);
        @memcpy(hist[0..prompt.len], prompt);
        var start: u32 = 0;
        var next: u32 = 0;
        while (start < prompt.len) {
            const n: u32 = @min(chunk, @as(u32, @intCast(prompt.len)) - start);
            next = (try run.step(hist, start, hist[start .. start + n], false)).next;
            start += n;
        }
        var pos: u32 = @intCast(prompt.len);
        const ahead: usize = if (long_tokens > 0 or text_path != null) 16 else max_rows;
        for (0..gen + ahead - 1) |_| {
            hist[pos] = next;
            next = (try run.step(hist, pos, hist[pos .. pos + 1], false)).next;
            pos += 1;
        }
        const at: u32 = @intCast(prompt.len + gen / 2);
        var one: f64 = 0;
        run.all_logits = true;
        if (skip_list.len > 0) {
            // one real pass of every timed width first, so each row's picks are its own
            _ = try run.step(hist, at, hist[at .. at + 16], false);
            mtl.skip_ids = skip_list;
        }
        for ([_]u32{ 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 32, 64, 128, 256, 512 }) |rows| {
            if (at + rows + 1 > capacity) break;
            var times: [7]f64 = undefined;
            for (&times) |*tm| tm.* = (try run.step(hist, at, hist[at .. at + rows], false)).gpu;
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            if (rows == 1) one = times[3];
            std.debug.print("ROWS {d:2}: {d:.3} ms GPU a pass (median of 7, min {d:.3}), {d:.2}x one row, {d:.3} ms a row\n", .{ rows, 1000 * times[3], 1000 * times[0], times[3] / one, 1000 * times[3] / @as(f64, @floatFromInt(rows)) });
        }
        mtl.skip_ids = &.{};
        if (std.c.getenv("TF_HY4_ACHECK") != null) try attentionCheck(&run, hist, at);
        if (profile_steps == 0) return;
        // --profile ROWS in this mode: profile three passes of ROWS rows
        for (0..3) |_| _ = try run.step(hist, at, hist[at .. at + profile_steps], true);
        lists.items.len = 0;
    }
    for (lists.items, 0..) |prompt, pi| {
        if (prompt.len + gen + 1 > capacity) return error.Hy4ProbeContextTooLong;
        const hist = try a.alloc(u32, prompt.len + gen + 16);
        @memcpy(hist[0..prompt.len], prompt);
        const tp = mtl.clock.seconds();
        var start: u32 = 0;
        var next: u32 = 0;
        var nll_sum: f64 = 0;
        var nll_count: usize = 0;
        run.all_logits = nll;
        while (start < prompt.len) {
            const n: u32 = @min(chunk, @as(u32, @intCast(prompt.len)) - start);
            next = (try run.step(hist, start, hist[start .. start + n], false)).next;
            if (nll) {
                // teacher-forced: each row's log-probability of the prompt's next token
                const all = run.frame.logits.buffer.slice(u16, @as(usize, n) * run.vocab);
                for (0..n) |r| {
                    const at = start + r + 1;
                    if (at >= prompt.len) break;
                    const row = all[r * run.vocab ..][0..run.vocab];
                    var top: f64 = -std.math.inf(f64);
                    for (row) |w| top = @max(top, bf16(w));
                    var z: f64 = 0;
                    for (row) |w| z += @exp(@as(f64, bf16(w)) - top);
                    const tok_nll = top + @log(z) - bf16(row[hist[at]]);
                    nll_sum += tok_nll;
                    nll_count += 1;
                    if (nll_dump) std.debug.print("TOKNLL {d} {d:.6}\n", .{ at, tok_nll });
                }
            }
            start += n;
        }
        run.all_logits = false;
        if (std.c.getenv("TF_HY4_ICHECK") != null and run.indexed) try indexCheck(&run);
        if (nll and nll_count > 0) std.debug.print("NLL: {d:.5} nats a token over {d} tokens (perplexity {d:.4})\n", .{ nll_sum / @as(f64, @floatFromInt(nll_count)), nll_count, @exp(nll_sum / @as(f64, @floatFromInt(nll_count))) });
        const prefill = mtl.clock.seconds() - tp;
        var pos: u32 = @intCast(prompt.len);
        var wall: f64 = 0;
        var gpu: f64 = 0;
        if (lanes) |*l| {
            // hist[pos] is the bonus token (not yet in the cache); the last pass's final row is the anchor.
            var source: u32 = (@as(u32, @intCast(prompt.len)) - 1) % chunk;
            const goal: u32 = @intCast(prompt.len + gen);
            const t0 = mtl.clock.seconds();
            const g0 = l.draft_gpu + l.verify_gpu;
            var width: u32 = lane_rows;
            while (pos < goal) {
                const round_pool = mtl.objc.Pool.push();
                defer round_pool.pop();
                const rows: u32 = @min(width, goal - pos);
                var drafts: [8]u32 = undefined;
                l.profile_rounds = @intFromBool(profile_steps > 0 and pi == 0 and l.rounds >= 2 and l.rounds < 2 + profile_steps);
                // the draft runs while the host encodes the verify pass over placeholder rows
                hist[pos] = next;
                const copied = if (copies and rows > 1) copyDraft(hist[0 .. pos + 1], drafts[0 .. rows - 1]) else 0;
                const flight = if (rows > 1 and copied < rows - 1) try l.launch(&run, pos, next, source, rows - 1) else null;
                if (copied == rows - 1) l.copy_rounds += 1;
                @memset(hist[pos + 1 .. pos + rows], l.mask);
                try run.stage(hist, pos, hist[pos .. pos + rows]);
                run.layout(pos, rows);
                const verify = try run.encodePicks(rows);
                defer verify.release();
                if (flight) |f| try l.land(&run, f, rows - 1, drafts[0 .. rows - 1]);
                @memcpy(hist[pos + 1 .. pos + rows], drafts[0 .. rows - 1]);
                const ts = mtl.clock.seconds();
                try run.stage(hist, pos, hist[pos .. pos + rows]);
                l.stage_wall += mtl.clock.seconds() - ts;
                verify.commit();
                verify.wait();
                if (verify.failure()) |why| {
                    std.debug.print("verify command buffer failed: {s}\n", .{why});
                    return error.Hy4ProbeCommandFailed;
                }
                l.verify_gpu += verify.gpuSeconds();
                const picks = run.picks.slice(u32, rows);
                var j: u32 = 0;
                while (j + 1 < rows and picks[j] == drafts[j]) j += 1;
                l.rounds += 1;
                l.kept += j;
                l.proposed += rows - 1;
                source = j;
                pos += j + 1;
                next = picks[j];
                // TF_HY4_ADAPT=lo,hi: widen after a round that kept every draft, narrow after one that kept none
                if (adapt) |range| {
                    if (j + 1 == rows and width < range[1]) width += 1 else if (j == 0 and width > range[0]) width -= 1;
                }
            }
            wall = mtl.clock.seconds() - t0;
            gpu = l.draft_gpu + l.verify_gpu - g0;
        } else for (0..gen) |g| {
            hist[pos] = next;
            const profile = profile_steps > 0 and pi == profile_prompt and g >= 2 and g < 2 + profile_steps;
            const t0 = mtl.clock.seconds();
            const r = try run.step(hist, pos, hist[pos .. pos + 1], profile);
            if (!profile) {
                wall += mtl.clock.seconds() - t0;
                gpu += r.gpu;
            }
            next = r.next;
            pos += 1;
        }
        const timed: f64 = @floatFromInt(gen - @min(gen, if (pi == 0 and lanes == null) profile_steps else 0));
        const ids = hist[prompt.len .. prompt.len + gen];
        if (warm_lanes and pi == 0) {
            if (lanes) |*l| {
                l.rounds = 0;
                l.kept = 0;
                l.proposed = 0;
                l.draft_gpu = 0;
                l.verify_gpu = 0;
                l.stage_wall = 0;
                l.copy_rounds = 0;
            }
            std.debug.print("warm: lanes over prompt 0, untimed\n", .{});
            continue;
        }
        std.debug.print("prompt {d}, {d} input tokens: prefill {d:.1} tok/s, decode {d:.3} tok/s ({d:.3} ms wall, {d:.3} ms GPU a token), hash {x:0>16}\n", .{
            pi, prompt.len, @as(f64, @floatFromInt(prompt.len)) / prefill, timed / wall, 1000 * wall / timed, 1000 * gpu / timed, fnv(ids),
        });
        std.debug.print("  ids {any}\n", .{ids});
        const text = try tok.decode(a, ids, false);
        std.debug.print("  text {s}\n", .{text});
        total_wall += wall;
        total_gpu += gpu;
        total_tokens += @intFromFloat(timed);
        try all_ids.appendSlice(a, ids);
    }
    if (lanes) |l| std.debug.print("LANES: {d} rounds, {d}/{d} drafts kept, {d:.3} tokens a round, draft {d:.3} ms and verify {d:.3} ms GPU a round, Engram restage {d:.3} ms a round, {d} copy rounds\n", .{
        l.rounds, l.kept, l.proposed, @as(f64, @floatFromInt(l.kept + l.rounds)) / @as(f64, @floatFromInt(l.rounds)), 1000 * l.draft_gpu / @as(f64, @floatFromInt(l.rounds)), 1000 * l.verify_gpu / @as(f64, @floatFromInt(l.rounds)), 1000 * l.stage_wall / @as(f64, @floatFromInt(l.rounds)), l.copy_rounds,
    });
    const tt: f64 = @floatFromInt(total_tokens);
    std.debug.print("SUMMARY: {d} prompts, decode {d:.3} tok/s, {d:.3} ms wall and {d:.3} ms GPU a token, all-ids hash {x:0>16}\n", .{ lists.items.len - @intFromBool(warm_lanes), tt / total_wall, 1000 * total_wall / tt, 1000 * total_gpu / tt, fnv(all_ids.items) });

    if (run.profiled > 0) {
        const steps: f64 = @floatFromInt(run.profiled);
        const items = run.stages.values();
        std.mem.sort(Stage, items, {}, struct {
            fn more(_: void, x: Stage, y: Stage) bool {
                return x.ns > y.ns;
            }
        }.more);
        var sum: f64 = 0;
        for (items) |s| sum += s.ns;
        std.debug.print("PROFILE over {d} decode steps: {d:.3} ms a token in {d} timed segments a token\n", .{ run.profiled, sum / steps / 1e6, @divTrunc(blk: {
            var c: usize = 0;
            for (items) |s| c += s.count;
            break :blk c;
        }, run.profiled) });
        for (items) |s| {
            std.debug.print("  {d:8.3} ms {d:5.1}%  x{d:<4} {s}\n", .{ s.ns / steps / 1e6, 100 * s.ns / sum, s.count / run.profiled, s.label });
        }
    }
}
