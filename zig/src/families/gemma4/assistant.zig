//! Gemma 4's MTP assistant as a drafter: a stream's last kept row, then chains over the target's own last caches.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const aw = @import("assistant_weights.zig");
const kernels = @import("kernels.zig");
const weights = @import("weights.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const backend = @import("backend.zig");
const round = @import("tree_round_gpu");
const lane = @import("../../core/lane_projection.zig");
const Model = @import("model.zig").Model;

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Ref = lane.Ref;
const Pick = round.contract.Pick;
const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const private = mtl.ResourceOptions.private | mtl.ResourceOptions.untracked;

/// Drafts a round at most (the reference's default; the head was trained for 3).
pub const depth = 6;
/// Streams one hold or absorb call takes.
const max_items = 32;
/// Keys a gemma_decode_attention threadgroup takes (its CD), and a sliding partial's (the sliding kind's CK).
const global_chunk = 256;
const sliding_chunk = 128;

/// One stream's assistant state: the target's final-normed last kept row, its position, the token after it.
const Lane = struct {
    h: mtl.Buffer,
    position: u64 = 0,
    token: u32 = 0,
    rows: bool = false, // a row was absorbed
    held: [depth]u32 = @splat(0),
    held_len: u32 = 0,
};

/// A stream's chain in a hold: its lane and the target cache it reads, from the row's position.
const Item = struct { lane: *Lane, cache: *const st.Cache, pending: u32, position: u64, depth: u32 };

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    target: *Model,
    metal: *backend.Metal,
    config: aw.Config,
    w: aw.Weights,
    /// The assistant's glue at its own width: sliding and global kinds (q-only qkv_prep, the target's key heads).
    kinds: [2]kernels.Kind,
    projections: weights.Projections,
    follow: ?mtl.ResidencySet.Follow = null,
    lanes: std.AutoHashMapUnmanaged(*Stream, *Lane) = .empty,
    /// The target's layers each kind reads: its last sliding and its last global layer.
    sources: [2]usize,
    tap: [1]u32,
    step_ms: f64 = 0,
    chain_ms: [2]f64 = .{ 0, 0 }, // a 1-draft and a 6-draft chain's wall time at load
    trace: bool = false, // GEMMA_DRAFT_TRACE: each hold's rows and drafts on stderr
    /// An absorb's command buffer, committed without a wait: the hold that follows (the same queue) runs after it.
    pending: ?mtl.CommandBuffer = null,
    // one chain's scratch, a row a stream
    ids: mtl.Buffer,
    pos: mtl.Buffer,
    lo: mtl.Buffer,
    meta: [2]mtl.Buffer,
    picks: mtl.Buffer,
    hin: mtl.Buffer,
    x: mtl.Buffer,
    h: [2]mtl.Buffer,
    normed: mtl.Buffer,
    hn: mtl.Buffer,
    n_mlp: mtl.Buffer,
    q_raw: mtl.Buffer,
    q: mtl.Buffer,
    attn: mtl.Buffer,
    o: mtl.Buffer,
    gate_up: mtl.Buffer,
    act: mtl.Buffer,
    y: mtl.Buffer,
    pm: mtl.Buffer,
    pl: mtl.Buffer,
    po: mtl.Buffer,
    sums: mtl.Buffer,
    logits: mtl.Buffer,

    const scratch = .{ "ids", "pos", "lo", "picks", "hin", "x", "normed", "hn", "n_mlp", "q_raw", "q", "attn", "o", "gate_up", "act", "y", "pm", "pl", "po", "sums", "logits" };

    /// `metal`: the target's backend, whose stream caches the assistant reads; its streams' capacity sizes the scratch.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, metal: *backend.Metal, dir: []const u8) !*Drafter {
        const target = metal.model;
        // the assistant's own weights are the lane layout: M5 only. The target's row path does not serve them.
        if (!target.device.tensorUnits()) return error.GemmaDraftNeedsTensorUnits;
        const geo = target.config.geo;
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = undefined;
        d.gpa = gpa;
        d.io = io;
        d.target = target;
        d.metal = metal;
        d.lanes = .empty;
        d.step_ms = 0;
        d.chain_ms = .{ 0, 0 };
        d.pending = null;
        d.follow = null;
        d.trace = std.c.getenv("GEMMA_DRAFT_TRACE") != null;
        d.projections = .{ .geo = geo };
        errdefer d.projections.deinit(gpa); // warm's chains fill it
        d.config = try aw.Config.read(gpa, io, dir, geo);
        const c = &d.config;
        d.sources = .{ 0, 0 };
        for (0..geo.layers) |i| d.sources[@intFromBool(cfg.global(i))] = i;
        d.tap = .{@intCast(geo.layers - 1)};
        d.w = try aw.load(gpa, target.device, io, dir, c, geo);
        errdefer d.w.deinit(gpa);
        d.kinds[0] = try kernels.Kind.init(gpa, target.device, c.glue, false, true, target.weights.embed_bits);
        errdefer d.kinds[0].deinit();
        d.kinds[1] = try kernels.Kind.init(gpa, target.device, c.glue, true, true, target.weights.embed_bits);
        errdefer d.kinds[1].deinit();
        const n = max_items;
        const hid: usize = c.hidden;
        const qw = c.heads * cfg.global_dim;
        const chunks = @max(metal.options.capacity / global_chunk + 2, (cfg.window + sliding_chunk) / sliding_chunk + 2);
        const slots = c.heads * chunks;
        const sizes = .{ n * 4, n * 4, n * 4, depth * n * @sizeOf(Pick), n * geo.hidden * 2, n * 2 * geo.hidden * 2, n * hid * 2, n * hid * 2, n * hid * 2, n * qw * 2, n * qw * 2, n * qw * 2, n * hid * 2, n * 2 * c.intermediate * 2, n * c.intermediate * 2, n * hid * 2, slots * 4, slots * 4, slots * cfg.global_dim * 4, n * (@max(c.intermediate, qw) / 64) * 4, n * cfg.vocab * 2 };
        const host = .{ true, true, true, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false };
        var made: usize = 0;
        errdefer d.free(made);
        inline for (scratch, sizes, host) |f, bytes, on_host| {
            @field(d, f) = try target.device.buffer(bytes, if (on_host) shared else private);
            made += 1;
        }
        for (&d.meta) |*b| {
            b.* = try target.device.buffer(n * st.meta_ints * 4, shared);
            made += 1;
        }
        for (&d.h) |*b| {
            b.* = try target.device.buffer(n * hid * 2, private);
            made += 1;
        }
        try d.warm();
        metal.resume_needs_draft = false;
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        d.settle() catch {};
        d.setResident(null);
        var it = d.lanes.valueIterator();
        while (it.next()) |l| d.dropLane(l.*);
        d.lanes.deinit(d.gpa);
        d.free(std.math.maxInt(usize));
        d.projections.deinit(d.gpa);
        for (d.kinds) |k| k.deinit();
        d.w.deinit(d.gpa);
        d.gpa.destroy(d);
    }

    pub fn setResident(d: *Drafter, hook: ?mtl.ResidencySet.Follow) void {
        @import("draft_residency.zig").set(d, hook);
    }

    /// Releases the first `made` scratch buffers in init's order, so a failed init frees what it made.
    fn free(d: *Drafter, made: usize) void {
        var i: usize = 0;
        inline for (scratch) |f| {
            if (i < made) @field(d, f).deinit();
            i += 1;
        }
        for (d.meta ++ d.h) |b| {
            if (i < made) b.deinit();
            i += 1;
        }
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .kind = "MTP assistant", .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
    }

    fn cast(ptr: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(ptr));
    }

    /// The target's last layer: its output before the final norm, which absorb applies.
    fn tapsFn(ptr: *anyopaque) []const u32 {
        return &cast(ptr).tap;
    }

    /// Chains of up to 6, one assistant forward a draft: the rule prices each draft and may choose none.
    fn factsFn(ptr: *anyopaque) dr.Facts {
        const d = cast(ptr);
        return .{ .depth = depth, .step_ms = d.step_ms, .batched = true, .stream_bytes = d.target.config.geo.hidden * 2 };
    }

    fn openFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const d = cast(ptr);
        if (d.lanes.contains(s)) return error.DraftStreamOpen;
        const l = try d.newLane();
        errdefer d.dropLane(l);
        try d.lanes.put(d.gpa, s, l);
    }

    pub fn newLane(d: *Drafter) !*Lane {
        const l = try d.gpa.create(Lane);
        errdefer d.gpa.destroy(l);
        l.* = .{ .h = try d.target.device.buffer(d.target.config.geo.hidden * 2, private) };
        if (d.follow) |f| @import("draft_residency.zig").lane(d, l, f, true);
        return l;
    }

    pub fn dropLane(d: *Drafter, l: *Lane) void {
        if (d.follow) |f| @import("draft_residency.zig").lane(d, l, f, false);
        l.h.deinit();
        d.gpa.destroy(l);
    }

    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const d = cast(ptr);
        if (d.lanes.fetchRemove(s)) |e| d.dropLane(e.value);
    }

    fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
        const d = cast(ptr);
        for (streams, out) |s, o| {
            const l = d.lanes.get(s) orelse return error.DraftStreamUnknown;
            if (o.len > l.held_len) return error.DraftHeldFewer;
            @memcpy(o, l.held[0..o.len]);
        }
    }

    fn failed(msg: [*:0]const u8) anyerror {
        std.log.err("gemma4 assistant: command buffer failed: {s}", .{msg});
        return error.GpuFailed;
    }

    /// Waits for a pending absorb's command buffer (a hold ordered after it on the queue has run it already).
    fn settle(d: *Drafter) !void {
        const cb = d.pending orelse return;
        d.pending = null;
        defer mtl.objc.release(cb.id); // retained when it was left pending (an autoreleased object otherwise)
        cb.wait();
        if (cb.failure()) |msg| return failed(msg);
    }

    /// Each stream's last absorbed row through the target's final norm into its lane; committed, not awaited.
    fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = cast(ptr);
        const t = d.target;
        const width = t.config.geo.hidden * 2;
        try d.settle();
        const cb = t.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (items) |it| {
            const l = d.lanes.get(it.stream) orelse return error.DraftStreamUnknown;
            const f = it.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != width or f.rows == 0 or it.follow.len < f.rows) return error.DraftFeaturesUnsupported;
            const last = f.rows - 1;
            e.setPipeline(t.kernels.sliding.rms);
            e.setBuffer(mtl.Buffer{ .id = @ptrFromInt(f.buffer) }, f.offset + last * f.row_bytes, 0);
            e.setBuffer(t.weights.final_norm.buf, t.weights.final_norm.off, 1);
            e.setBuffer(l.h, 0, 2);
            e.setValue(cfg.eps, 3);
            e.setBuffer(d.sums, 0, 4);
            e.setValue(@as(i32, 0), 5); // no group sums: the row is copied into a chain's input before it is read
            e.dispatchGroups(.{ .width = 1 }, .{ .width = 1024 });
            l.position = it.start + last;
            l.token = it.follow[last];
            l.rows = true;
            l.held_len = 0;
            if (d.trace) std.debug.print("absorb {s}: rows [{d}, {d}), kept row {d}, token {d}\n", .{ it.stream.id, it.start, it.start + f.rows, l.position, l.token });
        }
        e.end();
        cb.commit();
        _ = mtl.objc.retain(cb.id);
        d.pending = cb;
    }

    /// A projection; `summed`: its input's group sums are in d.sums (stride rows), written by the input's own kernel.
    fn project(d: *Drafter, e: mtl.ComputeEncoder, l: aw.Linear, x: mtl.Buffer, y: mtl.Buffer, rows: usize, summed: bool) !void {
        const p = try d.projections.get(d.gpa, d.target.device, .{ .w = l.w, .sb = l.sb, .n = l.n, .k = l.k });
        if (summed) return p.encodeSums(e, .{ .buf = x }, .{ .buf = l.w }, .{ .buf = l.sb }, .{ .ref = .{ .buf = d.sums }, .stride = @intCast(rows) }, .{ .buf = y }, @intCast(rows));
        try p.encode(e, .{ .buf = x }, .{ .buf = l.w }, .{ .buf = l.sb }, .{ .buf = y }, @intCast(rows));
    }

    fn sumsOut(d: *Drafter, e: mtl.ComputeEncoder, rows: usize, index: usize) void {
        e.setBuffer(d.sums, 0, index);
        e.setValue(@as(i32, @intCast(rows)), index + 1);
    }

    fn bindRef(e: mtl.ComputeEncoder, r: Ref, index: usize) void {
        e.setBuffer(r.buf, r.off, index);
    }

    /// The host inputs of a chain: each row's pending token, position, first sliding key and both kinds' meta.
    fn prepare(d: *Drafter, items: []const Item) void {
        const ids = d.ids.slice(u32, max_items);
        const pos = d.pos.slice(i32, max_items);
        const lo = d.lo.slice(i32, max_items);
        for (items, 0..) |it, r| {
            const p = it.position;
            ids[r] = it.pending;
            pos[r] = @intCast(p);
            lo[r] = @intCast(if (p + 1 <= cfg.window) 0 else p + 1 - cfg.window);
            const c0: usize = @as(usize, @intCast(lo[r])) / sliding_chunk;
            const metas = [2][5]i32{ .{ @intCast(c0), @intCast(p / sliding_chunk - c0 + 1), st.ring, 1, std.math.maxInt(i32) }, .{ 0, @intCast(p / global_chunk + 1), 0, 1, std.math.maxInt(i32) } };
            for (d.meta, metas) |m, v| m.slice(i32, max_items * st.meta_ints)[r * st.meta_ints ..][0..5].* = v;
        }
    }

    /// Item r's queries over its target cache's layer of kind `global` (keys up to its position), merged into d.attn.
    fn attend(d: *Drafter, e: mtl.ComputeEncoder, it: Item, r: usize, rows: usize, global: bool) void {
        const kd = &d.kinds[@intFromBool(global)];
        const src = d.sources[@intFromBool(global)];
        const c = it.cache;
        const nk = if (global) d.config.glue.global_kv_heads else d.config.glue.sliding_kv_heads;
        const dh: usize = if (global) cfg.global_dim else cfg.sliding_dim;
        const q_off = r * d.config.heads * dh * 2;
        const meta = d.meta[@intFromBool(global)];
        const nch: usize = @intCast(meta.slice(i32, max_items * st.meta_ints)[r * st.meta_ints + 1]);
        if (global) {
            e.setPipeline(kd.decode_attn);
            e.setBuffer(d.q, q_off, 0);
            e.setBuffer(c.k[src], 0, 1);
            e.setBuffer(c.v[src], 0, 2);
            e.setValue([4]i32{ 1, @intCast(it.position), @intCast(st.slotsOf(src, c.capacity)), @intCast(nch) }, 3);
            for ([_]mtl.Buffer{ d.pm, d.pl, d.po }, 4..) |b, i| e.setBuffer(b, 0, i);
            e.dispatchGroups(.{ .width = nch, .height = nk, .depth = (kd.heads_per_kv + 15) / 16 }, .{ .width = 32 * dh / 128 });
        } else {
            e.setPipeline(kd.partial);
            e.setBuffer(d.q, q_off, 0);
            for ([_]mtl.Buffer{ c.k[src], c.v[src], c.k[src], c.v[src] }, 1..) |b, i| e.setBuffer(b, 0, i); // none new: P0 past all
            e.setBuffer(d.pos, r * 4, 5);
            e.setBuffer(d.lo, r * 4, 6);
            e.setBuffer(meta, r * st.meta_ints * 4, 7);
            e.setValue(@as(i32, @intCast(st.slotsOf(src, c.capacity))), 8);
            for ([_]mtl.Buffer{ d.pm, d.pl, d.po, d.q_raw }, 9..) |b, i| e.setBuffer(b, 0, i);
            const qn = d.w.layers[0].q_norm;
            bindRef(e, qn, 13);
            bindRef(e, qn, 14);
            e.setBuffer(d.target.inv_sliding, 0, 15);
            e.setValue(cfg.eps, 16);
            e.setValue([2]i32{ 0, 0 }, 17); // q prepped already, partials merged below
            e.setBuffer(d.attn, q_off, 18);
            e.setBuffer(d.sums, r * 4, 19);
            e.setValue(@as(i32, @intCast(rows)), 20);
            e.setBuffer(c.k[src], 0, 21);
            e.setBuffer(c.v[src], 0, 22);
            e.dispatchGroups(.{ .width = nch, .height = nk }, .{ .width = 32 * kd.heads_per_kv * kd.split });
        }
        e.setPipeline(kd.merge);
        for ([_]mtl.Buffer{ d.pm, d.pl, d.po }, 0..) |b, i| e.setBuffer(b, 0, i);
        e.setBuffer(meta, r * st.meta_ints * 4, 3);
        e.setBuffer(d.attn, q_off, 4);
        e.setBuffer(d.sums, r * 4, 5); // this row's o projection sums
        e.setValue(@as(i32, @intCast(rows)), 6);
        e.dispatchGroups(.{ .width = 1, .height = d.config.heads, .depth = 1 }, .{ .width = 32 });
    }

    /// Assistant layer i on the rows: input normed in d.normed (sums written), residual d.h[cur] into d.h[1 - cur].
    fn layer(d: *Drafter, e: mtl.ComputeEncoder, items: []const Item, i: usize, cur: usize) !void {
        const c = &d.config;
        const rows = items.len;
        const global = !c.sliding[i];
        const kd = &d.kinds[@intFromBool(global)];
        const glue = &d.kinds[0];
        const l = &d.w.layers[i];
        try d.project(e, l.q, d.normed, d.q_raw, rows, true);
        e.setPipeline(kd.qkv);
        e.setBuffer(d.q_raw, 0, 0);
        bindRef(e, l.q_norm, 1);
        bindRef(e, l.q_norm, 2);
        e.setBuffer(if (global) d.target.inv_global else d.target.inv_sliding, 0, 3);
        e.setBuffer(d.pos, 0, 4);
        e.setValue(cfg.eps, 5);
        for ([_]mtl.Buffer{ d.q, d.o, d.o, d.o, d.o }, 6..) |b, j| e.setBuffer(b, 0, j); // k, v and the caches unwritten
        e.setValue([3]i32{ 0, 0, 0 }, 11);
        e.dispatchGroups(.{ .width = c.heads, .height = rows }, .{ .width = 32 });
        for (items, 0..) |it, r| d.attend(e, it, r, rows, global);
        try d.project(e, l.o, d.attn, d.o, rows, true);
        e.setPipeline(glue.attn_tail);
        e.setBuffer(d.h[cur], 0, 0);
        e.setBuffer(d.o, 0, 1);
        bindRef(e, l.post_attention, 2);
        bindRef(e, l.pre_feedforward, 3);
        e.setValue(cfg.eps, 4);
        e.setBuffer(d.hn, 0, 5);
        e.setBuffer(d.n_mlp, 0, 6);
        d.sumsOut(e, rows, 7);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
        try d.project(e, l.gate_up, d.n_mlp, d.gate_up, rows, true);
        e.setPipeline(glue.geglu);
        e.setBuffer(d.gate_up, 0, 0);
        e.setBuffer(d.act, 0, 1);
        e.setValue(@as(u32, c.intermediate), 2);
        d.sumsOut(e, rows, 3);
        e.setValue(@as(i32, 0), 5); // gate rows then up rows, as quantized
        e.dispatchGroups(.{ .width = c.intermediate / 64, .height = rows }, .{ .width = 64 });
        try d.project(e, l.down, d.act, d.y, rows, true);
        e.setPipeline(glue.dense_tail);
        e.setBuffer(d.hn, 0, 0);
        e.setBuffer(d.y, 0, 1);
        bindRef(e, l.post_feedforward, 2);
        bindRef(e, l.scalar, 3);
        bindRef(e, if (i + 1 < c.layers) d.w.layers[i + 1].input_norm else d.w.norm, 4);
        e.setValue(cfg.eps, 5);
        e.setBuffer(d.h[1 - cur], 0, 6);
        e.setBuffer(d.normed, 0, 7);
        d.sumsOut(e, rows, 8);
        e.setBuffer(d.hn, 0, 10);
        e.setValue(@as(i32, 0), 11); // no tap
        e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
    }

    /// Every item's chain of `steps` drafts in one pass: step s's picks at d.picks[s * max_items + row].
    fn encodeChain(d: *Drafter, e: mtl.ComputeEncoder, items: []const Item, steps: u32) !void {
        const c = &d.config;
        const t = d.target;
        const geo = t.config.geo;
        const rows = items.len;
        d.prepare(items);
        for (items, 0..) |it, r| { // step 0's hidden rows: each lane's normed row
            e.setPipeline(t.kernels.sliding.draft_store);
            e.setBuffer(it.lane.h, 0, 0);
            e.setBuffer(d.hin, 0, 1);
            e.setValue([4]u32{ @intCast(geo.hidden), @intCast(geo.hidden), @intCast(r), 0 }, 2);
            e.dispatchThreads(.{ .width = geo.hidden, .height = 1 }, .{ .width = 256 });
        }
        for (0..steps) |s| {
            e.setPipeline(t.kernels.sliding.assist_input);
            if (s == 0) e.setBuffer(d.ids, 0, 0) else e.setBuffer(d.picks, (s - 1) * max_items * @sizeOf(Pick), 0);
            e.setValue([2]u32{ if (s == 0) 1 else @sizeOf(Pick) / 4, cfg.vocab }, 1);
            bindRef(e, t.weights.embed_w, 2);
            bindRef(e, t.weights.embed_s, 3);
            bindRef(e, t.weights.embed_b, 4);
            e.setBuffer(d.hin, 0, 5);
            e.setBuffer(d.x, 0, 6);
            e.setValue(fwd.embedScale(geo.hidden), 7);
            e.dispatchThreads(.{ .width = geo.hidden / 2, .height = rows }, .{ .width = 256 });
            try d.project(e, d.w.pre, d.x, d.h[0], rows, false);
            e.setPipeline(d.kinds[0].rms);
            e.setBuffer(d.h[0], 0, 0);
            bindRef(e, d.w.layers[0].input_norm, 1);
            e.setBuffer(d.normed, 0, 2);
            e.setValue(cfg.eps, 3);
            d.sumsOut(e, rows, 4);
            e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
            var cur: usize = 0;
            for (0..c.layers) |i| {
                try d.layer(e, items, i, cur);
                cur = 1 - cur;
            }
            try d.project(e, d.w.head, d.normed, d.logits, rows, true);
            var first: usize = 0;
            while (first < rows) : (first += 16) { // tree_round's argmax in 16-row pieces, as the target's forward
                const out = (s * max_items + first) * @sizeOf(Pick);
                try t.picks.argmax(e, .{ .buf = d.logits, .off = first * cfg.vocab * 2 }, .{ .buf = d.picks, .off = out }, .{ .rows = @intCast(@min(16, rows - first)), .vocab = cfg.vocab, .stride = cfg.vocab });
            }
            if (s + 1 < steps) try d.project(e, d.w.post, d.normed, d.hin, rows, true); // the next step's hidden rows
        }
    }

    /// Each stream's chain of `depth` drafts after `pending`, all streams' rows through each step together.
    fn holdFn(ptr: *anyopaque, holds: []const dr.Hold) anyerror!void {
        const d = cast(ptr);
        if (holds.len > max_items) return error.DraftTooManyStreams;
        const t0 = std.Io.Clock.awake.now(d.io);
        var items: [max_items]Item = undefined;
        var steps: u32 = 0;
        for (holds, items[0..holds.len]) |h, *it| {
            const l = d.lanes.get(h.stream) orelse return error.DraftStreamUnknown;
            const c = d.metal.caches.get(h.stream) orelse return error.DraftStreamUnknown;
            if (h.depth == 0 or h.depth > depth) return error.DraftDepth;
            if (!l.rows or l.position + 2 != h.position or c.len < h.position - 1) return error.DraftContextGap;
            it.* = .{ .lane = l, .cache = c, .pending = h.pending, .position = l.position, .depth = h.depth };
            steps = @max(steps, h.depth);
        }
        try d.run(items[0..holds.len], steps);
        const picks = d.picks.slice(Pick, depth * max_items);
        for (items[0..holds.len], holds, 0..) |it, h, r| {
            for (it.lane.held[0..it.depth], 0..) |*tok, s| {
                const p = picks[s * max_items + r];
                if (p.nonfinite != 0 or p.token >= cfg.vocab) return error.DraftNonfinite;
                tok.* = p.token;
            }
            it.lane.held_len = it.depth;
            if (d.trace) std.debug.print("draft {s} row {d} pending {d} depth {d}: {any}\n", .{ h.stream.id, it.position, it.pending, it.depth, it.lane.held[0..it.depth] });
        }
        if (d.trace) std.debug.print("hold of {d} streams: {d:.2} ms\n", .{ holds.len, @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(d.io)).toNanoseconds())) / 1e6 });
    }

    /// One command buffer of chains, waited on; the absorb before it (the same queue) has run by then.
    fn run(d: *Drafter, items: []const Item, steps: u32) !void {
        const cb = d.target.queue.commandBuffer();
        const e = cb.compute(.serial);
        try d.encodeChain(e, items, steps);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| return failed(msg);
        try d.settle();
    }

    /// Chains over a throwaway cache (the two layers it reads): the pipelines compiled, then 1 and 6 drafts timed.
    fn warm(d: *Drafter) !void {
        const geo = d.target.config.geo;
        var cache: st.Cache = .{ .k = undefined, .v = undefined, .geo = geo, .capacity = 256, .logits = undefined };
        var made: usize = 0;
        defer for (d.sources[0..made]) |i| {
            cache.k[i].deinit();
            cache.v[i].deinit();
        };
        for (d.sources) |i| {
            const bytes = geo.kvHeads(i) * st.slotsOf(i, cache.capacity) * cfg.headDim(i) * 2;
            cache.k[i] = try d.target.device.buffer(bytes, private);
            cache.v[i] = d.target.device.buffer(bytes, private) catch |e| {
                cache.k[i].deinit();
                return e;
            };
            made += 1;
        }
        const l = try d.newLane();
        defer d.dropLane(l);
        var ms: [2]f64 = undefined;
        for ([_]u32{ depth, 1, depth, 1, depth }, 0..) |steps, i| {
            const t0 = std.Io.Clock.awake.now(d.io);
            try d.run(&.{.{ .lane = l, .cache = &cache, .pending = 2, .position = 128, .depth = steps }}, steps);
            if (i >= 3) ms[i - 3] = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(d.io)).toNanoseconds())) / 1e6;
        }
        d.chain_ms = ms;
        d.step_ms = @max(ms[1] - ms[0], 0) / @as(f64, depth - 1);
    }
};

test "a hold's key range and chunk counts follow the kept row's position" {
    const cases = [_][3]u64{ .{ 0, 0, 1 }, .{ 1023, 0, 8 }, .{ 1024, 1, 9 }, .{ 5000, 3977, 9 } };
    for (cases) |k| {
        const p = k[0];
        const lo: u64 = if (p + 1 <= cfg.window) 0 else p + 1 - cfg.window;
        try std.testing.expectEqual(k[1], lo);
        try std.testing.expectEqual(k[2], p / sliding_chunk - lo / sliding_chunk + 1);
        try std.testing.expect(p + 1 - lo <= cfg.window);
    }
}
