//! One command buffer verifies every layer; cache publication waits for the accepted path.
const std = @import("std");
const mtl = @import("metal");
const source = @import("qwen_runtime_sources");
const Model = @import("model.zig").Model;
const Frame = @import("gpu_frame.zig").Frame;
const forward = @import("forward.zig");
/// Layers a verify encodes before committing them, so the GPU starts while the CPU encodes the rest.
const commit_every = 4;
const wts = @import("gpu_weights.zig");
const attn = @import("attention.zig");
const abi = @import("gdn_contract.zig");
const GdnPool = @import("gdn_pool_gpu.zig").Pool;
const GdnHook = @import("gdn_hook.zig").Hook;
const GdnPlan = @import("gdn_plan.zig").Plan;
const Round = @import("round_plan.zig").Round;
const Taps = @import("taps.zig").Taps;

pub const Runner = struct {
    allocator: std.mem.Allocator,
    model: *Model,
    gdn: GdnPool,
    attention: attn.Attention,
    scratch: attn.Scratch,
    bindings: []attn.Binding,
    caches: []attn.Cache,
    linear_index: []u32,
    attention_index: []u32,
    offsets: []u32,
    owned: std.ArrayList(mtl.Buffer),
    taps: Taps,
    slots: u32,
    ready: u32, // slots [0, ready) have their K/V rows; a later slot takes its own when first used (ensure)
    capacity: u32,
    active: ?[32]u8 = null,
    failed: bool = false,
    capture_taps: bool = false,
    accepted_chain: bool = false,
    last_gpu_seconds: f64 = 0,
    current_round: ?*const Round = null,
    current_gdn: ?*const GdnPlan = null,
    current_attention: ?*const attn.Plan = null,

    pub fn init(a: std.mem.Allocator, m: *Model, slots: u32, capacity: u32) !Runner {
        return initReady(a, m, slots, capacity, slots);
    }

    /// A runner of `slots` whose first `ready` take their K/V rows now, the rest when first used (ensure).
    pub fn initReady(a: std.mem.Allocator, m: *Model, slots: u32, capacity: u32, ready: u32) !Runner {
        if (slots == 0 or slots > 8 or capacity == 0 or m.frame.capacity % 16 != 0 or ready == 0 or ready > slots) return error.BadRoundCapacity;
        const linear_index = try a.alloc(u32, m.config.layers);
        errdefer a.free(linear_index);
        const attention_index = try a.alloc(u32, m.config.layers);
        errdefer a.free(attention_index);
        var ng: u32 = 0;
        var na: u32 = 0;
        for (linear_index, attention_index, 0..) |*g, *v, block| {
            g.* = std.math.maxInt(u32);
            v.* = std.math.maxInt(u32);
            if (m.config.kind(block) == .linear) {
                g.* = ng;
                ng += 1;
            } else {
                v.* = na;
                na += 1;
            }
        }
        var pool = try GdnPool.init(m.device, try abi.Shape.init(m.config), ng, slots, m.frame.capacity);
        errdefer pool.deinit();
        const units = m.device.tensorUnits();
        // before M5 the tail attention reads its tensor-op results in the simdgroup-matrix layout (simd_attention.zig)
        if (!units) try @import("simd_attention.zig").check(m.device, m.queue);
        const text = if (units) try a.dupe(u8, source.glue ++ source.attention_mpp ++ source.attention_io ++ source.attention_prompt) else try @import("simd_attention.zig").rewrite(a, source.glue ++ source.attention_mpp ++ source.attention_io);
        defer a.free(text);
        const library = try mtl.Library.fromSource(m.device, text, mtl.CompileOptions.mlx());
        defer library.deinit();
        var attention = try attn.Attention.init(m.device, library, units);
        errdefer attention.deinit();
        const limits = attn.Limits{ .rows = m.frame.capacity, .streams = slots, .keys = capacity, .shared_prefix = m.device.tensorUnits() };
        var scratch = try attn.Scratch.init(m.device, limits);
        errdefer scratch.deinit();
        const bindings = try a.alloc(attn.Binding, na);
        var made: usize = 0;
        errdefer {
            for (bindings[0..made]) |*b| b.deinit();
            a.free(bindings);
        }
        for (bindings) |*b| {
            b.* = try attn.Binding.init(m.device, limits);
            made += 1;
        }
        const caches = try a.alloc(attn.Cache, na * slots);
        errdefer a.free(caches);
        var owned: std.ArrayList(mtl.Buffer) = .empty;
        errdefer {
            for (owned.items) |b| b.deinit();
            owned.deinit(a);
        }
        for (0..na) |layer| for (0..ready) |slot| {
            caches[layer * slots + slot] = try slotCache(a, m, &owned, capacity);
        };
        const offsets = try a.alloc(u32, slots);
        errdefer a.free(offsets);
        @memset(offsets, 0);
        const taps = try Taps.init(m.device, m.frame.capacity, @intCast(m.config.hidden));
        return .{ .allocator = a, .model = m, .gdn = pool, .attention = attention, .scratch = scratch, .bindings = bindings, .caches = caches, .linear_index = linear_index, .attention_index = attention_index, .offsets = offsets, .owned = owned, .taps = taps, .slots = slots, .ready = ready, .capacity = capacity };
    }

    /// One slot's K/V rows across the attention layers.
    pub fn slotBytes(r: *const Runner) usize {
        return r.caches.len / r.slots * 2 * @as(usize, r.capacity) * r.model.config.kvDim() * 2;
    }

    /// Slot `slot`'s K/V rows (the next unready slot, as free slots are taken lowest first); its new buffers.
    pub fn ensure(r: *Runner, slot: u32) ![]const mtl.Buffer {
        if (slot < r.ready) return &.{};
        if (slot != r.ready) return error.SlotOutOfOrder;
        const first = r.owned.items.len;
        errdefer {
            for (r.owned.items[first..]) |b| b.deinit();
            r.owned.shrinkRetainingCapacity(first);
        }
        const layers = r.caches.len / r.slots;
        for (0..layers) |layer| r.caches[layer * r.slots + slot] = try slotCache(r.allocator, r.model, &r.owned, r.capacity);
        r.ready += 1;
        return r.owned.items[first..];
    }

    /// The K/V buffers of the ready slots (for a residency set).
    pub fn readyBuffers(r: *const Runner, a: std.mem.Allocator, out: *std.ArrayList(mtl.Buffer)) !void {
        for (0..r.caches.len / r.slots) |layer| for (0..r.ready) |slot| {
            const c = r.caches[layer * r.slots + slot];
            try out.appendSlice(a, &.{ c.keys.buffer, c.values.buffer });
        };
    }

    pub fn deinit(r: *Runner) void {
        r.taps.deinit();
        r.gdn.deinit();
        r.attention.deinit();
        r.scratch.deinit();
        for (r.bindings) |*b| b.deinit();
        for (r.owned.items) |b| b.deinit();
        r.owned.deinit(r.allocator);
        r.allocator.free(r.bindings);
        r.allocator.free(r.caches);
        r.allocator.free(r.linear_index);
        r.allocator.free(r.attention_index);
        r.allocator.free(r.offsets);
    }

    pub fn profile(r: *Runner, trace: ?*@import("core").gpu_profile.Trace) void {
        r.model.kernels.profiler = trace;
        r.model.glue.profiler = trace;
        r.gdn.profiler = trace;
        r.attention.profiler = trace;
    }

    pub fn verify(r: *Runner, round: *const Round) !void {
        return r.verifyHead(round, .all);
    }

    /// Prompt chunks may skip nonfinal logits or project just their last row, using the identical lane math.
    pub fn verifyHead(r: *Runner, round: *const Round, head: forward.Head) !void {
        return r.verifyImpl(round, head, false);
    }

    /// Ordinary decode and prompt: every chain row is kept, no snapshots or replay; a GPU failure ends the runner.
    pub fn advance(r: *Runner, round: *const Round, head: forward.Head) !void {
        return r.verifyImpl(round, head, true);
    }

    fn verifyImpl(r: *Runner, round: *const Round, head: forward.Head, accepted: bool) !void {
        if (r.failed or r.active != null or round.ids.len > r.model.frame.capacity) return error.RoundNotReady;
        for (round.slots, round.windows) |slot, window| if (slot >= r.ready or r.offsets[slot] != window.start) return error.CachePositionDiffers;
        const paths = try r.allocator.alloc([]const u32, round.windows.len);
        defer r.allocator.free(paths);
        @memset(paths, &.{});
        var accepted_rows: [128]u32 = undefined;
        if (accepted) for (round.windows, round.firsts, paths) |window, first, *path| {
            for (window.parents, 0..) |parent, row| {
                if (parent != (if (row == 0) @as(i32, -1) else @as(i32, @intCast(row - 1)))) return error.NotAcceptedChain;
                accepted_rows[first + row] = @intCast(row);
            }
            path.* = accepted_rows[first..][0..window.parents.len];
        };
        r.accepted_chain = accepted;
        defer r.accepted_chain = false;
        var recurrent = try round.recurrent(try r.params(round.ids.len), paths);
        defer recurrent.deinit();
        var attention_plan = try r.attentionPlan(round);
        defer attention_plan.deinit();
        try r.model.frame.setRows(round.ids, round.feed_positions);
        try r.scratch.upload(attention_plan);
        r.current_round = round;
        r.current_gdn = &recurrent;
        r.current_attention = &attention_plan;
        defer {
            r.current_round = null;
            r.current_gdn = null;
            r.current_attention = null;
        }
        if (r.capture_taps) r.taps.begin();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        var stages = forward.Stages{ .model = r.model, .rows = @intCast(round.ids.len), .hooks = .{ .ptr = r, .gdn = gdnForward, .attention = attentionForward, .tap = if (r.capture_taps) tap else null }, .head = head };
        r.last_gpu_seconds = @import("core").segments.runCommitting(r.model.device, &.{r.model.queue}, r.model.config.layers, .concurrent, &stages, commit_every) catch |err| {
            r.failed = true;
            return err;
        };
        if (r.model.ane) |share| share.drain() catch |err| {
            r.failed = true;
            return err;
        };
        if (r.capture_taps) try r.taps.complete();
        if (accepted) {
            for (round.slots, round.windows) |slot, window| r.offsets[slot] = window.start + @as(u32, @intCast(window.parents.len));
        } else r.active = fingerprint(round);
    }

    pub fn keep(r: *Runner, round: *const Round, paths: []const []const u32) !void {
        const active = r.active orelse return error.RoundNotReady;
        if (r.failed or !std.mem.eql(u8, &active, &fingerprint(round))) return error.RoundNotReady;
        var recurrent = try round.recurrent(try r.params(round.ids.len), paths);
        defer recurrent.deinit();
        var attention_plan = try r.attentionPlan(round);
        defer attention_plan.deinit();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = r.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        var gi: usize = 0;
        var ai: usize = 0;
        for (r.model.weights.layers) |layer| switch (layer.mixer) {
            .linear => {
                try r.gdn.commit(gi, .{ .host = recurrent, .metadata = r.metadata() }, e);
                gi += 1;
            },
            .attention => {
                var views: [8]attn.Cache = undefined;
                const caches = r.layerCaches(ai, round, &views);
                const kept = try r.bindings[ai].uploadKeep(attention_plan, caches, paths);
                try r.attention.keep(e, &r.scratch, &r.bindings[ai], kept, caches);
                ai += 1;
            },
        };
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) {
            r.failed = true;
            return error.RoundGpuFailure;
        }
        for (round.slots, round.windows, paths) |slot, window, path| r.offsets[slot] = (try window.keep(path)).length;
        r.active = null;
    }

    /// A validated single-window round consumes authoritative GPU keep metadata without host path selection.
    pub fn keepDevice(r: *Runner, round: *const Round, e: mtl.ComputeEncoder, ops: @import("core").tree_commit_gpu.Ops, result: @import("core").tree_commit_gpu.Ref) !void {
        const active = r.active orelse return error.RoundNotReady;
        if (r.failed or round.windows.len != 1 or round.ids.len > 16 or !std.mem.eql(u8, &active, &fingerprint(round))) return error.RoundNotReady;
        var recurrent = try round.recurrent(try r.params(round.ids.len), &.{&.{}});
        defer recurrent.deinit();
        var attention_plan = try r.attentionPlan(round);
        defer attention_plan.deinit();
        const m = r.metadata();
        try m.upload(recurrent);
        const p = @import("core").tree_commit_gpu.Meta{ .rows = @intCast(round.ids.len), .slot = round.slots[0] };
        const keep_ref = @import("core").tree_commit_gpu.Ref{ .buf = m.keeps.buffer, .off = m.keeps.offset };
        const rows = @import("core").tree_commit_gpu.Ref{ .buf = m.kept_rows.buffer, .off = m.kept_rows.offset };
        try ops.metadata(e, result, keep_ref, rows, null, p);
        e.barrier();
        const plan = @import("gdn_pool_gpu.zig").DevicePlan{ .host = recurrent, .metadata = m };
        for (0..r.gdn.budget.layers) |gi| try r.gdn.commitScan(gi, plan, e);
        e.barrier();
        for (0..r.gdn.budget.layers) |gi| try r.gdn.commitPublish(gi, plan, e);
        e.barrier();
        var ai: usize = 0;
        for (r.model.weights.layers) |layer| switch (layer.mixer) {
            .linear => {},
            .attention => {
                var views: [8]attn.Cache = undefined;
                const caches = r.layerCaches(ai, round, &views);
                try r.bindings[ai].admitDeviceKeep(attention_plan, caches, p.rows);
                const map = r.bindings[ai].get(.keep_map);
                try ops.metadata(e, result, keep_ref, rows, .{ .buf = map.buffer, .off = map.offset }, p);
                e.barrier();
                try r.attention.keep(e, &r.scratch, &r.bindings[ai], p.rows, caches);
                ai += 1;
            },
        };
    }

    pub fn completeDeviceKeep(r: *Runner, round: *const Round, path: []const u32) !void {
        const active = r.active orelse return error.RoundNotReady;
        if (r.failed or round.windows.len != 1 or path.len == 0 or !std.mem.eql(u8, &active, &fingerprint(round))) return error.RoundNotReady;
        r.offsets[round.slots[0]] = (try round.windows[0].keep(path)).length;
        r.active = null;
    }

    pub fn reset(r: *Runner, slot: u32) !void {
        if (r.failed or r.active != null or slot >= r.ready) return error.RoundNotReady;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = r.model.queue.commandBuffer();
        const e = cb.compute(.serial);
        try r.gdn.clearSlot(slot, e);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) {
            r.failed = true;
            return error.RoundGpuFailure;
        }
        r.offsets[slot] = 0;
    }

    fn params(r: *Runner, rows: usize) !abi.Params {
        return r.gdn.budget.shape.params(rows, r.slots, .{ .conv = .bf16, .a_log = .bf16, .dt = .bf16, .norm = .bf16 });
    }

    fn metadata(r: *Runner) @import("gdn_pool_gpu.zig").Metadata {
        const f = &r.model.frame;
        return .{ .parents = f.get(.parents), .windows = f.get(.windows), .row_slots = f.get(.row_slots), .segments = f.get(.segments), .keeps = f.get(.keeps), .kept_rows = f.get(.kept_rows) };
    }

    fn attentionPlan(r: *Runner, round: *const Round) !attn.Plan {
        var windows: [8]attn.planning.Window = undefined;
        if (round.windows.len > windows.len) return error.AttentionStreams;
        for (round.windows, 0..) |window, i| windows[i] = .{ .parents = window.parents, .start = window.start, .capacity = r.capacity };
        return attn.Plan.init(r.allocator, windows[0..round.windows.len], .{ .shared_prefix = r.model.device.tensorUnits() });
    }

    fn layerCaches(r: *Runner, layer: usize, round: *const Round, views: *[8]attn.Cache) []const attn.Cache {
        for (round.slots, 0..) |slot, i| views[i] = r.caches[layer * r.slots + slot];
        return views[0..round.slots.len];
    }

    fn gdnForward(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: wts.Gdn, frame: *Frame, rows: u32) !void {
        const r: *Runner = @ptrCast(@alignCast(ptr));
        var hook = GdnHook{ .pool = &r.gdn, .quant = &r.model.kernels, .glue = r.model.glue, .plan = r.current_gdn orelse return error.RoundNotReady, .linear_index = r.linear_index, .accepted_chain = r.accepted_chain };
        try GdnHook.forward(&hook, e, block, weights, frame, rows);
    }

    fn attentionForward(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: wts.Attention, f: *Frame, rows: u32) !void {
        const r: *Runner = @ptrCast(@alignCast(ptr));
        const plan = (r.current_attention orelse return error.RoundNotReady).*;
        const layer = r.attention_index[block];
        var views: [8]attn.Cache = undefined;
        const caches = r.layerCaches(layer, r.current_round orelse return error.RoundNotReady, &views);
        try r.bindings[layer].upload(plan, caches);
        try r.model.kernels.quant(e, weights.q, f.get(.input), forward.sums(f, .sums, rows), f.get(.dims), f.get(.q_gate), rows);
        try r.model.kernels.quant(e, weights.kv, f.get(.input), forward.sums(f, .sums, rows), f.get(.dims), f.get(.mixed), rows);
        e.barrier();
        try r.model.glue.unstack(e, f.get(.mixed), f.get(.key), .{ .rows = rows, .width = 1024, .stride = 2048, .offset = 0 });
        try r.model.glue.unstack(e, f.get(.mixed), f.get(.value), .{ .rows = rows, .width = 1024, .stride = 2048, .offset = 1024 });
        e.barrier();
        try r.attention.preprocess(e, .{ .qg = f.get(.q_gate), .key = f.get(.key), .q_gain = wts.ref(weights.q_norm), .k_gain = wts.ref(weights.k_norm), .norm_q = f.get(.norm_query), .norm_k = f.get(.norm_key), .query = f.get(.query), .rotated_key = f.get(.rotated_key), .positions = f.get(.positions), .rows = rows, .eps = r.model.config.eps, .theta = r.model.config.rope_theta });
        try r.attention.encode(e, &r.scratch, &r.bindings[layer], plan, f.get(.query), f.get(.rotated_key), f.get(.value), f.get(.mixer_out), caches, f.get(.q_gate), r.model.kernels.prompt);
    }

    fn tap(ptr: *anyopaque, e: mtl.ComputeEncoder, layer: usize, hidden: @import("projection.zig").Ref, rows: u32) !void {
        const r: *Runner = @ptrCast(@alignCast(ptr));
        try r.taps.record(r.model.glue, e, layer, hidden, rows);
    }
};

fn slotCache(a: std.mem.Allocator, m: *Model, owned: *std.ArrayList(mtl.Buffer), capacity: u32) !attn.Cache {
    const bytes = @as(usize, capacity) * m.config.kvDim() * 2;
    const key = try allocation(a, m.device, owned, bytes);
    const value = try allocation(a, m.device, owned, bytes);
    return .{ .keys = .{ .buffer = key }, .values = .{ .buffer = value }, .capacity = capacity };
}

fn allocation(a: std.mem.Allocator, device: mtl.Device, owned: *std.ArrayList(mtl.Buffer), bytes: usize) !mtl.Buffer {
    const b = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    owned.append(a, b) catch |err| {
        b.deinit();
        return err;
    };
    return b;
}

fn fingerprint(round: *const Round) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(std.mem.sliceAsBytes(round.ids));
    hash.update(std.mem.sliceAsBytes(round.slots));
    hash.update(std.mem.sliceAsBytes(round.feed_positions));
    hash.update(std.mem.sliceAsBytes(round.draw_positions));
    hash.update(std.mem.sliceAsBytes(round.firsts));
    for (round.windows) |window| {
        hash.update(std.mem.asBytes(&window.start));
        hash.update(std.mem.asBytes(&window.taps));
        hash.update(std.mem.sliceAsBytes(window.parents));
        hash.update(std.mem.sliceAsBytes(window.conv));
    }
    var out: [32]u8 = undefined;
    hash.final(&out);
    return out;
}
