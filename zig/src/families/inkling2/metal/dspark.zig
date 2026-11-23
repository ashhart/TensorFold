//! Inkling 2 Mini DSpark on Metal, the kernels behind the shared lane host: features, blocks, folded Markov logits.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const gpu_rule = lanes.gpu_rule;
const ck = @import("../../../core/checkpoint_metal.zig");
const Model = @import("model.zig").Model;
const fwd = @import("forward.zig");
const glue = @import("glue.zig");
const k = @import("dspark_kernels.zig");
const weight_loader = @import("dspark_weights.zig");
const state = @import("dspark_state.zig");
const host = @import("../dspark_lanes.zig");

const Ref = k.Ref;
/// Drafts a hold offers at most: the block after its anchor. The depth rule prices how many a round takes.
const hold_depth: u32 = host.depth;
/// The partials of the tensor-unit attention at a window under 1,024 tokens, where the host has fewer score bytes.
const units_floor = k.unitsBytes(1024 / k.units_chunk + 2);

/// A lane and what the shared host books of it: its rings and its committed rows [from, end).
const Context = struct { lane: *const state.Lane, rings: [3]usize, from: u64, end: u64 };
/// Streams one greedy hold can chain on the GPU; a wider hold samples on the host as sampled streams do.
const chain_streams = 16;

fn at(buf: mtl.Buffer, off: usize) Ref {
    return .{ .buf = buf, .off = off };
}
fn r(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}
fn failed(cb: mtl.CommandBuffer, what: []const u8) !void {
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2_loop DSpark {s}: {s}", .{ what, msg });
        return error.GpuFailed;
    }
}

pub const Drafter = struct {
    /// What the shared host opens a stream: its rings and its newest row.
    pub const Lane = state.Lane;

    gpa: std.mem.Allocator,
    target: *const Model,
    weights: weight_loader.Loaded,
    pipelines: k.Pipelines,
    scratch: state.Scratch,
    /// Scratch for the second and later streams of one hold; the first uses `scratch`, as absorb does.
    more: std.ArrayList(state.Scratch) = .empty,
    /// The greedy chain's kernels, and each held stream's eight tokens: the anchor, then its drafts.
    chain: k.Chain = undefined,
    chain_tokens: mtl.Buffer = undefined,
    scores: ?mtl.Buffer = null,
    /// The block attention's tensor-unit kernels, on the chips that have the units.
    units: ?k.Units = null,
    /// A draft's share of a full hold, timed by `measure`.
    step_ms: f64 = 0,
    /// The round loop's drafter: the shared host's books over these kernels.
    lanes: host.Drafter(Drafter) = undefined,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, target: *const Model, drafter_dir: []const u8) !*Drafter {
        _ = io; // Native factory convention; Checkpoint.addFile uses platform file calls.
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{
            .gpa = gpa,
            .target = target,
            .weights = weight_loader.load(gpa, target, drafter_dir) catch |err| return err,
            .pipelines = undefined,
            .scratch = undefined,
        };
        d.lanes = host.Drafter(Drafter).init(gpa, d);
        errdefer d.weights.deinit();
        d.pipelines = try k.Pipelines.load(device);
        errdefer d.pipelines.deinit();
        d.scratch = try state.Scratch.init(device);
        errdefer d.scratch.deinit();
        d.chain = try k.Chain.load(device, @import("inkling2_metal").inkling2_dspark);
        errdefer d.chain.deinit();
        d.chain_tokens = try device.buffer(chain_streams * 8 * @sizeOf(u32), mtl.ResourceOptions.shared);
        errdefer d.chain_tokens.deinit();
        if (device.tensorUnits()) d.units = try k.Units.load(device, @import("inkling2_metal").inkling2_dspark_units);
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        d.lanes.deinit();
        if (d.scores) |scores| scores.deinit();
        if (d.units) |*u| u.deinit();
        d.chain_tokens.deinit();
        d.chain.deinit();
        for (d.more.items) |*extra| extra.deinit();
        d.more.deinit(d.gpa);
        d.scratch.deinit();
        d.pipelines.deinit();
        d.weights.deinit();
        d.gpa.destroy(d);
    }

    pub fn fixedBytes(d: *const Drafter) u64 {
        return d.weights.fixedBytes() + state.scratchBytes() + units_floor;
    }

    pub fn openLane(d: *Drafter, rings: [3]usize) !Lane {
        return Lane.init(d.target.device, rings);
    }

    pub fn closeLane(_: *Drafter, lane: *Lane) void {
        lane.deinit();
    }

    /// The holds' shared attention scratch: the host's bytes, or the tensor-unit partials' floor where that is more.
    pub fn setScores(d: *Drafter, bytes: u64) !void {
        const scores = try d.target.device.buffer(@intCast(@max(bytes, units_floor)), mtl.ResourceOptions.private);
        if (d.scores) |old| old.deinit();
        d.scores = scores;
    }

    pub fn policy(d: *const Drafter) host.Policy {
        // No plain rounds: one draft pays from about a third landing, and the guard's estimate near that line is noise.
        return .{ .step_ms = d.step_ms, .plain_guard = false, .batched = true };
    }

    fn absorbRows(d: *Drafter, e: mtl.ComputeEncoder, lane: *const Lane, rings: [3]usize, taps: Ref, start: u64, rows: usize) void {
        const n: u32 = @intCast(rows);
        k.fuseTaps(&d.pipelines, e, taps, r(d.weights.weights.fuse), at(d.weights.dflare, 0), at(d.scratch.fused, 0), at(lane.newest, 0), n);
        for (d.weights.weights.layers, 0..) |layer, i| {
            const c = at(d.scratch.fused, i * rows * state.hidden * 2);
            fwd.dense(d.target, e, layer.k_ctx, c, at(d.scratch.ck, 0), rows);
            fwd.dense(d.target, e, layer.v_ctx, c, at(d.scratch.cv, 0), rows);
            k.commitContext(&d.pipelines, e, at(d.scratch.ck, 0), at(d.scratch.cv, 0), r(layer.k_norm_ctx), at(lane.k[i], 0), at(lane.v[i], 0), @intCast(start), n, @intCast(rings[i]));
        }
    }

    /// Every stream's pieces of an absorb in one command buffer, in the order the host hands them.
    pub fn absorb(d: *Drafter, pieces: []const host.Absorb(Lane)) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.target.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (pieces) |p| {
            const source = mtl.Buffer{ .id = @ptrFromInt(p.features.buffer) };
            d.absorbRows(e, p.lane, p.rings, at(source, @intCast(p.offset)), p.position, p.rows);
        }
        e.end();
        cb.commit();
        try failed(cb, "absorb");
    }

    fn encodeBlock(d: *Drafter, e: mtl.ComputeEncoder, s: *const state.Scratch, c: Context, pending: u32, depth: u32) void {
        s.ids.slice(u32, 1)[0] = pending;
        glue.embedNorm(&d.target.glue, e, at(s.ids, 0), r(d.target.weights.embed), r(d.target.weights.embed_norm), at(s.x, 0), 1);
        k.initializeRows(&d.pipelines, e, at(s.x, 0), r(d.weights.weights.mask), at(s.x, 0));
        k.residualNorm(&d.pipelines, e, at(s.x, 0), at(s.resid, 0), r(d.weights.weights.layers[0].input_norm), at(s.resid, 0), at(s.x, 0), 8, true);
        for (d.weights.weights.layers, 0..) |layer, i| {
            fwd.dense(d.target, e, layer.q, at(s.x, 0), at(s.q_raw, 0), 8);
            fwd.dense(d.target, e, layer.k, at(s.x, 0), at(s.k_raw, 0), 8);
            fwd.dense(d.target, e, layer.v, at(s.x, 0), at(s.v_raw, 0), 8);
            k.blockQkv(&d.pipelines, e, at(s.q_raw, 0), at(s.k_raw, 0), at(s.v_raw, 0), r(layer.q_norm), r(layer.k_norm), at(s.q, 0), at(s.kb, 0), at(s.vb, 0), @intCast(c.end));
            if (d.units) |*u| k.attendUnits(u, e, at(s.q, 0), at(s.kb, 0), at(s.vb, 0), at(c.lane.k[i], 0), at(c.lane.v[i], 0), at(d.scores.?, 0), at(s.attn, 0), @intCast(c.end), @intCast(c.rings[i]), i != 1, @intCast(c.from)) else k.attend(&d.pipelines, e, at(s.q, 0), at(s.kb, 0), at(s.vb, 0), at(c.lane.k[i], 0), at(c.lane.v[i], 0), at(d.scores.?, 0), at(s.attn, 0), @intCast(c.end), @intCast(c.rings[i]), i != 1, @intCast(c.from));
            fwd.dense(d.target, e, layer.o, at(s.attn, 0), at(s.y, 0), 8);
            k.residualNorm(&d.pipelines, e, at(s.y, 0), at(s.resid, 0), r(layer.post_norm), at(s.next_resid, 0), at(s.x, 0), 8, false);
            fwd.dense(d.target, e, layer.gate, at(s.x, 0), at(s.gate, 0), 8);
            fwd.dense(d.target, e, layer.up, at(s.x, 0), at(s.up, 0), 8);
            k.swiglu(&d.pipelines, e, at(s.gate, 0), at(s.up, 0), at(s.act, 0), 8 * state.intermediate);
            fwd.dense(d.target, e, layer.down, at(s.act, 0), at(s.y, 0), 8);
            const next_norm = if (i + 1 < d.weights.weights.layers.len) d.weights.weights.layers[i + 1].input_norm else d.weights.weights.norm;
            k.residualNorm(&d.pipelines, e, at(s.y, 0), at(s.next_resid, 0), r(next_norm), at(s.resid, 0), at(s.x, 0), 8, false);
        }
        k.base(&d.pipelines, e, r(d.target.weights.unembed), at(s.x, state.hidden * 2), at(s.base, 0), depth, state.vocabulary);
        k.kappa(&d.pipelines, e, at(s.x, state.hidden * 2), at(c.lane.newest, 0), r(d.weights.weights.rnn), r(d.weights.weights.rnn_bias), at(s.kappa, 0), depth);
    }

    fn recurrent(d: *Drafter, e: mtl.ComputeEncoder, s: *const state.Scratch, step: usize, previous: u32, second_bank: bool) void {
        const current = if (second_bank) s.next_state else s.state;
        const next = if (second_bank) s.state else s.next_state;
        k.recurrentStep(&d.pipelines, e, at(current, 0), at(s.kappa, step * 3 * state.rank * 4), r(d.weights.weights.rnn), r(d.weights.weights.markov_w1), at(s.z, 0), at(next, 0), at(s.lambda, 0), previous);
        k.finalLogits(&d.pipelines, e, at(s.base, step * state.vocabulary * 2), at(d.weights.phi, 0), at(d.weights.beta, 0), at(s.lambda, 0), at(s.logits, 0));
    }

    fn slot(d: *Drafter, index: usize) !*const state.Scratch {
        if (index == 0) return &d.scratch;
        while (d.more.items.len < index) {
            var extra = try state.Scratch.init(d.target.device);
            errdefer extra.deinit();
            try d.more.append(d.gpa, extra);
        }
        return &d.more.items[index - 1];
    }

    /// One stream's greedy hold: its block, then each step's recurrent pass, logits and pick; `held` takes the drafts.
    fn encodeChain(d: *Drafter, e: mtl.ComputeEncoder, s: *const state.Scratch, c: Context, held: Ref, pending: u32, depth: usize) void {
        d.encodeBlock(e, s, c, pending, @intCast(depth));
        var second_bank = false;
        for (0..depth) |step| {
            const current = if (second_bank) s.next_state else s.state;
            const next = if (second_bank) s.state else s.next_state;
            k.chainStep(&d.chain, e, at(current, 0), at(s.kappa, step * 3 * state.rank * 4), r(d.weights.weights.rnn), r(d.weights.weights.markov_w1), at(s.z, 0), at(next, 0), at(s.lambda, 0), held, @intCast(step));
            k.finalLogits(&d.pipelines, e, at(s.base, step * state.vocabulary * 2), at(d.weights.phi, 0), at(d.weights.beta, 0), at(s.lambda, 0), at(s.logits, 0));
            k.chainPick(&d.chain, e, at(s.logits, 0), held, @intCast(step));
            second_bank = !second_bank;
        }
    }

    /// A full hold on an empty context, the third of three runs timed: the depth rule prices a draft at its share.
    pub fn measure(d: *Drafter) !void {
        const rings = host.rings(d.lanes.capacity);
        var lane = try Lane.init(d.target.device, rings);
        defer lane.deinit();
        const held = Ref{ .buf = d.chain_tokens, .off = 0 };
        var ms: f64 = 0;
        for (0..3) |attempt| {
            var t0: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &t0);
            @memset(d.scratch.state.slice(u16, state.rank), 0);
            d.chain_tokens.slice(u32, 8)[0] = 2;
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = d.target.queue.commandBuffer();
            const e = cb.compute(.serial);
            d.encodeChain(e, &d.scratch, .{ .lane = &lane, .rings = rings, .from = 0, .end = 0 }, held, 2, hold_depth);
            e.end();
            cb.commit();
            try failed(cb, "timed hold");
            var t1: std.c.timespec = undefined;
            _ = std.c.clock_gettime(.MONOTONIC, &t1);
            if (attempt == 2) ms = @as(f64, @floatFromInt(t1.sec - t0.sec)) * 1e3 + @as(f64, @floatFromInt(t1.nsec - t0.nsec)) / 1e6;
        }
        d.step_ms = ms / @as(f64, @floatFromInt(hold_depth));
    }

    /// Greedy streams: the blocks and every step in one command buffer, each draft picked on the GPU for the next.
    fn holdChained(d: *Drafter, items: []const host.Hold(Lane), drafts: [][host.depth]u32) !void {
        const tokens = d.chain_tokens.slice(u32, chain_streams * 8);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = d.target.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (items, 0..) |item, i| {
            const s = try d.slot(i);
            const held = Ref{ .buf = d.chain_tokens, .off = i * 8 * @sizeOf(u32) };
            tokens[i * 8] = item.pending;
            d.encodeChain(e, s, .{ .lane = item.lane, .rings = item.rings, .from = item.from, .end = item.anchor }, held, item.pending, item.depth);
        }
        e.end();
        cb.commit();
        try failed(cb, "chained block");
        // A token past the vocabulary says the logits were not finite; the host refuses it when it keeps the drafts.
        for (items, drafts, 0..) |item, *out, i| @memcpy(out[0..item.depth], tokens[i * 8 + 1 ..][0..item.depth]);
    }

    /// One call a hold for all its streams: greedy ones chain in one command buffer, sampled ones a step a buffer.
    pub fn hold(d: *Drafter, items: []const host.Hold(Lane), drafts: [][host.depth]u32) !void {
        var deepest: usize = 0;
        for (items, 0..) |item, i| {
            deepest = @max(deepest, item.depth);
            @memset((try d.slot(i)).state.slice(u16, state.rank), 0);
        }
        var greedy = items.len <= chain_streams;
        for (items) |item| greedy = greedy and item.sampling == null;
        if (greedy) return d.holdChained(items, drafts);
        const previous = try d.gpa.alloc(u32, items.len);
        defer d.gpa.free(previous);
        const values = try d.gpa.alloc(f32, state.vocabulary);
        defer d.gpa.free(values);
        var second_bank = false;
        for (0..deepest) |step| {
            {
                const pool = mtl.objc.Pool.push();
                defer pool.pop();
                const cb = d.target.queue.commandBuffer();
                const e = cb.compute(.serial);
                for (items, 0..) |item, i| {
                    if (step >= item.depth) continue;
                    const s = try d.slot(i);
                    if (step == 0) {
                        d.encodeBlock(e, s, .{ .lane = item.lane, .rings = item.rings, .from = item.from, .end = item.anchor }, item.pending, item.depth);
                        previous[i] = item.pending;
                    }
                    d.recurrent(e, s, step, previous[i], second_bank);
                }
                e.end();
                cb.commit();
                try failed(cb, if (step == 0) "block" else "recurrent step");
            }
            for (items, drafts, 0..) |item, *out, i| {
                if (step >= item.depth) continue;
                const raw = (try d.slot(i)).logits.slice(u16, state.vocabulary);
                for (raw, values) |bits, *value| value.* = @bitCast(@as(u32, bits) << 16);
                const base_position = std.math.cast(u32, item.position) orelse return error.DraftDepth;
                const position = std.math.add(u32, base_position, @intCast(step)) catch return error.DraftDepth;
                previous[i] = try gpu_rule.sample(d.gpa, values, item.sampling, position, null);
                if (previous[i] >= state.vocabulary) return error.DraftNonfinite;
                out[step] = previous[i];
            }
            second_bank = !second_bank;
        }
    }
};

test "Mini taps and block depth match checkpoint section eight" {
    try std.testing.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, &host.taps);
    try std.testing.expectEqual(@as(u32, 48), host.taps[3]);
    try std.testing.expectEqual(@as(u32, 7), hold_depth);
    try std.testing.expectEqual(@as(u32, state.block), host.block);
}

test "the host's score bytes hold the tensor-unit partials from 1,024 tokens up, and the floor holds them below" {
    for ([_]usize{ 1024, 1025, 4096, 65536, 262144, 1 << 20 }) |n| try std.testing.expect(host.scoreBytes(n) >= k.unitsBytes(n / k.units_chunk + 2));
    for ([_]usize{ 1, 255, 256, 1023 }) |n| try std.testing.expect(units_floor >= k.unitsBytes(@max(n, 1024) / k.units_chunk + 2));
}
