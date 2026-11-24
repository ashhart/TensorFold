//! Generated draft tensors exercise GLM's callback backend on the GPU, embedding and head included.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const tf = @import("tensorfold");
const glm = tf.glm;
const cfg = glm.config;
const wts = glm.weights;
const kernels = glm.kernels;
const Backend = glm.metal_dflash_backend.Backend;
const Target = glm.metal_dflash_backend.Target;
const Model = glm.metal_dflash_model.Model;
const affine = glm.affine;
const wide = glm.wide;
const Taps = glm.metal_dflash_taps.Taps;
const fwd = glm.forward;

pub fn bf16FromF32(v: f32) u16 {
    const u = @as(u32, @bitCast(v));
    const h = u & 0xffff0000;
    const l = u & 0x0000ffff;
    const half: u32 = 0x8000;
    if (l > half or (l == half and (h & 0x10000) != 0)) {
        return @intCast((h +% 0x10000) >> 16);
    }
    return @intCast(h >> 16);
}

fn f32FromBf16(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

// The toy drafter: the stock incoai contract at fixture scale, five taps into 45 layers.
pub const config = tf.qwen27.dflash.config.Config{ .hidden = 512, .intermediate = 512, .vocab = 96, .heads = 8, .kv_heads = 2, .head_dim = 64, .target_layers = 45, .mask = 95, .window = 64, .rank = 32, .topk = 4, .taps = glm.metal_dflash_config.stock.taps, .tap_count = glm.metal_dflash_config.stock.tap_count };
const target_config = cfg.Config{ .hidden = 512, .vocab = 96, .layers = 45 };

fn dimensions(name: []const u8) []const usize {
    if (std.mem.eql(u8, name, "fc.weight")) return &.{ 512, 2560 };
    if (std.mem.eql(u8, name, "candidate_selector.hidden_projection.weight")) return &.{ 32, 512 };
    if (std.mem.startsWith(u8, name, "candidate_selector.")) return &.{ 96, 32 };
    if (std.mem.endsWith(u8, name, "base_kernel")) return &.{ 2, 2, 512 };
    if (std.mem.endsWith(u8, name, "kernel_projection.weight")) return &.{ 128, 512 };
    if (std.mem.endsWith(u8, name, "k_proj.weight") or std.mem.endsWith(u8, name, "v_proj.weight")) return &.{ 128, 512 };
    if (std.mem.endsWith(u8, name, "q_norm.weight") or std.mem.endsWith(u8, name, "k_norm.weight")) return &.{64};
    if (std.mem.endsWith(u8, name, "proj.weight")) return &.{ 512, 512 };
    return &.{512};
}

/// A tensor's generated bf16 values: norms 1, base kernels a fixed pattern, the rest small random multiples of 1/1024.
fn fill(name: []const u8, out: []u16, rng: std.Random) void {
    const norm = std.mem.endsWith(u8, name, "norm.weight") or std.mem.endsWith(u8, name, "layernorm.weight");
    for (out, 0..) |*item, i| {
        const value: f32 = if (norm) 1 else if (std.mem.endsWith(u8, name, "base_kernel")) (if ((i / 512) % 2 == 0) @as(f32, 1) else 0) else @as(f32, @floatFromInt(rng.intRangeLessThan(i32, -8, 9))) / 1024;
        item.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    }
}

/// The target stand-in: the drafter's own device, queue, kernels and a 4-bit embed / 8-bit head at its shape.
pub const Harness = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    k: *kernels.Kernels,
    w: wts.Weights,
    arena: std.heap.ArenaAllocator,
    fn init(gpa: std.mem.Allocator) !Harness {
        const device = try mtl.Device.init();
        errdefer device.deinit();
        const queue = try device.queue();
        errdefer queue.deinit();
        var h = Harness{ .device = device, .queue = queue, .k = undefined, .w = undefined, .arena = std.heap.ArenaAllocator.init(gpa) };
        errdefer h.deinit();
        h.k = try kernels.load(gpa, device);
        h.w = .{ .gpa = gpa };
        h.w.embed = try matrix(device, config.vocab, config.hidden, .{ .bits = 4, .group = 64 });
        h.w.head = try matrix(device, config.vocab, config.hidden, .{ .bits = 8, .group = 64 });
        return h;
    }
    fn deinit(h: *Harness) void {
        h.w.deinit();
        h.k.deinit();
        h.arena.deinit();
        h.queue.deinit();
        h.device.deinit();
    }
    fn target(h: *Harness) Target {
        return .{ .device = h.device, .queue = h.queue, .k = h.k, .w = &h.w, .c = target_config };
    }
};

/// One bf16 [n, k] affine matrix on the device: packed codes and bf16 tables at `width`.
fn matrix(device: mtl.Device, n: usize, k: usize, width: affine.Width) !affine.Affine {
    const wb = affine.wBytes(n, k, width);
    const sb = affine.sBytes(n, k, width);
    const buffer = try device.buffer(wb + 2 * sb, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    var random: std.Random.DefaultPrng = .init(7919);
    const rng = random.random();
    const codes = buffer.slice(u8, wb);
    rng.bytes(codes);
    const words = buffer.slice(u16, (wb + 2 * sb) / 2);
    for (words[wb / 2 ..][0 .. sb / 2]) |*scale| scale.* = 0x3f80; // 1.0
    @memset(words[wb / 2 + sb / 2 ..], 0);
    return .{ .w = .{ .buf = buffer, .off = 0 }, .s = .{ .buf = buffer, .off = wb }, .b = .{ .buf = buffer, .off = wb + sb }, .n = @intCast(n), .k = @intCast(k), .bits = width.bits, .group = width.group };
}

/// The draft model over generated tensors, every byte owned by `h`; `q4`: its linears read through 4-bit copies.
pub fn create(a: std.mem.Allocator, h: *Harness, q4: bool) !*Model {
    const names = try glm.metal_dflash_weights.names(a, config);
    defer glm.metal_dflash_weights.freeNames(a, names);
    const m = try a.create(Model);
    errdefer a.destroy(m);
    m.allocator = a;
    m.tree_block = 16;
    m.committed_end = 0;
    m.checkpoint = tf.checkpoint_metal.Checkpoint.init(a);
    errdefer m.checkpoint.deinit();
    var random: std.Random.DefaultPrng = .init(5051);
    const rng = random.random();
    for (names) |name| {
        const shape = dimensions(name);
        var count: usize = 1;
        for (shape) |dimension| count *= dimension;
        const buffer = try h.device.buffer(count * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        m.checkpoint.shards.append(a, .{ .buffer = buffer, .bytes = count * 2 }) catch |err| {
            buffer.deinit();
            return err;
        };
        fill(name, buffer.slice(u16, count), rng);
        var tensor = tf.checkpoint_metal.Tensor{ .buffer = buffer, .offset = 0, .bytes = count * 2, .dtype = .bf16, .rank = shape.len };
        @memcpy(tensor.shape[0..shape.len], shape);
        try m.checkpoint.tensors.put(a, try a.dupe(u8, name), tensor);
    }
    const t = try a.create(Target);
    t.* = h.target();
    m.backend = try Backend.init(a, t, &m.checkpoint, config);
    errdefer m.backend.deinit();
    m.graph = try tf.glm.metal_dflash_model.loadGraph(a, &m.backend);
    m.pending = try h.device.buffer(@as(usize, config.window) * config.tapWidth() * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer m.pending.deinit();
    m.session = try tf.qwen27.dflash.session.Session.init(config, 1, 0);
    m.preparation = .{ .mode = .bf16_reference };
    if (q4) {
        m.backend.copies = try glm.metal_dflash_prepared.Copies.prepare(a, h.device, &m.checkpoint, config.linearCount());
        m.preparation = .{ .mode = .prepared_q4_reference, .quantization_verified = true, .source_sha256 = tf.affine4.signature() };
    }
    return m;
}

fn testTapCapture(a: std.mem.Allocator, h: *Harness) !void {
    const rows = 7;
    const pos = 3;
    const D = target_config.hidden;
    const window = config.window;
    const stream_bytes = @as(usize, rows) * 4 * D * 2;
    const streams = try h.device.buffer(stream_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer streams.deinit();
    const raw = try h.device.buffer(@as(usize, rows) * D * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer raw.deinit();
    const ring = try h.device.buffer(@as(usize, window) * 5 * D * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer ring.deinit();
    const taps = try Taps.init(h.device, ring, window, D, a, glm.state.max_rows);
    var taps_mut = taps;
    defer taps_mut.deinit(a);
    const words = streams.slice(u16, rows * 4 * D);
    for (0..rows) |r| {
        for (0..4) |s| {
            for (0..D) |d| {
                const v: f32 = @floatFromInt(r * 10000 + s * 1000 + d * 10 + 1);
                words[(r * 4 + s) * D + d] = bf16FromF32(v);
            }
        }
    }
    const streams_ref = wts.Ref{ .buf = streams, .off = 0 };
    var sc: glm.state.Scratch = undefined;
    sc.x = .{ streams_ref, streams_ref };
    sc.raw = .{ .buf = raw, .off = 0 };
    var dummy_state: glm.state.State = undefined;
    taps_mut.setRoute(0, &dummy_state, ring);
    var x: fwd.Ctx = .{ .k = h.k, .c = &target_config, .w = undefined, .s = &dummy_state, .sc = &sc, .taps = &taps_mut };
    const cb = h.queue.commandBuffer();
    const e = cb.compute(.serial);
    taps_mut.after(&x, e, 5, rows, pos);
    e.end();
    cb.commit();
    cb.wait();
    const ring_bytes = ring.contents();
    for (0..rows) |r| {
        const off = taps_mut.offset(5, pos + @as(u32, @intCast(r))).?;
        for (0..D) |d| {
            var sum: f32 = 0;
            for (0..4) |s| {
                const w = words[(r * 4 + s) * D + d];
                sum += f32FromBf16(w);
            }
            const want = bf16FromF32(sum * 0.25);
            const got = @as(*const u16, @ptrCast(@alignCast(ring_bytes + off + d * 2))).*;
            if (want != got) return error.TapMismatch;
        }
    }
    @memset(ring_bytes[0..ring.length()], 0);
    taps_mut.setActive(0, false);
    const cb2 = h.queue.commandBuffer();
    const e2 = cb2.compute(.serial);
    taps_mut.after(&x, e2, 5, rows, pos);
    e2.end();
    cb2.commit();
    cb2.wait();
    const first_off = taps_mut.offset(5, pos).?;
    if (@as(*const u16, @ptrCast(@alignCast(ring_bytes + first_off))).* != 0) return error.TapMismatch;
}

/// The keeps go ahead of a DFlash2 pass without a wait, and stay for the window when no stream drafts with it.
fn testCommitAhead(h: *Harness) !void {
    var e: glm.engine.Engine = undefined;
    e.device = h.device;
    e.queue = h.queue;
    e.ev = 0;
    e.ep = null;
    e.event = try h.device.sharedEvent();
    defer e.event.deinit();
    var sl: glm.slots.Slots = undefined;
    sl.e = &e;
    sl.launched = null;
    sl.fence = null;
    sl.ahead = .{ 0, 0 };
    sl.log = false;
    defer if (sl.fence) |f| f.deinit();
    const open = e.begin();
    sl.open = .{ .cb = open.cb, .enc = open.enc, .pool = mtl.objc.Pool.push() };
    var b: glm.backend.Backend = .{ .gpa = std.heap.page_allocator, .sl = &sl };
    var rounds: glm.metal_dflash_rounds.Rounds = undefined;
    rounds.inline_pass = null;
    // no stream drafts with DFlash2 this round: the keeps stay in the open buffer for the window, as without a drafter
    try glm.metal_dflash_rounds.draft(&b, &rounds, &.{});
    if (sl.open == null or sl.launched != null) return error.KeepsCommittedForNoDrafter;
    try sl.commitAhead(); // a round with a DFlash2 pass commits them ahead; the next window's wait reaps them
    if (sl.open != null or sl.launched == null) return error.KeepsNotCommittedAhead;
    try sl.flush();
    if (sl.launched != null) return error.KeepsNotReaped;
}

/// --drafter-bits 4: every linear has its copy, a round drafts through them, and a linear without one is refused.
fn testFourBit(a: std.mem.Allocator, h: *Harness) !void {
    const m = try create(a, h, true);
    defer m.deinit();
    const copies = &m.backend.copies.?;
    try std.testing.expectEqual(config.linearCount(), copies.adapters.count() + copies.tiled.count());
    try m.absorb(.{ .buf = m.pending, .off = 0 }, 2);
    var tree = try m.propose(7, 15);
    defer tree.deinit();
    try std.testing.expect(tree.tokens.len > 0);
    for (tree.tokens) |token| try std.testing.expect(token < config.vocab);
    var empty: glm.metal_dflash_prepared.Copies = .{ .gpa = a };
    std.mem.swap(glm.metal_dflash_prepared.Copies, copies, &empty);
    defer std.mem.swap(glm.metal_dflash_prepared.Copies, copies, &empty);
    try m.absorb(.{ .buf = m.pending, .off = 0 }, 1);
    try std.testing.expectError(error.DraftPreparationUnqualified, m.propose(3, 7));
}

/// `rows` of seeded taps for stream `k` from row `from` on, into `x` (the model holds it meanwhile).
pub fn feedStream(m: *Model, x: *glm.metal_dflash_model.Context, taps: mtl.Buffer, from: u32, rows: u32) !void {
    try m.swap(x);
    defer m.swap(x) catch unreachable;
    try m.absorb(.{ .buf = taps, .off = @as(usize, from) * config.tapWidth() * 2 }, rows);
}

/// Batched passes: each stream's tree equals its own pass's, whichever streams share the pass, wrapped rings and all.
fn testBatch(a: std.mem.Allocator, h: *Harness, q4: bool) !void {
    const m = try create(a, h, q4);
    defer m.deinit();
    const Context = glm.metal_dflash_model.Context;
    const lens = [3]u32{ 40, 25, 9 };
    var taps: [3]mtl.Buffer = undefined;
    // four copies of three streams fed the same taps: alone, all three, streams 0 and 1, streams 0 and 2
    var ctx: [4][3]*Context = undefined;
    for (0..3) |k| {
        taps[k] = try h.device.buffer(@as(usize, 120) * config.tapWidth() * 2, mtl.ResourceOptions.shared);
        var random: std.Random.DefaultPrng = .init(77 + k);
        for (taps[k].slice(u16, 120 * config.tapWidth())) |*v| v.* = bf16FromF32((random.random().float(f32) * 2 - 1) * 2);
    }
    defer for (taps) |t| t.deinit();
    for (0..4) |set| for (0..3) |k| {
        ctx[set][k] = try Context.init(a, m);
        try ctx[set][k].reset(config, 0);
        try feedStream(m, ctx[set][k], taps[k], 0, lens[k]);
    };
    defer for (0..4) |set| for (0..3) |k| ctx[set][k].deinit(a);
    const Tree = tf.qwen27.dflash.selector.Tree;
    for (0..6) |r| {
        const nodes: u32 = if (r % 2 == 0) 15 else 7; // block 16, then 8
        var fed: [3]u32 = undefined;
        for (0..3) |k| {
            fed[k] = lens[k] + @as(u32, @intCast(r)) * 3;
            for (0..4) |set| try feedStream(m, ctx[set][k], taps[k], fed[k], 3);
        }
        var alone: [3]Tree = undefined;
        for (0..3) |k| {
            try m.swap(ctx[0][k]);
            alone[k] = try m.proposeOptions(@intCast(k * 11 + r), .{ .nodes = nodes });
            try m.swap(ctx[0][k]);
        }
        defer for (&alone) |*t| t.deinit();
        const left = ctx[2][2].session.context;
        const pairs = [3][]const usize{ &.{ 0, 1, 2 }, &.{ 0, 1 }, &.{ 0, 2 } };
        for (pairs, 1..) |members, set| {
            var entries: [3]Model.Proposal = undefined;
            for (members, 0..) |k, i| entries[i] = .{ .context = ctx[set][k], .pending = @intCast(k * 11 + r), .options = .{ .nodes = nodes } };
            var trees: [3]Tree = undefined;
            try m.proposeBatch(entries[0..members.len], trees[0..members.len]);
            defer for (trees[0..members.len]) |*t| t.deinit();
            for (members, trees[0..members.len]) |k, t| {
                try std.testing.expectEqualSlices(u32, alone[k].tokens, t.tokens);
                try std.testing.expectEqualSlices(i32, alone[k].parents, t.parents);
                try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(alone[k].scores), std.mem.sliceAsBytes(t.scores));
            }
        }
        // a stream left out of a pass is untouched: no absorb, no pass, its context where it was
        try std.testing.expectEqualDeep(left, ctx[2][2].session.context);
    }
    // a pass of streams of two block sizes is refused rather than drafting one at the other's size
    var mixed = [2]Model.Proposal{ .{ .context = ctx[1][0], .pending = 1, .options = .{ .nodes = 15 } }, .{ .context = ctx[1][1], .pending = 2, .options = .{ .nodes = 3 } } };
    var none: [2]Tree = undefined;
    try std.testing.expectError(error.BadDraftOptions, m.proposeBatch(&mixed, &none));
}

/// Active-row slicing in single and batched passes matches full 8-row block prefixes identically.
fn testSlicedActive(a: std.mem.Allocator, h: *Harness) !void {
    const m = try create(a, h, false);
    defer m.deinit();
    const Context = glm.metal_dflash_model.Context;
    const Tree = tf.qwen27.dflash.selector.Tree;
    var taps: [2]mtl.Buffer = undefined;
    for (0..2) |k| {
        taps[k] = try h.device.buffer(@as(usize, 64) * config.tapWidth() * 2, mtl.ResourceOptions.shared);
        var random: std.Random.DefaultPrng = .init(131 + k);
        for (taps[k].slice(u16, 64 * config.tapWidth())) |*v| v.* = bf16FromF32((random.random().float(f32) * 2 - 1) * 2);
    }
    defer for (taps) |t| t.deinit();
    var ctx: [3][2]*Context = undefined; // [full, solo_sliced, batch_sliced][stream]
    for (0..3) |set| for (0..2) |k| {
        ctx[set][k] = try Context.init(a, m);
        try ctx[set][k].reset(config, 0);
        try feedStream(m, ctx[set][k], taps[k], 0, 12 + @as(u32, @intCast(k)) * 7);
    };
    defer for (0..3) |set| for (0..2) |k| ctx[set][k].deinit(a);
    const depths = [2]u32{ 1, 3 };
    const pendings = [2]u32{ 11, 22 };
    var full: [2]Tree = undefined;
    var solo: [2]Tree = undefined;
    for (0..2) |k| {
        try m.swap(ctx[0][k]);
        full[k] = try m.proposeOptions(pendings[k], .{ .nodes = 7, .children = 1 });
        try m.swap(ctx[0][k]);
        try m.swap(ctx[1][k]);
        solo[k] = try m.proposeBlock(pendings[k], 8, .{ .nodes = depths[k], .children = 1 });
        try m.swap(ctx[1][k]);
        const d = depths[k];
        try std.testing.expectEqual(d, @as(u32, @intCast(solo[k].tokens.len)));
        try std.testing.expectEqualSlices(u32, full[k].tokens[0..d], solo[k].tokens);
        try std.testing.expectEqualSlices(i32, full[k].parents[0..d], solo[k].parents);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(full[k].scores[0..d]), std.mem.sliceAsBytes(solo[k].scores));
    }
    defer for (&full) |*t| t.deinit();
    defer for (&solo) |*t| t.deinit();
    var entries = [2]Model.Proposal{
        .{ .context = ctx[2][0], .pending = pendings[0], .options = .{ .nodes = depths[0], .children = 1 }, .block = 8 },
        .{ .context = ctx[2][1], .pending = pendings[1], .options = .{ .nodes = depths[1], .children = 1 }, .block = 8 },
    };
    var batch: [2]Tree = undefined;
    try m.proposeBatch(&entries, &batch);
    defer for (&batch) |*t| t.deinit();
    for (0..2) |k| {
        try std.testing.expectEqualSlices(u32, solo[k].tokens, batch[k].tokens);
        try std.testing.expectEqualSlices(i32, solo[k].parents, batch[k].parents);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(solo[k].scores), std.mem.sliceAsBytes(batch[k].scores));
    }
}

/// A stream back on DFlash2 after MTP rounds catches its taps up: its tree is that of a stream drafting every round.
fn testCatchUp(a: std.mem.Allocator, h: *Harness) !void {
    const m = try create(a, h, true);
    defer m.deinit();
    const Context = glm.metal_dflash_model.Context;
    const taps = try h.device.buffer(@as(usize, 400) * config.tapWidth() * 2, mtl.ResourceOptions.shared);
    defer taps.deinit();
    var random: std.Random.DefaultPrng = .init(91);
    for (taps.slice(u16, 400 * config.tapWidth())) |*v| v.* = bf16FromF32((random.random().float(f32) * 2 - 1) * 2);
    const every = try Context.init(a, m);
    defer every.deinit(a);
    const sometimes = try Context.init(a, m);
    defer sometimes.deinit(a);
    for ([_]*Context{ every, sometimes }) |x| {
        try x.reset(config, 0);
        try feedStream(m, x, taps, 0, 20);
    }
    var fed: u32 = 20;
    // rounds of 3 taps; the second stream drafts only at rounds 5 (12 behind) and 40 (105 behind, past the window)
    for (0..41) |r| {
        for ([_]*Context{ every, sometimes }) |x| try feedStream(m, x, taps, fed, 3);
        fed += 3;
        try m.swap(every);
        var t1 = try m.proposeOptions(@intCast(r), .{ .nodes = 7, .children = 1 });
        try m.swap(every);
        defer t1.deinit();
        if (r != 5 and r != 40) continue;
        try m.swap(sometimes);
        var t2 = try m.proposeOptions(@intCast(r), .{ .nodes = 7, .children = 1 });
        try m.swap(sometimes);
        defer t2.deinit();
        try std.testing.expectEqualSlices(u32, t1.tokens, t2.tokens);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(t1.scores), std.mem.sliceAsBytes(t2.scores));
    }
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var h = try Harness.init(a);
    defer h.deinit();
    try testTapCapture(a, &h);
    try testCommitAhead(&h);
    try testFourBit(a, &h);
    try testBatch(a, &h, false);
    try testBatch(a, &h, true);
    try testSlicedActive(a, &h);
    try testCatchUp(a, &h);
    try @import("glm_metal_dflash_ahead.zig").run(a, &h);
    try @import("glm_metal_dflash_follow.zig").run(a, &h);
    try @import("glm_metal_dflash_inline.zig").run(a, &h);
    const m = try create(a, &h, false);
    defer m.deinit();
    // two rounds of taps: the first absorbs, the second rides the staged block path
    try m.absorb(.{ .buf = m.pending, .off = 0 }, 2);
    try m.flush();
    var tree = try m.propose(7, 15);
    defer tree.deinit();
    try std.testing.expect(tree.tokens.len > 0);
    for (tree.tokens) |token| try std.testing.expect(token < config.vocab);
    var more = try m.propose(3, 8);
    defer more.deinit();
    try std.testing.expectEqual(m.committed_end, m.session.context.end);
    // a backlog of 40 rows, contiguous in the ring, catches up in the frame that drafts
    const backlog = try h.device.buffer(40 * @as(usize, config.tapWidth()) * 2, mtl.ResourceOptions.shared);
    defer backlog.deinit();
    @memset(backlog.contents()[0..backlog.length()], 0);
    try m.absorb(.{ .buf = backlog, .off = 0 }, 40);
    var caught = try m.propose(5, 7);
    defer caught.deinit();
    try std.testing.expectEqual(@as(u64, 42), m.session.context.end);
    try std.testing.expectEqual(m.committed_end, m.session.context.end);
}

test {
    _ = Backend;
    _ = Model;
    _ = Target;
    _ = wide;
    _ = affine;
}
