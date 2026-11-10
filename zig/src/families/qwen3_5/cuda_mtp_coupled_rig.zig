//! Host Qwen `Engine` + `Mtp` + causal `Target` harness for end-to-end MTP coupled-draw verification.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const mtp = @import("cuda_mtp.zig");
const sim_mod = @import("cuda_mtp_coupled_sim.zig");
const t = std.testing;

const Sim = sim_mod.Sim;
const vocab = sim_mod.vocab;
const hidden = sim_mod.hidden;
const q_heads = sim_mod.q_heads;
const kv_heads = sim_mod.kv_heads;
const aw = sim_mod.aw;
const kw = sim_mod.kw;
const init_state = sim_mod.init_state;
const stepState = sim_mod.stepState;
const drawState = sim_mod.drawState;

fn func(d: *const cuda.Driver, id: usize) cuda.Function {
    return .{ .d = d, .handle = @ptrFromInt(id) };
}

fn affine(tag: u64, n: usize, k: usize) cuda.qlinear.Affine4 {
    return .{ .w = tag, .s = tag, .b = tag, .n = n, .k = k, .npad = std.mem.alignForward(usize, n, 64) };
}

/// Self-contained Qwen `Engine` + `Mtp` fixture running real production MTP staging, draws, and context pools on host.
pub const Rig = struct {
    drv: cuda.Driver = undefined,
    e: engine.Engine = undefined,
    d: mtp.Mtp = undefined,

    pub fn init(r: *Rig, gpa: std.mem.Allocator, trees: bool, batch: bool) !void {
        Sim.head_bias = 0;
        Sim.resetCounters();
        r.drv = undefined;
        r.drv.api.cuMemAlloc_v2 = Sim.memAlloc;
        r.drv.api.cuMemFree_v2 = Sim.memFree;
        r.drv.api.cuMemHostAlloc = Sim.hostAlloc;
        r.drv.api.cuMemFreeHost = Sim.hostFree;
        r.drv.api.cuMemcpyHtoD_v2 = Sim.htod;
        r.drv.api.cuMemcpyHtoDAsync_v2 = Sim.htodAsync;
        r.drv.api.cuMemcpyDtoHAsync_v2 = Sim.dtohAsync;
        r.drv.api.cuMemcpyDtoD_v2 = Sim.dtod;
        r.drv.api.cuMemcpyDtoDAsync_v2 = Sim.dtodAsync;
        r.drv.api.cuMemsetD8_v2 = Sim.memset8;
        r.drv.api.cuMemsetD8Async = Sim.memset8Async;
        r.drv.api.cuStreamSynchronize = Sim.syncStream;
        r.drv.api.cuCtxSynchronize = Sim.syncCtx;
        r.drv.api.cuEventCreate = Sim.evCreate;
        r.drv.api.cuEventRecord = Sim.evRecord;
        r.drv.api.cuEventQuery = Sim.evSync;
        r.drv.api.cuEventSynchronize = Sim.evSync;
        r.drv.api.cuEventDestroy_v2 = Sim.evDestroy;
        r.drv.api.cuLaunchKernel = Sim.launch;
        r.drv.api.cuLaunchKernelEx = Sim.launchEx;
        r.e = undefined;
        r.e.d = &r.drv;
        r.e.stream = .{ .d = &r.drv, .handle = null };
        r.e.kit = null;
        r.e.mo = null;
        r.e.int8 = .{};
        r.e.embed_bits = 4;
        r.e.embed_w = 0;
        r.e.embed_s = 0;
        r.e.embed_b = 0;
        r.e.final_norm = sim_mod.w_final_norm;
        r.e.rope_table = 0;
        r.e.picks = 0;
        r.e.head = .{ .a4 = affine(sim_mod.tag_head, vocab, hidden) };
        r.e.g = .{
            .hidden = hidden,
            .intermediate = hidden,
            .layers = 1,
            .vocab = vocab,
            .k_heads = 1,
            .v_heads = 2,
            .q_heads = q_heads,
            .kv_heads = kv_heads,
            .eps = 1e-6,
            .tied = false,
            .context = 512,
            .linear = @splat(false),
            .experts = 4,
            .top_k = 2,
        };
        r.e.fns.sums = func(&r.drv, sim_mod.fn_sums);
        r.e.fns.embed = func(&r.drv, sim_mod.fn_embed);
        r.e.fns.norm = func(&r.drv, sim_mod.fn_norm);
        r.e.fns.argmax = func(&r.drv, sim_mod.fn_argmax);
        r.e.affine = undefined;
        r.e.affine.pdl = false;
        r.e.affine.wide_verify = false;
        r.e.affine.half_prompt = false;
        r.e.affine.mlp128 = false;
        r.e.affine.split = null;
        r.e.affine.gemv_blocks = 1;
        r.e.affine.gemv_fn = func(&r.drv, sim_mod.fn_gemv);
        r.e.affine.group_fn = func(&r.drv, sim_mod.fn_group);
        r.e.affine.prefill_fn = func(&r.drv, sim_mod.fn_prefill);
        r.e.affine.prefill22_fn = func(&r.drv, sim_mod.fn_prefill);
        r.e.affine.verify_fn = func(&r.drv, sim_mod.fn_verify);
        r.e.affine.verify64_fn = func(&r.drv, sim_mod.fn_verify);
        r.e.affine.verify128_fn = func(&r.drv, sim_mod.fn_verify);
        r.e.affine.many_fn = func(&r.drv, sim_mod.fn_noop);
        r.e.draws = .{
            .draw_fn = func(&r.drv, sim_mod.fn_noop),
            .draw_ids_fn = func(&r.drv, sim_mod.fn_noop),
            .logprob_rows_fn = func(&r.drv, sim_mod.fn_logprob_rows),
            .topk_slices_fn = func(&r.drv, sim_mod.fn_noop),
            .topk_merge_fn = func(&r.drv, sim_mod.fn_noop),
            .draw_rows_fn = func(&r.drv, sim_mod.fn_draw_rows),
        };
        const noop = func(&r.drv, sim_mod.fn_noop);
        r.e.moe = .{ .router_fn = noop, .topk_fn = noop, .combine_fn = noop, .router_tiles_fn = noop };
        r.e.plan = .{ .plan_small = noop, .plan_rows = noop, .plan_rank = noop, .plan_offsets = noop, .plan_scatter = noop, .plan_rank_rows = noop, .plan_scatter_rows = noop };
        r.e.experts = .{ .up_fn = noop, .down_fn = noop, .pair_fn = noop, .pre_up_fn = noop, .pre_down_fn = noop, .pre_pair_fn = noop, .pack_fn = noop, .resident = .{ 1, 1, 1 } };
        r.e.experts3 = undefined;
        r.e.routed = .{};
        r.d = .{
            .gpa = gpa,
            .io = t.io,
            .e = &r.e,
            .w = .{
                .arena = .{ .d = &r.drv, .ptr = 0, .len = 0, .borrowed = true },
                .fc_e = affine(sim_mod.tag_fc_e, hidden, hidden),
                .fc_h = affine(sim_mod.tag_fc_h, hidden, hidden),
                .norm_e = sim_mod.w_norm_e,
                .norm_h = sim_mod.w_norm_h,
                .input_norm = sim_mod.w_input_norm,
                .post_norm = sim_mod.w_post_norm,
                .q_norm = sim_mod.w_q_norm,
                .k_norm = sim_mod.w_k_norm,
                .norm = sim_mod.w_norm,
                .q = .{ .a4 = affine(sim_mod.tag_q, 2 * aw, hidden) },
                .k = .{ .a4 = affine(sim_mod.tag_k, kw, hidden) },
                .v = .{ .a4 = affine(sim_mod.tag_v, kw, hidden) },
                .o = .{ .a4 = affine(sim_mod.tag_o, hidden, aw) },
                .mlp = .{ .moe = .{ .router = 0, .experts = .{ .format = .affine4, .up = 0, .down = 0, .width = hidden, .dims = hidden, .experts = 5 } } },
            },
            .mod = undefined,
            .dmod = undefined,
            .fns = .{ .norm_rope = func(&r.drv, sim_mod.fn_norm_rope), .attn = func(&r.drv, sim_mod.fn_attn), .attn_tree = func(&r.drv, sim_mod.fn_attn_tree), .attn_rows = func(&r.drv, sim_mod.fn_attn_rows) },
            .ring_fns = .{ .gather_taps = func(&r.drv, sim_mod.fn_gather_taps), .gather_rows = func(&r.drv, sim_mod.fn_gather_rows), .ring_put = func(&r.drv, sim_mod.fn_ring_put) },
            .scratch = undefined,
            .capacity = 512,
            .depth = 3,
            .trees = trees,
            .tap = .{0},
            .taps_in = 0,
            .hin = 0,
            .hn = 0,
            .emb = 0,
            .en = 0,
            .h = 0,
            .r = 0,
            .x = 0,
            .qraw = 0,
            .q = 0,
            .kraw = 0,
            .kb = 0,
            .vb = 0,
            .att = 0,
            .xs = 0,
            .logits = 0,
            .words = 0,
            .picks = 0,
            .lane_state = 0,
            .lane_k = 0,
            .lane_v = 0,
            .anc = 0,
            .ids = 0,
            .tok = 0,
            .staged = undefined,
            .tab = 0,
            .batch = batch,
            .step_ms = 0.01,
        };
        const total = try r.d.layout(false);
        r.d.scratch = try cuda.DeviceBuffer.alloc(&r.drv, total);
        errdefer r.d.scratch.free();
        _ = try r.d.layout(true);
        r.d.staged = try cuda.HostBuffer.alloc(&r.drv, mtp.Mtp.staged_len);
        Sim.rules_addr = r.d.rules;
        Sim.rules_end = r.d.rules + mtp.max_lanes * @sizeOf(cuda.sample.Rule);
        try r.d.reserve(17);
        Sim.resetCounters();
    }

    pub fn deinit(r: *Rig) void {
        r.e.stream.synchronize() catch {};
        var it = r.d.streams.valueIterator();
        while (it.next()) |ctx| r.d.drop(ctx.*);
        r.d.streams.deinit(r.d.gpa);
        for (r.d.pool.items) |ctx| r.d.drop(ctx);
        r.d.pool.deinit(r.d.gpa);
        r.d.staged.free();
        r.d.scratch.free();
        Sim.rules_addr = 0;
        Sim.rules_end = 0;
    }
};

/// Causal target backend matching `Sim`'s state and `sample.cu`'s draws, writing hidden states into device tap rings.
pub const Target = struct {
    const be = lanes.backend;
    const Slot = struct {
        tap: cuda.DeviceBuffer,
        states: std.ArrayList(u64) = .empty,
        win_base: usize = 0,
        win_states: [engine.most_window]u64 = @splat(0),
        written: u64 = 0,
    };

    gpa: std.mem.Allocator,
    drv: *const cuda.Driver,
    map: std.AutoHashMapUnmanaged(*lanes.Stream, Slot) = .empty,

    pub fn deinit(self: *Target) void {
        var it = self.map.valueIterator();
        while (it.next()) |sl| {
            sl.states.deinit(self.gpa);
            var tap = sl.tap;
            tap.free();
        }
        self.map.deinit(self.gpa);
    }

    pub fn writeTap(tap: cuda.DeviceBuffer, pos: u64, st: u64) void {
        const slot: usize = @intCast(pos % engine.tap_rows);
        Sim.cell(tap.ptr, hidden * 2, slot).* = .{ st, 0, 0 };
    }

    fn cast(p: *anyopaque) *Target {
        return @ptrCast(@alignCast(p));
    }

    fn prefillFn(p: *anyopaque, s: *lanes.Stream) !void {
        const self = cast(p);
        const tap = try cuda.DeviceBuffer.alloc(self.drv, engine.tap_rows * hidden * 2);
        errdefer {
            var tmp = tap;
            tmp.free();
        }
        var states: std.ArrayList(u64) = .empty;
        errdefer states.deinit(self.gpa);
        var prev = init_state;
        for (s.prompt(), 0..) |tok, pos| {
            prev = stepState(prev, tok, pos + 1);
            try states.append(self.gpa, prev);
            writeTap(tap, pos, prev);
        }
        try self.map.put(self.gpa, s, .{ .tap = tap, .states = states, .written = s.prompt().len });
    }

    fn firstFn(p: *anyopaque, s: *lanes.Stream, position: u64) !u64 {
        const self = cast(p);
        const sl = self.map.getPtr(s).?;
        try t.expectEqual(@as(u64, sl.states.items.len), position);
        return try drawState(self.gpa, sl.states.items[sl.states.items.len - 1], s.sampling, position);
    }

    fn queueFn(p: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
        const self = cast(p);
        const sl = self.map.getPtr(s).?;
        const tok: u32 = switch (feed) {
            .value => |v| v,
            .handle => |h| @intCast(h),
        };
        const pos = sl.states.items.len;
        try t.expectEqual(@as(u64, pos + 1), position);
        const next_st = stepState(sl.states.items[pos - 1], tok, pos + 1);
        try sl.states.append(self.gpa, next_st);
        writeTap(sl.tap, pos, next_st);
        sl.written = pos + 1;
        return try drawState(self.gpa, next_st, s.sampling, position);
    }

    fn readFn(_: *anyopaque, handle: u64) !u32 {
        return @intCast(handle);
    }

    fn verifyFn(p: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const self = cast(p);
        for (windows, out) |w, o| {
            const sl = self.map.getPtr(w.stream).?;
            const base = sl.states.items.len;
            sl.win_base = base;
            var depths: [engine.most_window]usize = @splat(0);
            sl.win_states[0] = stepState(sl.states.items[base - 1], w.pending, base + 1);
            writeTap(sl.tap, base, sl.win_states[0]);
            for (w.tokens, 0..) |tok, d_idx| {
                const r = d_idx + 1;
                const par: usize = if (w.parents) |ps| @intCast(ps[d_idx] + 1) else d_idx;
                depths[r] = depths[par] + 1;
                sl.win_states[r] = stepState(sl.win_states[par], tok, base + depths[r] + 1);
                writeTap(sl.tap, base + r, sl.win_states[r]);
            }
            for (0..w.rows()) |r| try sl.states.append(self.gpa, sl.win_states[r]);
            sl.written = base + w.rows();
            for (o.sampled, w.positions, 0..) |*dst, pos, r| {
                try t.expectEqual(@as(u64, base + depths[r] + 1), pos);
                dst.* = try drawState(self.gpa, sl.win_states[r], w.stream.sampling, pos);
            }
            @memcpy(o.drafts, w.tokens);
        }
    }

    fn keepFn(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        const self = cast(p);
        for (windows, paths) |w, path| {
            const sl = self.map.getPtr(w.stream).?;
            sl.states.shrinkRetainingCapacity(sl.win_base);
            for (path) |row| try sl.states.append(self.gpa, sl.win_states[row]);
        }
    }

    fn draftFn(_: *anyopaque, requests: []const be.DraftRequest) !void {
        for (requests) |r| if (r.depth != 0) return error.UnexpectedTargetDraft;
    }

    fn featuresFn(p: *anyopaque, s: *lanes.Stream, _: []const u32, start: u64, count: u32) !be.Features {
        const sl = cast(p).map.getPtr(s).?;
        if (start + count > sl.written) return error.RowsNotHeld;
        return .{ .buffer = sl.tap.ptr, .offset = start, .rows = count, .row_bytes = hidden * 2, .space = .device, .dtype = .bf16 };
    }

    fn releaseFn(p: *anyopaque, s: *lanes.Stream) void {
        const self = cast(p);
        if (self.map.fetchRemove(s)) |kv| {
            var sl = kv.value;
            sl.states.deinit(self.gpa);
            sl.tap.free();
        }
    }

    const vtable: be.Backend.VTable = .{
        .prefill = prefillFn,
        .first = firstFn,
        .queue = queueFn,
        .read = readFn,
        .verify = verifyFn,
        .keep = keepFn,
        .draft = draftFn,
        .features = featuresFn,
        .release = releaseFn,
    };

    pub fn backend(self: *Target) be.Backend {
        return .{ .ptr = self, .vtable = &vtable };
    }
};

pub const RunOpts = struct {
    drafts: bool,
    trees: bool = false,
    batch: bool = true,
    head_bias: u64 = 0,
    prompt_seed: usize = 0,
    max_new: u32 = 20,
};

pub const RunStats = struct {
    replies: [4][24]u32 = @splat(@splat(0)),
    accepted: [4]u64 = @splat(0),
    drafted: [4]u64 = @splat(0),
    batch_passes: u64 = 0,
};

pub fn runEndToEnd(samplings: []const ?lanes.Sampling, opts: RunOpts) !RunStats {
    const gpa = t.allocator;
    var rig: Rig = .{};
    try rig.init(gpa, opts.trees, opts.batch);
    defer rig.deinit();
    Sim.head_bias = opts.head_bias;
    var target: Target = .{ .gpa = gpa, .drv = &rig.drv };
    defer target.deinit();
    const costs: [64]lanes.config.Cost = blk: {
        var c: [64]lanes.config.Cost = undefined;
        for (&c, 1..) |*v, i| v.* = .{ .width = @intCast(i), .ms = 5.0 };
        break :blk c;
    };
    for (costs) |c| rig.d.setCost(c.width, c.ms);
    var wrapped = try lanes.drafted.Drafted.init(gpa, target.backend(), rig.d.drafter());
    defer wrapped.deinit();
    var facts = if (opts.drafts) wrapped.facts(@import("cuda_native.zig").laneFacts(64, 128)) else @import("cuda_native.zig").laneFacts(64, 128);
    facts.copy_drafts = false;
    facts.window_costs = &costs;
    facts.lane_costs = &costs;
    if (opts.drafts) facts.plain_guard = false;
    var cfg = try lanes.Config.init(gpa, facts, 64, 63);
    defer cfg.deinit(gpa);
    cfg.timed = false;
    var clock: lanes.fake.FixedClock = .{};
    var eng = lanes.Engine.init(gpa, &cfg, if (opts.drafts) wrapped.backend() else target.backend(), clock.clock());
    defer eng.deinit();
    var streams: [4]lanes.Stream = undefined;
    var prompts: [4][6]u32 = undefined;
    var opened: usize = 0;
    defer for (streams[0..opened]) |*s| s.deinit(gpa);
    for (samplings, 0..) |samp, i| {
        for (&prompts[i], 0..) |*tok, j| tok.* = @intCast(1 + ((opts.prompt_seed + i) * 11 + j * 7) % (vocab - 1));
        streams[i] = try lanes.Stream.init(gpa, .{
            .id = "s",
            .prompt = &prompts[i],
            .max_new = opts.max_new,
            .drafts = opts.drafts,
            .sampling = samp,
            .loop_guard = false,
        });
        opened += 1;
        try eng.addStream(&streams[i]);
    }
    while (eng.activeCount() > 0) try eng.step();
    try t.expect(!Sim.ring_fault);
    var out: RunStats = .{ .batch_passes = rig.d.batch_passes };
    for (streams[0..samplings.len], 0..) |*s, i| {
        @memcpy(out.replies[i][0..opts.max_new], s.emitted());
        out.accepted[i] = s.accepted;
        out.drafted[i] = s.drafted;
    }
    return out;
}
