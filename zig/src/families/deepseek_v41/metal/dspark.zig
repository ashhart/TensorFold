//! The DSpark drafter on Metal: per-slot key rings, main rows absorbed from the taps, blocks of drafts proposed.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const ctx_ = @import("ctx.zig");
const kv = @import("kv.zig");
const rope = @import("rope.zig");
const scratch = @import("scratch.zig");
const weights = @import("weights.zig");
const pipelines = @import("pipelines.zig");
const layer = @import("layer.zig");
const forward = @import("forward.zig");
const plan_ = @import("dspark_plan.zig");
const dw = @import("dspark_weights.zig");
const st = @import("dspark_stage.zig");

const Config = host.config.Config;
const Params = host.dspark.Params;
const Ctx = ctx_.Ctx;
const Ref = ctx_.Ref;
const R = pipelines.max_rows;

pub const Weights = dw.Weights;
pub const loadWeights = dw.load;
pub const Run = plan_.Run;

pub const Error = error{ Slots, NotAPrefix, DuplicateSlot, BadOutput, PastLimit, NoDspark };

/// A slot's committed rows from the target: bf16 [rows][targets * hidden] at `off` of `buf` (the slot's taps).
pub const Taps = struct { buf: mtl.Buffer, off: usize = 0, rows: u32 };

/// One slot of a pass: its anchor (the pending token) at `start` and the keyed draft parameters.
pub const Ask = struct { slot: u32, anchor: u32, start: u64, params: Params };

/// One slot's results: `block` drafts and confidences; `cand` and `base` (candidate ids, base logits) for trees.
pub const Proposal = struct { drafts: []u32, conf: []f32, cand: ?[]i32 = null, base: ?[]f32 = null };

/// A launch prepared for a group of asks: its rows and their positions; the records are in the pass's plan buffer.
pub const Launch = struct {
    rows: u32,
    asks: []const Ask,
    at: [R]u32 = undefined,
};

pub const Options = struct {
    /// slots with rings
    slots: u32,
    /// candidates a draft row keeps from its base logits (the Markov bias is added to these only)
    candidates: u32 = 64,
};

/// The drafter's state and working buffers; the config, kernels, scratch, rope tables and heads are borrowed.
pub const Pass = struct {
    a: std.mem.Allocator,
    c: *const Config,
    p: *const pipelines.Pipelines,
    s: *const scratch.Scratch,
    tables: *const rope.Tables,
    top: *const weights.Top,
    w: *const Weights,
    device: mtl.Device,
    queue: mtl.Queue,
    opt: Options,
    b: forward.Bufs,
    plan: plan_.Plan,
    /// act_quant of the taps rows [16][targets * hidden]
    quant: mtl.Buffer,
    /// one ring buffer a stage: [slots][window + block][head_dim] bf16
    rings: []mtl.Buffer,
    runs: []Run,
    /// the context-free slot the layer encoders' Ctx names (the drafter reads its own rings)
    noslot: kv.Slot,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, c: *const Config, p: *const pipelines.Pipelines, s: *const scratch.Scratch, tables: *const rope.Tables, top: *const weights.Top, w: *const Weights, opt: Options) !*Pass {
        if (c.mtp_layers == 0 or p.route_dspark == null or w.stages.len != c.mtp_layers) return error.NoDspark;
        if (opt.slots == 0 or c.dspark_block == 0 or c.dspark_block > R or opt.candidates == 0) return error.Slots;
        const x = try a.create(Pass);
        errdefer a.destroy(x);
        x.* = .{ .a = a, .c = c, .p = p, .s = s, .tables = tables, .top = top, .w = w, .device = device, .queue = queue, .opt = opt, .b = undefined, .plan = undefined, .quant = undefined, .rings = undefined, .runs = undefined, .noslot = .{ .regions = &.{}, .limit = 0, .a = a } };
        x.b = try forward.Bufs.init(device, c);
        errdefer x.b.deinit();
        x.plan = try plan_.Plan.init(device);
        errdefer x.plan.deinit();
        x.quant = try device.buffer(@as(usize, R) * c.dspark_targets.len * c.hidden * 2, mtl.ResourceOptions.shared);
        errdefer x.quant.deinit();
        x.rings = try a.alloc(mtl.Buffer, c.mtp_layers);
        errdefer a.free(x.rings);
        var made: usize = 0;
        errdefer for (x.rings[0..made]) |r| r.deinit();
        const bytes = @as(usize, opt.slots) * plan_.poolRows(c) * c.head_dim * 2;
        for (x.rings) |*r| {
            r.* = try device.buffer(bytes, mtl.ResourceOptions.shared);
            @memset(r.contents()[0..bytes], 0);
            made += 1;
        }
        x.runs = try a.alloc(Run, opt.slots);
        @memset(x.runs, .{});
        return x;
    }

    pub fn deinit(x: *Pass) void {
        for (x.rings) |r| r.deinit();
        x.a.free(x.rings);
        x.a.free(x.runs);
        x.quant.deinit();
        x.plan.deinit();
        x.b.deinit();
        x.a.destroy(x);
    }

    /// The launch context of `rows` rows (the drafter reads its own rings, so no slot and no first position).
    pub fn ctx(x: *const Pass, rows: u32) Ctx {
        return .{ .c = x.c, .p = x.p, .s = x.s, .slot = &x.noslot, .tables = x.tables, .p0 = 0, .rows = rows };
    }

    /// The heads the family's chain and trees read.
    pub fn markov(x: *const Pass) host.dspark.Markov {
        return x.w.markov(x.c);
    }

    /// A new request on the slot: its context restarts at position 0.
    pub fn reset(x: *Pass, slot: u32) void {
        if (slot < x.runs.len) x.runs[slot] = .{};
    }

    /// The slot's context restarts after `at` (rows that are not on the device).
    pub fn restart(x: *Pass, slot: u32, at: u64) void {
        if (slot < x.runs.len) x.runs[slot] = .{ .first = at, .end = at };
    }

    /// The command buffer of one serial encoder: `f` encodes, then it is committed and waited for.
    fn run(x: *const Pass, comptime F: type, f: F) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = x.queue.commandBuffer();
        const e = cb.compute(.serial);
        f.encode(e) catch |err| {
            e.end();
            return err;
        };
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("dspark command buffer failed: {s}\n", .{msg});
            return error.CommandBufferFailed;
        }
    }

    // ingest -------------------------------------------------------------------------------------------------------

    /// Committed rows into the slot's rings: `taps` rows `rows[i]` (a contiguous run) land at positions start + i.
    pub fn ingest(x: *Pass, slot: u32, start: u64, taps: Taps, rows: []const u32) !void {
        if (slot >= x.runs.len) return error.Slots;
        if (rows.len == 0) return;
        for (rows, 0..) |r, i| if (r != rows[0] + i) return error.NotAPrefix;
        if (rows[rows.len - 1] >= taps.rows) return error.NotAPrefix;
        // older rows than the window are never read
        const sp = ingestSpan(rows.len, x.c.window);
        const keep = sp.count;
        const skip = sp.skip;
        var at: usize = 0;
        while (at < keep) : (at += R) {
            const n: u32 = @intCast(@min(R, keep - at));
            const row0: usize = rows[skip] + at;
            try x.ingestRows(slot, start + skip + at, taps, row0, n);
        }
        x.runs[slot].land(start + skip, keep);
    }

    /// `n` main rows from taps row `row0`, landing at `start`.
    fn ingestRows(x: *Pass, slot: u32, start: u64, taps: Taps, row0: usize, n: u32) !void {
        if (start + n > x.tables.positions) return error.PastLimit;
        try x.plan.writeIngest(x.c, slot, start, n);
        const row_bytes = x.c.dspark_targets.len * x.c.hidden * 2;
        const refs = x.plan.ingestRefs();
        const job = struct {
            x: *const Pass,
            n: u32,
            taps: Ref,
            refs: @TypeOf(refs),
            fn encode(j: @This(), e: mtl.ComputeEncoder) !void {
                const cx = j.x.ctx(j.n);
                try st.ingest(&cx, e, j.x.w, j.taps, .{ .buf = j.x.quant }, j.x.rings, j.refs.pos, j.refs.dest);
            }
        };
        try x.run(job, .{ .x = x, .n = n, .taps = .{ .buf = taps.buf, .off = taps.off + row0 * row_bytes }, .refs = refs });
    }

    // propose ------------------------------------------------------------------------------------------------------

    /// The most slots a launch carries (rows are limited to 16).
    pub fn group(x: *const Pass) u32 {
        return @max(1, R / x.c.dspark_block);
    }

    /// The plan and the block ids of a launch for these asks (at most `group`, distinct slots).
    pub fn prepare(x: *Pass, asks: []const Ask) !Launch {
        const n = x.c.dspark_block;
        if (asks.len == 0 or asks.len > x.group()) return error.Slots;
        var l: Launch = .{ .rows = @intCast(asks.len * n), .asks = asks };
        const ids: []u32 = @alignCast(std.mem.bytesAsSlice(u32, x.b.bytes(x.b.ids, 4 * l.rows)));
        for (asks, 0..) |a, j| {
            if (a.slot >= x.runs.len) return error.Slots;
            for (asks[0..j]) |o| if (o.slot == a.slot) return error.DuplicateSlot;
            if (a.start + n > x.tables.positions or a.anchor >= x.c.vocab) return error.PastLimit;
            try x.plan.writeBlock(x.c, @intCast(j), a.slot, a.start, x.runs[a.slot]);
            host.dspark.blockIds(.{ .block = n, .noise = x.c.dspark_noise_token }, a.anchor, ids[j * n ..][0..n]);
            for (0..n) |i| l.at[j * n + i] = @intCast(a.start + i);
        }
        return l;
    }

    pub fn rowsOf(x: *const Pass) st.Rows {
        const refs = x.plan.refs();
        return .{ .pos = refs.pos, .dest = refs.dest, .meta = refs.meta, .sel = refs.sel };
    }

    /// The stage's input and output streams: stage i reads streams[i % 2], writes the other.
    pub fn io(x: *const Pass, i: usize) layer.Io {
        const a = i % 2;
        return .{ .x = x.b.streams[a], .pre = x.b.pres[a], .x_out = x.b.streams[1 - a], .pre_out = x.b.pres[1 - a] };
    }

    pub fn encodeEmbed(x: *const Pass, e: mtl.ComputeEncoder, l: *const Launch) void {
        const cx = x.ctx(l.rows);
        st.embed(&cx, e, x.top, &x.b);
    }

    pub fn encodeStage(x: *const Pass, e: mtl.ComputeEncoder, l: *const Launch, i: usize) !void {
        const cx = x.ctx(l.rows);
        try st.stage(&cx, e, &x.w.stages[i], x.io(i), x.rings[i], x.rowsOf());
    }

    pub fn encodeHead(x: *const Pass, e: mtl.ComputeEncoder, l: *const Launch) void {
        const cx = x.ctx(l.rows);
        st.head(&cx, e, x.top, x.w, &x.b);
    }

    /// The whole launch: embed, every stage, the head.
    pub fn encodeAll(x: *const Pass, e: mtl.ComputeEncoder, l: *const Launch) !void {
        x.encodeEmbed(e, l);
        for (0..x.w.stages.len) |i| try x.encodeStage(e, l, i);
        x.encodeHead(e, l);
    }

    /// Candidates and the Markov chain over a launch's head outputs, each ask's results into `out` (its order).
    pub fn finish(x: *Pass, l: *const Launch, out: []Proposal) !void {
        const n: usize = x.c.dspark_block;
        const k: usize = @min(x.opt.candidates, x.c.vocab);
        const D: usize = x.c.hidden;
        const V: usize = x.c.vocab;
        if (out.len != l.asks.len) return error.BadOutput;
        var arena = std.heap.ArenaAllocator.init(x.a);
        defer arena.deinit();
        const sa = arena.allocator();
        const logits = x.b.logitsOf(x.c, l.rows);
        const hf: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, x.b.bytes(x.b.hf, 2 * l.rows * D)));
        const cand = try sa.alloc(i32, l.rows * k);
        const base = try sa.alloc(f32, l.rows * k);
        const hid = try sa.alloc(f32, n * D);
        const mk = x.markov();
        for (0..l.rows) |r| try host.dspark.candidates(sa, logits[r * V ..][0..V], 0, cand[r * k ..][0..k], base[r * k ..][0..k]);
        for (l.asks, out, 0..) |a, *o, j| {
            if (o.drafts.len != n or o.conf.len != n) return error.BadOutput;
            const cd = cand[j * n * k ..][0 .. n * k];
            const cv = base[j * n * k ..][0 .. n * k];
            for (hid, hf[j * n * D ..][0 .. n * D]) |*h, v| h.* = host.dspark.bf16(v);
            try host.dspark.chain(sa, mk, cd, cv, k, a.anchor, a.params, if (mk.conf != null) hid else null, o.drafts, o.conf);
            if (o.cand) |t| {
                if (t.len != cd.len) return error.BadOutput;
                @memcpy(t, cd);
            }
            if (o.base) |t| {
                if (t.len != cv.len) return error.BadOutput;
                @memcpy(t, cv);
            }
        }
    }

    /// Drafts for every ask: groups of at most `group` slots, one launch each; `out[i]` answers `asks[i]`.
    pub fn propose(x: *Pass, asks: []const Ask, out: []Proposal) !void {
        if (out.len != asks.len) return error.BadOutput;
        const g = x.group();
        var at: usize = 0;
        while (at < asks.len) : (at += g) {
            const m = @min(g, asks.len - at);
            const l = try x.prepare(asks[at..][0..m]);
            const job = struct {
                x: *const Pass,
                l: *const Launch,
                fn encode(j: @This(), e: mtl.ComputeEncoder) !void {
                    try j.x.encodeAll(e, j.l);
                }
            };
            try x.run(job, .{ .x = x, .l = &l });
            try x.finish(&l, out[at..][0..m]);
        }
    }

    /// One slot's context for a block starting at `start`, as the attention reads it.
    pub fn context(x: *const Pass, slot: u32, start: u64) plan_.Context {
        return x.runs[slot].context(x.c.window, start);
    }
};

/// Which committed rows an ingest writes: only the last `window` (older ones are never read).
pub fn ingestSpan(n: usize, window: usize) struct { skip: usize, count: usize } {
    const skip = n -| window;
    return .{ .skip = skip, .count = n - skip };
}

test "an ingest keeps the last window of its rows" {
    const s = ingestSpan(300, 128);
    try std.testing.expectEqual(@as(usize, 172), s.skip);
    try std.testing.expectEqual(@as(usize, 128), s.count);
    try std.testing.expectEqual(@as(usize, 0), ingestSpan(5, 128).skip);
}
