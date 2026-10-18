//! DSpark's drafting pass on the GPU: draft/iface.zig's `Pass` over dspark_emit.zig's launches, every rank; one slot.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const run = @import("run.zig");
const buffers = @import("buffers.zig");
const fwd = @import("forward.zig");
const emit = @import("dspark_emit.zig");
const iface = @import("draft/iface.zig");
const dspark = @import("draft/dspark.zig");
const dsd = @import("dspark_dev.zig");
const rows_mod = @import("dspark_rows.zig");
const ph = @import("phases.zig");
const gpu_device = @import("dspark_gpu_device.zig");

pub const Error = error{ Slots, NotAPrefix, NoWeight, CandidateWidth, BadCandidates };

pub const GpuPass = struct {
    gpa: std.mem.Allocator,
    f: *fwd.Forward,
    shape: dspark.Shape,
    /// candidates a rank (drafter.k: min(64, 128 / world))
    k: u32,
    /// the most rows an ingest takes at once (its buffers are planned for it)
    ingest_rows: u32,
    markov: dspark.Markov,
    w1: []u16,
    w2: []u16,
    conf: ?[]f32,
    arena: std.heap.ArenaAllocator,
    pass_calls: []const calls.Call = &.{},
    /// where the pass's host step ("glue.ds_out") starts: `begin` launches the calls before it, `collect` the rest
    head: usize = 0,
    /// the ask a `begin` launched (TF_DSV41_SPEC_DRAFT, draft/spec.zig), until its `collect`
    begun: ?iface.Ask = null,
    /// the first position whose ring rows are this slot's
    valid: u64 = 0,
    st: emit.Statics,
    h64: []i64,
    h32: []i32,
    send: ?cuda.DeviceBuffer = null,
    gathered: ?cuda.DeviceBuffer = null,
    /// the last pass's results on the host: gathered candidates / base logits [n][W k], head hidden [n][D]
    cand: []i32,
    cval: []f32,
    hid: []f32,
    /// TF_DSV41_DRAFT_GRAPHS (dspark_dev.zig): candidates on the device, the pass graphed; null: the host step
    dev: ?*dsd.Dev = null,
    /// the last eager pass's logits (a check pass compares the device candidates with the host's on them)
    last_logits: u64 = 0,
    last_v: usize = 0,
    armed: ?struct { start: u64, n: u32, sampled: bool } = null, // window awaiting its device pick, then its pass
    /// `wait`: the host values come from the accept's copy (a sampled window): `begin` waits for `staged` first
    dl: ?struct { start: u64, n: u32, host: [*]const i64, rows: u32, wait: bool = false } = null,
    /// a sampled window's accept: device ids / segment / result, its pinned copy and event
    sscr: ?cuda.DeviceBuffer = null,
    shost: ?cuda.HostBuffer = null,
    staged: ?cuda.Event = null,
    /// `begin` took a device-launched pass: `collect` waits and unpacks without counting a pass again
    begun_dev: bool = false,
    prev_ext: ?fwd.Forward.Ext = null,
    stats: struct { device_starts: u64 = 0, device_dropped: u64 = 0 } = .{},

    /// The buffer roles DSpark adds to the forward's plan (call before Runner.bind).
    pub fn plan(a: std.mem.Allocator, f: *const fwd.Forward, p: *buffers.Plan, ingest_rows: u32) !void {
        try p.add(try emit.emitIngest(a, f.cfg, f.widths, f.opts, ingest_rows, 0));
        try p.add(try emit.emitPass(a, f.cfg, f.widths, f.opts, f.cfg.dspark_block));
    }

    pub fn init(gpa: std.mem.Allocator, f: *fwd.Forward, ingest_rows: u32) !*GpuPass {
        const cfg = f.cfg;
        const r = f.runner;
        const world = f.comm.world();
        const x = try gpa.create(GpuPass);
        var assembled = false;
        var registered = false;
        errdefer if (!registered) {
            if (assembled) x.deinit() else gpa.destroy(x);
        };
        const n: i64 = cfg.dspark_block;
        const k: u32 = @min(64, 128 / world);
        const rank = cfg.dspark_markov_rank;
        const V: usize = cfg.vocab;
        const w1 = try gpa.alloc(u16, V * rank);
        errdefer if (!assembled) gpa.free(w1);
        const w2 = try gpa.alloc(u16, V * rank);
        errdefer if (!assembled) gpa.free(w2);
        var conf: ?[]f32 = null;
        errdefer if (!assembled) {
            if (conf) |c| gpa.free(c);
        };
        if (r.weights.get("dspark.conf")) |t| {
            conf = try gpa.alloc(f32, t.len / 4);
        }
        const st: emit.Statics = .{ .n = n, .window = cfg.window };
        const W: usize = world * k;
        const h64 = try gpa.alloc(i64, @intCast(st.len64()));
        errdefer if (!assembled) gpa.free(h64);
        const h32 = try gpa.alloc(i32, @intCast(st.len32()));
        errdefer if (!assembled) gpa.free(h32);
        const cand = try gpa.alloc(i32, @as(usize, @intCast(n)) * W);
        errdefer if (!assembled) gpa.free(cand);
        const cval = try gpa.alloc(f32, @as(usize, @intCast(n)) * W);
        errdefer if (!assembled) gpa.free(cval);
        const hid = try gpa.alloc(f32, @as(usize, @intCast(n)) * cfg.hidden);
        errdefer if (!assembled) gpa.free(hid);
        x.* = .{
            .gpa = gpa,
            .f = f,
            .shape = .{ .block = cfg.dspark_block, .noise = cfg.dspark_noise_token, .window = cfg.window, .rank = rank, .hidden = cfg.hidden, .candidates = world * k },
            .k = k,
            .ingest_rows = ingest_rows,
            .markov = .{ .w1 = w1, .w2 = w2, .rank = rank, .conf = conf },
            .w1 = w1,
            .w2 = w2,
            .conf = conf,
            .arena = std.heap.ArenaAllocator.init(gpa),
            .st = st,
            .h64 = h64,
            .h32 = h32,
            .cand = cand,
            .cval = cval,
            .hid = hid,
        };
        assembled = true;
        const Drop = struct {
            fn run(ptr: *anyopaque) void {
                const owner: *GpuPass = @ptrCast(@alignCast(ptr));
                owner.deinit();
            }
        };
        try @import("construction.zig").attach(f, x, Drop.run);
        registered = true;
        errdefer @import("construction.zig").failed(f, x);
        try fetch(r, "dspark.markov.w1", std.mem.sliceAsBytes(w1));
        try fetch(r, "dspark.markov.w2", std.mem.sliceAsBytes(w2));
        if (conf) |c| try fetch(r, "dspark.conf", std.mem.sliceAsBytes(c));
        x.pass_calls = try emit.emitPass(x.arena.allocator(), cfg, f.widths, f.opts, n);
        x.head = x.pass_calls.len;
        for (x.pass_calls, 0..) |c, i| if (std.mem.eql(u8, c.name, "glue.ds_out")) {
            x.head = i;
        };
        var publish_pick = false;
        const mode = try dsd.modeFromEnv();
        if (mode != .host) {
            if (r.kernels.others(r.stream).f.ds_cands == null) {
                if (!quiet) std.log.scoped(.dsv41).warn("draft graphs: TF_DSV41_DRAFT_GRAPHS set, but the glue fatbin has no ds_cands kernel: the host step", .{});
            } else {
                // every buffer a graphed pass bakes in, at its largest: the embedding's and exchanges' send
                _ = try x.device(&x.send, 2 * @as(usize, @intCast(n)) * cfg.hidden);
                x.dev = try dsd.Dev.init(gpa, r.d, f.comm, r.stream, mode, k, cfg.hidden, @intCast(n), conf != null, @intCast(8 * st.len64() + 4 * st.len32()), 1);
                // the speculative pass from the device pick (needs TF_DSV41_GREEDY_GPU's pick and ds_stage)
                if (r.kernels.others(r.stream).f.ds_stage != null) {
                    publish_pick = true;
                    if (r.kernels.others(r.stream).f.ds_accept_rows != null) {
                        x.sscr = try cuda.DeviceBuffer.alloc(r.d, 8 * 128);
                        x.shost = try cuda.HostBuffer.alloc(r.d, 8 * 128);
                        x.staged = try cuda.Event.init(r.d, false);
                    }
                    x.prev_ext = f.ext;
                }
            }
        }
        f.drafter = .{ .ptr = x, .ingest = followIngest, .propose = followPropose, .reset = followReset };
        if (publish_pick) {
            f.after_pick = .{ .ctx = x, .run = afterPick };
            if (x.staged != null) f.after_sample = .{ .ctx = x, .run = afterSample };
            f.ext = .{ .ptr = x, .run = followExt };
        }
        @import("construction.zig").detach(f, x);
        return x;
    }

    pub fn deinit(x: *GpuPass) void {
        const gpa = x.gpa;
        if (x.f.drafter) |d| if (d.ptr == @as(*anyopaque, x)) {
            x.f.drafter = null;
        };
        if (x.f.after_pick != null and x.f.after_pick.?.ctx == @as(*anyopaque, x)) {
            x.f.after_pick = null;
            x.f.ext = x.prev_ext;
        }
        if (x.f.after_sample != null and x.f.after_sample.?.ctx == @as(*anyopaque, x)) x.f.after_sample = null;
        if (x.sscr) |*b| b.free();
        if (x.shost) |*b| b.free();
        if (x.staged) |*e| e.deinit();
        if (x.dev) |d| {
            if (x.stats.device_starts + x.stats.device_dropped > 0) std.log.scoped(.dsv41).info("draft graphs: {d} speculative passes started from the device pick, {d} unused", .{ x.stats.device_starts, x.stats.device_dropped });
            d.deinit();
        }
        if (x.send) |*b| b.free();
        if (x.gathered) |*b| b.free();
        gpa.free(x.w1);
        gpa.free(x.w2);
        if (x.conf) |c| gpa.free(c);
        gpa.free(x.h64);
        gpa.free(x.h32);
        gpa.free(x.cand);
        gpa.free(x.cval);
        gpa.free(x.hid);
        x.arena.deinit();
        gpa.destroy(x);
    }

    fn fetch(r: *run.Runner, name: []const u8, out: []u8) !void {
        const t = r.weights.get(name) orelse return error.NoWeight;
        if (t.len < out.len) return error.NoWeight;
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = t.ptr, .len = out.len }, 0, out);
    }

    pub fn pass(x: *GpuPass) iface.Pass {
        return .{ .ptr = x, .vtable = &vtable };
    }

    const vtable: iface.Pass.VTable = .{ .ingest = ingest, .propose = propose, .reset = reset, .markov = heads, .begin = begin, .collect = collect, .arm = arm };

    /// [op, start, rows, sampled]: the armed window, every rank; SlotPass's has the slot first (5 words)
    pub const op_arm: i64 = 43;

    /// Pass.arm (rank 0): the slot's coming window, for the speculative pass from its device pick; once checks passed.
    fn arm(p: *anyopaque, slot: u32, start: u64, n_rows: u32, sampled: bool) void {
        const x = self(p);
        x.dropDevice();
        if (slot != 0 or x.f.after_pick == null or !x.devReady()) return;
        if (sampled and x.f.after_sample == null) return;
        x.f.sendDrafter(&.{ op_arm, @intCast(start), n_rows, @intFromBool(sampled) }) catch |e| {
            std.log.err("dspark: the arm did not reach the followers ({t})", .{e});
            return;
        };
        x.armed = .{ .start = start, .n = n_rows, .sampled = sampled };
    }

    fn followExt(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const x: *GpuPass = @ptrCast(@alignCast(ptr));
        if (msg[0] == op_arm) {
            if (msg.len != 4) return error.BadPlan;
            x.armed = .{ .start = @intCast(msg[1]), .n = @intCast(msg[2]), .sampled = msg[3] != 0 };
            return;
        }
        const e = x.prev_ext orelse return error.BadPlan;
        return e.run(e.ptr, msg);
    }

    pub const devReady = gpu_device.devReady;
    pub const afterPick = gpu_device.afterPick;
    pub const afterSample = gpu_device.afterSample;
    pub const launchFrom = gpu_device.launchFrom;
    pub const stageOffsets = gpu_device.stageOffsets;
    pub const dropDevice = gpu_device.dropDevice;

    fn self(p: *anyopaque) *GpuPass {
        return @ptrCast(@alignCast(p));
    }

    /// A new request's draft context, on every rank: the followers' `valid` decides their half of the candidates.
    fn reset(p: *anyopaque, slot: u32) void {
        _ = slot;
        const x = self(p);
        x.dropDevice();
        if (x.begun != null) std.log.err("dspark: a reset while a speculative pass is begun", .{});
        const op = fwd.Forward.dsResetOp();
        x.f.sendDrafter(&op) catch |e| std.log.err("dspark: the draft reset did not reach the followers ({t})", .{e});
        x.valid = 0;
    }

    fn followReset(p: *anyopaque) void {
        self(p).valid = 0;
        self(p).armed = null;
    }

    fn heads(p: *anyopaque) ?dspark.Markov {
        return self(p).markov;
    }

    /// `rows`: the taps' rows (window or prompt row indices, the last `window` kept), landing at start ..; contiguous.
    fn ingest(p: *anyopaque, slot: u32, start: u64, taps: iface.Taps, rows: []const u32) anyerror!void {
        const x = self(p);
        if (slot != 0) return error.Slots;
        if (x.begun != null) return error.PassBegun;
        if (rows.len == 0) return;
        for (rows, 0..) |r, i| if (r != rows[0] + i) return error.NotAPrefix;
        // the device-launched pass already ingested its window's rows (afterPick): the same rows at the same positions
        if (x.dl) |dl| if (start == dl.start and rows[0] == 0 and rows.len <= dl.rows) return;
        const n: u32 = @intCast(rows.len);
        const ops = fwd.Forward.dsIngestOp(start, rows[0], n, taps.rows);
        try x.f.sendDrafter(&ops);
        try x.doIngest(start, rows[0], n, taps.rows);
    }

    fn followIngest(p: *anyopaque, start: u64, row0: u32, n: u32, avail: u32) anyerror!void {
        return self(p).doIngest(start, row0, n, avail);
    }

    /// An ingest of rows on the device without draining the stream: the position as a memset (setPos), then launches.
    pub fn ingestAsync(x: *GpuPass, start: u64, row0: u32, n: u32) !void {
        const r = x.f.runner;
        const pos: i32 = @intCast(start);
        try setPos(r.d, r.addressOf("w.ds.pos") orelse return error.Unbound, pos, r.stream);
        var la = std.heap.ArenaAllocator.init(x.gpa);
        defer la.deinit();
        const cs = try emit.emitIngest(la.allocator(), x.f.cfg, x.f.widths, x.f.opts, n, row0);
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(cs);
    }

    fn doIngest(x: *GpuPass, start: u64, row0: u32, n: u32, avail: u32) !void {
        if (n == 0) return;
        if (row0 + n > avail or n > x.ingest_rows) {
            // rows not on the device (a resumed slot's prompt): the context restarts after them
            x.valid = start + n;
            return;
        }
        const r = x.f.runner;
        try r.stream.synchronize();
        const pos: i32 = @intCast(start);
        try setPos(r.d, r.addressOf("w.ds.pos") orelse return error.Unbound, pos, r.stream);
        var la = std.heap.ArenaAllocator.init(x.gpa);
        defer la.deinit();
        const cs = try emit.emitIngest(la.allocator(), x.f.cfg, x.f.widths, x.f.opts, n, row0);
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(cs);
    }

    fn propose(p: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const x = self(p);
        if (asks.len != 1) return error.Slots;
        if (x.begun != null) return error.PassBegun;
        x.dropDevice();
        const a = asks[0];
        const ops = fwd.Forward.dsProposeOp(a.anchor, a.start);
        try x.f.sendDrafter(&ops);
        try x.runPass(a.anchor, a.start);
        return x.finish(a, out);
    }

    /// The speculative pass: launched up to its host step; until `collect` only collective-free work may follow.
    fn begin(p: *anyopaque, asks: []const iface.Ask) anyerror!void {
        const x = self(p);
        if (asks.len != 1) return error.Slots;
        if (x.begun != null) return error.PassBegun;
        const a = asks[0];
        if (x.dl) |dl| {
            // a pass from the window's device pick is this ask's when its (bonus, next position) match the host's
            const h = dl.host;
            x.dl = null;
            if (dl.wait) try x.staged.?.synchronize(); // the accept's copy (not the pass)
            if (h[dl.n + 1] == a.anchor and h[dl.n + 2] == @as(i64, @intCast(a.start))) {
                x.stats.device_starts += 1;
                x.begun = a;
                x.begun_dev = true;
                return;
            }
            x.stats.device_dropped += 1;
            if (!quiet) std.log.warn("dspark: the device pick's pass ({d} at {d}) is not the ask ({d} at {d}); run again", .{ h[dl.n + 1], h[dl.n + 2], a.anchor, a.start });
        }
        const ops = fwd.Forward.dsProposeOp(a.anchor, a.start);
        try x.f.sendDrafter(&ops);
        if (x.devOn()) {
            // the whole pass, its candidates and their copy back: nothing in it waits for the host
            try x.devLaunch(a.anchor, a.start);
            x.begun = a;
            return;
        }
        try x.stagePass(a.anchor, a.start);
        try x.f.runner.window(x.pass_calls[0..x.head]);
        x.begun = a;
    }

    fn collect(p: *anyopaque, out: []iface.Proposal) anyerror!void {
        const x = self(p);
        const a = x.begun orelse return error.NoPassBegun;
        if (out.len != 1) return error.Slots;
        if (x.begun_dev) {
            try x.devResults();
        } else if (x.devOn()) {
            try x.devCollect(true);
        } else {
            const r = x.f.runner;
            r.glue = .{ .ctx = x, .run = glueFn };
            try r.window(x.pass_calls[x.head..]);
        }
        try x.finish(a, out);
        x.begun_dev = false;
        x.begun = null;
    }

    /// The host's chain over the pass's gathered candidates (the last `runPass` or `collect`).
    fn finish(x: *GpuPass, a: iface.Ask, out: []iface.Proposal) !void {
        // the chain: greedy at T = 0, keyed sampling at the ask's parameters
        const c: usize = x.f.comm.world() * x.k;
        var scratch = std.heap.ArenaAllocator.init(x.gpa);
        defer scratch.deinit();
        const tc = ph.start(x.f.phases, .@"draft.chain");
        defer tc.stop();
        try dspark.chain(scratch.allocator(), x.markov, x.cand, x.cval, c, a.anchor, a.params, if (x.conf != null) x.hid else null, out[0].drafts, out[0].conf);
        // trees (lanes' siblings): the gathered candidates and base logits [block][world x k]
        if (out[0].cand) |cd| {
            if (cd.len != x.cand.len) return error.CandidateWidth;
            @memcpy(cd, x.cand);
        }
        if (out[0].base) |b| {
            if (b.len != x.cval.len) return error.CandidateWidth;
            @memcpy(b, x.cval);
        }
    }

    fn followPropose(p: *anyopaque, anchor: u32, start: u64) anyerror!void {
        return self(p).runPass(anchor, start);
    }

    /// The pass on this rank: statics staged, the launches, the host's candidates (every rank: the gather pairs up).
    fn runPass(x: *GpuPass, anchor: u32, start: u64) !void {
        if (x.devOn()) {
            try x.devLaunch(anchor, start);
            return x.devCollect(x.f.comm.rank() == 0 or x.f.link == null);
        }
        try x.stagePass(anchor, start);
        try x.f.runner.window(x.pass_calls);
    }

    pub const devOn = gpu_device.devOn;
    pub const devLaunch = gpu_device.devLaunch;
    pub const passBody = gpu_device.passBody;
    pub const devCollect = gpu_device.devCollect;
    pub const devResults = gpu_device.devResults;

    /// The pass's statics on the device, and its glue in place.
    fn stagePass(x: *GpuPass, anchor: u32, start: u64) !void {
        const r = x.f.runner;
        try r.stream.synchronize();
        x.fillStatics(anchor, start, x.h64, x.h32);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i64") orelse return error.Unbound, .len = 8 * x.h64.len }, 0, std.mem.sliceAsBytes(x.h64), r.stream.handle);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i32") orelse return error.Unbound, .len = 4 * x.h32.len }, 0, std.mem.sliceAsBytes(x.h32), r.stream.handle);
        r.glue = .{ .ctx = x, .run = glueFn };
    }

    /// The statics (one slot) into `h64` / `h32`.
    pub fn fillStatics(x: *GpuPass, anchor: u32, start: u64, h64: []i64, h32: []i32) void {
        return rows_mod.stageOne(x.st, @intCast(emit.ringRows(x.f.cfg)), x.shape.noise, x.valid, anchor, start, h64, h32);
    }

    pub const device = gpu_device.device;
    pub const numel = gpu_device.numel;
    pub const glueFn = gpu_device.glueFn;
    pub const hostStep = gpu_device.hostStep;
};

/// The gathered candidates are token ids (or -1); anything else is a racing gather or a mis-paired collective.
pub fn checkCandidates(cand: []const i32, vocab: usize) error{BadCandidates}!void {
    for (cand, 0..) |id, i| if (id < -1 or id >= @as(i64, @intCast(vocab))) {
        std.log.err("dspark: gathered candidate {d} at {d} is not a token id", .{ id, i });
        return error.BadCandidates;
    };
}

/// The drafter's int32 position word set on `stream` as a 32-bit memset: no host buffer, legal inside a capture.
pub fn setPos(d: *const cuda.Driver, ptr: u64, pos: i32, stream: cuda.Stream) !void {
    try d.check(d.api.cuMemsetD32Async(ptr, @bitCast(pos), 1, stream.handle), "cuMemsetD32Async");
}
