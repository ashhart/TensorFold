//! DSpark drafting over several live slots on the GPU (TF_DSV41_SLOT_DRAFTS=1 with TF_DSV41_SLOTS > 1).

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const run = @import("run.zig");
const buffers = @import("buffers.zig");
const fwd = @import("forward.zig");
const emit = @import("dspark_emit.zig");
const rows = @import("dspark_rows.zig");
const gpu = @import("dspark_gpu.zig");
const batch = @import("batch.zig");
const iface = @import("draft/iface.zig");
const dspark = @import("draft/dspark.zig");
const dsd = @import("dspark_dev.zig");
const ph = @import("phases.zig");
const rowtab = @import("rowtab.zig");
const slots_device = @import("dspark_slots_device.zig");
const slots_ingest = @import("dspark_slots_ingest.zig");

pub const Error = error{ Slots, NotAPrefix, CandidateWidth, Unbound };

pub const SlotPass = struct {
    gpa: std.mem.Allocator,
    f: *fwd.Forward,
    one: *gpu.GpuPass,
    /// live slots (the rings' count)
    nslots: u32,
    /// slots a pass at most
    group: u32,
    valid: []u64,
    arena: std.heap.ArenaAllocator,
    /// a pass's launches by its slot count - 1
    programs: [][]const calls.Call,
    h64: []i64,
    h32: []i32,
    /// the last pass's host results, [rows][W k] / [rows][D]; the host step's scratch
    cand: []i32,
    cval: []f32,
    hid: []f32,
    logits: []f32,
    hid16: []u16,
    packed_: []f32,
    all: []f32,
    ids: []i32,
    send: ?cuda.DeviceBuffer = null,
    gathered: ?cuda.DeviceBuffer = null,
    prev_ext: ?fwd.Forward.Ext = null,
    /// batch.zig's taps stash (base, bytes; 0, 0: none)
    stash: [2]u64,
    stats: struct { passes: u64 = 0, slots: u64 = 0 } = .{},
    /// every proposal's (slot, drafts) folded in lane order: two runs drafted alike iff equal
    digest: u64 = 0,
    /// TF_DSV41_DRAFT_GRAPHS (dspark_dev.zig): candidates on the device, each pass size graphed; null: the host step
    dev: ?*dsd.Dev = null,
    /// the group being issued: its first row in the round (the device step's region)
    cur_at: usize = 0,
    /// the group being issued: its slot count (its program)
    cur_k: usize = 1,
    last_logits: u64 = 0,
    last_v: usize = 0,
    /// the speculative pass's asks (sorted by slot) and its groups, between `begin` and `collect`
    begun: ?Begun = null,
    begun_host: bool = false,
    order: [16]usize = undefined,
    begun_asks: [16]iface.Ask = undefined,

    /// the speculative round from the row window's device pick: the row windows' state, armed windows, scratch
    bat: ?*const batch.Batch = null,
    armed: [16]?Arm = @splat(null),
    dl: ?Dl = null,
    begun_dev: bool = false,
    dscr: ?cuda.DeviceBuffer = null,
    hscr: ?cuda.HostBuffer = null,
    staged: ?cuda.Event = null,
    prev_pick: ?fwd.Forward.AfterPick = null,
    prev_sample: ?fwd.Forward.AfterSample = null,
    dstats: struct { device_starts: u64 = 0, device_dropped: u64 = 0 } = .{},
    /// TF_DSV41_DRAFT_OVERLAP=1: the ingests' stream and its done event (the runner's taps mark is the start)
    side: ?Side = null,
    /// TF_DSV41_DRAFT_BATCH_PROJ: one main_proj over a device-started round's slots (`projRound`)
    bproj: BatchProj = .off,
    bstats: struct { rounds: u64 = 0, slots: u64 = 0, checked: u64 = 0, check_rows: u64 = 0, differ: u64 = 0 } = .{},
    /// TF_DSV41_DRAFT_CAPTURE=1: pass keys (size << 16 | row) regular rounds issued since graphs mode; others run once.
    capture_first: bool = false,
    seen: [32]u32 = undefined,
    nseen: usize = 0,
    declined: u64 = 0,

    const Side = struct { stream: cuda.Stream, done: cuda.Event, mark: cuda.Event, rounds: u64 = 0, slots: u64 = 0, serial: u64 = 0 };

    pub const Begun = struct { n: usize, groups: [16][2]usize, ng: usize };

    const Arm = struct { start: u64, n: u32, sampled: bool, chosen: bool = false };
    const Dl = struct { k: usize, slots: [16]u32 };
    /// dscr / hscr layout (i64): ids [64], picks [64], segs [16][3], acc [16][3], mem [16][16][3]
    pub const scr_ids = 0;
    pub const scr_picks = 64;
    pub const scr_segs = 128;
    pub const scr_acc = scr_segs + 48;
    pub const scr_mem = scr_acc + 48;
    pub const scr_len = scr_mem + 16 * 48;

    /// The roles several-slot passes add to the forward's plan: stacked rings, each pass size's statics and rows.
    pub fn plan(a: std.mem.Allocator, f: *const fwd.Forward, p: *buffers.Plan, nslots: u32, group: u32) !void {
        const n: i64 = f.cfg.dspark_block;
        for (1..group + 1) |k| try p.add(try emit.emitPassSlots(a, f.cfg, f.widths, f.opts, n, @intCast(k), nslots));
        // the overlapped ingests' own scratch, at a row window's most rows a slot
        const ovl = try overlapFromEnv();
        if (ovl) try p.add(try emit.sideRoles(a, try emit.emitIngestAt(a, f.cfg, f.widths, f.opts, rowtab.seg_max, 0, .{ .ring_slots = nslots })));
        // the batched projection at a row window's most rows; the check's one-slot projections on their own roles
        const bp = try batchProjFromEnv();
        if (bp != .off) {
            const pj = try emit.emitIngestProj(a, f.cfg, f.widths, f.opts, proj_rows_max, 0, "w.taps");
            try p.add(pj);
            if (ovl) try p.add(try emit.sideRoles(a, pj));
        }
        if (bp == .check) try p.add(try emit.renameRoles(a, try emit.emitIngestProj(a, f.cfg, f.widths, f.opts, rowtab.seg_max, 0, "w.taps"), "w.chk."));
    }

    /// On every rank after the one-slot pass and the row windows (`prev`: the forward's extension so far, slots').
    pub fn init(gpa: std.mem.Allocator, f: *fwd.Forward, one: *gpu.GpuPass, b: *const batch.Batch, nslots: u32, group: u32) !*SlotPass {
        const x = try gpa.create(SlotPass);
        var assembled = false;
        var registered = false;
        errdefer if (!registered) {
            if (assembled) x.deinit() else gpa.destroy(x);
        };
        const n: usize = f.cfg.dspark_block;
        const R = n * group;
        // the device round unpacks every group at its slots' offset, so the buffers cover every live slot's rows
        const Rr = n * @max(group, nslots);
        const c: usize = f.comm.world() * one.k;
        const lay = emit.Layout.of(group, @intCast(n), f.cfg.window);
        const V: usize = f.cfg.vocab / f.comm.world();
        const valid = try gpa.alloc(u64, nslots);
        errdefer if (!assembled) gpa.free(valid);
        const h64 = try gpa.alloc(i64, @intCast(lay.len64));
        errdefer if (!assembled) gpa.free(h64);
        const h32 = try gpa.alloc(i32, @intCast(lay.len32));
        errdefer if (!assembled) gpa.free(h32);
        const cand = try gpa.alloc(i32, Rr * c);
        errdefer if (!assembled) gpa.free(cand);
        const cval = try gpa.alloc(f32, Rr * c);
        errdefer if (!assembled) gpa.free(cval);
        const hid = try gpa.alloc(f32, Rr * f.cfg.hidden);
        errdefer if (!assembled) gpa.free(hid);
        const logits = try gpa.alloc(f32, R * V);
        errdefer if (!assembled) gpa.free(logits);
        const hid16 = try gpa.alloc(u16, R * f.cfg.hidden);
        errdefer if (!assembled) gpa.free(hid16);
        const packed_ = try gpa.alloc(f32, R * 2 * one.k);
        errdefer if (!assembled) gpa.free(packed_);
        const all = try gpa.alloc(f32, R * 2 * c);
        errdefer if (!assembled) gpa.free(all);
        const ids = try gpa.alloc(i32, one.k);
        errdefer if (!assembled) gpa.free(ids);
        x.* = .{
            .gpa = gpa,
            .f = f,
            .one = one,
            .nslots = nslots,
            .group = group,
            .valid = valid,
            .arena = std.heap.ArenaAllocator.init(gpa),
            .programs = undefined,
            .h64 = h64,
            .h32 = h32,
            .cand = cand,
            .cval = cval,
            .hid = hid,
            .logits = logits,
            .hid16 = hid16,
            .packed_ = packed_,
            .all = all,
            .ids = ids,
            .prev_ext = f.ext,
            .stash = if (b.taps_stash) |t| .{ t.ptr, t.len } else .{ 0, 0 },
        };
        assembled = true;
        const Drop = struct {
            fn run(ptr: *anyopaque) void {
                const owner: *SlotPass = @ptrCast(@alignCast(ptr));
                owner.deinit();
            }
        };
        try @import("construction.zig").attach(f, x, Drop.run);
        registered = true;
        errdefer @import("construction.zig").failed(f, x);
        @memset(x.valid, 0);
        const a = x.arena.allocator();
        x.programs = try a.alloc([]const calls.Call, group);
        for (x.programs, 1..) |*pg, k| pg.* = try emit.emitPassSlots(a, f.cfg, f.widths, f.opts, @intCast(n), @intCast(k), nslots);
        const mode = try dsd.modeFromEnv();
        if (mode != .host) {
            const r = f.runner;
            if (r.kernels.others(r.stream).f.ds_cands == null) {
                if (!quiet) std.log.scoped(.dsv41).warn("draft graphs: TF_DSV41_DRAFT_GRAPHS set, but the glue fatbin has no ds_cands kernel: the host step", .{});
            } else {
                _ = try x.device(&x.send, 2 * R * f.cfg.hidden); // the embedding's and exchanges' send, at its largest
                x.dev = try dsd.Dev.init(gpa, r.d, f.comm, r.stream, mode, one.k, f.cfg.hidden, n * nslots, one.conf != null, @intCast(8 * lay.len64 + 4 * lay.len32), nslots);
                // the speculative round from the row window's device pick: this pass takes the forward's hook over
                const ko = r.kernels.others(r.stream).f;
                if (ko.ds_accept_rows != null and ko.ds_stage_rows != null and nslots <= 16) {
                    x.bat = b;
                    x.dscr = try cuda.DeviceBuffer.alloc(r.d, 8 * scr_len);
                    x.hscr = try cuda.HostBuffer.alloc(r.d, 8 * scr_len);
                    x.staged = try cuda.Event.init(r.d, false);
                    x.prev_pick = f.after_pick;
                    x.prev_sample = f.after_sample;
                }
                if (try dsd.chainFromEnv()) {
                    // refused, not a silent host chain: the variant comes from the AOT fill (aot-needs lists it)
                    const ch = try dsd.Chain.init(gpa, r, n, nslots, c, f.cfg.hidden, one.conf != null);
                    errdefer ch.deinit();
                    if (try ch.probe(r, f.cfg, f.widths, f.opts)) |S| {
                        std.log.err("draft chain: TF_DSV41_DRAFT_CHAIN=1, but the AOT set has no _chain variant for {d} slot(s) (N {d}, KP {d}, candidates {d}, confidence head {}): merge a fill that holds it (tf-dsv41 aot-needs lists it, recipe/triton_fill.py compile), or set TF_DSV41_DRAFT_CHAIN=0", .{ S, n, emit.chain_kp, c, one.conf != null });
                        return error.MissingChainVariant;
                    }
                    x.dev.?.chain = ch;
                }
                if (try overlapFromEnv()) {
                    if (x.bat == null) {
                        if (!quiet) std.log.scoped(.dsv41).warn("draft overlap: TF_DSV41_DRAFT_OVERLAP=1 needs the device-started round (TF_DSV41_GREEDY_GPU and the glue's ds_accept_rows): the serial ingests", .{});
                    } else {
                        var st = try cuda.Stream.initPriority(r.d, .highest);
                        errdefer st.deinit();
                        var done = try cuda.Event.init(r.d, false);
                        errdefer done.deinit();
                        x.side = .{ .stream = st, .done = done, .mark = try cuda.Event.init(r.d, false) };
                    }
                }
                if (try captureFromEnv()) {
                    if (x.bat == null) {
                        if (!quiet) std.log.scoped(.dsv41).warn("draft capture: TF_DSV41_DRAFT_CAPTURE=1 needs the device-started round: off", .{});
                    } else {
                        x.capture_first = true;
                        std.log.scoped(.dsv41).info("draft capture: a device-started round over a pass size not yet graphed runs the regular path once (its graph captured there)", .{});
                    }
                }
                const bp = try batchProjFromEnv();
                if (bp != .off) {
                    if (x.bat == null) {
                        if (!quiet) std.log.scoped(.dsv41).warn("draft batched main_proj: TF_DSV41_DRAFT_BATCH_PROJ needs the device-started round: one main_proj a slot", .{});
                    } else x.bproj = bp;
                }
            }
        } else if (try overlapFromEnv() or try dsd.chainFromEnv() or try batchProjFromEnv() != .off) {
            if (!quiet) std.log.scoped(.dsv41).warn("draft overlap / chain / batched main_proj: they need TF_DSV41_DRAFT_GRAPHS' device path: off", .{});
        }
        std.log.scoped(.dsv41).info("slot drafts: {d} slots, DSpark passes of up to {d} slots ({d} rows); chain on the {s}, ingests {s}, main_proj {t}", .{ nslots, group, R, if (x.dev != null and x.dev.?.chain != null) "device" else "host", if (x.side != null) "overlapped" else "serial", x.bproj });
        f.ext = .{ .ptr = x, .run = followFn };
        if (x.bat != null) {
            f.after_pick = .{ .ctx = x, .run = afterPick };
            if (f.runner.kernels.others(f.runner.stream).f.vs_choose != null) f.after_sample = .{ .ctx = x, .run = afterSample };
        }
        if (x.side) |side| f.runner.taps_mark = .{ .ev = side.mark };
        @import("construction.zig").detach(f, x);
        return x;
    }

    pub fn deinit(x: *SlotPass) void {
        const gpa = x.gpa;
        if (x.f.ext) |e| if (e.ptr == @as(*anyopaque, x)) {
            x.f.ext = x.prev_ext;
        };
        if (x.bproj != .off or x.bstats.rounds > 0) std.log.scoped(.dsv41).info("draft batched main_proj: {d} rounds ({d} slots) one main_proj, {d} check rounds, {d} of {d} check rows differ", .{ x.bstats.rounds, x.bstats.slots, x.bstats.checked, x.bstats.differ, x.bstats.check_rows });
        if (x.side) |*sd| {
            std.log.scoped(.dsv41).info("draft overlap: {d} rounds' ingests ({d} slots) on their own stream, {d} serial", .{ sd.rounds, sd.slots, sd.serial });
            if (x.f.runner.taps_mark) |mark| if (mark.ev.handle == sd.mark.handle) {
                x.f.runner.taps_mark = null;
            };
            sd.stream.deinit();
            sd.done.deinit();
            sd.mark.deinit();
        }
        if (x.f.after_pick) |pick| if (pick.ctx == @as(*anyopaque, x)) {
            x.f.after_pick = x.prev_pick;
        };
        if (x.f.after_sample) |sample| if (sample.ctx == @as(*anyopaque, x)) {
            x.f.after_sample = x.prev_sample;
        };
        if (x.capture_first) std.log.scoped(.dsv41).info("draft capture: {d} device-started rounds ran the regular path to capture {d} pass key(s)", .{ x.declined, x.nseen });
        if (x.dscr) |*b| b.free();
        if (x.hscr) |*b| b.free();
        if (x.staged) |*e| e.deinit();
        if (x.dev) |d| {
            if (x.dstats.device_starts + x.dstats.device_dropped > 0) std.log.scoped(.dsv41).info("slot drafts: {d} speculative rounds started from the device pick, {d} unused", .{ x.dstats.device_starts, x.dstats.device_dropped });
            d.deinit();
        }
        if (x.send) |*b| b.free();
        if (x.gathered) |*b| b.free();
        inline for (.{ "valid", "h64", "h32", "cand", "cval", "hid", "logits", "hid16", "packed_", "all", "ids" }) |name| gpa.free(@field(x, name));
        x.arena.deinit();
        gpa.destroy(x);
    }

    pub fn pass(x: *SlotPass) iface.Pass {
        return .{ .ptr = x, .vtable = if (x.dev != null) &vtable_begin else &vtable };
    }

    const vtable: iface.Pass.VTable = .{ .ingest = ingest, .propose = proposeFold, .reset = reset, .markov = heads };
    /// with the device path: the speculative pass too (draft/spec.zig), the whole round launched by `begin`
    const vtable_begin: iface.Pass.VTable = .{ .ingest = ingest, .propose = proposeFold, .reset = reset, .markov = heads, .begin = begin, .collect = collectFold, .arm = arm };

    fn proposeFold(p: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        try propose(p, asks, out);
        self(p).fold(asks, out);
    }

    fn collectFold(p: *anyopaque, out: []iface.Proposal) anyerror!void {
        const x = self(p);
        const n = if (x.begun) |b| b.n else 0;
        try collect(p, out);
        x.fold(x.begun_asks[0..n], out);
    }

    fn fold(x: *SlotPass, asks: []const iface.Ask, out: []const iface.Proposal) void {
        var h = std.hash.Wyhash.init(x.digest);
        for (asks, out) |a, o| {
            h.update(std.mem.asBytes(&a.slot));
            h.update(std.mem.sliceAsBytes(o.drafts));
        }
        x.digest = h.final();
    }

    pub const devReady = slots_device.devReady;
    pub const arm = slots_device.arm;
    pub const dropDevice = slots_device.dropDevice;
    pub const afterPick = slots_device.afterPick;
    pub const afterSample = slots_device.afterSample;
    pub const tryLaunch = slots_device.tryLaunch;

    pub const ingestRound = slots_ingest.ingestRound;
    pub const projRound = slots_ingest.projRound;
    pub const projCheck = slots_ingest.projCheck;
    pub const ingestAsync = slots_ingest.ingestAsync;

    pub fn self(p: *anyopaque) *SlotPass {
        return @ptrCast(@alignCast(p));
    }

    fn leads(x: *const SlotPass) bool {
        return x.f.link != null and !x.f.quiet and x.f.comm.rank() == 0;
    }

    pub fn sendOp(x: *SlotPass, msg: []const i64) !void {
        if (x.leads()) try x.f.link.?.send(msg);
    }

    fn heads(p: *anyopaque) ?dspark.Markov {
        return self(p).one.markov;
    }

    fn reset(p: *anyopaque, slot: u32) void {
        const x = self(p);
        x.dropDevice();
        x.sendOp(&.{ rows.op_reset, slot }) catch |e| std.log.err("slot drafts: the draft reset did not reach the followers ({t})", .{e});
        if (slot < x.nslots) x.valid[slot] = 0;
    }

    pub const ingest = slots_ingest.ingest;
    pub const doIngest = slots_ingest.doIngest;

    // propose ------------------------------------------------------------------------------------------------------

    pub fn bySlot(asks: []const iface.Ask, a: usize, b: usize) bool {
        return asks[a].slot < asks[b].slot;
    }

    /// Every ask's drafts: the asks sorted by slot, one pass a group (dspark_rows.groups).
    pub fn propose(p: *anyopaque, asks: []const iface.Ask, out: []iface.Proposal) anyerror!void {
        const x = self(p);
        if (asks.len == 0) return;
        if (asks.len > x.nslots) return error.Slots;
        if (x.begun != null) return error.PassBegun;
        x.dropDevice();
        if (x.devOn()) {
            try x.devLaunchRound(asks);
            return x.devFinishRound(asks, out);
        }
        var order: [16]usize = undefined;
        for (order[0..asks.len], 0..) |*o, i| o.* = i;
        std.sort.insertion(usize, order[0..asks.len], asks, bySlot);
        var gs: [16][2]usize = undefined;
        const ng = rows.groups(asks.len, x.group, &gs);
        const n: usize = x.f.cfg.dspark_block;
        const c: usize = x.f.comm.world() * x.one.k;
        const D: usize = x.f.cfg.hidden;
        var scratch = std.heap.ArenaAllocator.init(x.gpa);
        defer scratch.deinit();
        for (gs[0..ng]) |g| {
            var ms: [16]rows.Member = undefined;
            const idx = order[g[0]..g[1]];
            for (idx, ms[0..idx.len]) |i, *m| {
                const a = asks[i];
                if (a.slot >= x.nslots) return error.Slots;
                m.* = .{ .slot = a.slot, .anchor = a.anchor, .start = a.start, .valid = x.valid[a.slot] };
            }
            var buf: [2 + 3 * 16]i64 = undefined;
            try x.sendOp(try rows.proposeMsg(ms[0..idx.len], &buf));
            try x.runGroup(ms[0..idx.len]);
            // each slot's chain over its own rows of the pass
            for (idx, 0..) |i, j| {
                const a = asks[i];
                const o = out[i];
                const cd = x.cand[j * n * c ..][0 .. n * c];
                const cv = x.cval[j * n * c ..][0 .. n * c];
                try dspark.chain(scratch.allocator(), x.one.markov, cd, cv, c, a.anchor, a.params, if (x.one.conf != null) x.hid[j * n * D ..][0 .. n * D] else null, o.drafts, o.conf);
                if (o.cand) |t| {
                    if (t.len != cd.len) return error.CandidateWidth;
                    @memcpy(t, cd);
                }
                if (o.base) |t| {
                    if (t.len != cv.len) return error.CandidateWidth;
                    @memcpy(t, cv);
                }
            }
        }
    }

    pub const devOn = slots_device.devOn;
    pub const begin = slots_device.begin;
    pub const takeDevice = slots_device.takeDevice;
    pub const collect = slots_device.collect;
    pub const devLaunchRound = slots_device.devLaunchRound;
    pub const launchChain = slots_device.launchChain;
    pub const devFinishRound = slots_device.devFinishRound;
    pub const chainResults = slots_device.chainResults;
    pub const devGroup = slots_device.devGroup;
    pub const passKey = slots_device.passKey;
    pub const wasSeen = slots_device.wasSeen;
    pub const markSeen = slots_device.markSeen;
    pub const groupBody = slots_device.groupBody;

    /// One pass over `ms` on this rank: the statics staged, the launches, the host's candidates (every rank).
    fn runGroup(x: *SlotPass, ms: []const rows.Member) !void {
        const f = x.f;
        const n: u32 = f.cfg.dspark_block;
        const lay = emit.Layout.of(@intCast(ms.len), n, f.cfg.window);
        const r = f.runner;
        try r.stream.synchronize();
        rows.stage(lay, n, f.cfg.window, @intCast(emit.ringRows(f.cfg)), x.one.shape.noise, ms, x.h64, x.h32);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i64") orelse return error.Unbound, .len = 8 * @as(usize, @intCast(lay.len64)) }, 0, std.mem.sliceAsBytes(x.h64[0..@intCast(lay.len64)]), r.stream.handle);
        try cuda.DeviceBuffer.uploadAsync(.{ .d = r.d, .ptr = r.addressOf("w.ds.i32") orelse return error.Unbound, .len = 4 * @as(usize, @intCast(lay.end32)) }, 0, std.mem.sliceAsBytes(x.h32[0..@intCast(lay.end32)]), r.stream.handle);
        x.stats.passes += 1;
        x.stats.slots += ms.len;
        r.glue = .{ .ctx = x, .run = glueFn };
        try r.window(x.programs[ms.len - 1]);
    }

    // followers ----------------------------------------------------------------------------------------------------

    fn followFn(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const x: *SlotPass = @ptrCast(@alignCast(ptr));
        switch (msg[0]) {
            rows.op_reset => {
                if (msg.len != 2) return error.BadPlan;
                const s: usize = @intCast(msg[1]);
                if (s < x.nslots) x.valid[s] = 0;
            },
            rows.op_ingest => {
                if (msg.len != 6 or msg[1] < 0 or msg[1] >= x.nslots) return error.BadPlan;
                try x.doIngest(@intCast(msg[1]), @intCast(msg[2]), .{ .src = msg[3], .skip = msg[4] }, @intCast(msg[5]));
            },
            gpu.GpuPass.op_arm => {
                // [op, slot, start, rows, sampled]: this pass's arm; [op, start, rows, sampled]: the one-slot pass's
                if (msg.len == 5) {
                    const sl: usize = @intCast(msg[1]);
                    if (sl >= x.nslots) return error.BadPlan;
                    x.armed[sl] = .{ .start = @intCast(msg[2]), .n = @intCast(msg[3]), .sampled = msg[4] != 0 };
                    return;
                }
                const e = x.prev_ext orelse return error.BadPlan;
                try e.run(e.ptr, msg);
            },
            rows.op_propose => {
                var ms: [16]rows.Member = undefined;
                if (x.devOn()) {
                    // the leader's group, run alike at its row of the round: same graph keys and order on every rank
                    if (msg.len < 3) return error.BadPlan;
                    const at: usize = @intCast(msg[msg.len - 1]);
                    const got = try rows.proposeOf(msg[0 .. msg.len - 1], x.valid, ms[0..x.group]);
                    const d = x.dev.?;
                    try x.devGroup(got, at);
                    if (d.stats.passes < dsd.check_passes) {
                        try d.mark();
                        try d.wait();
                        try d.check(x.last_logits, x.last_v, got.len * x.f.cfg.dspark_block, at);
                    }
                    try d.settle();
                    return;
                }
                try x.runGroup(try rows.proposeOf(msg, x.valid, ms[0..x.group]));
            },
            else => {
                const e = x.prev_ext orelse return error.BadPlan;
                try e.run(e.ptr, msg);
            },
        }
    }

    pub const device = slots_device.device;
    pub const numel = slots_device.numel;
    pub const glueFn = slots_device.glueFn;
    pub const hostStep = slots_device.hostStep;
};

/// TF_DSV41_DRAFT_OVERLAP: 1 = a device-started round's ingests overlapped with its window's tail, unset / 0 = serial.
pub fn overlapFromEnv() !bool {
    const v = std.mem.span(std.c.getenv("TF_DSV41_DRAFT_OVERLAP") orelse return false);
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadDraftOverlap;
}

/// TF_DSV41_DRAFT_CAPTURE: 1 = a new pass key runs the regular path once (captured); unset / 0 = never captures.
pub fn captureFromEnv() !bool {
    const v = std.mem.span(std.c.getenv("TF_DSV41_DRAFT_CAPTURE") orelse return false);
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadDraftCapture;
}

/// TF_DSV41_DRAFT_BATCH_PROJ: 1 = one main_proj over a round's slots, `check` = also compared per slot, 0 = per slot.
pub const BatchProj = enum { off, on, check };

pub fn batchProjFromEnv() !BatchProj {
    const v = std.mem.span(std.c.getenv("TF_DSV41_DRAFT_BATCH_PROJ") orelse return .off);
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return .off;
    if (std.mem.eql(u8, v, "1")) return .on;
    if (std.mem.eql(u8, v, "check")) return .check;
    return error.BadDraftBatchProj;
}

/// The batched projection's most rows: a row window's (rowtab's largest bucket; linear.cu takes up to 128).
pub const proj_rows_max = rowtab.buckets[rowtab.buckets.len - 1];
/// TF_DSV41_DRAFT_BATCH_PROJ=check: rounds whose rows are compared
pub const proj_check_rounds = 8;
