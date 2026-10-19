//! Sessions on the GPU path: DeepSeek-V4.1's side of the family-neutral store over the paged pool (snapshot, resume).

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const pf = @import("forward_prefill.zig");
const cuda = @import("cuda");
const kv = @import("kv");
const fwd = @import("forward.zig");
const eh = @import("engram_host.zig");
const kvs = @import("cuda_kv.zig");
const buffers = @import("buffers.zig");
const ced = @import("ced.zig");
const rowmode = @import("rowmode.zig");
const slots_mod = @import("slots.zig");
const graphs = @import("graphs.zig");
const lockstep = @import("lockstep.zig");
const dspark_emit = @import("dspark_emit.zig");

const sessions = kv.sessions;
const Store = sessions.Store;
const Blob = sessions.disk.Blob;

pub const Error = error{ NoPool, BadBlob, SlotNotActive };

pub const Options = struct {
    /// TF_DSV41_SESSIONS (default 1)
    enabled: bool = true,
    disk: ?[]const u8 = null,
    disk_budget: u64 = 64 << 30,
    disk_min_tokens: u64 = 1024,
    /// TF_DSV41_SESSION_RESIDENT / _RESIDENT_GIB: idle chats and bytes kept in RAM (null: unlimited; 0: none idle)
    resident: ?u32 = null,
    resident_bytes: ?u64 = null,
    /// TF_DSV41_SESSION_RAM_MIB (default 256)
    ram_bytes: u64 = 256 << 20,
    staging_mib: u32 = 128,
    skip_covered: bool = false,
    delta: bool = false,
    /// TF_DSV41_SESSION_SLOTS=1: the store with several live slots too (default off)
    several: bool = false,
};

fn flag(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "1");
}

/// The store's policy from the family's options: an unset residency stays unlimited, as sessions.Policy reads null.
fn policy(o: Options) sessions.store.Policy {
    return .{ .ram_bytes = o.ram_bytes, .chats = o.resident, .resident_bytes = o.resident_bytes, .skip_covered = o.skip_covered, .delta = o.delta };
}

test "the default session options keep idle chats unlimited; an explicit 0 keeps none" {
    const unset = policy(.{});
    try std.testing.expectEqual(@as(?u32, null), unset.chats);
    try std.testing.expectEqual(@as(?u64, null), unset.resident_bytes);
    const zero = policy(.{ .resident = 0, .resident_bytes = 0 });
    try std.testing.expectEqual(@as(?u32, 0), zero.chats);
    try std.testing.expectEqual(@as(?u64, 0), zero.resident_bytes);
}

pub fn optionsFromEnv() !Options {
    var o: Options = .{};
    if (std.c.getenv("TF_DSV41_SESSIONS")) |v| o.enabled = !std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "0");
    if (std.c.getenv("TF_DSV41_SESSION_DISK")) |v| if (std.mem.trim(u8, std.mem.span(v), " ").len > 0) {
        o.disk = std.mem.trim(u8, std.mem.span(v), " ");
    };
    if (std.c.getenv("TF_DSV41_SESSION_DISK_GIB")) |v| if (std.mem.span(v).len > 0) {
        o.disk_budget = @intFromFloat(try std.fmt.parseFloat(f64, std.mem.span(v)) * (1 << 30));
    };
    if (std.c.getenv("TF_DSV41_SESSION_DISK_MIN")) |v| if (std.mem.span(v).len > 0) {
        o.disk_min_tokens = try std.fmt.parseInt(u64, std.mem.span(v), 10);
    };
    if (std.c.getenv("TF_DSV41_SESSION_RESIDENT")) |v| o.resident = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (std.c.getenv("TF_DSV41_SESSION_RESIDENT_GIB")) |v| o.resident_bytes = @intFromFloat(try std.fmt.parseFloat(f64, std.mem.span(v)) * (1 << 30));
    if (std.c.getenv("TF_DSV41_SESSION_RAM_MIB")) |v| if (std.mem.span(v).len > 0) {
        o.ram_bytes = (try std.fmt.parseInt(u64, std.mem.span(v), 10)) << 20;
    };
    if (std.c.getenv("TF_DSV41_SESSION_STAGING_MIB")) |v| o.staging_mib = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    o.skip_covered = flag("TF_DSV41_SESSION_SKIP_COVERED");
    o.delta = flag("TF_DSV41_SESSION_DELTA");
    o.several = flag("TF_DSV41_SESSION_SLOTS");
    return o;
}

/// What the store is opened over: the live slots and the drafter's blocks (0: drafts off; for the RAM measure only).
pub const Shape = struct { slots: u32 = 1, ds_blocks: u32 = 0, checkpoint: ?[32]u8 = null };

/// One device role of the bounded state at its address when run; `stride`: unviewed stacked role, base + slot x stride.
pub const Piece = struct { role: []const u8, len: usize, stride: u64 = 0 };

pub const Sessions = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    f: *fwd.Forward,
    kx: *kvs.Kv,
    store: Store,
    disk: ?*sessions.Disk = null,
    pieces: []Piece,
    bounded_bytes: u64,
    /// each slot's committed ids (prefill and kept windows append): a saved entry's ids
    hist: []std.ArrayList(i32),
    /// each slot's drafter context start (0 after a reset, pos - n after restoring n rows), for the measure
    ds_valid: []u64,
    measure: kv.sess.Measure,
    /// several slots: the operations run on the activated slot (checked)
    slot_set: ?*slots_mod.SlotSet = null,
    names: std.heap.ArenaAllocator,
    stats: struct { admits: u64 = 0, spills: u64 = 0, waits: u64 = 0, damaged: u64 = 0 } = .{},
    /// several ranks: a restore's verdict agreed (one rank's damaged file sends every rank to a fresh prefill)
    agreement: ?graphs.Agreement = null,
    /// full mode: the drafter's rings ride in the snapshot; a restore sets the pass's context start through `ds_set`
    ds_rings: bool = false,
    ds_set: ?DsSet = null,

    pub const DsSet = struct { ctx: *anyopaque, set: *const fn (ctx: *anyopaque, slot: u32, valid: u64) void, get: ?*const fn (*anyopaque, u32) u64 = null };

    /// The store over the forward's pool (after Runner.bind: the bounded roles have addresses).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, f: *fwd.Forward, plan: *const buffers.Plan, o: Options, shape: Shape) !*Sessions {
        const kx = f.kv orelse return error.NoPool;
        if (kx.slots.len != shape.slots) return error.SlotRange;
        const s = try gpa.create(Sessions);
        errdefer gpa.destroy(s);
        s.* = .{ .gpa = gpa, .io = io, .f = f, .kx = kx, .store = undefined, .pieces = &.{}, .bounded_bytes = 0, .hist = &.{}, .ds_valid = &.{}, .measure = undefined, .names = std.heap.ArenaAllocator.init(gpa) };
        errdefer s.names.deinit();
        const na = s.names.allocator();
        s.hist = try na.alloc(std.ArrayList(i32), shape.slots);
        @memset(s.hist, .empty);
        s.ds_valid = try na.alloc(u64, shape.slots);
        @memset(s.ds_valid, 0);
        // the bounded roles in a fixed order (both ranks); in row mode a slot's bytes are the plan's / S
        var list: std.ArrayList(Piece) = .empty;
        var la = std.heap.ArenaAllocator.init(gpa);
        defer la.deinit();
        const layers = try f.backbone(la.allocator());
        for (layers) |L| {
            for ([_][]const u8{ "swa.v", "swa.s", "carry" }) |part| {
                const role = try std.fmt.allocPrint(na, "s.L{d}.{s}", .{ L, part });
                const size = plan.sizes.get(role) orelse continue;
                if (f.runner.addressOf(role) == null) return error.Unbound;
                const stacked = shape.slots > 1 and (rowmode.isRing(role) or rowmode.isCarry(role));
                if (stacked and size % shape.slots != 0) return error.Unbound;
                const len = if (stacked) size / shape.slots else size;
                try list.append(na, .{ .role = role, .len = len });
                s.bounded_bytes += len;
            }
        }
        // CED replay: the stash ring (planned in replay mode only; one ring: a prompt's prefill runs on one slot alone)
        for (ced.roles) |role| if (plan.sizes.get(role)) |len| {
            if (f.runner.addressOf(role) == null) return error.Unbound;
            try list.append(na, .{ .role = role, .len = len });
            s.bounded_bytes += len;
        };
        // full mode with drafts: the drafter's rings; replay mode: the decoder replay rewrites what the drafter reads
        const replay = pf.state(f).mode == .replay;
        if (!replay and shape.ds_blocks > 0) {
            const rr: u64 = @intCast(dspark_emit.ringRows(f.cfg));
            for (plan.sizes.keys(), plan.sizes.values()) |role, size| {
                if (!std.mem.startsWith(u8, role, "s.L")) continue;
                const v = std.mem.endsWith(u8, role, ".ring.v");
                if (!v and !std.mem.endsWith(u8, role, ".ring.s")) continue;
                const L = std.fmt.parseInt(u32, role["s.L".len .. std.mem.indexOfScalarPos(u8, role, 3, '.') orelse continue], 10) catch continue;
                if (L < f.cfg.layers) continue;
                const row: u64 = if (v) 576 else 8;
                if (size % (rr * row) != 0) return error.Unbound;
                const ring_slots = size / (rr * row);
                if (f.runner.addressOf(role) == null) return error.Unbound;
                const len = size / ring_slots;
                try list.append(na, .{ .role = try na.dupe(u8, role), .len = len, .stride = if (ring_slots > 1) len else 0 });
                s.bounded_bytes += len;
                s.ds_rings = true;
            }
        }
        s.pieces = list.items;
        // the snapshot as the RAM tier measures it (kv/sess.zig): replay keeps the rings below the decoder
        var rings: u64 = 0;
        var carries: u64 = 0;
        const d = if (replay) try ced.decoderStart(f.cfg) else f.cfg.layers;
        for (layers) |L| {
            if (L < d) rings += 1;
            if (f.cfg.isKvSource(L) and f.cfg.compressRatio(L) == 2) carries += 1;
        }
        s.measure = .{ .rings = rings, .window = f.cfg.window, .carries = carries, .head_dim = f.cfg.head_dim, .hidden = f.cfg.hidden, .hc = f.cfg.hc_mult, .ds_blocks = shape.ds_blocks, .replay = replay };
        if (o.disk) |root| {
            // the tier's compat ident: the families' layout, the split and this rank (a file is one rank's rows)
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            for (kx.layout.families()) |fam| h.update(std.mem.asBytes(&[_]u32{ fam.ratio, fam.row_bytes, @intFromBool(fam.split) }));
            h.update(std.mem.asBytes(&[_]u32{ kx.world, kx.rank, kx.pool.page }));
            const checkpoint = shape.checkpoint orelse return error.MissingCheckpointIdentity;
            const identity = @import("session_identity.zig");
            const semantics = identity.semantics(.{ .config = f.cfg.*, .options = f.opts, .mode = pf.state(f).mode, .ds_blocks = shape.ds_blocks });
            identity.bind(&h, checkpoint, layers, s.pieces, &semantics);
            var full: [32]u8 = undefined;
            h.final(&full);
            // the staging set (TF_DSV41_SESSION_STAGING_MIB) in 8 MiB chunks: a park's or restore's host transient
            s.disk = try sessions.Disk.open(gpa, io, .{ .root = root, .compat = full[0..16].*, .rank = kx.rank, .world = kx.world, .budget = o.disk_budget, .min_tokens = o.disk_min_tokens, .staging = @max(2, o.staging_mib / 8), .chunk = 8 << 20 });
        }
        errdefer if (s.disk) |dd| dd.close();
        s.store = try Store.init(gpa, io, &kx.pool, kx.dev.store(), s.disk, policy(o));
        errdefer s.store.deinit();
        if (f.comm.world() > 1) s.agreement = try graphs.Agreement.init(f.runner.d, f.comm, f.runner.stream);
        f.sess = .{ .ptr = s, .reserve = reserveFn, .commit = commitFn, .save = saveFn, .restore = restoreFn, .reset = resetFn, .prompt = promptFn, .admit = admitFn };
        return s;
    }

    fn reserveFn(ptr: *anyopaque, ids: []const u32) !void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        try s.reserveAt(s.cur(), ids);
    }
    fn commitFn(ptr: *anyopaque, ids: []const u32) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        try s.commitAt(s.cur(), ids);
    }

    fn saveFn(ptr: *anyopaque) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        _ = try s.save();
    }

    fn promptFn(ptr: *anyopaque) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        _ = try s.savePrompt();
    }

    fn resetFn(ptr: *anyopaque) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        try s.reset();
    }

    fn restoreFn(ptr: *anyopaque, id: u32, ids: []const i64) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        const p = try s.gpa.alloc(u32, ids.len);
        defer s.gpa.free(p);
        for (ids, p) |t, *y| y.* = @intCast(t);
        _ = try s.restore(id, if (ids.len > 0) p else null);
    }

    /// A follower: the leader's admission (forward.zig op_sess_admit: [op, need, n, spilled ids...]).
    fn admitFn(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const s: *Sessions = @ptrCast(@alignCast(ptr));
        if (msg.len < 3) return error.BadPlan;
        const n: usize = @intCast(msg[2]);
        if (msg.len != 3 + n) return error.BadPlan;
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(s.gpa);
        for (msg[3..]) |x| try ids.append(s.gpa, @intCast(x));
        try s.applyAdmit(@intCast(msg[1]), ids.items);
    }

    pub fn drain(s: *Sessions) !void {
        try s.f.runner.stream.synchronize();
        try s.kx.dev.io.synchronize();
        try s.store.settleAll();
    }
    pub fn close(s: *Sessions) !void {
        try s.drain();
        s.deinitAfterDrain();
    }
    pub fn deinitAfterDrain(s: *Sessions) void {
        s.f.sess = null;
        if (s.agreement) |*a| a.deinit();
        s.store.deinit();
        if (s.disk) |d| d.close();
        for (s.hist) |*h| h.deinit(s.gpa);
        s.names.deinit();
        s.gpa.destroy(s);
    }

    /// The slot the one-slot operations run on (the pool's current slot: slots.zig `activate`).
    pub fn cur(s: *const Sessions) u32 {
        return s.kx.slot.id;
    }

    /// Several slots: the slot's views must be the bound ones (its rings, carries and `Forward.Slot`).
    fn active(s: *const Sessions) !void {
        const ss = s.slot_set orelse return;
        if (ss.active != s.cur()) return error.SlotNotActive;
    }

    /// Slot `slot` committed `ids` (prefill segments and kept windows on the active slot; row windows by slot).
    pub fn commitAt(s: *Sessions, slot: u32, ids: []const u32) !void {
        try s.reserveAt(slot, ids);
        for (ids) |t| s.hist[slot].appendAssumeCapacity(@intCast(t));
    }

    pub fn reserveAt(s: *Sessions, slot: u32, ids: []const u32) !void {
        if (slot >= s.hist.len) return error.SlotRange;
        for (ids) |t| if (t > std.math.maxInt(i32)) return error.BadToken;
        try s.hist[slot].ensureUnusedCapacity(s.gpa, ids.len);
    }
    // -- the bounded state (store.Bounded) ----

    pub const Snap = @import("session_snapshot.zig").Snap;

    /// The "slot" blob: position, tail length, tail; then (CED) the stash's from / hi and ids; then the drafter's rows.
    pub const ced_at: usize = 16 + 4 * eh.max_ngram;
    pub const ds_at: usize = ced_at + 16 + 4 * ced.ring;
    pub const slot_bytes: usize = ds_at + 8;

    pub fn validStart(s: *const Sessions) u64 {
        if (s.ds_set) |ds| if (ds.get) |get| return get(ds.ctx, s.cur());
        return s.ds_valid[s.cur()];
    }
    /// The active slot's bounded state now (device to host, after the stream's work).
    pub fn snapshot(s: *Sessions) !*Snap {
        try s.active();
        const r = s.f.runner;
        try r.stream.synchronize();
        const p = try s.gpa.create(Snap);
        errdefer s.gpa.destroy(p);
        const sl = &s.f.slot;
        const ds = if (s.measure.ds_blocks > 0) kv.sess.dsRows(sl.pos, s.validStart(), s.measure.window) else 0;
        const stash = if (s.measure.replay) kv.sess.stashRows(sl.pos, sl.ced.from) else 0;
        p.* = .{ .owner = s, .bytes = try s.gpa.alloc(u8, s.bounded_bytes), .pos = sl.pos, .tail = sl.tail, .tail_len = sl.tail_len, .stash = sl.ced, .ds_rows = ds, .measure = s.measure.bytes(sl.pos, stash, ds) };
        errdefer s.gpa.free(p.bytes);
        var at: usize = 0;
        for (s.pieces) |pc| {
            const addr = (r.addressOf(pc.role) orelse return error.Unbound) + s.cur() * pc.stride;
            try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = addr, .len = pc.len }, 0, p.bytes[at..][0..pc.len]);
            at += pc.len;
        }
        return p;
    }

    pub fn uploadSnap(s: *Sessions, p: *const Snap) !void {
        const r = s.f.runner;
        var at: usize = 0;
        for (s.pieces) |pc| {
            const addr = (r.addressOf(pc.role) orelse return error.Unbound) + s.cur() * pc.stride;
            try s.f.upload(addr, p.bytes[at..][0..pc.len]);
            at += pc.len;
        }
        s.f.slot.pos = p.pos;
        s.f.slot.tail = p.tail;
        s.f.slot.tail_len = p.tail_len;
        s.f.slot.ced = p.stash;
        s.ds_valid[s.cur()] = p.pos - p.ds_rows;
        s.dsRestored();
    }

    /// The drafter's rings came back with the entry: its context starts at pos - rows.
    fn dsRestored(s: *Sessions) void {
        if (!s.ds_rings) return;
        const h = s.ds_set orelse return;
        h.set(h.ctx, s.cur(), s.ds_valid[s.cur()]);
    }

    pub fn uploadBlobs(s: *Sessions, bs: []const Blob) !void {
        const m = try @import("session_codec.zig").validate(s, bs);
        const r = s.f.runner;
        for (s.pieces) |pc| {
            const blob = for (bs) |b| {
                if (std.mem.eql(u8, b.name, pc.role)) break b;
            } else unreachable;
            const addr = (r.addressOf(pc.role) orelse return error.Unbound) + s.cur() * pc.stride;
            try s.f.upload(addr, blob.bytes);
        }
        s.f.slot.pos = m.pos;
        s.f.slot.tail = m.tail;
        s.f.slot.tail_len = m.tail_len;
        s.f.slot.ced = m.stash;
        s.ds_valid[s.cur()] = m.pos - m.ds;
        s.dsRestored();
    }

    // -- the operations (every rank, the leader's order) ----

    /// An entry for the slot's committed ids (null: a duplicate), then the slot's pages go back (the entry holds them).
    pub fn save(s: *Sessions) !?u32 {
        const h = &s.hist[s.cur()];
        if (h.items.len != s.f.slot.pos or s.f.slot.pending != null) return error.BadWindow;
        try s.f.sendSess(.save, 0);
        if (h.items.len == 0) return null;
        const snap = try s.snapshot();
        const id = s.store.save(s.kx.slot, snap.bounded(), h.items, s.tag(), 0) catch |e| {
            Snap.release(snap);
            return e;
        };
        try s.reset();
        // the residency budget past the save: the least recently used chats parked (streamed) or dropped
        try s.store.settleAll();
        return id;
    }

    /// The served shape's prompt snapshot at its replay point; the slot keeps its pages and goes on prefilling.
    pub fn savePrompt(s: *Sessions) !?u32 {
        const h = &s.hist[s.cur()];
        if (h.items.len != s.f.slot.pos or s.f.slot.pending != null) return error.BadWindow;
        if (pf.state(s.f).mode == .replay and s.f.slot.ced.hi != s.f.slot.pos) return error.BadWindow;
        try s.f.sendSess(.prompt, 0);
        if (h.items.len == 0) return null;
        const snap = try s.snapshot();
        const id = s.store.save(s.kx.slot, snap.bounded(), h.items, s.tag(), prompt_kind) catch |e| {
            Snap.release(snap);
            return e;
        };
        try s.store.settleAll();
        return id;
    }

    /// sessions.Entry.kind: a prompt's replay point (a turn's end is 0)
    const prompt_kind: u32 = 1;

    /// The entries' tag (sessions.Tag's CED field): replay entries never resume full ones, nor full ones replay's.
    fn tag(s: *const Sessions) u32 {
        return if (pf.state(s.f).mode == .replay) ced.tag else ced.tag_fast;
    }

    /// The slot empty, its pages back (a session entry may hold them), its reservation gone.
    pub fn reset(s: *Sessions) !void {
        try s.kx.release();
        s.f.slot = .{};
        s.hist[s.cur()].clearRetainingCapacity();
        s.ds_valid[s.cur()] = 0;
    }

    /// Admits the request, then restores the longest entry strictly prefixing `prompt`; its position (0: none).
    pub fn resume_(s: *Sessions, prompt: []const u32, max_new: ?u64) !u64 {
        if (s.f.slot.pos != 0 or s.kx.slot.len != 0) return error.SlotNotEmpty;
        const hit = try s.find(prompt);
        if (max_new) |mn| try s.admit(hit, prompt.len, mn);
        s.ds_valid[s.cur()] = 0;
        return (try s.resumeHit(prompt, hit)).at;
    }

    /// The longest saved entry that strictly prefixes `prompt` (a row must remain to prefill), null: none.
    pub fn find(s: *Sessions, prompt: []const u32) !?sessions.store.Hit {
        const ids = try s.gpa.alloc(i32, prompt.len);
        defer s.gpa.free(ids);
        for (ids, prompt) |*d, t| d.* = @intCast(t);
        const h = (try s.store.find(s.tag(), ids)) orelse return null;
        return if (h.pos >= prompt.len) null else h;
    }

    /// The pool decision for a request (kv/sess.zig `admit`; the round planner's: kv/sched.zig).
    pub fn poolPlan(s: *Sessions, n: u64, max_new: u64, hit: ?sessions.store.Hit, held: []const u32, extra: u32, spills: *std.ArrayList(u32)) !kv.sess.Plan {
        return kv.sess.admit(&s.store, hit, n, max_new, s.kx.slot.capacity, held, extra, spills);
    }

    pub const Resumed = struct { at: u64 = 0, damaged: bool = false };

    /// The round planner's admission into the empty slot: `spills` evicted, `need` pages reserved, `hit` restored.
    pub fn resumePlanned(s: *Sessions, prompt: []const u32, need: u32, hit: ?sessions.store.Hit, spills: []const u32) !Resumed {
        if (s.f.slot.pos != 0 or s.kx.slot.len != 0) return error.SlotNotEmpty;
        s.stats.admits += 1;
        try s.f.sendSessAdmit(need, spills);
        try s.applyAdmit(need, spills);
        s.ds_valid[s.cur()] = 0;
        return s.resumeHit(prompt, hit);
    }

    fn resumeHit(s: *Sessions, prompt: []const u32, hit: ?sessions.store.Hit) !Resumed {
        const h = hit orelse return .{};
        const n = prompt.len;
        // out of RAM no tokens are indexed: the history is the prompt's prefix
        const ids: ?[]const u32 = if (s.store.entry(h.id).ram) null else prompt[0..@intCast(h.pos)];
        try s.f.sendRestore(h.id, ids);
        if (!try s.restore(h.id, ids)) return .{ .damaged = true };
        std.log.scoped(.dsv41).info("sessions: slot {d} resumed {d} of {d} prompt tokens from {s}", .{ s.cur(), h.pos, n, if (h.ram) "RAM" else "NVMe" });
        return .{ .at = h.pos };
    }

    /// The request's admission: every rank evicts the leader's picks and reserves pages, else goes on unreserved.
    fn admit(s: *Sessions, hit: ?sessions.store.Hit, n: usize, max_new: u64) !void {
        var spills: std.ArrayList(u32) = .empty;
        defer spills.deinit(s.gpa);
        const held: []const u32 = if (hit) |h| &.{h.id} else &.{};
        const plan = try kv.sess.admit(&s.store, hit, n, max_new, s.kx.slot.capacity, held, 0, &spills);
        s.stats.admits += 1;
        if (plan.wait) {
            s.stats.waits += 1;
            if (!quiet) std.log.scoped(.dsv41).warn("sessions: a {d}-token request (+{d}) needs {d} pool pages, {d} are free and unreserved even with every RAM entry gone; the reference server would queue it, it runs unreserved", .{ n, max_new, plan.want, s.kx.pool.available() });
            return;
        }
        try s.f.sendSessAdmit(plan.need, spills.items);
        try s.applyAdmit(plan.need, spills.items);
    }

    fn applyAdmit(s: *Sessions, need: u32, spills: []const u32) !void {
        for (spills) |id| _ = try s.store.evict(id);
        try s.store.settleAll();
        s.stats.spills += spills.len;
        s.kx.slot.reserve(need);
    }

    /// Entry `id` into the empty slot, agreed (lockstep.zig); false: not back on some rank, so a miss everywhere.
    pub fn restore(s: *Sessions, id: u32, ids: ?[]const u32) !bool {
        if (s.f.slot.pos != 0 or s.kx.slot.len != 0) return error.SlotNotEmpty;
        try s.active();
        const pos = s.store.entry(id).pos;
        const g: ?lockstep.Agree = if (s.agreement) |*a| blk: {
            const x = a.agree();
            break :blk .{ .ctx = x.ctx, .all = x.all };
        } else null;
        if (!try lockstep.agreed(g, s.restoreOn(id, ids), "a session restore", s.f.comm.rank())) {
            try s.kx.truncate(0);
            s.f.slot = .{};
            s.hist[s.cur()].clearRetainingCapacity();
            s.ds_valid[s.cur()] = 0;
            // a cache miss from here: the entry out of RAM and its file deleted, on every rank alike
            if (s.store.entry(id).ram) try s.store.drop(id);
            if (s.store.entry(id).live) try s.store.forget(id); // drop removed an entry no file holds
            s.stats.damaged += 1;
            if (!quiet) std.log.scoped(.dsv41).warn("sessions: slot {d}: the entry of {d} tokens did not come back; its prompt prefills from 0", .{ s.cur(), pos });
            return false;
        }
        try s.kx.dev.syncTables();
        return true;
    }

    /// This rank's restore of `id`: pages and bounded state (RAM or NVMe), the position, the slot's history.
    fn restoreOn(s: *Sessions, id: u32, ids: ?[]const u32) !void {
        const e = s.store.entry(id);
        const pos = e.pos;
        if (e.ram) {
            const b = try s.store.restoreRam(id, s.kx.slot);
            try s.uploadSnap(@ptrCast(@alignCast(b.ptr)));
        } else try s.restoreDisk(id);
        if (s.f.slot.pos != pos) return error.RestoredPosition;
        const h = &s.hist[s.cur()];
        try h.resize(s.gpa, pos);
        // RAM: the path's indexed tokens; NVMe: the ids the leader sent (its prompt's prefix)
        const given: ?[]i32 = if (ids) |x| try s.gpa.alloc(i32, x.len) else null;
        defer if (given) |gv| s.gpa.free(gv);
        if (given) |gv| for (ids.?, gv) |t, *y| {
            y.* = @intCast(t);
        };
        try s.store.historyOf(id, given, h.items);
    }

    fn restoreDisk(s: *Sessions, id: u32) !void {
        const j = try s.store.beginRestore(id, s.kx.slot);
        var restored = (try s.store.settle(j)) orelse return error.BadBlob;
        defer restored.deinit(s.gpa);
        try s.uploadBlobs(restored.blobs);
    }

    /// Every RAM entry parked to the NVMe tier now (a server parks through the budget).
    pub fn parkAll(s: *Sessions) !u32 {
        var n: u32 = 0;
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(s.gpa);
        var id = s.store.oldest;
        while (id != sessions.store.none) : (id = s.store.entry(id).extra.newer) try ids.append(s.gpa, id);
        for (ids.items) |e| if (try s.store.beginPark(e)) |j| {
            _ = try s.store.settle(j);
            n += 1;
        };
        return n;
    }
};
