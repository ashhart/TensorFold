//! Flash Next's kept states on disk (--learn-conversations): their image, copies out and in, and their identity.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const fx = tf.flashnext_engine;
const snap = tf.flashnext_snapshot;
const learned = tf.flashnext_learned;
const rows = api.prompt_cache.rows;
const Allocator = std.mem.Allocator;

/// The prompt cache's kept state: the snapshot and its name on both Macs of a pair (key 0: read back from disk).
pub const Kept = struct { st: *snap.State, key: u64 };

/// Engine.imprint for host type H: an ask queued for its engine thread, answered between requests.
pub fn ask(comptime H: type) *const fn (*anyopaque, *api.ImprintAsk) bool {
    return &struct {
        fn queue(ctx: *anyopaque, a: *api.ImprintAsk) bool {
            const h: *H = @ptrCast(@alignCast(ctx));
            h.lock();
            defer h.unlock();
            if (h.closing or (a.op == .flush and h.cache == null) or !h.disk.push(h.gpa, a)) return false; // no store: 501
            h.wake.signal(h.io);
            return true;
        }
    }.queue;
}

/// Answer asks, then stage one image or advance a synchronous write after the quiet; false when neither ran.
pub fn idleStep(h: anytype) bool {
    h.lock();
    const asks = h.disk.take(h.gpa);
    var flushes = h.disk.takeFlushes(); // between requests: none is in flight
    const store = if (h.cache) |*c| c else null;
    const write = if (store) |c| c.disk != null and rows.write.ready(c.disk.?) and h.queued.items.len == 0 and !h.closing and h.disk.quiet(h.served, h.now()) else false;
    h.unlock();
    defer h.gpa.free(asks);
    defer flushes.deinit(h.gpa);
    for (asks) |a| a.answer(store, .{ .starts = false, .spans = false }); // its lookups pass neither
    for (flushes.items) |a| a.answer(store, .{});
    if (flushes.items.len > 0) if (store) |s| { // a reply's end publishes these; with none to come, the flush does
        h.lock();
        h.cache_status = api.prompt_cache.status.PromptCache(s);
        h.disk_status = api.prompt_cache.status.Disk(s);
        h.unlock();
    };
    if (write) _ = rows.write.step(store.?.disk.?);
    return write or asks.len > 0 or flushes.items.len > 0;
}

/// Whether the idle loop has work: asks, or kept states waiting for disk (under the lock).
pub fn idleWork(h: anytype) bool {
    const store = if (h.cache) |*c| c else return h.disk.waiting();
    return h.disk.waiting() or (store.disk != null and rows.write.ready(store.disk.?));
}

/// The store's view of a kept state: the n-gram history at it (16 bytes), then its buffer as snapshot.zig lays it out.
pub fn image(comptime H: type) rows.Image {
    return .{ .parts = parts, .gather = gather, .blank = Blank(H).blank, .scatter = scatter };
}

/// The same image for a shared session's kept state, the bare snapshot (Backend.snapSave); B holds `e` and `gpa`.
pub fn stateImage(comptime B: type) rows.Image {
    return .{ .parts = parts, .gather = gatherState, .blank = BlankState(B).blank, .scatter = scatterState };
}

const hist = 16;

/// The history, the DeltaNet states, each layer's key heads, value heads and indexer keys by row, the tail.
fn parts(_: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
    const sh = snap.shape;
    try out.append(a, .{ .bytes = hist });
    try out.append(a, .{ .bytes = sh.deltanet });
    for (0..sh.layers) |_| {
        for (0..4) |_| try out.append(a, .{ .bytes = @as(u64, at) * sh.key_row, .row = sh.key_row });
        try out.append(a, .{ .bytes = @as(u64, at) * sh.raw_row, .row = sh.raw_row });
    }
    try out.append(a, .{ .bytes = sh.tail });
}

fn gather(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    const k: *const Kept = @ptrCast(@alignCast(saved));
    copyOut(k.st, ranges, out);
}

fn gatherState(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    copyOut(@ptrCast(@alignCast(saved)), ranges, out);
}

fn copyOut(st: *const snap.State, ranges: []const rows.image.Range, out: []u8) void {
    const buf = st.buf.contents();
    var o: usize = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        if (g.off < hist) @memcpy(out[o..][0..len], std.mem.asBytes(&st.hist)[@intCast(g.off)..][0..len]) else @memcpy(out[o..][0..len], buf[@intCast(g.off - hist)..][0..len]);
        o += len;
    }
}

fn Blank(comptime H: type) type {
    return struct {
        /// A pool buffer for `at` tokens (as a learned state's read takes one), filled by `scatter`.
        fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
            const h: *H = @ptrCast(@alignCast(ptr));
            const k = try h.gpa.create(Kept);
            errdefer h.gpa.destroy(k);
            k.* = .{ .st = try snap.blank(h.eng, h.gpa, at), .key = 0 };
            return k;
        }
    };
}

fn BlankState(comptime B: type) type {
    return struct {
        /// A pool buffer for `at` tokens, filled by `scatterState`, kept bare as Backend.snapSave keeps its states.
        fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
            const b: *B = @ptrCast(@alignCast(ptr));
            return try snap.blank(b.e, b.gpa, at);
        }
    };
}

fn scatter(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    const k: *Kept = @ptrCast(@alignCast(saved));
    copyIn(k.st, ranges, in);
}

fn scatterState(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    copyIn(@ptrCast(@alignCast(saved)), ranges, in);
}

fn copyIn(st: *snap.State, ranges: []const rows.image.Range, in: []const u8) void {
    const buf = st.buf.contents();
    var o: usize = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        if (g.off < hist) @memcpy(std.mem.asBytes(&st.hist)[@intCast(g.off)..][0..len], in[o..][0..len]) else @memcpy(buf[@intCast(g.off - hist)..][0..len], in[o..][0..len]);
        o += len;
    }
}

/// What learned and disk states read under: checkpoint, kernel sources, modes, the probe's bits, the OS build and chip.
pub fn identity(gpa: Allocator, io: std.Io, dir: []const u8, eng: *fx.Engine) !struct { id: u64, rate: f64 } {
    var h = std.hash.Wyhash.init(0x6678);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const probe = try learned.probe(eng, gpa);
    const ns = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    const modes = [_]u64{ @intFromBool(eng.r.xnew), @intFromBool(eng.r.hc_up), @intFromBool(eng.r.device.tensorUnits()) };
    for ([_]u64{ eng.identity_hash, learned.sourceHash(), modes[0], modes[1], modes[2], probe }) |x| h.update(std.mem.asBytes(&x));
    const variants = @import("variant_identity.zig");
    try variants.hash(&h, gpa, eng.r.variants);
    const precision = @import("tensorfold").frags.precision;
    variants.policy(&h, eng.r.window.capacity, eng.r.window.factor, eng.r.device.tensorUnits() and try precision.requested());
    var os: [64]u8 = undefined;
    h.update(api.prompt_imprint.osBuild(&os));
    h.update(std.mem.span(eng.r.device.name()));
    @import("disk_identity.zig").profile(&h, .{ .family = "qwen4_exp", .backend = "metal" });
    return .{ .id = try @import("disk_identity.zig").scoped(gpa, io, dir, try withInputs(withPolicy(h.final(), eng.r), gpa, io, eng.r)), .rate = 4143.0 / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9) };
}

/// --learn-conversations or --spill-gib: the store under `l.root`, attached to the cache (Imprint's cut files unread).
pub fn attach(gpa: Allocator, store: *api.prompt_cache.Store, id: anytype, l: anytype) !*rows.Rows {
    const r = try rows.attach(gpa, store, .{ .root = l.root, .identity = id.id, .cap = l.cap, .floor = l.floor, .daily = l.daily, .lookahead = store.rules.lookahead, .planned = store.rules.planned, .spill = l.spill, .slots = l.slots });
    r.prefill_rate = id.rate;
    std.log.info("flash next: conversation states on disk in {s} ({d} states, {d} MiB of {d} MiB)", .{ r.dir, r.states.items.len, r.held() >> 20, l.cap >> 20 });
    return r;
}

/// Extend a probe identity with the loaded execution policy (`serial` orders launches, not arithmetic: not keyed).
pub fn withPolicy(legacy: u64, run: anytype) u64 {
    return @import("persistence_policy.zig").extend(legacy, "flash-metal-policy-v1", &.{
        @intFromBool(run.fused_xsum),       @intFromBool(run.dense),            @intFromBool(run.dense_target),
        @intFromBool(run.split),            @intFromBool(run.gdn_step),         @intFromBool(run.hc_mma),
        @intFromBool(run.grouped),          @intFromBool(run.xpack),            @intFromBool(run.xfused),
        @intFromBool(run.xsx),              @intFromBool(run.copy),             @intFromBool(run.prefetch),
        @intFromBool(run.lane_new),         @intFromBool(run.rotary_enabled),   run.skip,
        run.hcskip,                         run.gskip,                          run.lane_pf,
        // fz_gdn and its kept-state form write the DeltaNet state and conv rows a snapshot copies (snapshot.zig)
        @intFromBool(run.gdn_pipe != null), @intFromBool(run.gdn_kept != null), @intFromBool(run.gx != null),
        if (run.gx) |g| g.min_rows else 0,  @intFromBool(run.tp != null),       if (run.tp) |tp| tp.rank else 0,
    });
}

/// Indexed weight files include the actual pack outside the source checkpoint.
pub fn withFiles(legacy: u64, a: Allocator, io: std.Io, files: []const c_int) !u64 {
    return @import("persistence_policy.zig").files(legacy, a, io, files);
}

/// The loaded weight files and metadata, including a supplied reference outside the checkpoint.
pub fn withInputs(legacy: u64, a: Allocator, io: std.Io, run: anytype) !u64 {
    return @import("persistence_policy.zig").mapped(try withFiles(legacy, a, io, run.files.items), a, run.maps.items);
}
