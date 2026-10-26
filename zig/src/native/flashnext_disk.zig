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
            if (h.closing or !h.disk.push(h.gpa, a)) return false;
            h.wake.signal(h.io);
            return true;
        }
    }.queue;
}

/// Asks answered and, past the quiet with nothing waiting, one piece of disk writing; false when neither ran.
pub fn idleStep(h: anytype) bool {
    h.lock();
    const asks = h.disk.take(h.gpa);
    const store = if (h.cache) |*c| c else null;
    const write = if (store) |c| c.disk != null and rows.write.pending(c.disk.?) and h.queued.items.len == 0 and !h.closing and h.disk.quiet(h.served, h.now()) else false;
    h.unlock();
    defer h.gpa.free(asks);
    for (asks) |a| a.answer(store, .{ .starts = false, .spans = false }); // its lookups pass neither
    if (write) _ = rows.write.step(store.?.disk.?);
    return write or asks.len > 0;
}

/// Whether the idle loop has work: asks, or kept states waiting for disk (under the lock).
pub fn idleWork(h: anytype) bool {
    const store = if (h.cache) |*c| c else return h.disk.asks.items.len > 0;
    return h.disk.asks.items.len > 0 or (store.disk != null and rows.write.pending(store.disk.?));
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
pub fn identity(gpa: Allocator, io: std.Io, eng: *fx.Engine) !struct { id: u64, rate: f64 } {
    var h = std.hash.Wyhash.init(0x6678);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const probe = try learned.probe(eng, gpa);
    const ns = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    const modes = [_]u64{ @intFromBool(eng.r.xnew), @intFromBool(eng.r.hc_up), @intFromBool(eng.r.device.tensorUnits()) };
    for ([_]u64{ eng.identity_hash, learned.sourceHash(), modes[0], modes[1], modes[2], probe }) |x| h.update(std.mem.asBytes(&x));
    var os: [64]u8 = undefined;
    h.update(api.prompt_imprint.osBuild(&os));
    h.update(std.mem.span(eng.r.device.name()));
    return .{ .id = h.final(), .rate = 4143.0 / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9) };
}

/// --learn-conversations or --spill-gib: the store under `l.root`, attached to the cache (Imprint's cut files unread).
pub fn attach(gpa: Allocator, store: *api.prompt_cache.Store, id: anytype, l: anytype) !*rows.Rows {
    const r = try rows.attach(gpa, store, .{ .root = l.root, .identity = id.id, .cap = l.cap, .floor = l.floor, .daily = l.daily, .lookahead = store.rules.lookahead, .planned = store.rules.planned, .spill = l.spill, .slots = l.slots });
    r.prefill_rate = id.rate;
    std.log.info("flash next: conversation states on disk in {s} ({d} states, {d} MiB of {d} MiB)", .{ r.dir, r.states.items.len, r.held() >> 20, l.cap >> 20 });
    return r;
}
