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

/// The store's view of a kept state: the n-gram history at it (16 bytes), then its buffer as snapshot.zig lays it out.
pub fn image(comptime H: type) rows.Image {
    return .{ .parts = parts, .gather = gather, .blank = Blank(H).blank, .scatter = scatter };
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
    const buf = k.st.buf.contents();
    var o: usize = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        if (g.off < hist) @memcpy(out[o..][0..len], std.mem.asBytes(&k.st.hist)[@intCast(g.off)..][0..len]) else @memcpy(out[o..][0..len], buf[@intCast(g.off - hist)..][0..len]);
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

fn scatter(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    const k: *Kept = @ptrCast(@alignCast(saved));
    const buf = k.st.buf.contents();
    var o: usize = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        if (g.off < hist) @memcpy(std.mem.asBytes(&k.st.hist)[@intCast(g.off)..][0..len], in[o..][0..len]) else @memcpy(buf[@intCast(g.off - hist)..][0..len], in[o..][0..len]);
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

/// --learn-conversations: the store under `root`, attached to the cache (Imprint's cut files go unread in this mode).
pub fn attach(gpa: Allocator, store: *api.prompt_cache.Store, id: anytype, root: []const u8, cap: u64, floor: u64, daily: u64) !*rows.Rows {
    const r = try rows.Rows.open(gpa, store.family, .{ .root = root, .identity = id.id, .cap = cap, .floor = floor, .daily = daily, .lookahead = store.rules.lookahead, .planned = store.rules.planned });
    r.prefill_rate = id.rate;
    store.disk = r;
    std.log.info("flash next: conversation states on disk in {s} ({d} states, {d} MiB of {d} MiB)", .{ r.dir, r.states.items.len, r.held() >> 20, cap >> 20 });
    return r;
}
