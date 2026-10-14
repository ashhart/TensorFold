//! The 27B's kept states on disk (--learn-conversations): their image, copies out and in, and their identity.
const std = @import("std");
const mtl = @import("metal");
const api = @import("engine_api");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lanes = tf.lanes;
const rows = api.prompt_cache.rows;
const disk_identity = @import("disk_identity.zig");
const Host = @import("qwen27_served.zig").Host;
const State = q.snapshot.State;
const Allocator = std.mem.Allocator;

/// The store's view of a kept state: the drafter's committed end, then the slot's copy as snapshot.zig lays it out.
pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter };

fn host(ptr: *anyopaque) *Host {
    return @ptrCast(@alignCast(ptr));
}

/// The committed end (8 bytes), the DeltaNet states, each attention layer's key then value rows by head, the taps ring.
fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
    const h = host(ptr);
    const r = &h.runner;
    const gdn = q.snapshot.slotBytes(r, null, 0);
    const n = r.caches.len / r.slots * 2 * r.model.config.kv_heads;
    const row = (q.snapshot.slotBytes(r, null, 1) - gdn) / n;
    try out.append(a, .{ .bytes = 8 });
    try out.append(a, .{ .bytes = gdn });
    for (0..n) |_| try out.append(a, .{ .bytes = @as(u64, at) * row, .row = @intCast(row) });
    if (h.back.ringOf(0)) |g| try out.append(a, .{ .bytes = g.bytes(at), .row = @intCast(g.stride), .window = g.window });
}

/// A shared buffer for copies out of private kept states, kept between steps and grown as needed.
fn staged(h: *Host, n: usize) !mtl.Buffer {
    if (h.disk_stage) |b| if (b.length() >= n) return b;
    if (h.disk_stage) |b| b.deinit();
    h.disk_stage = null;
    h.disk_stage = try h.model.device.buffer(@max(n, 32 << 20), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    return h.disk_stage.?;
}

/// Ranges of a kept state's image into `out`: one blit pass on the model's queue (the engine thread's, while it idles).
fn gather(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    const h = host(ptr);
    const st: *const State = @ptrCast(@alignCast(saved));
    const stage = try staged(h, out.len);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = h.model.queue.commandBuffer();
    const e = cb.blit();
    var o: usize = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        if (g.off < 8) @memcpy(out[o..][0..len], std.mem.asBytes(&st.draft_end)[@intCast(g.off)..][0..len]) else e.copy(st.buf, @intCast(g.off - 8), stage, o, len);
        o += len;
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.SnapshotGpuFailure;
    o = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        if (g.off >= 8) @memcpy(out[o..][0..len], stage.contents()[o..][0..len]);
        o += len;
    }
}

/// A kept state of `at` tokens in a shared buffer, which `scatter` fills straight from the disk's staging.
fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
    const h = host(ptr);
    const n = q.snapshot.slotBytes(&h.runner, h.back.ringOf(0), at);
    const buf = try h.model.device.buffer(n, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer buf.deinit();
    const st = try h.gpa.create(State);
    st.* = .{ .buf = buf, .at = at, .bytes = n, .draft_end = 0 };
    return st;
}

fn scatter(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    const st: *State = @ptrCast(@alignCast(saved));
    var o: usize = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        if (g.off < 8) @memcpy(std.mem.asBytes(&st.draft_end)[@intCast(g.off)..][0..len], in[o..][0..len]) else @memcpy(st.buf.contents()[@intCast(g.off - 8)..][0..len], in[o..][0..len]);
        o += len;
    }
}

/// What disk states are read under, and the probe's prompt rate: the checkpoint, the drafter, sources, modes, bits.
pub fn identity(h: *Host, io: std.Io, a: Allocator, dir: []const u8, drafter: ?[]const u8) !struct { id: u64, rate: f64 } {
    var w = std.hash.Wyhash.init(0x7132);
    try disk_identity.checkpoint(&w, a, io, dir);
    if (drafter) |d| try disk_identity.checkpoint(&w, a, io, d);
    const p = try probe(h, io);
    const words = [_]u64{ q.disk.sourceHash(), @intFromBool(h.model.device.tensorUnits()), @intFromBool(h.model.ane != null), @intFromBool(h.draft != null), p.hash };
    w.update(std.mem.sliceAsBytes(&words));
    disk_identity.machine(&w, std.mem.span(h.model.device.name()));
    return .{ .id = w.final(), .rate = p.rate };
}

/// A fixed prompt's state through slot 0 (whole 128-row chunks, a decoded span, a short tail), hashed, and its rate.
fn probe(h: *Host, io: std.Io) !struct { hash: u64, rate: f64 } {
    var toks: [1024 + 40 + 7]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(1000 + i * 7919 % 50000);
    const Hook = struct {
        h: *Host,
        saved: ?*State = null,
        failed: ?anyerror = null,
        fn at(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
            const p: *@This() = @ptrCast(@alignCast(ptr));
            const slot = p.h.back.slotOf(s) orelse return;
            p.saved = q.snapshot.saveSlot(p.h.gpa, &p.h.runner, slot, p.h.back.ringOf(slot), mark) catch |e| {
                p.failed = e;
                return;
            };
        }
    };
    var hook: Hook = .{ .h = h };
    const marks = [_]u32{1064};
    var s = try lanes.Stream.init(h.gpa, .{ .id = "probe", .prompt = &toks, .max_new = 1, .drafts = h.draft != null, .decode_spans = &.{.{ 1028, 1048 }}, .reuse = .{ .marks = &marks, .hook = .{ .ptr = &hook, .at = Hook.at } } });
    defer s.deinit(h.gpa);
    const be = h.back.backend();
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    be.prefill(&s) catch |e| {
        be.release(&s);
        return e;
    };
    const ns = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    be.release(&s);
    const st = hook.saved orelse return hook.failed orelse error.ProbeUnmarked;
    defer q.snapshot.drop(h.gpa, st);
    const bytes = try h.gpa.alloc(u8, st.bytes + 8);
    defer h.gpa.free(bytes);
    try gather(h, st, &.{ .{ .off = 0, .len = 8 }, .{ .off = 8, .len = st.bytes } }, bytes);
    const rate = @as(f64, @floatFromInt(toks.len)) / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9);
    return .{ .hash = std.hash.Wyhash.hash(0x7127, bytes), .rate = rate };
}

/// The store for --learn-conversations under `root`, attached to the cache; the probe seeds its prompt rate.
pub fn attach(h: *Host, io: std.Io, a: Allocator, o: Learn) !void {
    const store = &(h.cache orelse return std.log.warn("Qwen3.8-27B: --learn-conversations needs the prompt cache, which --prompt-cache-gib 0 turned off", .{}));
    const id = try identity(h, io, a, o.dir, o.drafter);
    const r = try rows.Rows.open(h.gpa, store.family, .{ .root = o.root, .identity = id.id, .cap = o.cap, .floor = o.floor, .daily = o.daily, .lookahead = store.rules.lookahead, .planned = store.rules.planned });
    r.prefill_rate = id.rate;
    store.disk = r;
    std.log.info("Qwen3.8-27B: conversation states on disk in {s} ({d} states, {d} MiB of {d} MiB)", .{ r.dir, r.states.items.len, r.held() >> 20, o.cap >> 20 });
}

/// --learn-conversations' settings, as the server read them.
pub const Learn = struct { root: []const u8, dir: []const u8, drafter: ?[]const u8, cap: u64, floor: u64, daily: u64 };
