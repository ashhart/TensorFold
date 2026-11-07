//! GLM-5.3-Flash's CUDA kept states on disk (--learn-conversations): every rank's part in rank 0's files.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const glm = @import("glm_cuda");
const cuda_disk = @import("cuda_disk.zig");
const img = glm.disk_image;
const ranks = glm.disk_ranks;
const rows = api.prompt_cache.rows;
const Engine = glm.engine.Engine;
const Kept = glm.engine.Kept;
const Allocator = std.mem.Allocator;

pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter, .whole = whole };

/// Every rank's prefix bytes, read at attach (the process serves one model; rank 0 alone serves).
var prefix_buf: [8]u64 = undefined;
var prefixes: []const u64 = &.{};
/// Where each rank's verify window rows begin in its prefix (the probe's hash leaves them out).
var window_buf: [prefix_buf.len]u64 = undefined;

fn engineOf(ptr: *anyopaque) *Engine {
    return @ptrCast(@alignCast(ptr));
}

fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
    var list: std.ArrayList(img.Part) = .empty;
    defer list.deinit(a);
    try img.storeParts(try ranks.shape(engineOf(ptr)), prefixes, at, &list, a);
    for (list.items) |p| try out.append(a, .{ .bytes = p.bytes, .row = p.row, .per = p.per });
}

/// The other ranks asleep before a disk call go back to sleep after it (their lines, not NCCL, wait meanwhile).
const Rest = struct {
    e: *Engine,
    asleep: bool,

    fn of(e: *Engine) Rest {
        return .{ .e = e, .asleep = ranks.asleep(e) };
    }

    fn done(r: Rest) void {
        if (r.asleep) ranks.rest(r.e) catch |err| std.log.warn("glm: the ranks did not sleep after a disk step ({s})", .{@errorName(err)});
    }
};

fn gather(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    const e = engineOf(ptr);
    const k: *Kept = @ptrCast(@alignCast(saved));
    const rest = Rest.of(e);
    defer rest.done();
    var pieces: std.ArrayList(img.Piece) = .empty;
    defer pieces.deinit(e.gpa);
    const sh = try ranks.shape(e);
    var o: usize = 0;
    for (ranges) |g| {
        pieces.clearRetainingCapacity();
        try img.route(sh, prefixes, k.at, g.off, g.len, &pieces, e.gpa);
        for (pieces.items) |p| {
            const n: usize = @intCast(p.len);
            try ranks.take(e, p.rank, k.id, p.off, out[o..][0..n]);
            o += n;
        }
    }
}

/// A new id leased at `at` on every rank, filled by `scatter` and sealed by the words in its last one.
fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
    const e = engineOf(ptr);
    const rest = Rest.of(e);
    defer rest.done();
    const k = try e.gpa.create(Kept);
    errdefer e.gpa.destroy(k);
    k.* = .{ .id = try ranks.blank(e, at), .at = at };
    return @ptrCast(k);
}

/// The last scatter brings the state's own file, words first: every rank seals it then (the words place the head).
fn scatter(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    const e = engineOf(ptr);
    const k: *Kept = @ptrCast(@alignCast(saved));
    const rest = Rest.of(e);
    defer rest.done();
    if (ranges.len > 0 and ranges[0].off == 0) {
        if (ranges[0].len < img.Words.bytes) return error.DiskWords;
        var x: [img.Words.count]u64 = undefined;
        @memcpy(std.mem.sliceAsBytes(&x), in[0..img.Words.bytes]);
        try ranks.seal(e, k.id, try ranks.decode(e, x, k.at));
    }
    var pieces: std.ArrayList(img.Piece) = .empty;
    defer pieces.deinit(e.gpa);
    const sh = try ranks.shape(e);
    var o: usize = 0;
    for (ranges) |g| {
        pieces.clearRetainingCapacity();
        try img.route(sh, prefixes, k.at, g.off, g.len, &pieces, e.gpa);
        for (pieces.items) |p| {
            const n: usize = @intCast(p.len);
            const bytes = in[o..][0..n];
            o += n;
            if (p.rank == 0 and p.off < img.Words.bytes) continue; // the words went with the seal
            try ranks.put(e, p.rank, k.id, p.off, bytes);
            if (p.shared) for (prefixes[1..], 1..) |q, r| try ranks.put(e, @intCast(r), k.id, p.off - prefixes[0] + q, bytes);
        }
    }
}

fn whole(ptr: *anyopaque, saved: api.prompt_cache.Saved) bool {
    const k: *Kept = @ptrCast(@alignCast(saved));
    const s = engineOf(ptr).snaps.get(k.id) orelse return false;
    return s.len != 0;
}

/// A fixed prompt's store image hashed, other ranks' shared bytes checked equal: both but the bytes it never writes.
fn probe(a: Allocator, io: std.Io, backend: lanes.backend.Backend, e: *Engine) !struct { hash: u64, rate: f64 } {
    const mark: u32 = glm.native.prompt_rows;
    var toks: [glm.native.prompt_rows + 40]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = @intCast(1000 + i * 7919 % 50000);
    const Hook = struct {
        e: *Engine,
        saved: ?*anyopaque = null,
        failed: ?anyerror = null,
        fn at(p: *anyopaque, _: *lanes.Stream, m: u32) void {
            const h: *@This() = @ptrCast(@alignCast(p));
            h.saved = glm.native.snap_save(h.e, null, m) catch |err| {
                h.failed = err;
                return;
            };
        }
    };
    var hook: Hook = .{ .e = e };
    const marks = [_]u32{mark};
    var s = try lanes.Stream.init(a, .{ .id = "probe", .prompt = &toks, .max_new = 1, .drafts = e.mtp, .reuse = .{ .marks = &marks, .hook = .{ .ptr = &hook, .at = Hook.at } } });
    defer s.deinit(a);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    backend.vtable.prefill(backend.ptr, &s) catch |err| {
        backend.vtable.release(backend.ptr, &s);
        return err;
    };
    const ns = std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    backend.vtable.release(backend.ptr, &s);
    const saved = hook.saved orelse return hook.failed orelse error.ProbeUnmarked;
    defer glm.native.snap_drop(e, saved);
    const k: *Kept = @ptrCast(@alignCast(saved));
    const sh = try ranks.shape(e);
    const own = img.imageBytes(sh, mark);
    var total = own;
    for (prefixes[1..]) |p| total += p;
    const bytes = try a.alloc(u8, @intCast(total));
    try gather(e, saved, &.{.{ .off = 0, .len = total }}, bytes);
    var loose: std.ArrayList(img.Span) = .empty;
    try img.unwritten(sh, prefixes, window_buf[0..prefixes.len], mark, (try ranks.wordsOf(e, k.id)).mtp_len, &loose, a);
    img.clear(loose.items, 0, bytes);
    const from = img.Words.bytes + sh.prefix;
    const shared = bytes[@intCast(from)..@intCast(own)];
    var past: std.ArrayList(img.Span) = .empty; // the spans past rank 0's prefix, as spans of `shared`
    for (loose.items) |x| if (x.off >= from) try past.append(a, .{ .off = x.off - from, .len = x.len });
    try ranks.same(e, k.id, prefixes, shared, past.items, try a.alloc(u8, shared.len));
    try ranks.rest(e);
    return .{ .hash = std.hash.Wyhash.hash(0x676c6d, bytes), .rate = @as(f64, @floatFromInt(toks.len)) / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9) };
}

/// What --learn-conversations refuses before it reads a rank: TF_KV_SHARE, and more ranks than `prefix_buf` holds.
fn refusal(e: *const Engine) !void {
    // TF_KV_SHARE leaves a kept state's prefix pages in the pool's prompt cache, outside the lease the image reads
    if (e.share_pages) return error.KvShareWithLearnConversations;
    if (e.share.world > prefix_buf.len) return error.TooManyRanks;
}

/// --learn-conversations: every rank's prefix read, the identity probed on this thread, the store attached.
pub fn attach(a: Allocator, io: std.Io, ctx: *const cuda.Context, o: api.Open, root: []const u8, kernels: ?[]const u8, loaded: anytype, store: *api.prompt_cache.Store) !void {
    const e: *Engine = loaded.target;
    refusal(e) catch |err| {
        if (err == error.KvShareWithLearnConversations) std.log.err("--learn-conversations: GLM-5.3-Flash's disk images read a kept state's whole lease, and TF_KV_SHARE leaves its prefix pages out of it; serve without one of them", .{});
        return err;
    };
    try ranks.prefixes(e, prefix_buf[0..e.share.world], window_buf[0..e.share.world]);
    prefixes = prefix_buf[0..e.share.world];
    const sh = try ranks.shape(e);
    var facts: std.ArrayList(u64) = .empty;
    try facts.appendSlice(a, &.{ e.share.world, sh.capacity, sh.ring_rows, sh.tail, @intFromBool(sh.head), @intFromBool(sh.scales), @intFromBool(e.drafter != null) });
    try facts.appendSlice(a, prefixes);
    const p = try probe(a, io, loaded.backend, e);
    const id = try cuda_disk.identityOf(a, io, ctx, o.dir, kernels, facts.items, p.hash, .{ .family = o.model_type, .backend = "cuda" });
    const gibs = struct {
        fn bytes(g: f64) u64 {
            return @intFromFloat(g * (1 << 30));
        }
    }.bytes;
    // spill stays off: an eviction would gather on rank 0 mid-pass while the other ranks wait in the gate's all-reduce
    const r = try rows.Rows.open(store.gpa, store.family, .{ .root = root, .identity = id, .cap = gibs(o.learn_gib), .floor = gibs(o.learn_min_free_gib), .daily = gibs(o.learn_daily_gib), .lookahead = store.rules.lookahead, .planned = store.rules.planned, .grid = store.rules.grid });
    r.prefill_rate = p.rate;
    store.disk = r;
    std.log.info("glm: conversation states on disk in {s} ({d} states, {d} MiB of {d} MiB; {d} ranks' parts)", .{ r.dir, r.states.items.len, r.held() >> 20, r.cap >> 20, prefixes.len });
}

test "GLM disk attach: TF_KV_SHARE and more ranks than the prefixes hold are refused by name" {
    const e = try std.testing.allocator.create(Engine);
    defer std.testing.allocator.destroy(e);
    e.share_pages = false;
    e.share.world = prefix_buf.len;
    try refusal(e);
    e.share.world = prefix_buf.len + 1;
    try std.testing.expectError(error.TooManyRanks, refusal(e));
    e.share.world = 2;
    e.share_pages = true;
    try std.testing.expectError(error.KvShareWithLearnConversations, refusal(e));
}

test {
    std.testing.refAllDecls(@This());
}
