//! Nemotron's CUDA kept states on disk (--learn-conversations): their image, copies on the engine's stream, identity.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const lanes = @import("lanes");
const nemotron = @import("nemotron");
const disk_identity = @import("disk_identity.zig");
const kept = nemotron.native.kept_states;
const rows = api.prompt_cache.rows;
const Target = kept.Target;
const Allocator = std.mem.Allocator;

/// The store's view of a kept state: the engine's two words, the key and value rows, the Mamba states.
pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter, .whole = whole };

const words = 16; // the parity and row count where the state stood

fn target(ptr: *anyopaque) *Target {
    const t: *Target = @ptrCast(@alignCast(ptr));
    if (t.ctx) |c| c.makeCurrent() catch {};
    return t;
}

/// The words, each attention layer's key rows, then their value rows, the head's two, then the Mamba states whole.
fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
    const sh = kept.shape(target(ptr), at);
    try out.append(a, .{ .bytes = words });
    const planes = 2 * sh.layers + @as(usize, if (sh.head) 2 else 0);
    for (0..planes) |_| try out.append(a, .{ .bytes = @as(u64, at) * sh.row, .row = @intCast(sh.row) });
    try out.append(a, .{ .bytes = sh.mamba });
}

/// A range of the image: the words, the key and value buffer, or the Mamba buffer, at its offset there.
fn where(sh: kept.Shape, off: u64) struct { part: enum { words, kv, mamba }, at: usize } {
    if (off < words) return .{ .part = .words, .at = @intCast(off) };
    if (off < words + sh.kv) return .{ .part = .kv, .at = @intCast(off - words) };
    return .{ .part = .mamba, .at = @intCast(off - words - sh.kv) };
}

fn gather(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    const t = target(ptr);
    const copy: *kept.Copy = @ptrCast(@alignCast(saved));
    const sh = kept.shape(t, copy.pos);
    var o: usize = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        const w = where(sh, g.off);
        switch (w.part) {
            .words => {
                const x = kept.words(saved);
                @memcpy(out[o..][0..len], std.mem.asBytes(&x)[w.at..][0..len]);
            },
            .kv, .mamba => try kept.hostCopy(t, saved, w.part == .mamba, w.at, out[o..][0..len], true),
        }
        o += len;
    }
    try kept.settle(t);
}

fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
    return @ptrCast(try kept.blank(target(ptr), at));
}

fn scatter(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    const t = target(ptr);
    const copy: *kept.Copy = @ptrCast(@alignCast(saved));
    const sh = kept.shape(t, copy.pos);
    var o: usize = 0;
    for (ranges) |g| {
        const len: usize = @intCast(g.len);
        const w = where(sh, g.off);
        switch (w.part) {
            .words => {
                var x = kept.words(saved);
                @memcpy(std.mem.asBytes(&x)[w.at..][0..len], in[o..][0..len]);
                kept.setWords(saved, x);
            },
            .kv, .mamba => try kept.hostCopy(t, saved, w.part == .mamba, w.at, @constCast(in[o..][0..len]), false),
        }
        o += len;
    }
    try kept.settle(t); // `in` is the store's staging, reused by its next read
}

/// An armed mark's state whose chunk failed before its rows were copied is not whole.
fn whole(_: *anyopaque, saved: api.prompt_cache.Saved) bool {
    const copy: *kept.Copy = @ptrCast(@alignCast(saved));
    return copy.whole;
}

/// What disk states are read under: the checkpoint, kernels (ours or a captured set), driver, device, a probe's bits.
pub fn identity(a: Allocator, io: std.Io, g: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, backend: lanes.backend.Backend, t: *Target) !struct { id: u64, rate: f64 } {
    const p = try probe(a, io, backend, t);
    return .{ .id = try identityOf(a, io, g, dir, kernels, &[_]u64{@intFromBool(t.head != null)}, p.hash), .rate = p.rate };
}

/// The identity's hash: the checkpoint, kernels, device, driver, a family's facts, then its probe's hash.
pub fn identityOf(a: Allocator, io: std.Io, g: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, facts: []const u64, probed: u64) !u64 {
    var h = std.hash.Wyhash.init(0x6375);
    try disk_identity.checkpoint(&h, a, io, dir);
    inline for (comptime std.meta.declarations(cuda.kernels)) |d| {
        const v = @field(cuda.kernels, d);
        if (comptime @TypeOf(v) == []const u8) h.update(v);
    }
    if (kernels) |k| h.update(try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ k, "aot.json" }), a, .limited(1 << 26)));
    var name_buf: [256]u8 = undefined;
    h.update(g.name(&name_buf) catch "GPU");
    const device = [_]u64{ try g.capability(), @intCast(try g.d.version()) };
    h.update(std.mem.sliceAsBytes(&device));
    h.update(std.mem.sliceAsBytes(facts));
    h.update(std.mem.asBytes(&probed));
    return h.final();
}

/// A fixed prompt's state at a mark inside its last chunk, hashed (a kernel, driver or arithmetic change moves it).
fn probe(a: Allocator, io: std.Io, backend: lanes.backend.Backend, t: *Target) !struct { hash: u64, rate: f64 } {
    var toks: [4096 + 40 + 7]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = @intCast(1000 + i * 7919 % 50000);
    const Hook = struct {
        t: *Target,
        saved: ?*anyopaque = null,
        failed: ?anyerror = null,
        fn at(ptr: *anyopaque, _: *lanes.Stream, mark: u32) void {
            const p: *@This() = @ptrCast(@alignCast(ptr));
            p.saved = kept.save(p.t, null, mark) catch |e| {
                p.failed = e;
                return;
            };
        }
    };
    var hook: Hook = .{ .t = t };
    defer if (hook.saved) |saved| kept.drop(t, saved);
    const marks = [_]u32{4136};
    var s = try lanes.Stream.init(a, .{ .id = "probe", .prompt = &toks, .max_new = 1, .drafts = t.head != null, .reuse = .{ .marks = &marks, .hook = .{ .ptr = &hook, .at = Hook.at } } });
    defer s.deinit(a);
    const ns = pass: {
        defer backend.vtable.release(backend.ptr, &s); // once, whether the pass, its wait or nothing failed
        const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
        try backend.vtable.prefill(backend.ptr, &s);
        try kept.settle(t);
        break :pass std.Io.Clock.awake.now(io).toNanoseconds() - t0;
    };
    const saved = hook.saved orelse return hook.failed orelse error.ProbeUnmarked;
    if (!whole(t, saved)) return error.ProbeUnmarked;
    const sh = kept.shape(t, 4136);
    const bytes = try a.alloc(u8, words + sh.kv + sh.mamba);
    try gather(t, saved, &.{ .{ .off = 0, .len = words }, .{ .off = words, .len = sh.kv }, .{ .off = words + sh.kv, .len = sh.mamba } }, bytes);
    // the row count: the last window's, from whatever ran first (a new build measures its windows)
    @memset(bytes[8..words], 0);
    return .{ .hash = std.hash.Wyhash.hash(0x6375, bytes), .rate = @as(f64, @floatFromInt(toks.len)) / (@as(f64, @floatFromInt(@max(ns, 1))) / 1e9) };
}

/// --learn-conversations: the identity probed on this thread (its context current), the store opened and attached.
pub fn attach(a: Allocator, io: std.Io, ctx: *const cuda.Context, o: api.Open, root: []const u8, kernels: ?[]const u8, loaded: anytype, store: *api.prompt_cache.Store) !void {
    loaded.target.ctx = ctx;
    const id = try identity(a, io, ctx, o.dir, kernels, loaded.backend, loaded.target);
    try open(o, root, id.id, id.rate, store, "nemotron");
}

/// The disk module of a family's kept states (image, probe, facts), or null: Nemotron's is this file's own.
pub fn keptDisk(comptime F: type) ?type {
    return if (@hasDecl(F, "kept_disk")) F.kept_disk else null;
}

/// Kept states through disk module `K` (keptDisk): the same store under the identity its probe and facts give.
pub fn attachKept(comptime K: type, name: []const u8, a: Allocator, io: std.Io, ctx: *const cuda.Context, o: api.Open, root: []const u8, kernels: ?[]const u8, loaded: anytype, store: *api.prompt_cache.Store) !void {
    // a module that opens the store itself (several ranks' parts, say) declares attach, and takes identityOf from here
    if (comptime @hasDecl(K, "attach")) return K.attach(a, io, ctx, o, root, kernels, loaded, store) else {
        if (@hasDecl(K, "bind")) K.bind(loaded.target, ctx); // the context the store's copies make current
        const p = try K.probe(a, io, loaded.backend, loaded.target);
        const facts = K.facts(loaded.target); // an array, or a slice
        const list: []const u64 = if (@typeInfo(@TypeOf(facts)) == .pointer) facts else &facts;
        try open(o, root, try identityOf(a, io, ctx, o.dir, kernels, list, p.hash), p.rate, store, name);
    }
}

fn open(o: api.Open, root: []const u8, id: u64, rate: f64, store: *api.prompt_cache.Store, name: []const u8) !void {
    const gibs = struct {
        fn bytes(g: f64) u64 {
            return @intFromFloat(g * (1 << 30));
        }
    }.bytes;
    const r = try api.prompt_cache.rows.Rows.open(store.gpa, store.family, .{ .root = root, .identity = id, .cap = gibs(o.learn_gib), .floor = gibs(o.learn_min_free_gib), .daily = gibs(o.learn_daily_gib), .lookahead = store.rules.lookahead, .planned = store.rules.planned, .grid = store.rules.grid });
    r.prefill_rate = rate;
    store.disk = r;
    std.log.info("{s}: conversation states on disk in {s} ({d} states, {d} MiB of {d} MiB)", .{ name, r.dir, r.states.items.len, r.held() >> 20, r.cap >> 20 });
}

test {
    std.testing.refAllDecls(@This());
}

test "a disk module that declares attach opens the store itself" {
    const K = struct {
        pub var calls: usize = 0;
        pub fn attach(_: Allocator, _: std.Io, _: *const cuda.Context, _: api.Open, root: []const u8, _: ?[]const u8, _: anytype, _: *api.prompt_cache.Store) !void {
            calls += 1;
            try std.testing.expectEqualStrings("/r", root);
        }
    };
    try attachKept(K, "k", std.testing.allocator, std.testing.io, undefined, .{ .dir = "d", .model_type = "m" }, "/r", null, @as(u8, 0), undefined);
    try std.testing.expectEqual(@as(usize, 1), K.calls);
}
