//! The sessions tier's writes on the real Store and Disk index: a file's bytes are its Disk.plan, its I/O is accounted.
const std = @import("std");
const sessions = @import("sessions");
const Store = sessions.Store;
const Disk = sessions.Disk;
const disk_mod = sessions.disk;
const jobs = sessions.jobs;
const Digest = sessions.prefix.Digest;

pub const page: u32 = 256;
const tag: u32 = 1;

/// DeepSeek-V4.1's release families (deepseek_v41/kv/layout.zig:99-118): 1,790 B a token, replicated.
pub const families = [_]sessions.Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = 584 },  .{ .name = "index_k.2", .ratio = 2, .row_bytes = 132 },
    .{ .name = "comp.8", .ratio = 2, .row_bytes = 584 },  .{ .name = "index_k.8", .ratio = 2, .row_bytes = 132 },
    .{ .name = "comp.14", .ratio = 2, .row_bytes = 584 }, .{ .name = "index_k.14", .ratio = 2, .row_bytes = 132 },
    .{ .name = "comp.20", .ratio = 1, .row_bytes = 584 }, .{ .name = "index_k.20", .ratio = 1, .row_bytes = 132 },
};

/// A DeepSeek-V4.1 full-mode snapshot without drafts (sessions_gpu.zig:131-146, session_snapshot.zig:32-64).
pub const Snapshot = struct {
    layers: u32 = 40,
    window: u32 = 128,
    carries: [3]u32 = .{ 2, 8, 14 },
    head_dim: u32 = 512,
    /// sessions_gpu.zig:297, Sessions.slot_bytes
    slot: u32 = 584,

    pub fn bytes(m: *const Snapshot) u64 {
        return @as(u64, m.layers) * m.window * (576 + 8) + m.carries.len * 8 * @as(u64, m.head_dim) + m.slot;
    }

    fn of(p: *anyopaque) *const Snapshot {
        return @ptrCast(@alignCast(p));
    }
    fn nbytes(p: *anyopaque) u64 {
        return of(p).bytes();
    }
    fn blobs(p: *anyopaque, gpa: std.mem.Allocator) anyerror![]disk_mod.Blob {
        const m = of(p);
        var out: std.ArrayList(disk_mod.Blob) = .empty;
        errdefer {
            for (out.items) |b| {
                gpa.free(b.name);
                gpa.free(b.bytes);
            }
            out.deinit(gpa);
        }
        for (0..m.layers) |L| {
            try add(gpa, &out, try std.fmt.allocPrint(gpa, "s.L{d}.swa.v", .{L}), m.window * 576);
            try add(gpa, &out, try std.fmt.allocPrint(gpa, "s.L{d}.swa.s", .{L}), m.window * 8);
        }
        for (m.carries) |L| try add(gpa, &out, try std.fmt.allocPrint(gpa, "s.L{d}.carry", .{L}), 8 * m.head_dim);
        try add(gpa, &out, try gpa.dupe(u8, "slot"), m.slot);
        return out.toOwnedSlice(gpa);
    }
    fn add(gpa: std.mem.Allocator, out: *std.ArrayList(disk_mod.Blob), name: []u8, n: usize) !void {
        errdefer gpa.free(name);
        const b = try gpa.alloc(u8, n);
        errdefer gpa.free(b);
        @memset(b, 0);
        try out.append(gpa, .{ .name = name, .bytes = b });
    }
    fn release(_: *anyopaque) void {}
    const vtable: sessions.store.Bounded.VTable = .{ .nbytes = nbytes, .blobs = blobs, .release = release };
};

/// Pages whose bytes nobody reads: the accounting never moves rows, a real park hashes zeros.
const Blank = struct {
    fn read(_: *anyopaque, _: u32, _: []const u32, dst: []u8) anyerror!void {
        @memset(dst, 0);
    }
    fn write(_: *anyopaque, _: u32, _: []const u32, _: []const u8) anyerror!void {}
    fn copyPage(_: *anyopaque, _: u32, _: u32, _: u32) anyerror!void {}
    fn fence(_: *anyopaque) anyerror!u64 {
        return 0;
    }
    fn waitFence(_: *anyopaque, _: u64) anyerror!void {}
    fn publish(_: *anyopaque) anyerror!void {}
    const vtable: sessions.PageStore.VTable = .{ .read = read, .write = write, .copyPage = copyPage, .fence = fence, .waitFence = waitFence, .publish = publish };
};

/// The served shape saves the prompt's snapshot (target.zig:95-101); TF_DSV41_PROMPT_TAIL=prefill the turn (:462-465).
pub const Saves = enum { prompt, turn };

pub const Config = struct {
    policy: sessions.store.Policy,
    min_tokens: u64 = 1024,
    max_chain: u32 = 8,
    pool_tokens: u64 = 1 << 20,
    capacity: u64 = 1 << 18,
    chunk: u32 = 8 << 20,
    staging: u32 = 16,
    saves: Saves = .prompt,
    /// real files under this root (Linux), else the accounted tier
    root: ?[]const u8 = null,
    snapshot: Snapshot = .{},
};

/// DeepSeek-V4.1's defaults (sessions_gpu.zig:25-50): RAM 256 MiB, idle chats unlimited, covered and delta off.
pub fn dsv41() Config {
    return .{ .policy = .{ .ram_bytes = 256 << 20, .skip_covered = false, .delta = false } };
}

pub const Counts = struct {
    requests: u64 = 0,
    prompt_tokens: u64 = 0,
    ram_tokens: u64 = 0,
    disk_tokens: u64 = 0,
    spills: u64 = 0,
    waits: u64 = 0,
    files: u64 = 0,
    deltas: u64 = 0,
    written: u64 = 0,
    read: u64 = 0,
    parks: u64 = 0,
    covered_drops: u64 = 0,
    drops: u64 = 0,

    pub fn minus(a: Counts, b: Counts) Counts {
        var out: Counts = undefined;
        inline for (@typeInfo(Counts).@"struct".field_names) |n| @field(out, n) = @field(a, n) - @field(b, n);
        return out;
    }
};

pub const Bench = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    cfg: Config,
    pool: sessions.Pool,
    disk: *Disk,
    store: Store,
    slot: *sessions.Slot,
    /// each accounted file's segments, for its restores (accounting thread only, while the round waits)
    files: std.AutoHashMapUnmanaged(Digest, []format.Seg) = .empty,
    spills: std.ArrayList(u32) = .empty,
    counts: Counts = .{},
    peak_used: u64 = 0,
    blank: u8 = 0,

    const format = sessions.format;

    pub fn init(gpa: std.mem.Allocator, io: std.Io, cfg: Config) !*Bench {
        const b = try gpa.create(Bench);
        errdefer gpa.destroy(b);
        b.* = .{ .gpa = gpa, .io = io, .cfg = cfg, .pool = undefined, .disk = undefined, .store = undefined, .slot = undefined };
        b.pool = try sessions.Pool.init(gpa, .{ .families = &families, .page = page }, cfg.pool_tokens, 1, 0);
        errdefer b.pool.deinit();
        const opts: disk_mod.Options = .{ .root = cfg.root orelse "", .compat = @splat(0xEF), .budget = std.math.maxInt(u64), .min_tokens = cfg.min_tokens, .max_chain = cfg.max_chain, .chunk = cfg.chunk, .staging = cfg.staging, .fresh_start = true };
        b.disk = if (cfg.root != null) try Disk.open(gpa, io, opts) else try accounted(gpa, io, opts);
        errdefer b.closeDisk();
        var policy = cfg.policy;
        policy.async_io = false;
        b.store = try Store.init(gpa, io, &b.pool, .{ .ptr = &b.blank, .vtable = &Blank.vtable }, b.disk, policy);
        errdefer b.store.deinit();
        // the tier thread's place: the runner queues each job and this thread accounts it, as jobs.zig:134-149 runs it
        if (cfg.root == null) b.store.runner.?.thread = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, loop, .{b});
        b.slot = try b.pool.newSlot(cfg.capacity);
        return b;
    }

    pub fn deinit(b: *Bench) void {
        b.spills.deinit(b.gpa);
        b.store.deinit();
        b.closeDisk();
        var it = b.files.valueIterator();
        while (it.next()) |v| b.gpa.free(v.*);
        b.files.deinit(b.gpa);
        b.pool.deinit();
        b.gpa.destroy(b);
    }

    fn accounted(gpa: std.mem.Allocator, io: std.Io, opts: disk_mod.Options) !*Disk {
        const d = try gpa.create(Disk);
        errdefer gpa.destroy(d);
        d.* = .{ .gpa = gpa, .io = io, .opts = opts, .dir = undefined, .staging = try sessions.lanes.Staging.init(gpa, io, @max(2, opts.staging), sessions.lanes.up(opts.chunk)), .lanes = undefined };
        return d;
    }

    fn closeDisk(b: *Bench) void {
        if (b.cfg.root != null) return b.disk.close();
        const d = b.disk;
        d.staging.deinit();
        d.recs.deinit(b.gpa);
        d.free_recs.deinit(b.gpa);
        d.by_key.deinit(b.gpa);
        b.gpa.destroy(d);
    }

    fn loop(b: *Bench) void {
        const r = b.store.runner.?;
        while (true) {
            const j = blk: {
                r.mutex.lockUncancelable(r.io);
                defer r.mutex.unlock(r.io);
                while (r.head == r.queue.items.len and !r.stop) r.cond.waitUncancelable(r.io, &r.mutex);
                if (r.head == r.queue.items.len) return;
                r.head += 1;
                break :blk r.queue.items[r.head - 1];
            };
            j.state.store(.running, .release);
            j.err = switch (j.kind) {
                .park => b.park(j),
                .restore => b.restore(j),
            };
            j.state.store(if (j.err == null) .done else .failed, .release);
            r.mutex.lockUncancelable(r.io);
            r.cond.broadcast(r.io);
            r.mutex.unlock(r.io);
        }
    }

    /// disk.zig:229 and :275: writeFile writes the plan's layout and returns its total.
    fn park(b: *Bench, j: *jobs.Job) ?anyerror {
        var lay = b.disk.plan(&j.src) catch |e| return e;
        defer lay.deinit(b.gpa);
        j.size = lay.head.total;
        const segs = b.gpa.dupe(format.Seg, lay.segs) catch |e| return e;
        const g = b.files.getOrPut(b.gpa, j.src.key) catch |e| {
            b.gpa.free(segs);
            return e;
        };
        if (g.found_existing) b.gpa.free(g.value_ptr.*);
        g.value_ptr.* = segs;
        return null;
    }

    /// disk.zig:321-355 and :372: a restore reads its files' chunks holding [lo, hi), and the last file's blobs.
    fn restore(b: *Bench, j: *jobs.Job) ?anyerror {
        const sink = &j.sink;
        for (j.extents, 0..) |ext, i| {
            const segs = b.files.get(ext.key) orelse return error.FileNotFound;
            var bytes: u64 = 0;
            for (segs) |*s| switch (s.kind) {
                .pool => {
                    const fam = sink.families[s.family];
                    const pb: u32 = @intCast(fam.pageBytes(sink.page));
                    const need = disk_mod.ownedIn(fam, sink, ext.first_page, ext.hi) - disk_mod.ownedIn(fam, sink, ext.first_page, ext.lo);
                    const skip = disk_mod.ownedIn(fam, sink, ext.first_page, ext.lo);
                    const kpp = s.chunk / pb;
                    const c0 = skip / kpp;
                    const c1 = if (need == 0) c0 else (skip + need + kpp - 1) / kpp;
                    for (c0..c1) |c| bytes += s.chunkAt(@intCast(c)).span;
                },
                .blob => if (i + 1 == j.extents.len) {
                    for (0..s.nchunks) |c| bytes += s.chunkAt(@intCast(c)).span;
                },
                else => {},
            };
            b.disk.stats.bytes_read += bytes;
        }
        return null;
    }

    /// The bytes Disk.plan lays out for an entry of `pos` tokens written from `first` (0: whole; else a delta).
    pub fn fileBytes(b: *Bench, pos: u64, first: u32) !u64 {
        const np = sessions.pool.pagesFor(pos, page);
        const zeros = try b.gpa.alloc(u32, np - first);
        defer b.gpa.free(zeros);
        @memset(zeros, 0);
        var lists: [families.len][]const u32 = undefined;
        @memset(&lists, zeros);
        const ids = try b.gpa.alloc(i32, @intCast(pos - @as(u64, first) * page));
        defer b.gpa.free(ids);
        const blobs = try Snapshot.blobs(&b.cfg.snapshot, b.gpa);
        defer {
            for (blobs) |x| {
                b.gpa.free(x.name);
                b.gpa.free(x.bytes);
            }
            b.gpa.free(blobs);
        }
        const src: disk_mod.WriteSrc = .{ .key = @splat(0), .base = if (first > 0) @splat(1) else null, .tag = tag, .page = page, .pos = pos, .first_page = first, .npages = np, .ids = ids, .store = undefined, .families = &families, .pages = &lists, .blobs = blobs };
        var lay = try b.disk.plan(&src);
        defer lay.deinit(b.gpa);
        return lay.head.total;
    }

    fn bounded(b: *Bench) sessions.store.Bounded {
        return .{ .ptr = &b.cfg.snapshot, .vtable = &Snapshot.vtable };
    }

    /// The tier's counters and this bench's, now.
    pub fn now(b: *const Bench) Counts {
        var c = b.counts;
        const st = b.store.stats;
        c.files = b.disk.stats.writes;
        c.written = b.disk.stats.bytes_written;
        c.read = b.disk.stats.bytes_read;
        c.parks = st.parks;
        c.deltas = st.delta_parks;
        c.covered_drops = st.covered_drops;
        c.drops = st.drops;
        return c;
    }

    fn settle(b: *Bench) !void {
        try b.store.settleAll();
        b.peak_used = @max(b.peak_used, b.disk.used);
    }

    /// One request as DeepSeek-V4.1 serves it on one slot: admit, resume, then the prompt's or the turn's entry.
    pub fn request(b: *Bench, ids: []const i32, prompt_len: usize) !void {
        const s = b.slot;
        const prompt = ids[0..prompt_len];
        b.counts.requests += 1;
        b.counts.prompt_tokens += prompt_len;
        // sessions_gpu.zig:432-438: the longest entry strictly prefixing the prompt
        const hit = if (try b.store.find(tag, prompt)) |h| (if (h.pos < prompt_len) h else null) else null;
        // kv/sess.zig:15-29 and sessions_gpu.zig:470-490: pages for prompt + reply + slack, spilling the coldest
        const need = b.pool.needPages(prompt_len, ids.len - prompt_len, s.capacity, 64);
        const shared: u32 = if (hit) |h| (if (h.ram) @intCast(h.pos / page) else 0) else 0;
        const want = need - @min(need, shared);
        var reserve = true;
        if (b.pool.available() < want) {
            const held: []const u32 = if (hit) |h| &.{h.id} else &.{};
            if (try b.store.planSpill(want, held, &b.spills)) {
                for (b.spills.items) |id| _ = try b.store.evict(id);
                b.counts.spills += b.spills.items.len;
            } else {
                b.counts.waits += 1;
                reserve = false;
            }
        }
        try b.settle();
        if (reserve) s.reserve(need);
        var cached: u64 = 0;
        if (hit) |h| {
            if (h.ram) {
                _ = try b.store.restoreRam(h.id, s);
                b.counts.ram_tokens += h.pos;
            } else {
                const j = try b.store.beginRestore(h.id, s);
                if (try b.store.settle(j)) |r| {
                    var rr = r;
                    rr.deinit(b.gpa);
                }
                b.counts.disk_tokens += h.pos;
            }
            cached = h.pos;
        }
        if (b.cfg.saves == .prompt) if (savePoint(prompt_len, cached)) |at| {
            try s.ensure(at);
            _ = try b.store.save(s, b.bounded(), ids[0..@intCast(at)], tag, 1);
            try b.settle();
        };
        try s.ensure(ids.len);
        if (b.cfg.saves == .turn) _ = try b.store.save(s, b.bounded(), ids, tag, 0);
        _ = try s.releaseAll();
        try b.settle();
    }
};

/// ced.zig:42-50: the last 16-grid point before the prompt's end, past the resumed position and at least 64.
pub fn savePoint(n: u64, cached: u64) ?u64 {
    const s: u64 = if (n > 0) (n - 1) / 16 * 16 else 0;
    return if (s > cached and s >= 64) s else null;
}
