//! The NVMe tier: a file an entry under <root>/<compat>/rank<r>/, written via .tmp + rename, streamed via the lanes.

const std = @import("std");
const linux = std.os.linux;
const Io = std.Io;
const lanes_mod = @import("lanes.zig");
const Lanes = lanes_mod.Lanes;
const Staging = lanes_mod.Staging;
const Op = lanes_mod.Op;
const up = lanes_mod.up;
const format = @import("format.zig");
const Layout = format.Layout;
const Head = format.Head;
const Seg = format.Seg;
const PageStore = @import("pagestore.zig").PageStore;
const pool_mod = @import("pool.zig");
const Family = pool_mod.Family;
const Digest = @import("prefix.zig").Digest;

pub const none = std.math.maxInt(u32);

pub const Options = struct {
    root: []const u8,
    compat: Digest,
    rank: u32 = 0,
    world: u32 = 1,
    /// bytes of files the tier keeps (least recently used files without dependents go first)
    budget: u64 = 64 << 30,
    /// entries shorter than this are dropped, not parked
    min_tokens: u64 = 1024,
    /// O_DIRECT where the filesystem takes it (buffered with the page cache dropped otherwise)
    direct: bool = true,
    /// staging buffers and their size: the host transient of any park or restore
    staging: u32 = 16,
    chunk: u32 = 8 << 20,
    lanes: u32 = 4,
    /// delta files a chain may stack before a park writes the entry whole
    max_chain: u32 = 8,
};

/// A piece of bounded state (rings, carries, taps): small, host bytes.
pub const Blob = struct {
    name: []const u8,
    bytes: []const u8,
};

/// What a park writes: the entry, its tokens from first_page on, and this rank's pages per family, in logical order.
pub const WriteSrc = struct {
    key: Digest,
    /// the file holding pages [0, first_page) (a delta), or null (a whole entry: first_page 0)
    base: ?Digest = null,
    tag: u32,
    kind: u32 = 0,
    page: u32,
    pos: u64,
    first_page: u32 = 0,
    npages: u32,
    /// tokens [first_page x page, pos)
    ids: []const i32,
    store: PageStore,
    fence: u64 = 0,
    families: []const Family,
    /// per family: family-tensor pages of this rank's logical pages [first_page, npages), in logical order
    pages: []const []const u32,
    blobs: []const Blob = &.{},
};

/// One file of a restore: read logical pages [lo, hi) of it into the slot.
pub const Extent = struct {
    key: Digest,
    first_page: u32,
    lo: u32,
    hi: u32,
};

/// Where a restore puts rows: per family, the family-tensor page of each of the slot's logical pages.
pub const ReadSink = struct {
    store: PageStore,
    families: []const Family,
    page: u32,
    world: u32,
    rank: u32,
    /// per family: family-tensor pages for logical pages [0, npages)
    pages: []const []const u32,
};

/// The bounded state a restore returns (owned).
pub const Restored = struct {
    blobs: []Blob,

    pub fn deinit(r: *Restored, gpa: std.mem.Allocator) void {
        for (r.blobs) |b| {
            gpa.free(b.name);
            gpa.free(b.bytes);
        }
        gpa.free(r.blobs);
    }
};

/// The index record of one file.
pub const Rec = struct {
    key: Digest,
    base: u32 = none,
    tag: u32,
    kind: u32,
    pos: u64,
    first_page: u32,
    npages: u32,
    size: u64,
    /// files stacked on this one
    refs: u32 = 0,
    /// files under it, this one included
    depth: u32 = 1,
    older: u32 = none,
    newer: u32 = none,
    live: bool = true,
};

pub const Stats = struct {
    writes: u64 = 0,
    reads: u64 = 0,
    bytes_written: u64 = 0,
    bytes_read: u64 = 0,
    bad: u64 = 0,
    evicted: u64 = 0,
};

pub const Disk = struct {
    gpa: std.mem.Allocator,
    io: Io,
    opts: Options,
    dir: Io.Dir,
    staging: Staging,
    lanes: *Lanes,
    recs: std.ArrayList(Rec) = .empty,
    free_recs: std.ArrayList(u32) = .empty,
    by_key: std.AutoHashMapUnmanaged(Digest, u32) = .empty,
    /// least recently used first
    oldest: u32 = none,
    newest: u32 = none,
    used: u64 = 0,
    stats: Stats = .{},

    pub fn open(gpa: std.mem.Allocator, io: Io, opts: Options) !*Disk {
        const d = try gpa.create(Disk);
        errdefer gpa.destroy(d);
        var name_buf: [64]u8 = undefined;
        const sub = try std.fmt.bufPrint(&name_buf, "{x}/rank{d}", .{ opts.compat[0..8], opts.rank });
        var root = try Io.Dir.cwd().createDirPathOpen(io, opts.root, .{});
        defer root.close(io);
        const dir = try root.createDirPathOpen(io, sub, .{ .open_options = .{ .iterate = true } });
        errdefer dir.close(io);
        d.* = .{ .gpa = gpa, .io = io, .opts = opts, .dir = dir, .staging = try Staging.init(gpa, io, @max(2, opts.staging), up(opts.chunk)), .lanes = undefined };
        errdefer d.staging.deinit();
        d.lanes = try Lanes.start(gpa, io, &d.staging, @max(1, opts.lanes));
        return d;
    }

    pub fn close(d: *Disk) void {
        d.lanes.shutdown();
        d.staging.deinit();
        d.dir.close(d.io);
        d.recs.deinit(d.gpa);
        d.free_recs.deinit(d.gpa);
        d.by_key.deinit(d.gpa);
        const gpa = d.gpa;
        gpa.destroy(d);
    }

    pub fn accepts(d: *const Disk, pos: u64) bool {
        return pos >= d.opts.min_tokens;
    }

    pub fn get(d: *Disk, key: Digest) ?*Rec {
        const i = d.by_key.get(key) orelse return null;
        return &d.recs.items[i];
    }

    fn fileName(key: Digest, ext: []const u8, buf: *[48]u8) [:0]const u8 {
        const n = (std.fmt.bufPrint(buf[0..47], "{x}.{s}", .{ &key, ext }) catch unreachable).len;
        buf[n] = 0;
        return buf[0..n :0];
    }

    fn openFile(d: *Disk, name: [:0]const u8, write: bool) !struct { fd: linux.fd_t, direct: bool } {
        var flags: linux.O = .{ .ACCMODE = if (write) .WRONLY else .RDONLY, .CLOEXEC = true };
        if (write) {
            flags.CREAT = true;
            flags.TRUNC = true;
        }
        if (d.opts.direct) {
            var f2 = flags;
            f2.DIRECT = true;
            const rc = linux.openat(d.dir.handle, name, f2, 0o644);
            if (linux.errno(rc) == .SUCCESS) return .{ .fd = @intCast(rc), .direct = true };
        }
        const rc = linux.openat(d.dir.handle, name, flags, 0o644);
        if (linux.errno(rc) != .SUCCESS) return if (linux.errno(rc) == .NOENT) error.FileNotFound else error.OpenFailed;
        return .{ .fd = @intCast(rc), .direct = false };
    }

    /// The layout a WriteSrc gets: segments, chunk sizes (whole pages a chunk for pool segments).
    pub fn plan(d: *Disk, src: *const WriteSrc) !Layout {
        var segs: std.ArrayList(Seg) = .empty;
        defer segs.deinit(d.gpa);
        const cap = d.opts.chunk;
        try segs.append(d.gpa, .{ .name = format.segName("ids"), .kind = .ids, .family = 0, .nbytes = src.ids.len * 4, .chunk = cap });
        for (src.families, 0..) |f, i| {
            const pb: u32 = @intCast(f.pageBytes(src.page));
            const k = @max(1, cap / pb);
            if (@as(u64, k) * pb > d.staging.size) return error.PageTooLarge;
            try segs.append(d.gpa, .{ .name = format.segName(f.name), .kind = .pool, .family = @intCast(i), .nbytes = @as(u64, src.pages[i].len) * pb, .chunk = k * pb });
        }
        for (src.blobs) |b| try segs.append(d.gpa, .{ .name = format.segName(b.name), .kind = .blob, .family = 0, .nbytes = b.bytes.len, .chunk = cap });
        const head: Head = .{ .key = src.key, .base = src.base orelse @splat(0), .compat = d.opts.compat, .tag = src.tag, .kind = src.kind, .page = src.page, .world = d.opts.world, .rank = d.opts.rank, .first_page = src.first_page, .npages = src.npages, .nseg = 0, .ndigest = 0, .pos = src.pos };
        return Layout.build(d.gpa, head, segs.items);
    }

    /// Writes an entry's file streamed (the pump copies pages to staging, lanes hash and write); returns its bytes.
    pub fn writeFile(d: *Disk, src: *const WriteSrc) !u64 {
        var lay = try d.plan(src);
        defer lay.deinit(d.gpa);
        var tb: [48]u8 = undefined;
        var fb: [48]u8 = undefined;
        const tmp = fileName(src.key, "tmp", &tb);
        const f = try d.openFile(tmp, true);
        var ok = false;
        defer {
            _ = linux.close(f.fd);
            if (!ok) _ = linux.unlinkat(d.dir.handle, tmp, 0);
        }
        try src.store.waitFence(src.fence);
        var op = try Op.init(d.gpa, d.io, &d.staging);
        defer op.deinit();
        errdefer op.wait() catch {};
        for (lay.segs) |*s| {
            for (0..s.nchunks) |ci| {
                if (op.failed()) |e| return e;
                const c = s.chunkAt(@intCast(ci));
                const sb = try d.staging.get();
                errdefer d.staging.put(sb);
                const buf = d.staging.buf(sb)[0..c.data];
                const at = @as(u64, ci) * s.chunk;
                switch (s.kind) {
                    .ids => @memcpy(buf, std.mem.sliceAsBytes(src.ids)[at..][0..c.data]),
                    .blob => @memcpy(buf, blobOf(src, s).bytes[at..][0..c.data]),
                    .pool => {
                        const pb: u32 = @intCast(src.families[s.family].pageBytes(src.page));
                        const p0: usize = @intCast(at / pb);
                        try src.store.read(s.family, src.pages[s.family][p0..][0 .. c.data / pb], buf);
                    },
                    _ => unreachable,
                }
                d.lanes.submit(.{ .kind = .write, .fd = f.fd, .staging = sb, .data = c.data, .span = c.span, .off = c.off, .digest = &lay.digests[s.first_digest + ci], .op = &op });
            }
        }
        try op.wait();
        const hb = try d.staging.get();
        defer d.staging.put(hb);
        if (lay.head_span > d.staging.size) return error.HeaderTooLarge;
        lay.encode(d.staging.buf(hb));
        try lanes_mod.pwriteAll(f.fd, d.staging.buf(hb)[0..lay.head_span], 0);
        if (linux.errno(linux.fdatasync(f.fd)) != .SUCCESS) return error.SyncFailed;
        if (!f.direct) _ = linux.fadvise(f.fd, 0, 0, linux.POSIX_FADV.DONTNEED);
        if (linux.errno(linux.renameat(d.dir.handle, tmp, d.dir.handle, fileName(src.key, "tfs", &fb))) != .SUCCESS) return error.RenameFailed;
        ok = true;
        return lay.head.total;
    }

    fn blobOf(src: *const WriteSrc, s: *const Seg) Blob {
        for (src.blobs) |b| if (std.mem.eql(u8, b.name, s.nameOf())) return b;
        unreachable;
    }

    /// A file's header, checked.
    pub fn readHead(d: *Disk, fd: linux.fd_t) !Layout {
        const hb = try d.staging.get();
        defer d.staging.put(hb);
        const buf = d.staging.buf(hb);
        try lanes_mod.preadAll(fd, buf[0..4096], 0);
        const hn = std.mem.readInt(u32, buf[12..16], .little);
        if (up(hn) > buf.len) return error.BadHeader;
        if (hn > 4096) try lanes_mod.preadAll(fd, buf[4096..up(hn)], 4096);
        return Layout.decode(d.gpa, buf[0..up(hn)]);
    }

    /// Reads pages [lo, hi) of one restore file into the sink, each chunk checked first; blobs too when want_blobs.
    pub fn readFile(d: *Disk, ext: Extent, sink: *const ReadSink, want_blobs: bool) !?Restored {
        var nb: [48]u8 = undefined;
        const f = try d.openFile(fileName(ext.key, "tfs", &nb), false);
        defer _ = linux.close(f.fd);
        var lay = try d.readHead(f.fd);
        defer lay.deinit(d.gpa);
        if (!std.mem.eql(u8, &lay.head.key, &ext.key) or lay.head.first_page != ext.first_page or lay.head.page != sink.page) return error.BadHeader;
        var st: linux.Statx = undefined;
        if (linux.errno(linux.statx(f.fd, "", linux.AT.EMPTY_PATH, .{ .SIZE = true }, &st)) != .SUCCESS or st.size < lay.head.total) return error.Truncated;
        var out: std.ArrayList(Blob) = .empty;
        errdefer {
            for (out.items) |b| {
                d.gpa.free(b.name);
                d.gpa.free(b.bytes);
            }
            out.deinit(d.gpa);
        }
        var op = try Op.init(d.gpa, d.io, &d.staging);
        defer op.deinit();
        errdefer op.wait() catch {};
        // (segment, chunk) jobs; the pump keeps at most staging - 1 in flight (one buffer for the header path)
        var jobs: std.ArrayList(struct { seg: u32, chunk: u32 }) = .empty;
        defer jobs.deinit(d.gpa);
        for (lay.segs, 0..) |*s, si| switch (s.kind) {
            .pool => {
                if (s.family >= sink.families.len) return error.BadHeader;
                const fam = sink.families[s.family];
                const pb: u32 = @intCast(fam.pageBytes(sink.page));
                if (s.chunk % pb != 0 or !std.mem.eql(u8, s.nameOf(), fam.name)) return error.BadHeader;
                const need = ownedIn(fam, sink, ext.first_page, ext.hi) - ownedIn(fam, sink, ext.first_page, ext.lo);
                const skip = ownedIn(fam, sink, ext.first_page, ext.lo);
                if (s.nbytes < (@as(u64, skip) + need) * pb) return error.BadHeader;
                const kpp = s.chunk / pb;
                const c0 = skip / kpp;
                const c1 = if (need == 0) c0 else (skip + need + kpp - 1) / kpp;
                for (c0..c1) |c| try jobs.append(d.gpa, .{ .seg = @intCast(si), .chunk = @intCast(c) });
            },
            .blob => if (want_blobs) {
                const buf = try d.gpa.alloc(u8, s.nbytes);
                errdefer d.gpa.free(buf);
                const name = try d.gpa.dupe(u8, s.nameOf());
                errdefer d.gpa.free(name);
                try out.append(d.gpa, .{ .name = name, .bytes = buf });
                for (0..s.nchunks) |c| try jobs.append(d.gpa, .{ .seg = @intCast(si), .chunk = @intCast(c) });
            },
            else => {},
        };
        const window = d.staging.count - 1;
        var sent: usize = 0;
        var seen: usize = 0;
        var bytes: u64 = 0;
        while (seen < jobs.items.len) {
            while (sent < jobs.items.len and sent - seen < window and op.failed() == null) : (sent += 1) {
                const jb = jobs.items[sent];
                const s = &lay.segs[jb.seg];
                const c = s.chunkAt(jb.chunk);
                d.lanes.submit(.{ .kind = .read, .fd = f.fd, .staging = try d.staging.get(), .data = c.data, .span = c.span, .off = c.off, .digest = &lay.digests[s.first_digest + jb.chunk], .op = &op, .tag = @intCast(sent) });
                bytes += c.span;
            }
            const j = op.next() orelse break;
            seen += 1;
            defer d.staging.put(j.staging);
            if (op.failed() != null) continue;
            const jb = jobs.items[j.tag];
            const s = &lay.segs[jb.seg];
            const data = d.staging.buf(j.staging)[0..j.data];
            if (s.kind == .blob) {
                for (out.items) |b| if (std.mem.eql(u8, b.name, s.nameOf())) @memcpy(@constCast(b.bytes)[@as(u64, jb.chunk) * s.chunk ..][0..data.len], data);
                continue;
            }
            try d.drain(s, jb.chunk, data, ext, sink);
        }
        try op.wait();
        if (!f.direct) _ = linux.fadvise(f.fd, 0, 0, linux.POSIX_FADV.DONTNEED);
        d.stats.bytes_read += bytes;
        if (!want_blobs) {
            out.deinit(d.gpa);
            return null;
        }
        return .{ .blobs = try out.toOwnedSlice(d.gpa) };
    }

    /// A checked pool chunk's pages that fall in [lo, hi) into the sink's pages.
    fn drain(d: *Disk, s: *const Seg, chunk: u32, data: []const u8, ext: Extent, sink: *const ReadSink) !void {
        _ = d;
        const fam = sink.families[s.family];
        const pb: u32 = @intCast(fam.pageBytes(sink.page));
        const kpp = s.chunk / pb;
        const skip = ownedIn(fam, sink, ext.first_page, ext.lo);
        const end = ownedIn(fam, sink, ext.first_page, ext.hi);
        const a = @max(chunk * kpp, skip);
        const b = @min(chunk * kpp + @as(u32, @intCast(data.len / pb)), end);
        if (a >= b) return;
        var list: [512]u32 = undefined;
        var n: u32 = 0;
        var j = a;
        while (j < b) : (j += 1) {
            list[n] = sink.pages[s.family][logicalOf(fam, sink, ext.first_page, j)];
            n += 1;
            if (n == list.len or j + 1 == b) {
                const first = j + 1 - n;
                try sink.store.write(s.family, list[0..n], data[(first - chunk * kpp) * pb ..][0 .. n * pb]);
                n = 0;
            }
        }
    }

    /// Records a written file in the index (round thread, at the plan's point); its base gains a dependent.
    pub fn register(d: *Disk, src: *const WriteSrc, size: u64) !u32 {
        const id = try d.add(src.key, src.base, src.tag, src.kind, src.pos, src.first_page, src.npages, size);
        d.stats.writes += 1;
        d.stats.bytes_written += size;
        return id;
    }

    fn add(d: *Disk, key: Digest, base: ?Digest, tag: u32, kind: u32, pos: u64, first_page: u32, npages: u32, size: u64) !u32 {
        var r: Rec = .{ .key = key, .tag = tag, .kind = kind, .pos = pos, .first_page = first_page, .npages = npages, .size = size };
        if (base) |b| {
            const bi = d.by_key.get(b) orelse return error.NoBase;
            r.base = bi;
            r.depth = d.recs.items[bi].depth + 1;
        }
        const id: u32 = if (d.free_recs.pop()) |i| i else blk: {
            try d.recs.append(d.gpa, undefined);
            break :blk @intCast(d.recs.items.len - 1);
        };
        try d.by_key.put(d.gpa, key, id);
        if (r.base != none) d.recs.items[r.base].refs += 1;
        d.recs.items[id] = r;
        d.link(id);
        d.used += size;
        return id;
    }

    /// A valid file found on restart: its header, size and verified ids (tokens [first_page x page, pos), owned).
    pub const Found = struct {
        head: Head,
        size: u64,
        mtime: i64,
        ids: []i32,

        fn older(_: void, a: Found, b: Found) bool {
            return a.mtime < b.mtime or (a.mtime == b.mtime and a.head.first_page < b.head.first_page);
        }
    };

    /// Indexes the directory on restart: drops .tmp and bad files, checks ids, registers bases before their deltas.
    pub fn scan(d: *Disk, out: *std.ArrayList(Found)) !void {
        var it = d.dir.iterate();
        var nb: [48]u8 = undefined;
        while (try it.next(d.io)) |ent| {
            if (ent.name.len >= nb.len) continue;
            @memcpy(nb[0..ent.name.len], ent.name);
            nb[ent.name.len] = 0;
            const name = nb[0..ent.name.len :0];
            if (std.mem.endsWith(u8, name, ".tmp")) {
                _ = linux.unlinkat(d.dir.handle, name, 0);
                continue;
            }
            if (!std.mem.endsWith(u8, name, ".tfs")) continue;
            const f = d.found(name) catch {
                d.stats.bad += 1;
                _ = linux.unlinkat(d.dir.handle, name, 0);
                continue;
            };
            try out.append(d.gpa, f);
        }
        std.mem.sort(Found, out.items, {}, Found.older);
        var i: usize = 0;
        while (i < out.items.len) {
            const f = out.items[i];
            const base: ?Digest = if (f.head.hasBase()) f.head.base else null;
            if (base != null and d.by_key.get(base.?) == null) {
                d.stats.bad += 1;
                var kb: [48]u8 = undefined;
                _ = linux.unlinkat(d.dir.handle, fileName(f.head.key, "tfs", &kb), 0);
                d.gpa.free(f.ids);
                _ = out.orderedRemove(i);
                continue;
            }
            _ = try d.add(f.head.key, base, f.head.tag, f.head.kind, f.head.pos, f.head.first_page, f.head.npages, f.size);
            i += 1;
        }
    }

    fn found(d: *Disk, name: [:0]const u8) !Found {
        const f = try d.openFile(name, false);
        defer _ = linux.close(f.fd);
        var lay = try d.readHead(f.fd);
        defer lay.deinit(d.gpa);
        const h = lay.head;
        var kb: [48]u8 = undefined;
        if (!std.mem.eql(u8, fileName(h.key, "tfs", &kb), name)) return error.BadHeader;
        if (!std.mem.eql(u8, &h.compat, &d.opts.compat) or h.rank != d.opts.rank or h.world != d.opts.world) return error.Foreign;
        var st: linux.Statx = undefined;
        if (linux.errno(linux.statx(f.fd, "", linux.AT.EMPTY_PATH, .{ .SIZE = true, .MTIME = true }, &st)) != .SUCCESS or st.size < h.total) return error.Truncated;
        const seg = lay.find(.ids, "ids") orelse return error.BadHeader;
        if (seg.nbytes % 4 != 0 or seg.nbytes / 4 != h.pos - @as(u64, h.first_page) * h.page) return error.BadHeader;
        const ids = try d.gpa.alloc(i32, seg.nbytes / 4);
        errdefer d.gpa.free(ids);
        const sb = try d.staging.get();
        defer d.staging.put(sb);
        for (0..seg.nchunks) |ci| {
            const c = seg.chunkAt(@intCast(ci));
            const buf = d.staging.buf(sb);
            try lanes_mod.preadAll(f.fd, buf[0..c.span], c.off);
            var dg: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(buf[0..c.data], &dg, .{});
            if (!std.mem.eql(u8, &dg, &lay.digests[seg.first_digest + ci])) return error.ChecksumMismatch;
            @memcpy(std.mem.sliceAsBytes(ids)[@as(usize, ci) * seg.chunk ..][0..c.data], buf[0..c.data]);
        }
        return .{ .head = h, .size = st.size, .mtime = st.mtime.sec * std.time.ns_per_s + st.mtime.nsec, .ids = ids };
    }

    fn link(d: *Disk, id: u32) void {
        const r = &d.recs.items[id];
        r.older = d.newest;
        r.newer = none;
        if (d.newest != none) d.recs.items[d.newest].newer = id else d.oldest = id;
        d.newest = id;
    }

    fn unlink(d: *Disk, id: u32) void {
        const r = &d.recs.items[id];
        if (r.older != none) d.recs.items[r.older].newer = r.newer else d.oldest = r.newer;
        if (r.newer != none) d.recs.items[r.newer].older = r.older else d.newest = r.older;
    }

    /// Marks a file (and its bases) used now: O(chain).
    pub fn touch(d: *Disk, key: Digest) void {
        var id = d.by_key.get(key) orelse return;
        while (id != none) : (id = d.recs.items[id].base) {
            d.unlink(id);
            d.link(id);
        }
        // bases end up newest: walk order keeps the entry itself older than its bases, so trims reach dependents first
    }

    /// The files a restore of key reads, base first, with the logical pages each gives.
    pub fn extents(d: *Disk, key: Digest, out: *std.ArrayList(Extent)) !void {
        out.clearRetainingCapacity();
        var id = d.by_key.get(key) orelse return error.FileNotFound;
        var hi = d.recs.items[id].npages;
        while (id != none) {
            const r = &d.recs.items[id];
            try out.append(d.gpa, .{ .key = r.key, .first_page = r.first_page, .lo = r.first_page, .hi = hi });
            hi = r.first_page;
            id = r.base;
        }
        std.mem.reverse(Extent, out.items);
    }

    /// Deletes a file without dependents (error when one stacks on it).
    pub fn remove(d: *Disk, key: Digest) !void {
        const id = d.by_key.get(key) orelse return;
        const r = &d.recs.items[id];
        if (r.refs > 0) return error.HasDependents;
        if (r.base != none) d.recs.items[r.base].refs -= 1;
        d.unlink(id);
        _ = d.by_key.remove(key);
        d.used -= r.size;
        r.live = false;
        var nb: [48]u8 = undefined;
        _ = linux.unlinkat(d.dir.handle, fileName(key, "tfs", &nb), 0);
        try d.free_recs.append(d.gpa, id);
    }

    /// Deletes least recently used files without dependents while over budget; their keys go to out.
    pub fn trim(d: *Disk, out: *std.ArrayList(Digest)) !void {
        var id = d.oldest;
        while (d.used > d.opts.budget and id != none) {
            const next = d.recs.items[id].newer;
            if (d.recs.items[id].refs == 0) {
                const key = d.recs.items[id].key;
                try d.remove(key);
                try out.append(d.gpa, key);
                d.stats.evicted += 1;
                id = d.oldest; // a removed file may free its base
                continue;
            }
            id = next;
        }
    }
};

/// Owned (this rank's) logical pages of family f in [first, k): the file's index of logical page k.
pub fn ownedIn(f: Family, sink: *const ReadSink, first: u32, k: u32) u32 {
    if (!(f.split and sink.world > 1)) return k - first;
    var by: [pool_mod.max_world]u32 = undefined;
    pool_mod.span(first, k, sink.world, &by);
    return by[sink.rank];
}

/// The logical page of a family's j-th stored page of a file that starts at first.
pub fn logicalOf(f: Family, sink: *const ReadSink, first: u32, j: u32) u32 {
    if (!(f.split and sink.world > 1)) return first + j;
    const w = sink.world;
    const r0 = first + (sink.rank + w - first % w) % w; // the first owned logical page at or after first
    return r0 + j * w;
}
