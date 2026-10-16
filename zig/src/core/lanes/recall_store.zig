//! Recall's record on disk: one tokenizer's ids in chunk files of a 0700 directory, the oldest dropped past the cap.
const std = @import("std");
const platform = @import("sys");
const recall = @import("recall.zig");
const rr = recall.record;
const Admission = @import("learned_disk.zig").Admission;
const Allocator = std.mem.Allocator;

const flock = platform.c.flock;
const lock_ex = platform.c.LOCK.EX;
const lock_nb = platform.c.LOCK.NB;

const magic = "TFRECALL";
const version: u32 = 1;
/// A chunk file's header: magic, version, the vocabulary's size and the identity's first 16 bytes.
const head_bytes = 32;
/// Match lengths that open each band of landing rates (lanes/recall.zig).
pub const bands = recall.bands;
/// A full chunk file's bytes: what the cap counts.
pub const chunk_bytes: u64 = head_bytes + 4 * @as(u64, rr.chunk_tokens);

/// What a record's ids mean: the tokenizer's identity and vocabulary size, and the model whose server first wrote it.
pub const Identity = struct {
    digest: [32]u8,
    vocab: u32,
    name: []const u8 = "",

    /// The digest's first 8 bytes in hex: the default directory's name.
    pub fn short(i: Identity) [16]u8 {
        return std.fmt.bytesToHex(i.digest[0..8].*, .lower);
    }
};

pub const Options = struct {
    dir: []const u8,
    cap_bytes: u64,
    read_only: bool = false,
};

/// The store's state for GET /v1/recall and the startup line.
pub const Stats = struct {
    writes: bool,
    left: bool, // the last empty left a chunk file: nothing is added until one succeeds
    read_only: bool,
    ids: u64,
    disk_bytes: u64,
    ram_bytes: u64,
    cap_bytes: u64,
    added: u64,
    held: u64,
    rates: [recall.bands.len]f64,
    probe_every: u32,
    proposals: u64,
    proposed: u64,
    accepted: u64,
};

pub const Store = struct {
    gpa: Allocator,
    shared: recall.Shared,
    dir: [:0]u8,
    name: []u8,
    identity: Identity,
    cap_chunks: usize,
    read_only: bool, // asked to read only: never writes, never takes the lock
    writes: bool, // adds what its requests read and write (false: read only, or another server holds the lock)
    lock_fd: c_int = -1,
    file: c_int = -1, // the newest chunk's file, open to append (-1: none)
    file_chunk: u32 = 0, // the record chunk that file holds
    seq0: u64 = 0, // the file number of record chunk 0
    failed: bool = false, // a write failed: this server adds nothing more
    admission: Admission = .{}, // the free-disk floor writes stay above
    added: u64 = 0, // ids this server added
    held: u64 = 0, // ids its requests read or wrote that the record held already
    problem_buf: [1200]u8 = undefined,
    problem: []const u8 = "", // why the last empty failed (in problem_buf, or a plain reason)

    /// Ids written before and after each new run: keys across its joins are found, matches there reach max_match.
    pub const overlap: usize = rr.max_match;
    /// A held run shorter than this is written again: a join costs two overlaps.
    const min_run: usize = 2 * overlap;
    /// Ids between the starts the dedup checks: runs of min_run ids or more are never missed.
    const stride: usize = 16;

    /// The record in `o.dir` (made 0700), or null with `problem` set when another tokenizer wrote it.
    pub fn open(
        gpa: Allocator,
        io: std.Io,
        identity: Identity,
        o: Options,
        problem: *[]const u8,
        a: Allocator,
    ) !?*Store {
        const s = try gpa.create(Store);
        errdefer gpa.destroy(s);
        const dir = try gpa.dupeSentinel(u8, o.dir, 0);
        errdefer gpa.free(dir);
        const name = try gpa.dupe(u8, identity.name);
        errdefer gpa.free(name);
        const cap_chunks: usize = @max(2, o.cap_bytes / chunk_bytes);
        s.* = .{
            .gpa = gpa,
            .shared = .{ .io = io, .record = try recall.Record.init(gpa, bucketBits(cap_chunks)) },
            .dir = dir,
            .name = name,
            .identity = identity,
            .cap_chunks = cap_chunks,
            .read_only = o.read_only,
            .writes = !o.read_only,
        };
        s.identity.name = name;
        errdefer s.shared.record.deinit();
        makePath(dir) catch {
            problem.* = try std.fmt.allocPrint(a, "--recall cannot make its directory {s}", .{o.dir});
            return null;
        };
        if (!try s.checkIdentity(a, problem)) {
            s.shared.record.deinit();
            gpa.free(name);
            gpa.free(dir);
            gpa.destroy(s);
            return null;
        }
        if (s.writes) s.takeLock();
        errdefer if (s.lock_fd >= 0) {
            _ = platform.c.close(s.lock_fd);
        };
        try s.load();
        return s;
    }

    pub fn close(s: *Store) void {
        s.closeFile();
        if (s.lock_fd >= 0) _ = platform.c.close(s.lock_fd);
        s.shared.record.deinit();
        s.gpa.free(s.name);
        s.gpa.free(s.dir);
        s.gpa.destroy(s);
    }

    fn closeFile(s: *Store) void {
        if (s.file < 0) return;
        _ = platform.c.fsync(s.file);
        _ = platform.c.close(s.file);
        s.file = -1;
    }

    /// The key table's size for a cap: about one bucket for every 8 ids, fixed so growth never rebuilds the index.
    fn bucketBits(chunks: usize) u5 {
        var bits: u5 = 16;
        while (bits < 28 and (@as(usize, 1) << bits) < chunks * (rr.chunk_tokens / 8)) bits += 1;
        return bits;
    }

    /// The identity file names the tokenizer that wrote the record: another one is refused by its name.
    fn checkIdentity(s: *Store, a: Allocator, problem: *[]const u8) !bool {
        var buf: [1100]u8 = undefined;
        const path = try std.fmt.bufPrintSentinel(&buf, "{s}/identity", .{s.dir}, 0);
        const fields = .{ version, &s.identity.digest, s.identity.vocab };
        const want = try std.fmt.allocPrint(a, "tensorfold recall {d}\ndigest {x}\nvocab {d}\nname ", fields);
        const got = readSmall(a, path) catch |e| if (e == error.FileNotFound) {
            if (s.writes) writeSmall(path, try std.mem.concat(a, u8, &.{ want, s.name, "\n" })) catch {
                problem.* = try std.fmt.allocPrint(a, "--recall cannot write {s}", .{path});
                return false;
            };
            return true;
        } else {
            problem.* = try std.fmt.allocPrint(a, "--recall cannot read {s}", .{path});
            return false;
        };
        if (std.mem.startsWith(u8, got, want)) return true;
        var lines = std.mem.splitScalar(u8, got, '\n');
        var digest: []const u8 = "?";
        var vocab: []const u8 = "?";
        var name: []const u8 = "an unnamed model";
        while (lines.next()) |l| {
            if (std.mem.startsWith(u8, l, "digest ")) digest = l[7..@min(l.len, 7 + 16)];
            if (std.mem.startsWith(u8, l, "vocab ")) vocab = l[6..];
            if (std.mem.startsWith(u8, l, "name ") and l.len > 5) name = l[5..];
        }
        const mine = s.identity.short();
        const refusal = "--recall: {s} holds a record written under the tokenizer of {s} ({s} ids, identity {s}); " ++
            "{s}'s tokenizer is another ({d} ids, identity {s}): name another --recall-dir";
        const args = .{ s.dir, name, vocab, digest, s.name, s.identity.vocab, &mine };
        problem.* = try std.fmt.allocPrint(a, refusal, args);
        return false;
    }

    /// The directory's lock: a second server with the same record reads it as it was and writes nothing.
    fn takeLock(s: *Store) void {
        var buf: [1100]u8 = undefined;
        const path = std.fmt.bufPrintSentinel(&buf, "{s}/lock", .{s.dir}, 0) catch return s.readOnly();
        const fd = platform.c.open(path, .{ .ACCMODE = .RDWR, .CREAT = true, .NOFOLLOW = true }, @as(platform.c.mode_t, 0o600));
        if (fd < 0) return s.readOnly();
        if (flock(fd, lock_ex | lock_nb) != 0) {
            _ = platform.c.close(fd);
            return s.readOnly();
        }
        s.lock_fd = fd;
    }

    fn readOnly(s: *Store) void {
        s.writes = false;
    }

    /// Chunk files in order from the first: each full but the last, ids read up to the last mark; the rest cut away.
    fn load(s: *Store) !void {
        var seqs: std.ArrayList(u64) = .empty;
        defer seqs.deinit(s.gpa);
        try s.listChunks(&seqs);
        if (seqs.items.len == 0) return;
        s.seq0 = seqs.items[0];
        var pending: std.ArrayList(u32) = .empty;
        defer pending.deinit(s.gpa);
        const ids = try s.gpa.alloc(u32, 1 << 16);
        defer s.gpa.free(ids);
        for (seqs.items, 0..) |seq, k| {
            if (seq != s.seq0 + k) break; // a gap: the files past it go
            const n = s.readChunk(seq, &pending, ids) catch break;
            if (n < rr.chunk_tokens) break; // a short file is the last
        }
        if (s.writes) s.cut(seqs.items);
        while (s.shared.record.chunks.items.len > s.cap_chunks) _ = s.drop();
    }

    /// One chunk file's ids into the record a segment at a time; its count of ids (error: a bad header or id).
    fn readChunk(s: *Store, seq: u64, pending: *std.ArrayList(u32), ids: []u32) !usize {
        var buf: [1100]u8 = undefined;
        const path = try s.chunkPath(&buf, seq);
        const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
        if (fd < 0) return error.RecallRead;
        defer _ = platform.c.close(fd);
        const size = platform.c.lseek(fd, 0, platform.c.SEEK.END);
        if (size < head_bytes) return error.RecallRead;
        var head: [head_bytes]u8 = undefined;
        if (!readAt(fd, &head, 0) or !std.mem.eql(u8, &head, &s.header())) return error.RecallRead;
        const count: usize = @intCast(@divTrunc(size - head_bytes, 4));
        if (count > rr.chunk_tokens) return error.RecallRead;
        var at: usize = 0;
        const r = &s.shared.record;
        while (at < count) {
            const n = @min(ids.len, count - at);
            if (!readAt(fd, std.mem.sliceAsBytes(ids[0..n]), head_bytes + 4 * at)) return error.RecallRead;
            for (ids[0..n]) |t| {
                if (t == rr.mark) {
                    if (!r.room(pending.items.len)) return error.RecallRead;
                    try r.add(pending.items);
                    pending.clearRetainingCapacity();
                } else if (t < s.identity.vocab) try pending.append(s.gpa, t) else return error.RecallRead;
            }
            at += n;
        }
        return count;
    }

    /// The files cut to what the record read: the last one's torn tail truncated, the files after it deleted.
    fn cut(s: *Store, seqs: []const u64) void {
        const r = &s.shared.record;
        const chunks: usize = (@as(usize, r.end) + rr.chunk_tokens - 1) / rr.chunk_tokens;
        for (seqs, 0..) |seq, k| {
            var buf: [1100]u8 = undefined;
            const path = s.chunkPath(&buf, seq) catch continue;
            if (k >= chunks or seq != s.seq0 + k) {
                _ = platform.c.unlink(path);
            } else if (k + 1 == chunks) {
                const keep = head_bytes + 4 * (@as(u64, r.end) - @as(u64, k) * rr.chunk_tokens);
                const fd = platform.c.open(path, .{ .ACCMODE = .WRONLY, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
                if (fd >= 0) {
                    _ = platform.c.ftruncate(fd, @intCast(keep));
                    _ = platform.c.close(fd);
                }
            }
        }
    }

    fn chunkPath(s: *const Store, buf: []u8, seq: u64) ![:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.ids", .{ s.dir, seq }, 0);
    }

    /// The chunk files' numbers, ascending.
    fn listChunks(s: *Store, out: *std.ArrayList(u64)) !void {
        const d = platform.c.opendir(s.dir) orelse return error.RecallRead;
        defer _ = platform.c.closedir(d);
        while (platform.c.readdir(d)) |ent| {
            const name = std.mem.sliceTo(&ent.name, 0);
            if (name.len != 20 or !std.mem.endsWith(u8, name, ".ids")) continue;
            const seq = std.fmt.parseInt(u64, name[0..16], 16) catch continue;
            try out.append(s.gpa, seq);
        }
        std.mem.sort(u64, out.items, {}, std.sort.asc(u64));
    }

    /// The oldest chunk freed and its file deleted.
    fn drop(s: *Store) bool {
        const r = &s.shared.record;
        if (!r.dropOldest()) return false;
        if (s.writes) {
            var buf: [1100]u8 = undefined;
            if (s.chunkPath(&buf, s.seq0 + r.first - 1)) |path| {
                _ = platform.c.unlink(path);
            } else |_| {}
        }
        return true;
    }

    /// A prompt read from its resume point: the ids an earlier request's state did not cover, less the overlap.
    pub fn readPrompt(s: *Store, prompt: []const u32, resumed: u32) void {
        s.shared.lock();
        defer s.shared.unlock();
        s.add(prompt[@min(prompt.len, resumed) -| overlap..]);
    }

    /// A finished reply, after its prompt's last `overlap` ids so the keys across the join are found.
    pub fn readReply(s: *Store, prompt: []const u32, reply: []const u32) void {
        if (reply.len == 0) return;
        const seg = std.mem.concat(s.gpa, u32, &.{ prompt[prompt.len -| overlap..], reply }) catch return;
        defer s.gpa.free(seg);
        s.shared.lock();
        defer s.shared.unlock();
        s.add(seg);
    }

    /// The runs of `seg` the record lacks, each with its overlaps (the caller holds the lock).
    fn add(s: *Store, seg: []const u32) void {
        if (!s.writes or s.failed) return;
        const r = &s.shared.record;
        var fresh: usize = 0; // where the ids not yet written start
        var i: usize = 0;
        while (i + rr.key_len <= seg.len) {
            const run = r.heldRun(seg[i..]);
            if (run.len >= stride) {
                var j = i; // the run reaches back past i, to the last id decided
                var q = run.at;
                while (j > fresh and q > r.base() and r.id(q - 1) == seg[j - 1]) : ({
                    j -= 1;
                    q -= 1;
                }) {}
                if (i + run.len - j >= min_run) {
                    s.put(seg, fresh, j);
                    s.held += i + run.len - j;
                    fresh = i + run.len;
                    i = fresh;
                    continue;
                }
            }
            i += stride;
        }
        s.put(seg, fresh, seg.len);
    }

    /// The ids `seg[from..to]` with their overlaps, added and written behind the free-disk floor.
    fn put(s: *Store, seg: []const u32, from: usize, to: usize) void {
        if (to <= from) return;
        const piece = seg[from -| overlap..@min(seg.len, to + overlap)];
        const r = &s.shared.record;
        if (piece.len < rr.key_len or !r.room(piece.len)) return;
        for (piece) |t| if (t >= s.identity.vocab) return; // a load stops at such an id: it never goes in
        const bytes = 4 * (@as(u64, piece.len) + 1) + head_bytes;
        if (s.admission.reserve(s.dir, bytes) != .ready) return;
        const start = r.end;
        r.add(piece) catch {
            s.admission.finish(s.dir, bytes, false);
            s.failed = true;
            return;
        };
        const ok = s.write(start, r.end);
        s.admission.finish(s.dir, bytes, ok);
        if (!ok) s.failed = true;
        s.added += piece.len;
        while (r.chunks.items.len > s.cap_chunks) _ = s.drop();
    }

    /// The record's positions [start, end) appended to their chunk files.
    fn write(s: *Store, start: u32, end: u32) bool {
        const r = &s.shared.record;
        var ids: [4096]u32 = undefined;
        var p = start;
        while (p < end) {
            const k = p / rr.chunk_tokens;
            if (!s.fileFor(k)) return false;
            const stop = @min(end, (k + 1) * rr.chunk_tokens);
            while (p < stop) {
                const n = @min(stop - p, ids.len);
                for (ids[0..n], 0..) |*t, i| t.* = r.id(p + @as(u32, @intCast(i)));
                writeAll(s.file, std.mem.sliceAsBytes(ids[0..n])) catch return false;
                p += n;
            }
        }
        return true;
    }

    /// The file for record chunk `k`, open to append: the one on disk, or a new one with its header.
    fn fileFor(s: *Store, k: u32) bool {
        if (s.file >= 0 and s.file_chunk == k) return true;
        s.closeFile();
        var buf: [1100]u8 = undefined;
        const path = s.chunkPath(&buf, s.seq0 + k) catch return false;
        var flags: platform.c.O = .{ .ACCMODE = .WRONLY, .APPEND = true, .NOFOLLOW = true };
        var fd = platform.c.open(path, flags, @as(platform.c.mode_t, 0));
        if (fd < 0) {
            flags.CREAT = true;
            flags.EXCL = true;
            fd = platform.c.open(path, flags, @as(platform.c.mode_t, 0o600));
            if (fd < 0) return false;
            const head = s.header();
            writeAll(fd, &head) catch {
                _ = platform.c.close(fd);
                return false;
            };
        }
        s.file = fd;
        s.file_chunk = k;
        return true;
    }

    /// A chunk file's header for this record: magic, version, vocabulary size, the identity's first 16 bytes.
    fn header(s: *const Store) [head_bytes]u8 {
        var head: [head_bytes]u8 = undefined;
        @memcpy(head[0..8], magic);
        std.mem.writeInt(u32, head[8..12], version, .little);
        std.mem.writeInt(u32, head[12..16], s.identity.vocab, .little);
        @memcpy(head[16..32], s.identity.digest[0..16]);
        return head;
    }

    /// Chunk files deleted; record, rates and counters emptied (identity, lock stay). Unlinked: `why` names one left.
    pub fn empty(s: *Store) error{ ReadOnly, Unlinked }!void {
        s.shared.lock();
        defer s.shared.unlock();
        if (!s.writes) return error.ReadOnly;
        s.closeFile();
        const r = &s.shared.record;
        const next = s.seq0 + (@as(u64, r.end) + rr.chunk_tokens) / rr.chunk_tokens;
        var seqs: std.ArrayList(u64) = .empty;
        defer seqs.deinit(s.gpa);
        s.problem = "";
        s.listChunks(&seqs) catch s.note("{s} could not be listed", .{s.dir});
        for (seqs.items) |seq| {
            var buf: [1100]u8 = undefined;
            const path = s.chunkPath(&buf, seq) catch continue;
            const rc = platform.c.unlink(path);
            const e = platform.c.errno(rc);
            if (rc == 0 or e == .NOENT or s.problem.len > 0) continue;
            s.note("{s} could not be removed ({s})", .{ path, @tagName(e) });
        }
        r.clear();
        s.shared.rates = .{};
        s.added = 0;
        s.held = 0;
        s.seq0 = @max(next, if (seqs.items.len > 0) seqs.items[seqs.items.len - 1] + 1 else 0);
        s.failed = s.problem.len > 0;
        if (s.failed) return error.Unlinked;
    }

    /// Why the last empty failed, for its reply.
    pub fn why(s: *const Store) []const u8 {
        return s.problem;
    }

    fn note(s: *Store, comptime fmt: []const u8, args: anytype) void {
        s.problem = std.fmt.bufPrint(&s.problem_buf, fmt, args) catch "a chunk file could not be removed";
    }

    pub fn stats(s: *Store) Stats {
        s.shared.lock();
        defer s.shared.unlock();
        const r = &s.shared.record;
        const rates = &s.shared.rates;
        var out: Stats = .{
            .writes = s.writes and !s.failed,
            .left = s.problem.len > 0,
            .read_only = s.read_only,
            .ids = r.held(),
            .disk_bytes = @as(u64, r.chunks.items.len) * head_bytes + 4 * @as(u64, r.held()),
            .ram_bytes = r.ramBytes(),
            .cap_bytes = @as(u64, s.cap_chunks) * chunk_bytes,
            .added = s.added,
            .held = s.held,
            .rates = undefined,
            .probe_every = rates.probe_every,
            .proposals = rates.proposals,
            .proposed = rates.proposed,
            .accepted = rates.accepted,
        };
        for (&out.rates, 0..) |*x, b| x.* = rates.rate(b);
        return out;
    }
};

/// `dir` and its missing parents, each made 0700.
fn makePath(dir: [:0]const u8) !void {
    var buf: [1100]u8 = undefined;
    if (dir.len >= buf.len) return error.RecallDir;
    @memcpy(buf[0..dir.len], dir);
    buf[dir.len] = 0;
    for (1..dir.len) |i| if (buf[i] == '/') {
        buf[i] = 0;
        _ = platform.c.mkdir(@ptrCast(&buf), 0o700);
        buf[i] = '/';
    };
    const rc = platform.c.mkdir(dir, 0o700);
    if (rc != 0 and platform.c.errno(rc) != .EXIST) return error.RecallDir;
}

fn readSmall(a: Allocator, path: [:0]const u8) ![]u8 {
    const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(platform.c.mode_t, 0));
    if (fd < 0) return if (platform.c.errno(fd) == .NOENT) error.FileNotFound else error.RecallRead;
    defer _ = platform.c.close(fd);
    const size = platform.c.lseek(fd, 0, platform.c.SEEK.END);
    if (size < 0 or size > 4096) return error.RecallRead;
    const out = try a.alloc(u8, @intCast(size));
    if (!readAt(fd, out, 0)) return error.RecallRead;
    return out;
}

/// `bytes` as the file at `path` (0600), written whole under another name first.
fn writeSmall(path: [:0]const u8, bytes: []const u8) !void {
    var buf: [1200]u8 = undefined;
    const part = try std.fmt.bufPrintSentinel(&buf, "{s}.part", .{path}, 0);
    const flags: platform.c.O = .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .NOFOLLOW = true };
    const fd = platform.c.open(part, flags, @as(platform.c.mode_t, 0o600));
    if (fd < 0) return error.RecallWrite;
    {
        defer _ = platform.c.close(fd);
        try writeAll(fd, bytes);
        if (platform.c.fsync(fd) != 0) return error.RecallWrite;
    }
    if (platform.c.rename(part, path) != 0) return error.RecallWrite;
}

fn writeAll(fd: c_int, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = platform.c.write(fd, bytes.ptr + done, bytes.len - done);
        if (n <= 0) return error.RecallWrite;
        done += @intCast(n);
    }
}

fn readAt(fd: c_int, dest: []u8, at: u64) bool {
    var done: usize = 0;
    while (done < dest.len) {
        const n = platform.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return true;
}

/// Where Recall keeps records unless --recall-dir names one: $HOME/.cache/tensorfold/recall/<tokenizer identity>.
pub fn defaultDir(a: Allocator, identity: Identity) ![]const u8 {
    const home = platform.c.getenv("HOME") orelse return error.RecallDir;
    const short = identity.short();
    return std.fmt.allocPrint(a, "{s}/.cache/tensorfold/recall/{s}", .{ std.mem.span(home), &short });
}

test {
    _ = @import("recall_store_test.zig");
}
