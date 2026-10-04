//! A rank's tensors onto its GPU: local files mapped zero-copy, missing ranges pulled from a peer, sizes and SHA-256 checked.
const std = @import("std");
const checkpoint = @import("checkpoint.zig");
const plan_mod = @import("plan.zig");

const Allocator = std.mem.Allocator;
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const page = std.heap.page_size_min;

/// A byte range of one checkpoint file.
pub const Range = struct { file: u32, offset: u64, len: u64 };

/// The ranges rank `r` reads: a column split's own rows, everything else whole (row splits are gathered after mapping).
pub fn needs(a: Allocator, p: *const plan_mod.Plan, c: checkpoint.Checkpoint, r: u32) ![]Range {
    var out: std.ArrayList(Range) = .empty;
    for (c.tensors, 0..) |t, i| {
        if (p.bytesOn(t, i, r) == 0) continue;
        var rg: Range = .{ .file = t.file, .offset = t.start, .len = t.bytes };
        if (p.place[i].kind == .column) {
            const st = p.stages[p.place[i].stage];
            const mine = st.slices[r - st.first];
            const row_bytes = t.bytes / t.shape[0];
            rg.offset += t.shape[0] * mine.begin / p.opts.slices * row_bytes;
            rg.len = t.shape[0] * mine.len() / p.opts.slices * row_bytes;
        }
        try out.append(a, rg);
    }
    std.mem.sort(Range, out.items, {}, struct {
        fn less(_: void, x: Range, y: Range) bool {
            return x.file < y.file or (x.file == y.file and x.offset < y.offset);
        }
    }.less);
    var merged: std.ArrayList(Range) = .empty;
    for (out.items) |rg| {
        if (merged.items.len > 0) {
            const last = &merged.items[merged.items.len - 1];
            if (last.file == rg.file and rg.offset <= last.offset + last.len) {
                last.len = @max(last.len, rg.offset + rg.len - last.offset);
                continue;
            }
        }
        try merged.append(a, rg);
    }
    return merged.items;
}

/// A peer that holds the file: copies `out.len` bytes from `offset` and returns their SHA-256 as it computed it.
pub const Source = struct {
    ptr: *anyopaque,
    pull: *const fn (ptr: *anyopaque, file: u32, offset: u64, out: []u8) anyerror![32]u8,
};

/// Where mapped bytes go: Metal wraps each page-aligned view in a no-copy buffer; only `range`'s bytes inside it are valid.
pub const Sink = struct {
    ptr: *anyopaque,
    run: *const fn (ptr: *anyopaque, range: Range, view_offset: u64, view: []align(page) const u8) anyerror!void,
};

pub const Progress = struct {
    ptr: *anyopaque,
    report: *const fn (ptr: *anyopaque, done: u64, total: u64) void,
};

pub const Report = struct {
    local: u32 = 0,
    pulled: u32 = 0,
    pulled_bytes: u64 = 0,
    mapped_bytes: u64 = 0,
    hashed: u32 = 0,
    /// The file mappings the sink's buffers point into; they must outlive the model.
    maps: std.ArrayList([]align(page) u8) = .empty,

    pub fn release(r: *Report, a: Allocator) void {
        for (r.maps.items) |m| std.posix.munmap(m);
        r.maps.deinit(a);
    }
};

pub const Options = struct {
    files: []const checkpoint.File,
    /// Bytes a pull asks for at once (the mailbox's reply half).
    chunk: usize = 64 << 20,
    /// Hash whole local files against the manifest's SHA-256 (cached by size and mtime in .tensorfold/).
    verify: bool = true,
};

pub const Error = error{ MissingShard, HashMismatch, SizeMismatch };

const state_dir = ".tensorfold";

/// Make every range of `ranges` resident and hand it to `sink`; `dir` is this node's model folder.
pub fn load(io: Io, a: Allocator, dir: Io.Dir, o: Options, ranges: []const Range, source: ?Source, sink: Sink, progress: ?Progress) !Report {
    var rep: Report = .{};
    var total: u64 = 0;
    for (ranges) |r| total += r.len;
    var done: u64 = 0;
    var i: usize = 0;
    try dir.createDirPath(io, state_dir);
    while (i < ranges.len) {
        var j = i;
        while (j < ranges.len and ranges[j].file == ranges[i].file) j += 1;
        const f = o.files[ranges[i].file];
        const complete = if (dir.statFile(io, f.name, .{})) |st| st.size == f.size else |_| false;
        const pieces = ranges[i..j];
        if (complete and !try isSparse(io, a, dir, f)) {
            if (o.verify and f.sha256 != null) rep.hashed += @intFromBool(try verifyWhole(io, a, dir, f));
            rep.local += 1;
        } else {
            const src = source orelse return error.MissingShard;
            rep.pulled_bytes += try pull(io, a, dir, o, ranges[i].file, f, pieces, src, progress, &done, total);
            rep.pulled += 1;
        }
        const mem = try mapInto(io, dir, f, pieces, sink, &rep.mapped_bytes);
        try rep.maps.append(a, mem);
        for (pieces) |r| done += r.len;
        if (progress) |p| p.report(p.ptr, done, total);
        i = j;
    }
    return rep;
}

fn sidecar(a: Allocator, f: checkpoint.File, ext: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, state_dir ++ "/{s}.{s}", .{ f.name, ext });
}

/// A file this node only partly holds has a ranges record beside it.
fn isSparse(io: Io, a: Allocator, dir: Io.Dir, f: checkpoint.File) !bool {
    const path = try sidecar(a, f, "ranges");
    defer a.free(path);
    _ = dir.statFile(io, path, .{}) catch return false;
    return true;
}

/// The whole file's SHA-256 against the manifest, remembered by size and mtime; true when it hashed now.
fn verifyWhole(io: Io, a: Allocator, dir: Io.Dir, f: checkpoint.File) !bool {
    const st = try dir.statFile(io, f.name, .{});
    var key_buf: [96]u8 = undefined;
    const key = try std.fmt.bufPrint(&key_buf, "{d} {d} {x}", .{ st.size, st.mtime.nanoseconds, f.sha256.? });
    const path = try sidecar(a, f, "sha256");
    defer a.free(path);
    if (dir.readFileAlloc(io, path, a, .limited(256))) |seen| {
        defer a.free(seen);
        if (std.mem.eql(u8, seen, key)) return false;
    } else |_| {}
    const file = try dir.openFile(io, f.name, .{});
    defer file.close(io);
    var h = Sha256.init(.{});
    const buf = try a.alloc(u8, 16 << 20);
    defer a.free(buf);
    var at: u64 = 0;
    while (at < st.size) {
        const n = try file.readPositionalAll(io, buf[0..@min(buf.len, st.size - at)], at);
        if (n == 0) return error.SizeMismatch;
        h.update(buf[0..n]);
        at += n;
    }
    if (!std.mem.eql(u8, &h.finalResult(), &f.sha256.?)) return error.HashMismatch;
    try dir.writeFile(io, .{ .sub_path = path, .data = key });
    return true;
}

/// Pull the ranges this sparse copy still lacks, checking each chunk's SHA-256, and record what landed.
fn pull(io: Io, a: Allocator, dir: Io.Dir, o: Options, index: u32, f: checkpoint.File, pieces: []const Range, src: Source, progress: ?Progress, done: *u64, total: u64) !u64 {
    const record = try sidecar(a, f, "ranges");
    defer a.free(record);
    var have: std.ArrayList([2]u64) = .empty;
    defer have.deinit(a);
    if (dir.readFileAlloc(io, record, a, .limited(64 << 20))) |text| {
        defer a.free(text);
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |line| {
            var it = std.mem.tokenizeScalar(u8, line, ' ');
            const off = std.fmt.parseInt(u64, it.next() orelse continue, 10) catch continue;
            const len = std.fmt.parseInt(u64, it.next() orelse continue, 10) catch continue;
            try have.append(a, .{ off, len });
        }
    } else |_| {}
    const file = dir.openFile(io, f.name, .{ .mode = .read_write }) catch try dir.createFile(io, f.name, .{ .read = true, .truncate = false });
    defer file.close(io);
    if (try file.length(io) != f.size) try file.setLength(io, f.size);
    const log = try dir.createFile(io, record, .{ .truncate = false });
    defer log.close(io);
    var log_at = try log.length(io);
    const buf = try a.alloc(u8, o.chunk);
    defer a.free(buf);
    var pulled: u64 = 0;
    for (pieces) |r| {
        var at = r.offset;
        while (at < r.offset + r.len) {
            const n: usize = @intCast(@min(o.chunk, r.offset + r.len - at));
            if (covered(have.items, at, n)) {
                at += n;
                continue;
            }
            const sent = try src.pull(src.ptr, index, at, buf[0..n]);
            var got: [32]u8 = undefined;
            Sha256.hash(buf[0..n], &got, .{});
            if (!std.mem.eql(u8, &got, &sent)) return error.HashMismatch;
            try file.writePositionalAll(io, buf[0..n], at);
            var line_buf: [160]u8 = undefined;
            const line = try std.fmt.bufPrint(&line_buf, "{d} {d} {x}\n", .{ at, n, got });
            try log.writePositionalAll(io, line, log_at);
            log_at += line.len;
            try have.append(a, .{ at, n });
            pulled += n;
            at += n;
            if (progress) |p| p.report(p.ptr, done.* + at - r.offset, total);
        }
    }
    return pulled;
}

fn covered(have: []const [2]u64, at: u64, n: usize) bool {
    for (have) |h| if (h[0] <= at and at + n <= h[0] + h[1]) return true;
    return false;
}

/// Map the file once, read-only, and hand each range's page-aligned span to the sink.
fn mapInto(io: Io, dir: Io.Dir, f: checkpoint.File, pieces: []const Range, sink: Sink, mapped: *u64) ![]align(page) u8 {
    const file = try dir.openFile(io, f.name, .{});
    defer file.close(io);
    const size = try file.length(io);
    if (size != f.size) return error.SizeMismatch;
    const len = std.mem.alignForward(usize, @intCast(size), page);
    const mem = try std.posix.mmap(null, len, .{ .READ = true }, .{ .TYPE = .SHARED }, file.handle, 0);
    errdefer std.posix.munmap(mem);
    for (pieces) |r| {
        const begin = std.mem.alignBackward(u64, r.offset, page);
        const end = std.mem.alignForward(u64, r.offset + r.len, page);
        try sink.run(sink.ptr, r, begin, @alignCast(mem[@intCast(begin)..@intCast(end)]));
        mapped.* += r.len;
    }
    return mem;
}

test {
    _ = @import("loader_test.zig");
}
