//! Local BF16 Engram reader: 192-byte safetensors rows, aligned O_DIRECT extents, no fallback or decode.

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const engram = @import("engram.zig");

/// Logical block size an O_DIRECT read must align its offset, length and buffer to.
pub const alignment = 4096;
/// One embedding row: 96 BF16 values, as safetensors lays them out.
pub const row_bytes = 192;
pub const width = row_bytes / 2;
/// A header larger than this is refused before allocating memory for it.
pub const max_header_bytes = 16 * 1024 * 1024;

/// A positioned read with a caller context; `sysPread` supplies it, fixtures inject a recording one.
pub const Pread = *const fn (ctx: *anyopaque, fd: linux.fd_t, buf: [*]u8, len: usize, off: i64) usize;

pub const Error = std.mem.Allocator.Error || error{
    BadTable, // geometry, safetensors header or row width is not the BF16 table
    MissingTensor, // the named tensor is not in the safetensors header
    OpenFailed, // O_DIRECT open refused (never a buffered retry)
    StatFailed,
    ReadFailed, // pread failed, including O_DIRECT EINVAL (never a buffered retry)
    ShortRow, // a selected row extends past end of file (truncated data)
    BadOutput, // caller's output has the wrong length
};

/// File geometry a plan builds against; `data_start` (row 0) may be unaligned, extents are not.
pub const Geometry = struct {
    fd: linux.fd_t,
    size: u64,
    data_start: u64,
    count: u64,
    ctx: *anyopaque = undefined,
    read: Pread,

    /// Validate `row` and return its (possibly unaligned) file offset.
    pub fn rowOffset(self: Geometry, row: u32) Error!u64 {
        if (row >= self.count) return error.BadTable;
        const begin = std.math.add(u64, self.data_start, std.math.mul(u64, row, row_bytes) catch return error.BadTable) catch return error.BadTable;
        if (begin + row_bytes > self.size) return error.ShortRow;
        return begin;
    }
};

/// One merged extent: aligned file `[off, off+len)` into the arena at aligned `at`.
pub const Extent = struct {
    off: u64,
    len: usize,
    at: usize,
    first: usize = 0,
    count: usize = 0,
};

/// A reusable read plan for one batch of windows: build once, execute as often as needed.
pub const Plan = struct {
    /// Selected table row for each output, in token/head order.
    rows: []u32,
    /// Arena byte offset of each output's 192 bytes, in token/head order.
    at: []usize,
    /// Aligned, sorted, merged reads to issue.
    ext: []Extent,
    /// Padded length of the 4 KiB-aligned arena.
    arena_len: usize = 0,

    pub fn deinit(self: *Plan, a: std.mem.Allocator) void {
        a.free(self.rows);
        a.free(self.at);
        a.free(self.ext);
        self.* = undefined;
    }
};

const Pick = struct { off: u64, out: usize };

fn lessPick(_: void, x: Pick, y: Pick) bool {
    return if (x.off != y.off) x.off < y.off else x.out < y.out;
}

/// Hash windows into rows, then build sorted, merged, aligned extents; deterministic.
pub fn buildPlan(a: std.mem.Allocator, g: Geometry, map: engram.TokenMap, hash: engram.Layer, windows: []const [engram.order]u32) Error!Plan {
    const outputs = std.math.mul(usize, windows.len, engram.columns) catch return error.BadTable;
    var rows = try a.alloc(u32, outputs);
    errdefer a.free(rows);
    var at = try a.alloc(usize, outputs);
    errdefer a.free(at);
    var picks: std.ArrayList(Pick) = .empty;
    defer picks.deinit(a);
    try picks.ensureTotalCapacity(a, outputs);

    for (windows, 0..) |window, token| {
        for (hash.hash(map, window), 0..) |row, head| {
            const out = token * engram.columns + head;
            rows[out] = row;
            const off = try g.rowOffset(row);
            picks.appendAssumeCapacity(.{ .off = off, .out = out });
        }
    }
    std.mem.sort(Pick, picks.items, {}, lessPick);

    var ext: std.ArrayList(Extent) = .empty;
    errdefer ext.deinit(a);
    var arena: usize = 0;
    for (picks.items) |p| {
        const lo = std.mem.alignBackward(u64, p.off, alignment);
        const hi = std.mem.alignForward(u64, p.off + row_bytes, alignment);
        if (ext.items.len > 0) {
            const last = &ext.items[ext.items.len - 1];
            if (lo <= last.off + last.len) {
                const end = @max(last.off + last.len, hi);
                arena += @intCast(end - (last.off + last.len));
                last.len = @intCast(end - last.off);
                last.count += 1;
                at[p.out] = last.at + @as(usize, @intCast(p.off - last.off));
                continue;
            }
        }
        const len: usize = @intCast(hi - lo);
        // `lo` and `hi` are aligned, so `arena` stays 4 KiB-aligned.
        try ext.append(a, .{ .off = lo, .len = len, .at = arena, .first = p.out, .count = 1 });
        at[p.out] = arena + @as(usize, @intCast(p.off - lo));
        arena += len;
    }
    return .{
        .rows = rows,
        .at = at,
        .ext = try ext.toOwnedSlice(a),
        .arena_len = arena,
    };
}

/// One aligned O_DIRECT read: EINVAL is fatal (no fallback), EOF short reads are accepted.
fn readExtent(g: Geometry, buf: []align(alignment) u8, off: u64, len: usize) Error!void {
    const want: usize = @intCast(@min(@as(u64, len), g.size - off));
    var done: usize = 0;
    while (done < want) {
        const at = std.mem.alignBackward(usize, done, alignment);
        const ask = len - at; // len is a multiple of alignment
        const rc = g.read(g.ctx, g.fd, buf.ptr + at, ask, @intCast(off + at));
        switch (linux.errno(rc)) {
            .SUCCESS => {},
            .INTR => continue,
            else => return error.ReadFailed,
        }
        if (rc == 0) return error.ShortRow;
        const next = at + @as(usize, @intCast(rc));
        if (next <= done) return error.ShortRow;
        done = next;
    }
}

/// Execute `plan` into raw BF16 words `[token][head][width]`, in that order.
pub fn execute(a: std.mem.Allocator, g: Geometry, plan: Plan, output: []u16) Error!void {
    if (output.len != plan.rows.len * width) return error.BadOutput;
    if (plan.rows.len == 0) return;
    const arena = try a.alignedAlloc(u8, .fromByteUnits(alignment), plan.arena_len);
    defer a.free(arena);
    try readRows(g, plan, output, arena);
}

fn readRows(g: Geometry, plan: Plan, output: []u16, arena: []align(alignment) u8) Error!void {
    for (plan.ext) |e| {
        const buf: []align(alignment) u8 = @alignCast(arena[e.at..][0..e.len]);
        try readExtent(g, buf, e.off, e.len);
    }
    // Little-endian host: copying bytes preserves each BF16 bit pattern.
    for (plan.at, 0..) |src, i| {
        @memcpy(std.mem.asBytes(output[i * width ..][0..width]), arena[src..][0..row_bytes]);
    }
}

/// In-flight direct read; keep the plan, file, and destination alive through deinit.
pub const Prefetch = struct {
    arena: []align(alignment) u8,
    future: std.Io.Future(Error!void),

    pub fn start(a: std.mem.Allocator, io: std.Io, g: Geometry, plan: Plan, output: []u16) !Prefetch {
        if (output.len != plan.rows.len * width) return error.BadOutput;
        const arena = try a.alignedAlloc(u8, .fromByteUnits(alignment), plan.arena_len);
        errdefer a.free(arena);
        return .{ .arena = arena, .future = try io.concurrent(readRows, .{ g, plan, output, arena }) };
    }

    pub fn finish(self: *Prefetch, io: std.Io) Error!void {
        try self.future.await(io);
    }

    pub fn deinit(self: *Prefetch, a: std.mem.Allocator, io: std.Io) void {
        self.finish(io) catch {};
        a.free(self.arena);
        self.* = undefined;
    }
};

/// Batch read: hash the windows, plan once, then gather their 192-byte rows.
pub fn gatherRows(a: std.mem.Allocator, g: Geometry, map: engram.TokenMap, hash: engram.Layer, windows: []const [engram.order]u32, output: []u16) Error!void {
    var plan = try buildPlan(a, g, map, hash, windows);
    defer plan.deinit(a);
    try execute(a, g, plan, output);
}

pub fn tokenElements() usize {
    return engram.columns * width;
}

/// Production wrapper: geometry plus the released vocabulary map and layer hash.
pub const Table = struct {
    geometry: Geometry,
    map: engram.TokenMap,
    hash: engram.Layer,

    pub fn gather(self: Table, a: std.mem.Allocator, windows: []const [engram.order]u32, output: []u16) Error!void {
        return gatherRows(a, self.geometry, self.map, self.hash, windows, output);
    }

    pub fn elementsPerToken(self: Table) usize {
        _ = self;
        return tokenElements();
    }

    pub fn close(self: *Table) void {
        _ = linux.close(self.geometry.fd);
        self.* = undefined;
    }
};

/// The real positioned read: the context is unused.
fn sysPread(_: *anyopaque, fd: linux.fd_t, buf: [*]u8, len: usize, off: i64) usize {
    return linux.pread(fd, buf, len, off);
}

var no_ctx: u8 = 0; // stable, never-dereferenced context for the system reader

/// Read `len` aligned bytes at `off` of an O_DIRECT `fd` into an aligned buffer.
fn readDirect(fd: linux.fd_t, buf: []align(alignment) u8, off: u64, len: usize, size: u64) Error!void {
    const g: Geometry = .{ .fd = fd, .size = size, .data_start = 0, .count = 1, .ctx = &no_ctx, .read = sysPread };
    return readExtent(g, buf, off, len);
}

/// Open a one-tensor BF16 safetensors table O_DIRECT; even the JSON header is read direct.
pub fn openDirect(a: std.mem.Allocator, path: [:0]const u8, tensor: []const u8, map: engram.TokenMap, hash: engram.Layer) Error!Table {
    const rc = linux.openat(linux.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .DIRECT = true }, 0);
    if (linux.errno(rc) != .SUCCESS) return error.OpenFailed;
    const fd: linux.fd_t = @intCast(rc);
    errdefer _ = linux.close(fd);
    var st: linux.Statx = undefined;
    if (linux.errno(linux.statx(fd, "", linux.AT.EMPTY_PATH, .{ .SIZE = true }, &st)) != .SUCCESS) return error.StatFailed;
    const size = st.size;
    if (size < 8) return error.BadTable;

    // First aligned block carries the header length; one bounded allocation follows.
    var first_block: [alignment]u8 align(alignment) = undefined;
    try readDirect(fd, &first_block, 0, alignment, size);
    const header_len = std.mem.readInt(u64, first_block[0..8], .little);
    if (header_len == 0 or header_len > max_header_bytes or header_len > size - 8) return error.BadTable;
    const header_end = std.math.add(u64, 8, header_len) catch return error.BadTable;
    const json_len: usize = @intCast(header_len);
    const buf_len = std.mem.alignForward(u64, header_end, alignment);
    const header_buf = try a.alignedAlloc(u8, .fromByteUnits(alignment), @intCast(buf_len));
    defer a.free(header_buf); // one allocation, one owner: no free-then-replace on OOM
    try readDirect(fd, header_buf, 0, header_buf.len, size);
    const json = header_buf[8..][0..json_len];

    const parsed = std.json.parseFromSlice(std.json.Value, a, json, .{}) catch return error.BadTable;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadTable;
    const entry = parsed.value.object.get(tensor) orelse return error.MissingTensor;
    if (entry != .object) return error.BadTable;
    const dtype = entry.object.get("dtype") orelse return error.BadTable;
    if (dtype != .string or !std.mem.eql(u8, dtype.string, "BF16")) return error.BadTable;

    const offsets = entry.object.get("data_offsets") orelse return error.BadTable;
    if (offsets != .array or offsets.array.items.len != 2) return error.BadTable;
    const begin = offsets.array.items[0];
    const last = offsets.array.items[1];
    if (begin != .integer or last != .integer) return error.BadTable;
    if (begin.integer < 0 or last.integer < 0) return error.BadTable; // no negative offsets
    const lo: u64 = @intCast(begin.integer);
    const hi: u64 = @intCast(last.integer);
    if (hi < lo) return error.BadTable;
    const bytes = hi - lo;
    if (bytes == 0 or bytes % row_bytes != 0) return error.BadTable;
    const count = bytes / row_bytes;

    // Shape and byte range must describe [count, width] BF16 and fit the reported size.
    const shape = entry.object.get("shape") orelse return error.BadTable;
    if (shape != .array or shape.array.items.len != 2) return error.BadTable;
    for (shape.array.items) |dim| if (dim != .integer or dim.integer < 0) return error.BadTable;
    if (count > std.math.maxInt(i64)) return error.BadTable;
    if (shape.array.items[0].integer != @as(i64, @intCast(count)) or
        shape.array.items[1].integer != @as(i64, width)) return error.BadTable;
    const expected = std.math.mul(u64, count, row_bytes) catch return error.BadTable;
    if (expected != bytes) return error.BadTable;

    const data_start = std.math.add(u64, header_end, lo) catch return error.BadTable;
    const payload_end = std.math.add(u64, data_start, bytes) catch return error.BadTable;
    if (payload_end > size or count > std.math.maxInt(u32)) return error.BadTable;

    return .{
        .geometry = .{ .fd = fd, .size = size, .data_start = data_start, .count = count, .ctx = &no_ctx, .read = sysPread },
        .map = map,
        .hash = hash,
    };
}

test "a plan's extents are aligned and cover every selected 192-byte row" {
    const a = std.testing.allocator;
    const g: Geometry = .{ .fd = -1, .size = 1 << 20, .data_start = 4096 - 100, .count = 1000, .read = undefined };
    const lookup = [4]u32{ 0, 1, 2, 3 };
    const map: engram.TokenMap = .{ .bytes = std.mem.asBytes(&lookup), .pad = 0 };
    const offsets = [engram.columns]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    const hash: engram.Layer = .{ .multipliers = .{ 3, 5, 7 }, .primes = @splat(17), .offsets = offsets };
    var plan = try buildPlan(a, g, map, hash, &.{ .{ 1, 2, 3 }, .{ 3, 2, 1 } });
    defer plan.deinit(a);
    try std.testing.expectEqual(@as(usize, 2 * engram.columns), plan.rows.len);
    for (plan.ext) |e| {
        try std.testing.expectEqual(@as(u64, 0), e.off % alignment);
        try std.testing.expectEqual(@as(usize, 0), e.len % alignment);
        try std.testing.expectEqual(@as(usize, 0), e.at % alignment);
        try std.testing.expect(e.len > 0);
    }
    for (plan.rows, plan.at) |row, src| {
        var found = false;
        for (plan.ext) |e| {
            const off = g.data_start + @as(u64, row) * row_bytes;
            if (off >= e.off and off + row_bytes <= e.off + e.len and src == e.at + @as(usize, @intCast(off - e.off))) found = true;
        }
        try std.testing.expect(found);
    }
}
