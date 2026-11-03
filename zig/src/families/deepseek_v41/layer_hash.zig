//! Opt-in rank-specific JSONL receipts of existing per-layer BF16 trace outputs.
const std = @import("std");
const Allocator = std.mem.Allocator;
pub const Phase = enum { prefill, replay, decode };
pub const Request = struct { sequence: u64, sha256: [64]u8 };
pub const Span = struct { slot: u32, start: u64, row0: u32, rows: u32, ids: []const u32 };
const Key = struct { phase: Phase, start: u64, rows: u32, ids_sha256: [64]u8 };
const Window = struct { number: u64, last_layer: u32 };
const Slot = struct { request: Request = .{ .sequence = 0, .sha256 = @splat('0') }, next: u64 = 0, seen: std.AutoHashMapUnmanaged(Key, Window) = .empty };
pub const op_request: i64 = 66;

pub fn tokenSha(ids: []const u32) [64]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (ids) |id| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, id, .little);
        h.update(&bytes);
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

pub fn frame(slot: u32, req: Request) [67]i64 {
    var words: [67]i64 = undefined;
    words[0] = op_request;
    words[1] = slot;
    words[2] = @intCast(req.sequence);
    for (req.sha256, words[3..]) |c, *v| v.* = c;
    return words;
}

pub fn readFrame(words: []const i64) !struct { slot: u32, request: Request } {
    if (words.len != 67 or words[0] != op_request or words[1] < 0 or words[1] >= 64 or words[2] <= 0) return error.BadHashRequest;
    var req: Request = .{ .sequence = @intCast(words[2]), .sha256 = undefined };
    for (words[3..], &req.sha256) |c, *v| {
        if (!((c >= '0' and c <= '9') or (c >= 'a' and c <= 'f'))) return error.BadHashRequest;
        v.* = @intCast(c);
    }
    return .{ .slot = @intCast(words[1]), .request = req };
}

pub const Writer = struct {
    gpa: Allocator,
    io: std.Io,
    rank: u32,
    file: std.Io.File,
    offset: u64 = 0,
    next_request: u64 = 0,
    slots: [64]Slot = @splat(.{}),

    /// Null or empty does not allocate or touch the filesystem; each enabled rank owns a separate new file.
    pub fn open(gpa: Allocator, io: std.Io, prefix: ?[]const u8, rank: u32) !?*Writer {
        const path = prefix orelse return null;
        if (path.len == 0) return null;
        const name = try std.fmt.allocPrint(gpa, "{s}.rank{d}.jsonl", .{ path, rank });
        defer gpa.free(name);
        const w = try gpa.create(Writer);
        errdefer gpa.destroy(w);
        const file = try std.Io.Dir.cwd().createFile(io, name, .{ .exclusive = true });
        errdefer file.close(io);
        w.* = .{ .gpa = gpa, .io = io, .rank = rank, .file = file };
        try w.line(.{ .version = 1, .kind = "header", .rank = rank, .algorithm = "sha256", .tensor = "post-only BF16 streams", .token_encoding = "u32 little-endian", .request_zero = "boot or calibration" });
        return w;
    }

    pub fn close(w: *Writer) void {
        for (&w.slots) |*s| s.seen.deinit(w.gpa);
        w.file.close(w.io);
        w.gpa.destroy(w);
    }

    pub fn begin(w: *Writer, slot: u32, ids: []const u32) !Request {
        const seq = try std.math.add(u64, w.next_request, 1);
        if (seq > std.math.maxInt(i64)) return error.HashRequestOverflow;
        w.next_request = seq;
        const req: Request = .{ .sequence = seq, .sha256 = tokenSha(ids) };
        try w.accept(slot, req);
        return req;
    }

    pub fn accept(w: *Writer, slot: u32, req: Request) !void {
        if (slot >= w.slots.len) return error.BadHashSlot;
        const s = &w.slots[slot];
        s.request = req;
        s.next = 0;
        s.seen.clearRetainingCapacity();
        try w.line(.{ .version = 1, .kind = "request", .rank = w.rank, .slot = slot, .request = req.sequence, .request_sha256 = @as([]const u8, &req.sha256) });
    }

    pub fn layer(w: *Writer, phase: Phase, L: u32, span: Span, columns: usize, bytes: []const u8) !void {
        if (span.slot >= w.slots.len or span.rows == 0 or span.ids.len != span.rows) return error.BadHashSpan;
        const row_bytes = try std.math.mul(usize, 2, columns);
        const lo = try std.math.mul(usize, span.row0, row_bytes);
        const n = try std.math.mul(usize, span.rows, row_bytes);
        if (lo > bytes.len or n > bytes.len - lo) return error.BadHashSpan;
        const s = &w.slots[span.slot];
        const key: Key = .{ .phase = phase, .start = span.start, .rows = span.rows, .ids_sha256 = tokenSha(span.ids) };
        const item = try s.seen.getOrPut(w.gpa, key);
        if (!item.found_existing or L <= item.value_ptr.last_layer) {
            s.next = try std.math.add(u64, s.next, 1);
            item.value_ptr.* = .{ .number = s.next, .last_layer = L };
        } else item.value_ptr.last_layer = L;
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes[lo..][0..n], &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        try w.line(.{ .version = 1, .kind = "layer", .rank = w.rank, .slot = span.slot, .request = s.request.sequence, .request_sha256 = @as([]const u8, &s.request.sha256), .window = item.value_ptr.number, .phase = @tagName(phase), .start = span.start, .rows = span.rows, .row0 = span.row0, .columns = columns, .layer = L, .ids_sha256 = @as([]const u8, &key.ids_sha256), .sha256 = @as([]const u8, &hex) });
    }

    fn line(w: *Writer, value: anytype) !void {
        const body = try std.json.Stringify.valueAlloc(w.gpa, value, .{});
        defer w.gpa.free(body);
        const end = try std.math.add(u64, w.offset, body.len + 1);
        try w.file.writePositionalAll(w.io, body, w.offset);
        try w.file.writePositionalAll(w.io, "\n", w.offset + body.len);
        w.offset = end;
    }
};
