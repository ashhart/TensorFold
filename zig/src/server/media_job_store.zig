//! Checksummed records and immutable media files with atomic publication and reclaim.
const std = @import("std");
const platform = @import("sys");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Kind = enum(u8) { image = 1, audio = 2, video = 3 };
pub const State = enum(u8) {
    queued = 1,
    running = 2,
    succeeded = 3,
    failed = 4,
    cancelled = 5,

    pub fn terminal(s: State) bool {
        return @backingInt(s) >= @backingInt(State.succeeded);
    }
};

pub const Owner = ?[32]u8;
pub const max_artifacts = 64;
pub const max_media_type = 128;
pub const max_message = 1024;
pub const id_len = 38;
const magic = "TFMJOB01";
const fixed_bytes = 104;
const artifact_fixed = 42;

pub const Artifact = struct {
    media_type: []const u8,
    bytes: u64,
    digest: [32]u8,
};

pub const Record = struct {
    serial: u64,
    id: []const u8,
    kind: Kind,
    owner: Owner,
    created: i64,
    state: State,
    cancel_requested: bool = false,
    request: []const u8 = &.{},
    message: ?[]const u8 = null,
    artifacts: []Artifact = &.{},

    pub fn deinit(r: *Record, a: Allocator) void {
        if (r.id.len > 0) a.free(r.id);
        if (r.request.len > 0) a.free(r.request);
        if (r.message) |message| if (message.len > 0) a.free(message);
        for (r.artifacts) |artifact| if (artifact.media_type.len > 0) a.free(artifact.media_type);
        if (r.artifacts.len > 0) a.free(r.artifacts);
        r.* = undefined;
    }

    pub fn resultBytes(r: *const Record) usize {
        var total: usize = 0;
        for (r.artifacts) |artifact| total += @intCast(artifact.bytes);
        return total;
    }
};

pub fn validId(id: []const u8) bool {
    if (id.len != id_len or !std.mem.startsWith(u8, id, "media-")) return false;
    for (id[6..]) |ch| if (!std.ascii.isDigit(ch) and (ch < 'a' or ch > 'f')) return false;
    return true;
}

pub fn validMediaType(text: []const u8) bool {
    if (text.len == 0 or text.len > max_media_type or std.mem.indexOfScalar(u8, text, '/') == null) return false;
    for (text) |ch| if (ch < 0x21 or ch > 0x7e) return false;
    return true;
}

fn kindOf(value: u8) !Kind {
    return switch (value) {
        1 => .image,
        2 => .audio,
        3 => .video,
        else => error.CorruptRecord,
    };
}

fn stateOf(value: u8) !State {
    return switch (value) {
        1 => .queued,
        2 => .running,
        3 => .succeeded,
        4 => .failed,
        5 => .cancelled,
        else => error.CorruptRecord,
    };
}

fn encodedLen(r: *const Record) !usize {
    var total: usize = fixed_bytes;
    total = try std.math.add(usize, total, r.id.len);
    total = try std.math.add(usize, total, r.request.len);
    total = try std.math.add(usize, total, if (r.message) |m| m.len else 0);
    for (r.artifacts) |artifact| {
        total = try std.math.add(usize, total, artifact_fixed);
        total = try std.math.add(usize, total, artifact.media_type.len);
    }
    return total;
}

pub fn encode(a: Allocator, r: *const Record) ![]u8 {
    if (!validId(r.id) or r.artifacts.len > max_artifacts) return error.CorruptRecord;
    if (r.message) |message| if (message.len == 0 or message.len > max_message) return error.CorruptRecord;
    if (r.state.terminal() and r.cancel_requested and r.state != .cancelled) return error.CorruptRecord;
    if (r.state != .succeeded and r.artifacts.len != 0) return error.CorruptRecord;
    for (r.artifacts) |artifact| if (!validMediaType(artifact.media_type)) return error.CorruptRecord;
    const bytes = try a.alloc(u8, try encodedLen(r));
    errdefer a.free(bytes);
    @memcpy(bytes[0..8], magic);
    std.mem.writeInt(u64, bytes[8..16], r.serial, .little);
    std.mem.writeInt(i64, bytes[16..24], r.created, .little);
    std.mem.writeInt(u16, bytes[24..26], id_len, .little);
    bytes[26] = @backingInt(r.kind);
    bytes[27] = @backingInt(r.state);
    bytes[28] = @intFromBool(r.owner != null) | (@as(u8, @intFromBool(r.cancel_requested)) << 1);
    bytes[29] = @intCast(r.artifacts.len);
    std.mem.writeInt(u16, bytes[30..32], 0, .little);
    if (r.owner) |owner| @memcpy(bytes[32..64], &owner) else @memset(bytes[32..64], 0);
    std.mem.writeInt(u32, bytes[64..68], std.math.cast(u32, r.request.len) orelse return error.CorruptRecord, .little);
    std.mem.writeInt(u32, bytes[68..72], std.math.cast(u32, if (r.message) |m| m.len else 0) orelse return error.CorruptRecord, .little);
    var at: usize = 72;
    @memcpy(bytes[at..][0..r.id.len], r.id);
    at += r.id.len;
    @memcpy(bytes[at..][0..r.request.len], r.request);
    at += r.request.len;
    if (r.message) |message| {
        @memcpy(bytes[at..][0..message.len], message);
        at += message.len;
    }
    for (r.artifacts) |artifact| {
        std.mem.writeInt(u64, bytes[at..][0..8], artifact.bytes, .little);
        std.mem.writeInt(u16, bytes[at + 8 ..][0..2], @intCast(artifact.media_type.len), .little);
        @memcpy(bytes[at + 10 ..][0..32], &artifact.digest);
        at += artifact_fixed;
        @memcpy(bytes[at..][0..artifact.media_type.len], artifact.media_type);
        at += artifact.media_type.len;
    }
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes[0..at], &digest, .{});
    @memcpy(bytes[at..][0..32], &digest);
    return bytes;
}

pub fn decode(a: Allocator, bytes: []const u8) !Record {
    if (bytes.len < fixed_bytes or !std.mem.eql(u8, bytes[0..8], magic)) return error.CorruptRecord;
    var actual: [32]u8 = undefined;
    Sha256.hash(bytes[0 .. bytes.len - 32], &actual, .{});
    if (!std.mem.eql(u8, &actual, bytes[bytes.len - 32 ..])) return error.CorruptRecord;
    if (std.mem.readInt(u16, bytes[24..26], .little) != id_len or std.mem.readInt(u16, bytes[30..32], .little) != 0) return error.CorruptRecord;
    if (bytes[28] & ~@as(u8, 3) != 0 or bytes[29] > max_artifacts) return error.CorruptRecord;
    const request_len = std.mem.readInt(u32, bytes[64..68], .little);
    const message_len = std.mem.readInt(u32, bytes[68..72], .little);
    if (message_len > max_message) return error.CorruptRecord;
    if (std.mem.readInt(u64, bytes[8..16], .little) == 0) return error.CorruptRecord;
    var r: Record = .{
        .serial = std.mem.readInt(u64, bytes[8..16], .little),
        .id = &.{},
        .kind = try kindOf(bytes[26]),
        .owner = if (bytes[28] & 1 != 0) bytes[32..64].* else null,
        .created = std.mem.readInt(i64, bytes[16..24], .little),
        .state = try stateOf(bytes[27]),
        .cancel_requested = bytes[28] & 2 != 0,
    };
    errdefer r.deinit(a);
    var at: usize = 72;
    if (!validId(bytes[at..][0..id_len])) return error.CorruptRecord;
    r.id = try a.dupe(u8, bytes[at..][0..id_len]);
    at += id_len;
    const body_end = bytes.len - 32;
    if (request_len > body_end -| at or at + request_len > body_end) return error.CorruptRecord;
    r.request = if (request_len == 0) &.{} else try a.dupe(u8, bytes[at..][0..request_len]);
    at += request_len;
    if (message_len > body_end -| at or at + message_len > body_end) return error.CorruptRecord;
    r.message = if (message_len == 0) null else try a.dupe(u8, bytes[at..][0..message_len]);
    at += message_len;
    if (r.state.terminal() and r.cancel_requested and r.state != .cancelled) return error.CorruptRecord;
    if (r.state != .succeeded and bytes[29] != 0) return error.CorruptRecord;
    if (bytes[29] > 0) {
        r.artifacts = try a.alloc(Artifact, bytes[29]);
        for (r.artifacts) |*artifact| artifact.* = .{ .media_type = &.{}, .bytes = 0, .digest = @splat(0) };
        for (r.artifacts) |*artifact| {
            if (at + artifact_fixed > body_end) return error.CorruptRecord;
            artifact.bytes = std.mem.readInt(u64, bytes[at..][0..8], .little);
            const media_len = std.mem.readInt(u16, bytes[at + 8 ..][0..2], .little);
            artifact.digest = bytes[at + 10 ..][0..32].*;
            at += artifact_fixed;
            if (media_len > body_end -| at or at + media_len > body_end) return error.CorruptRecord;
            const media = bytes[at..][0..media_len];
            if (!validMediaType(media)) return error.CorruptRecord;
            artifact.media_type = try a.dupe(u8, media);
            at += media_len;
        }
    }
    if (at != body_end) return error.CorruptRecord;
    return r;
}

fn recordName(a: Allocator, id: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "{s}.job", .{id});
}

pub fn artifactName(a: Allocator, id: []const u8, index: usize) ![]u8 {
    return std.fmt.allocPrint(a, "{s}.{x:0>8}.blob", .{ id, index });
}

fn fullPath(a: Allocator, dir: []const u8, name: []const u8) ![]u8 {
    return std.fs.path.join(a, &.{ dir, name });
}

pub const Store = struct {
    gpa: Allocator,
    io: std.Io,
    dir: []const u8,
    max_record_bytes: usize,

    pub fn init(s: *Store, gpa: Allocator, io: std.Io, dir: []const u8, max_record_bytes: usize) !void {
        s.* = .{ .gpa = gpa, .io = io, .dir = try gpa.dupe(u8, dir), .max_record_bytes = max_record_bytes };
        errdefer gpa.free(s.dir);
        try std.Io.Dir.cwd().createDirPath(io, dir);
    }

    pub fn deinit(s: *Store) void {
        s.gpa.free(s.dir);
    }

    pub fn save(s: *Store, r: *const Record) !void {
        const name = try recordName(s.gpa, r.id);
        defer s.gpa.free(name);
        const path = try fullPath(s.gpa, s.dir, name);
        defer s.gpa.free(path);
        const bytes = try encode(s.gpa, r);
        defer s.gpa.free(bytes);
        try platform.atomic_file.replacePrivate(s.gpa, path, &.{bytes});
    }

    pub fn saveArtifact(s: *Store, id: []const u8, index: usize, bytes: []const u8) ![32]u8 {
        const name = try artifactName(s.gpa, id, index);
        defer s.gpa.free(name);
        const path = try fullPath(s.gpa, s.dir, name);
        defer s.gpa.free(path);
        try platform.atomic_file.replacePrivate(s.gpa, path, &.{bytes});
        var digest: [32]u8 = undefined;
        Sha256.hash(bytes, &digest, .{});
        return digest;
    }

    pub fn readArtifact(s: *Store, a: Allocator, id: []const u8, index: usize, artifact: Artifact) ![]u8 {
        const name = try artifactName(s.gpa, id, index);
        defer s.gpa.free(name);
        const path = try fullPath(s.gpa, s.dir, name);
        defer s.gpa.free(path);
        const expected = std.math.cast(usize, artifact.bytes) orelse return error.CorruptResult;
        const limit = std.math.add(usize, expected, 1) catch return error.CorruptResult;
        const bytes = std.Io.Dir.cwd().readFileAlloc(s.io, path, a, .limited(limit)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.CorruptResult,
        };
        errdefer a.free(bytes);
        if (bytes.len != expected) return error.CorruptResult;
        var digest: [32]u8 = undefined;
        Sha256.hash(bytes, &digest, .{});
        if (!std.mem.eql(u8, &digest, &artifact.digest)) return error.CorruptResult;
        return bytes;
    }

    fn delete(s: *Store, name: []const u8) !void {
        const path = try fullPath(s.gpa, s.dir, name);
        defer s.gpa.free(path);
        std.Io.Dir.cwd().deleteFile(s.io, path) catch |e| if (e != error.FileNotFound) return e;
    }

    pub fn deleteArtifact(s: *Store, id: []const u8, index: usize) !void {
        const name = try artifactName(s.gpa, id, index);
        defer s.gpa.free(name);
        try s.delete(name);
    }

    pub fn remove(s: *Store, r: *const Record) !void {
        const name = try recordName(s.gpa, r.id);
        defer s.gpa.free(name);
        var failure: ?anyerror = null;
        s.delete(name) catch |e| {
            failure = e;
        };
        for (r.artifacts, 0..) |_, index| {
            const artifact = artifactName(s.gpa, r.id, index) catch |e| return e;
            defer s.gpa.free(artifact);
            s.delete(artifact) catch |e| if (failure == null) {
                failure = e;
            };
        }
        if (failure) |e| return e;
    }

    pub fn load(s: *Store, max_records: usize) !std.ArrayList(Record) {
        var records: std.ArrayList(Record) = .empty;
        errdefer {
            for (records.items) |*r| r.deinit(s.gpa);
            records.deinit(s.gpa);
        }
        var dir = try std.Io.Dir.cwd().openDir(s.io, s.dir, .{ .iterate = true });
        defer dir.close(s.io);
        platform.atomic_file.discardStale(s.io, s.dir, null);
        var it = dir.iterate();
        while (try it.next(s.io)) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".job")) continue;
            if (records.items.len >= max_records) return error.TooManyRecords;
            const bytes = dir.readFileAlloc(s.io, entry.name, s.gpa, .limited(s.max_record_bytes)) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.CorruptRecord,
            };
            defer s.gpa.free(bytes);
            var record = try decode(s.gpa, bytes);
            errdefer record.deinit(s.gpa);
            const expected = try recordName(s.gpa, record.id);
            defer s.gpa.free(expected);
            if (!std.mem.eql(u8, expected, entry.name)) return error.CorruptRecord;
            for (records.items) |old| if (old.serial == record.serial or std.mem.eql(u8, old.id, record.id)) return error.CorruptRecord;
            var result_total: usize = 0;
            for (record.artifacts, 0..) |artifact, index| {
                const name = try artifactName(s.gpa, record.id, index);
                defer s.gpa.free(name);
                const stat = dir.statFile(s.io, name, .{}) catch return error.CorruptResult;
                if (stat.kind != .file or stat.size != artifact.bytes) return error.CorruptResult;
                const size = std.math.cast(usize, artifact.bytes) orelse return error.CorruptRecord;
                result_total = std.math.add(usize, result_total, size) catch return error.CorruptRecord;
            }
            try records.append(s.gpa, record);
        }
        try s.sweep(dir, records.items);
        return records;
    }

    fn sweep(s: *Store, dir: std.Io.Dir, records: []const Record) !void {
        var it = dir.iterate();
        while (try it.next(s.io)) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".blob")) continue;
            var referenced = false;
            for (records) |record| for (record.artifacts, 0..) |_, index| {
                var buffer: [id_len + 14]u8 = undefined;
                const name = std.fmt.bufPrint(&buffer, "{s}.{x:0>8}.blob", .{ record.id, index }) catch return error.CorruptRecord;
                if (std.mem.eql(u8, name, entry.name)) referenced = true;
            };
            if (!referenced) dir.deleteFile(s.io, entry.name) catch |e| if (e != error.FileNotFound) return e;
        }
    }
};
