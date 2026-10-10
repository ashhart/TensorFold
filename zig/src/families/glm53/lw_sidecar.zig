//! Living Weights on the full GLM-5.3 (glm53): the sidecar that keeps a model folder's committed low-rank change.
//! Same format as the Flash one (core/living_sidecar.zig, tf-living-weights-glm/1), its site and shapes the full
//! model's: layer 77's shared-expert down projection, in 2,048 -> out 6,144.
//!   <site>.lw_a bf16 [R, 2048], <site>.lw_b bf16 [R, 6144], metadata format/scale/site/ranks/base_sha (strings).
//! The file always holds the WHOLE a and b (every input column), so any rank count loads it: a 4-way engine's rank r
//! reads a's columns [512 r, 512 r + 512) on the GPU (its share of the shared expert's intermediate rows), b whole.
//! Host only (libc file calls), unit-tested on Linux.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const file_name = "living_weights.safetensors";
pub const format = "tf-living-weights-glm/1";
pub const layer: usize = 77; // the last trunk layer (78 = the MTP head)
pub const site = "model.layers.77.mlp.shared_experts.down_proj";
pub const scale_text = "10";
pub const scale: f32 = 10;
pub const in: usize = 2048;
pub const out: usize = 6144;
pub const block: usize = 16;
pub const max_rank: usize = 512;
pub const a_name = site ++ ".lw_a";
pub const b_name = site ++ ".lw_b";
pub const weight_name = site ++ ".weight";
/// sha256 of the stored weight bytes of orcarouter--GLM-5.3-MLX-6bit (the published 6-bit MLX checkpoint).
pub const orcarouter_sha = "b55452de1ab6e82a0048035780a7c79ea01f9e3118d376a0d923436c758d626e";

pub fn toBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
    const lsb = (u >> 16) & 1;
    return @intCast((u +% (0x7fff + lsb)) >> 16);
}

pub fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn roundSlice(v: []f32) void {
    for (v) |*x| x.* = fromBf16(toBf16(x.*));
}

/// A decoded sidecar: f32 copies of the bf16 rows (owned by the arena it was decoded with).
pub const Loaded = struct {
    ranks: usize,
    a: []f32, // [ranks, in]
    b: []f32, // [ranks, out]
    base_sha: []const u8,
};

const Ent = struct { dtype: []const u8, shape: []const std.json.Value, begin: usize, end: usize };

fn entry(root: std.json.Value, name: []const u8, data_len: usize) !Ent {
    const v = root.object.get(name) orelse return error.BadLivingWeights;
    if (v != .object) return error.BadLivingWeights;
    const dt = v.object.get("dtype") orelse return error.BadLivingWeights;
    const sh = v.object.get("shape") orelse return error.BadLivingWeights;
    const off = v.object.get("data_offsets") orelse return error.BadLivingWeights;
    if (dt != .string or sh != .array or off != .array or off.array.items.len != 2) return error.BadLivingWeights;
    const b0 = off.array.items[0];
    const e0 = off.array.items[1];
    if (b0 != .integer or e0 != .integer or b0.integer < 0 or e0.integer < b0.integer) return error.BadLivingWeights;
    const b: usize = @intCast(b0.integer);
    const e: usize = @intCast(e0.integer);
    if (e > data_len) return error.BadLivingWeights;
    return .{ .dtype = dt.string, .shape = sh.array.items, .begin = b, .end = e };
}

fn dims2(e: Ent, r: usize, c: usize) bool {
    if (e.shape.len != 2) return false;
    for (e.shape) |d| if (d != .integer) return false;
    return e.shape[0].integer == @as(i64, @intCast(r)) and e.shape[1].integer == @as(i64, @intCast(c));
}

/// The sidecar's bytes checked against the contract, decoded to f32 rows.
pub fn decode(arena: Allocator, bytes: []const u8) !Loaded {
    if (bytes.len < 8) return error.BadLivingWeights;
    const hl: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    if (hl > bytes.len - 8) return error.BadLivingWeights;
    const json = bytes[8..][0..hl];
    const data = bytes[8 + hl ..];
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{});
    if (root != .object) return error.BadLivingWeights;
    const meta = root.object.get("__metadata__") orelse return error.BadLivingWeights;
    if (meta != .object) return error.BadLivingWeights;
    const text = struct {
        fn of(mm: std.json.Value, key: []const u8) ![]const u8 {
            const v = mm.object.get(key) orelse return error.BadLivingWeights;
            return if (v == .string) v.string else error.BadLivingWeights;
        }
    };
    if (!std.mem.eql(u8, try text.of(meta, "format"), format)) return error.LivingFormat;
    if (!std.mem.eql(u8, try text.of(meta, "site"), site)) return error.LivingSite;
    if (!std.mem.eql(u8, try text.of(meta, "scale"), scale_text)) return error.LivingScale;
    const ranks = std.fmt.parseInt(usize, try text.of(meta, "ranks"), 10) catch return error.BadLivingWeights;
    const sha = try text.of(meta, "base_sha");
    if (sha.len != 64) return error.BadLivingWeights;
    if (ranks == 0 or ranks % block != 0 or ranks > max_rank) return error.LivingRanks;
    const ea = try entry(root, a_name, data.len);
    const eb = try entry(root, b_name, data.len);
    if (!std.mem.eql(u8, ea.dtype, "BF16") or !dims2(ea, ranks, in) or ea.end - ea.begin != ranks * in * 2) return error.LivingShape;
    if (!std.mem.eql(u8, eb.dtype, "BF16") or !dims2(eb, ranks, out) or eb.end - eb.begin != ranks * out * 2) return error.LivingShape;
    const a = try arena.alloc(f32, ranks * in);
    const b = try arena.alloc(f32, ranks * out);
    for (a, 0..) |*d, i| d.* = fromBf16(std.mem.readInt(u16, data[ea.begin + 2 * i ..][0..2], .little));
    for (b, 0..) |*d, i| d.* = fromBf16(std.mem.readInt(u16, data[eb.begin + 2 * i ..][0..2], .little));
    return .{ .ranks = ranks, .a = a, .b = b, .base_sha = try arena.dupe(u8, sha) };
}

/// The sidecar's bytes for `ranks` committed ranks (f32 rounded to bf16 here), the same layout the Flash writer uses.
pub fn encode(gpa: Allocator, a: []const f32, b: []const f32, ranks: usize, base_sha: []const u8) ![]u8 {
    if (ranks == 0 or ranks % block != 0 or ranks > max_rank) return error.LivingRanks;
    if (a.len < ranks * in or b.len < ranks * out or base_sha.len != 64) return error.BadLivingWeights;
    const a_bytes = ranks * in * 2;
    const b_bytes = ranks * out * 2;
    var head: std.ArrayList(u8) = .empty;
    defer head.deinit(gpa);
    try head.print(gpa, "{{\"__metadata__\":{{\"format\":\"{s}\",\"scale\":\"{s}\",\"site\":\"{s}\",\"ranks\":\"{d}\",\"base_sha\":\"{s}\"}}," ++
        "\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}}," ++
        "\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}}}", .{ format, scale_text, site, ranks, base_sha, a_name, ranks, in, a_bytes, b_name, ranks, out, a_bytes, a_bytes + b_bytes });
    while (head.items.len % 8 != 0) try head.append(gpa, ' ');
    const buf = try gpa.alloc(u8, 8 + head.items.len + a_bytes + b_bytes);
    std.mem.writeInt(u64, buf[0..8], head.items.len, .little);
    @memcpy(buf[8..][0..head.items.len], head.items);
    var at = 8 + head.items.len;
    for (a[0 .. ranks * in]) |x| {
        std.mem.writeInt(u16, buf[at..][0..2], toBf16(x), .little);
        at += 2;
    }
    for (b[0 .. ranks * out]) |x| {
        std.mem.writeInt(u16, buf[at..][0..2], toBf16(x), .little);
        at += 2;
    }
    return buf;
}

pub fn readFile(gpa: Allocator, path: []const u8) !?[]u8 {
    const z = try gpa.dupeSentinel(u8, path, 0);
    defer gpa.free(z);
    const fd = std.c.open(z, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return if (std.c._errno().* == @backingInt(std.c.E.NOENT)) null else error.LivingRead;
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0) return error.LivingRead;
    const buf = try gpa.alloc(u8, @intCast(end));
    errdefer gpa.free(buf);
    try preadAll(fd, buf, 0);
    return buf;
}

fn preadAll(fd: c_int, buf: []u8, offset: usize) !void {
    var done: usize = 0;
    while (done < buf.len) {
        const r = std.c.pread(fd, buf.ptr + done, @min(buf.len - done, 1 << 30), @intCast(offset + done));
        if (r <= 0) return error.LivingRead;
        done += @intCast(r);
    }
}

/// `bytes` to `path` through `path`.tmp, synced, renamed over it: a reader sees the old sidecar or the new one.
pub fn writeAtomic(gpa: Allocator, path: []const u8, bytes: []const u8) !void {
    const tmp = try std.fmt.allocPrintSentinel(gpa, "{s}.tmp", .{path}, 0);
    defer gpa.free(tmp);
    const dest = try gpa.dupeSentinel(u8, path, 0);
    defer gpa.free(dest);
    const fd = std.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return error.LivingWrite;
    errdefer _ = std.c.unlink(tmp);
    {
        defer _ = std.c.close(fd);
        var done: usize = 0;
        while (done < bytes.len) {
            const n = std.c.write(fd, bytes.ptr + done, @min(bytes.len - done, 1 << 30));
            if (n <= 0) return error.LivingWrite;
            done += @intCast(n);
        }
        if (std.c.fsync(fd) != 0) return error.LivingWrite;
    }
    if (std.c.rename(tmp, dest) != 0) return error.LivingWrite;
}

/// sha256 (hex) of the site's stored `.weight` bytes in the checkpoint at `dir` (index -> shard -> the tensor's range).
pub fn siteSha(gpa: Allocator, dir: []const u8) ![64]u8 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const index_path = try std.fs.path.join(arena, &.{ dir, "model.safetensors.index.json" });
    const ix_bytes = (try readFile(arena, index_path)) orelse return error.LivingSiteMissing;
    const ix = try std.json.parseFromSliceLeaky(std.json.Value, arena, ix_bytes, .{});
    const map = (ix.object.get("weight_map") orelse return error.BadIndex).object;
    const shard_v = map.get(weight_name) orelse return error.LivingSiteMissing;
    if (shard_v != .string) return error.BadIndex;
    const path = try std.fs.path.joinZ(arena, &.{ dir, shard_v.string });
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return error.LivingRead;
    defer _ = std.c.close(fd);
    var lenb: [8]u8 = undefined;
    try preadAll(fd, &lenb, 0);
    const hl: usize = @intCast(std.mem.readInt(u64, &lenb, .little));
    if (hl > 1 << 28) return error.BadSafetensors;
    const json = try arena.alloc(u8, hl);
    try preadAll(fd, json, 8);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0) return error.LivingRead;
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{});
    const ent = try entry(root, weight_name, @as(usize, @intCast(end)) - 8 - hl);
    var h = Sha256.init(.{});
    const chunk = try arena.alloc(u8, 1 << 22);
    var at: usize = ent.begin;
    while (at < ent.end) {
        const n = @min(chunk.len, ent.end - at);
        try preadAll(fd, chunk[0..n], 8 + hl + at);
        h.update(chunk[0..n]);
        at += n;
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    var hex: [64]u8 = undefined;
    _ = std.fmt.bufPrint(&hex, "{x}", .{digest}) catch unreachable;
    return hex;
}

/// A stable 64-bit tag of a sidecar's bytes (ranks must agree on it before serving: folded into the exchange's start word).
pub fn tag(bytes: []const u8) u64 {
    return std.hash.Wyhash.hash(0x6c77_3533, bytes) | 1;
}

// ------------------------------------------------------------------ v2: spawned shards
// Learned blocks live in spawned files `living-<NNNN>.safetensors` (each a v1-format sidecar holding ONLY its session's ranks),
// listed in `living_weights.index.json` {"format":"tf-living-weights-glm/2","shards":[{file,site,ranks,created,topic,base_sha,
// sha256}]}. The engine applies every listed shard's blocks in index order (R = sum of ranks). Append-only: a shard is never
// rewritten; rollback = drop the newest entry. A v1 single file in an index is simply shard 0001. LW_TOPICS = an allow-list.

pub const index_name = "living_weights.index.json";
pub const format2 = "tf-living-weights-glm/2";

pub const Entry = struct {
    file: []const u8,
    site: []const u8 = site,
    ranks: usize,
    created: []const u8 = "",
    topic: []const u8 = "",
    base_sha: []const u8,
    sha256: []const u8,
};

pub const Index = struct {
    format: []const u8 = format2,
    shards: []const Entry = &.{},
};

fn hexSha(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    Sha256.hash(bytes, &d, .{});
    var hex: [64]u8 = undefined;
    _ = std.fmt.bufPrint(&hex, "{x}", .{d}) catch unreachable;
    return hex;
}

fn topicAllowed(topic: []const u8, topics: ?[]const u8) bool {
    const list = topics orelse return true;
    if (list.len == 0) return true;
    var it = std.mem.tokenizeScalar(u8, list, ',');
    while (it.next()) |t| if (std.mem.eql(u8, std.mem.trim(u8, t, " "), topic)) return true;
    return false;
}

pub fn readIndex(arena: Allocator, dir: []const u8) !?Index {
    const path = try std.fs.path.join(arena, &.{ dir, index_name });
    const bytes = (try readFile(arena, path)) orelse return null;
    const ix = try std.json.parseFromSliceLeaky(Index, arena, bytes, .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    if (!std.mem.eql(u8, ix.format, format2)) return error.LivingFormat;
    return ix;
}

/// Every shard of the index at `dir` (allowed by `topics`), checked (file sha256, site, base_sha, ranks) and concatenated in index
/// order. No index: null.
pub fn loadIndex(arena: Allocator, dir: []const u8, topics: ?[]const u8) !?Loaded {
    const ix = (try readIndex(arena, dir)) orelse return null;
    var a: std.ArrayList(f32) = .empty;
    var b: std.ArrayList(f32) = .empty;
    var total: usize = 0;
    var base: ?[]const u8 = null;
    for (ix.shards) |e| {
        if (!std.mem.eql(u8, e.site, site)) return error.LivingSite;
        if (base) |x| {
            if (!std.mem.eql(u8, x, e.base_sha)) return error.LivingBaseMismatch;
        } else base = e.base_sha;
        if (!topicAllowed(e.topic, topics)) continue;
        const path = try std.fs.path.join(arena, &.{ dir, e.file });
        const bytes = (try readFile(arena, path)) orelse return error.LivingShardMissing;
        if (!std.mem.eql(u8, &hexSha(bytes), e.sha256)) return error.LivingShardChanged;
        const v = try decode(arena, bytes);
        if (v.ranks != e.ranks or !std.mem.eql(u8, v.base_sha, e.base_sha)) return error.LivingShardChanged;
        total += v.ranks;
        if (total > max_rank) return error.LivingRanks;
        try a.appendSlice(arena, v.a);
        try b.appendSlice(arena, v.b);
    }
    if (total == 0) return .{ .ranks = 0, .a = &.{}, .b = &.{}, .base_sha = base orelse "" };
    return .{ .ranks = total, .a = a.items, .b = b.items, .base_sha = base.? };
}

/// A new shard with rows [0, ranks) of `a`/`b` (this session's blocks only) and its index entry appended (index written atomically
/// after the shard). Returns the shard's file name (arena).
pub fn spawn(gpa: Allocator, arena: Allocator, dir: []const u8, a: []const f32, b: []const f32, ranks: usize, base_sha: []const u8, topic: []const u8, created: []const u8) ![]const u8 {
    const old = (try readIndex(arena, dir)) orelse Index{};
    var used: usize = 0;
    for (old.shards) |e| used += e.ranks;
    if (used + ranks > max_rank) return error.LivingRanks;
    const name = try std.fmt.allocPrint(arena, "living-{d:0>4}.safetensors", .{old.shards.len + 1});
    const bytes = try encode(gpa, a, b, ranks, base_sha);
    defer gpa.free(bytes);
    try writeAtomic(gpa, try std.fs.path.join(arena, &.{ dir, name }), bytes);
    const shards = try arena.alloc(Entry, old.shards.len + 1);
    @memcpy(shards[0..old.shards.len], old.shards);
    shards[old.shards.len] = .{ .file = name, .ranks = ranks, .created = created, .topic = topic, .base_sha = base_sha, .sha256 = try arena.dupe(u8, &hexSha(bytes)) };
    try writeIndex(gpa, arena, dir, .{ .shards = shards });
    return name;
}

fn writeIndex(gpa: Allocator, arena: Allocator, dir: []const u8, ix: Index) !void {
    var w: std.Io.Writer.Allocating = .init(arena);
    try std.json.Stringify.value(ix, .{ .whitespace = .indent_1 }, &w.writer);
    try writeAtomic(gpa, try std.fs.path.join(arena, &.{ dir, index_name }), w.written());
}

/// Rollback: the newest entry out of the index (its file stays on disk, never edited). Returns its name, or null when empty.
pub fn dropNewest(gpa: Allocator, arena: Allocator, dir: []const u8) !?[]const u8 {
    const ix = (try readIndex(arena, dir)) orelse return null;
    if (ix.shards.len == 0) return null;
    try writeIndex(gpa, arena, dir, .{ .shards = ix.shards[0 .. ix.shards.len - 1] });
    return ix.shards[ix.shards.len - 1].file;
}

test "v2 shards: spawn appends, loading concatenates in order, topics filter, a changed shard is refused, rollback drops the newest" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fs.path.join(arena, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    try std.testing.expect((try loadIndex(arena, dir, null)) == null);
    var rng = std.Random.DefaultPrng.init(2);
    var a1: [16 * in]f32 = undefined;
    var b1: [16 * out]f32 = undefined;
    var a2: [32 * in]f32 = undefined;
    var b2: [32 * out]f32 = undefined;
    for (&a1) |*x| x.* = rng.random().floatNorm(f32);
    for (&b1) |*x| x.* = rng.random().floatNorm(f32);
    for (&a2) |*x| x.* = rng.random().floatNorm(f32);
    for (&b2) |*x| x.* = rng.random().floatNorm(f32);
    try std.testing.expectEqualStrings("living-0001.safetensors", try spawn(gpa, arena, dir, &a1, &b1, 16, orcarouter_sha, "a", "t1"));
    try std.testing.expectEqualStrings("living-0002.safetensors", try spawn(gpa, arena, dir, &a2, &b2, 32, orcarouter_sha, "b", "t2"));
    const all = (try loadIndex(arena, dir, null)).?;
    try std.testing.expectEqual(@as(usize, 48), all.ranks);
    try std.testing.expectEqual(fromBf16(toBf16(a1[0])), all.a[0]);
    try std.testing.expectEqual(fromBf16(toBf16(a2[0])), all.a[16 * in]);
    try std.testing.expectEqual(fromBf16(toBf16(b2[32 * out - 1])), all.b[48 * out - 1]);
    const only_b = (try loadIndex(arena, dir, "b")).?;
    try std.testing.expectEqual(@as(usize, 32), only_b.ranks);
    try std.testing.expectEqual(fromBf16(toBf16(a2[0])), only_b.a[0]);
    // rollback: the newest out; the file stays
    try std.testing.expectEqualStrings("living-0002.safetensors", (try dropNewest(gpa, arena, dir)).?);
    try std.testing.expectEqual(@as(usize, 16), (try loadIndex(arena, dir, null)).?.ranks);
    // a shard edited after it was written is refused
    const p1 = try std.fs.path.join(arena, &.{ dir, "living-0001.safetensors" });
    const bytes = (try readFile(arena, p1)).?;
    bytes[bytes.len - 1] ^= 1;
    try writeAtomic(gpa, p1, bytes);
    try std.testing.expectError(error.LivingShardChanged, loadIndex(arena, dir, null));
}

test "glm53 sidecar: encode/decode round trip to bf16, full a and b, contract refusals" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const R = 32;
    const a = try arena.alloc(f32, R * in);
    const b = try arena.alloc(f32, R * out);
    var rng = std.Random.DefaultPrng.init(53);
    for (a) |*x| x.* = rng.random().floatNorm(f32) * 0.02;
    for (b) |*x| x.* = rng.random().floatNorm(f32) * 0.002;
    const bytes = try encode(gpa, a, b, R, orcarouter_sha);
    defer gpa.free(bytes);
    const v = try decode(arena, bytes);
    try std.testing.expectEqual(@as(usize, R), v.ranks);
    try std.testing.expectEqualStrings(orcarouter_sha, v.base_sha);
    for (a, v.a) |x, y| try std.testing.expectEqual(fromBf16(toBf16(x)), y);
    for (b, v.b) |x, y| try std.testing.expectEqual(fromBf16(toBf16(x)), y);
    // a re-encode of the decoded values is byte-identical (bf16 is a fixed point)
    const again = try encode(gpa, v.a, v.b, R, v.base_sha);
    defer gpa.free(again);
    try std.testing.expectEqualSlices(u8, bytes, again);
    try std.testing.expectError(error.LivingRanks, encode(gpa, a, b, 8, orcarouter_sha));
    const bad = try gpa.dupe(u8, bytes);
    defer gpa.free(bad);
    const at = std.mem.indexOf(u8, bad, "layers.77").?;
    bad[at + 7] = '6'; // the Flash-style site name of another layer
    try std.testing.expectError(error.LivingSite, decode(arena, bad));
}
