//! Living Weights on GLM-5.3-Flash: the sidecar file that keeps a model folder's committed low-rank change.
//! `living_weights.safetensors` beside the shards, never folded into the 4-bit weights (re-quantizing would erase it):
//!   <site>.lw_a bf16 [R, in], <site>.lw_b bf16 [R, out], metadata format/scale/site/ranks/base_sha (all strings).
//! The forward adds scale * (x a^T) b after the site's projection. Absent file: the engine is the stock engine.
//! Host only (libc file calls, no Metal): the engine maps it, the learner writes it, the unit tests run anywhere.
const std = @import("std");
const st = @import("safetensors.zig");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const file_name = "living_weights.safetensors";
pub const format = "tf-living-weights-glm/1";
/// The one site: layer 44's shared-expert down projection (in 2048, out 4096).
pub const site = "model.layers.44.mlp.shared_experts.down_proj";
pub const scale_text = "10";
pub const scale: f32 = 10;
pub const in: usize = 2048;
pub const out: usize = 4096;
pub const block: usize = 16; // ranks a committed lesson adds
pub const max_rank: usize = 512;
pub const a_name = site ++ ".lw_a";
pub const b_name = site ++ ".lw_b";
/// The site's original weight as the checkpoint names it (MLX: under `model.language_model.`), then the plain name.
pub const weight_names = [_][]const u8{ "model.language_model.layers.44.mlp.shared_experts.down_proj.weight", site ++ ".weight" };

/// f32 to bf16 by round-to-nearest-even (Metal's `bfloat(float)`), NaN kept quiet.
pub fn toBf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((u >> 16) | 0x40);
    const lsb = (u >> 16) & 1;
    return @intCast((u +% (0x7fff + lsb)) >> 16);
}

pub fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// Each value replaced by the nearest bf16 (what the file keeps, so RAM and a reloaded folder compute the same bits).
pub fn roundSlice(v: []f32) void {
    for (v) |*x| x.* = fromBf16(toBf16(x.*));
}

/// A parsed sidecar: views into its bytes (bf16, unaligned).
pub const View = struct {
    ranks: usize,
    a: []align(1) const u16, // [ranks, in]
    b: []align(1) const u16, // [ranks, out]
    base_sha: []const u8, // 64 hex digits

    pub fn aF32(v: View, dst: []f32) void {
        for (dst[0 .. v.ranks * in], v.a) |*d, h| d.* = fromBf16(h);
    }
    pub fn bF32(v: View, dst: []f32) void {
        for (dst[0 .. v.ranks * out], v.b) |*d, h| d.* = fromBf16(h);
    }
};

/// The sidecar's bytes checked against the contract (format, site, scale, shapes, ranks a multiple of 16, at most 512).
pub fn decode(arena: Allocator, bytes: []const u8) !View {
    if (bytes.len < 8) return error.BadLivingWeights;
    const hl: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    if (hl > bytes.len - 8) return error.BadLivingWeights;
    const json = bytes[8..][0..hl];
    const data = bytes[8 + hl ..];
    const names = try st.parseHeader(arena, json, data.len);
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{});
    const meta = (root.object.get("__metadata__") orelse return error.BadLivingWeights);
    if (meta != .object) return error.BadLivingWeights;
    const text = struct {
        fn of(m: std.json.Value, key: []const u8) ![]const u8 {
            const v = m.object.get(key) orelse return error.BadLivingWeights;
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
    const ea = names.get(a_name) orelse return error.BadLivingWeights;
    const eb = names.get(b_name) orelse return error.BadLivingWeights;
    if (ea.dtype != .bf16 or ea.rank != 2 or ea.shape[0] != ranks or ea.shape[1] != in) return error.LivingShape;
    if (eb.dtype != .bf16 or eb.rank != 2 or eb.shape[0] != ranks or eb.shape[1] != out) return error.LivingShape;
    const a: []align(1) const u16 = std.mem.bytesAsSlice(u16, data[ea.begin..ea.end]);
    const b: []align(1) const u16 = std.mem.bytesAsSlice(u16, data[eb.begin..eb.end]);
    return .{ .ranks = ranks, .a = a, .b = b, .base_sha = sha };
}

/// The sidecar's bytes for `ranks` committed ranks (a [ranks, in], b [ranks, out], f32 rounded to bf16 here).
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
    const total = 8 + head.items.len + a_bytes + b_bytes;
    const buf = try gpa.alloc(u8, total);
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

/// The whole file at `path` (libc), or null when it does not exist.
pub fn readFile(gpa: Allocator, path: []const u8) !?[]u8 {
    const z = try gpa.dupeSentinel(u8, path, 0);
    defer gpa.free(z);
    const fd = std.c.open(z, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return if (std.c._errno().* == @backingInt(std.c.E.NOENT)) null else error.LivingRead;
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0) return error.LivingRead;
    const n: usize = @intCast(end);
    const buf = try gpa.alloc(u8, n);
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

/// sha256 (hex) of the site's original `.weight` bytes in the checkpoint at `dir` (index -> shard -> the tensor's range).
pub fn siteSha(gpa: Allocator, dir: []const u8) ![64]u8 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const index_path = try std.fs.path.join(arena, &.{ dir, "model.safetensors.index.json" });
    var shard: []const u8 = "model.safetensors";
    var wname: []const u8 = weight_names[0];
    if (try readFile(arena, index_path)) |ix_bytes| {
        const ix = try std.json.parseFromSliceLeaky(std.json.Value, arena, ix_bytes, .{});
        const map = (ix.object.get("weight_map") orelse return error.BadIndex).object;
        const found = for (weight_names) |n| {
            if (map.get(n)) |v| if (v == .string) break .{ n, v.string };
        } else return error.LivingSiteMissing;
        wname, shard = found;
    }
    const path = try std.fs.path.joinZ(arena, &.{ dir, shard });
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
    const names = try st.parseHeader(arena, json, @as(usize, @intCast(end)) - 8 - hl);
    var e: ?st.Entry = names.get(wname);
    if (e == null) for (weight_names) |n| if (names.get(n)) |x| {
        e = x;
        break;
    };
    const ent = e orelse return error.LivingSiteMissing;
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

// ---- v2: spawned shards ----
// `living_weights.index.json` {"format":"tf-living-weights-glm/2","site":...,"shards":[{"file","site","ranks","created",
// "topic","base_sha","sha256"}]} lists append-only shard files (`living-NNNN.safetensors`, each in the v1 file format);
// the engine sums every listed shard's ranks in index order. A v1 `living_weights.safetensors` with no index = shard 0001.
// Rollback = the newest entry dropped from the index (its file stays). LW_TOPICS=a,b loads only shards of those topics.

pub const index_name = "living_weights.index.json";
pub const format2 = "tf-living-weights-glm/2";

pub const Shard = struct { file: []const u8, ranks: usize, created: i64 = 0, topic: []const u8 = "", base_sha: []const u8, sha256: []const u8 };

/// The folder's shard list: the index's entries, else the v1 file as the one shard, else none (null).
pub fn listShards(arena: Allocator, dir: []const u8) !?[]Shard {
    const ipath = try std.fs.path.join(arena, &.{ dir, index_name });
    if (try readFile(arena, ipath)) |bytes| {
        const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, bytes, .{});
        if (root != .object) return error.BadLivingIndex;
        const f = root.object.get("format") orelse return error.BadLivingIndex;
        if (f != .string or !std.mem.eql(u8, f.string, format2)) return error.LivingFormat;
        const arr = root.object.get("shards") orelse return error.BadLivingIndex;
        if (arr != .array) return error.BadLivingIndex;
        const list = try arena.alloc(Shard, arr.array.items.len);
        for (arr.array.items, list) |v, *o| {
            if (v != .object) return error.BadLivingIndex;
            const g = struct {
                fn s(obj: std.json.Value, k: []const u8, need: bool) ![]const u8 {
                    const x = obj.object.get(k) orelse return if (need) error.BadLivingIndex else "";
                    return if (x == .string) x.string else error.BadLivingIndex;
                }
                fn n(obj: std.json.Value, k: []const u8) !i64 {
                    const x = obj.object.get(k) orelse return 0;
                    return switch (x) {
                        .integer => |i| i,
                        .string => |t| std.fmt.parseInt(i64, t, 10) catch error.BadLivingIndex,
                        else => error.BadLivingIndex,
                    };
                }
            };
            const site_v = try g.s(v, "site", true);
            if (!std.mem.eql(u8, site_v, site)) return error.LivingSite;
            const file = try g.s(v, "file", true);
            if (std.mem.indexOfScalar(u8, file, '/') != null or std.mem.eql(u8, file, "..")) return error.BadLivingIndex;
            o.* = .{ .file = file, .ranks = @intCast(try g.n(v, "ranks")), .created = try g.n(v, "created"), .topic = try g.s(v, "topic", false), .base_sha = try g.s(v, "base_sha", true), .sha256 = try g.s(v, "sha256", false) };
        }
        return list;
    }
    const v1 = try std.fs.path.join(arena, &.{ dir, file_name });
    const bytes = (try readFile(arena, v1)) orelse return null;
    const v = try decode(arena, bytes);
    const one = try arena.alloc(Shard, 1);
    one[0] = .{ .file = file_name, .ranks = v.ranks, .base_sha = try arena.dupe(u8, v.base_sha), .sha256 = "" };
    return one;
}

/// Whether a shard's topic passes the allow-list (comma separated; null or empty: every shard).
pub fn topicAllowed(allow: ?[]const u8, topic: []const u8) bool {
    const list = allow orelse return true;
    if (list.len == 0) return true;
    var it = std.mem.tokenizeScalar(u8, list, ',');
    while (it.next()) |t| if (std.mem.eql(u8, std.mem.trim(u8, t, " "), topic)) return true;
    return false;
}

/// Every allowed shard's rows summed in index order: a [ranks, in], b [ranks, out] f32 (exact from bf16).
pub const Loaded = struct { a: []f32, b: []f32, ranks: usize, shards: usize, digest: [32]u8 };

/// The folder's change (null: no sidecar, no index). Each shard's sha256 (when listed) and base_sha are checked.
pub fn loadAll(arena: Allocator, dir: []const u8, base_sha: []const u8, allow: ?[]const u8) !?Loaded {
    const shards = (try listShards(arena, dir)) orelse return null;
    var total: usize = 0;
    for (shards) |sh| if (topicAllowed(allow, sh.topic)) {
        total += sh.ranks;
    };
    if (total > max_rank) return error.LivingRanks;
    var got: Loaded = .{ .a = try arena.alloc(f32, total * in), .b = try arena.alloc(f32, total * out), .ranks = 0, .shards = 0, .digest = undefined };
    var h = Sha256.init(.{});
    for (shards) |sh| {
        if (!topicAllowed(allow, sh.topic)) continue;
        const path = try std.fs.path.join(arena, &.{ dir, sh.file });
        const bytes = (try readFile(arena, path)) orelse return error.LivingShardMissing;
        var d: [32]u8 = undefined;
        Sha256.hash(bytes, &d, .{});
        var hex: [64]u8 = undefined;
        _ = std.fmt.bufPrint(&hex, "{x}", .{d}) catch unreachable;
        if (sh.sha256.len > 0 and !std.mem.eql(u8, sh.sha256, &hex)) return error.LivingShardChanged;
        const v = try decode(arena, bytes);
        if (v.ranks != sh.ranks) return error.LivingRanks;
        if (!std.mem.eql(u8, v.base_sha, base_sha) or !std.mem.eql(u8, sh.base_sha, base_sha)) return error.LivingBaseMismatch;
        v.aF32(got.a[got.ranks * in ..]);
        v.bF32(got.b[got.ranks * out ..]);
        got.ranks += v.ranks;
        got.shards += 1;
        h.update(&d);
    }
    h.final(&got.digest);
    return got;
}

/// rows [0, ranks) of a/b as a new append-only shard `living-NNNN.safetensors`, then the index (atomic rename). A v1
/// file with no index becomes entry 0001 first. Returns the new shard's file name (in `gpa`, caller frees).
pub fn appendShard(gpa: Allocator, dir: []const u8, a: []const f32, b: []const f32, ranks: usize, topic: []const u8, base_sha: []const u8) ![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const old = (try listShards(arena, dir)) orelse &.{};
    var list: std.ArrayList(Shard) = .empty;
    for (old) |sh| {
        var e = sh;
        if (e.sha256.len == 0) { // the v1 file joining the index: its hash recorded now
            const bytes = (try readFile(arena, try std.fs.path.join(arena, &.{ dir, e.file }))).?;
            var d: [32]u8 = undefined;
            Sha256.hash(bytes, &d, .{});
            e.sha256 = try std.fmt.allocPrint(arena, "{x}", .{d});
        }
        try list.append(arena, e);
    }
    // the next free number (never reuse a dropped shard's file)
    var n: usize = list.items.len + 1;
    while (true) : (n += 1) {
        const name = try std.fmt.allocPrint(arena, "living-{d:0>4}.safetensors", .{n});
        const p = try std.fs.path.joinZ(arena, &.{ dir, name });
        if (std.c.access(p, std.c.F_OK) != 0) break;
    }
    const name = try std.fmt.allocPrint(gpa, "living-{d:0>4}.safetensors", .{n});
    errdefer gpa.free(name);
    const bytes = try encode(gpa, a, b, ranks, base_sha);
    defer gpa.free(bytes);
    var d: [32]u8 = undefined;
    Sha256.hash(bytes, &d, .{});
    try writeAtomic(gpa, try std.fs.path.join(arena, &.{ dir, name }), bytes);
    try list.append(arena, .{ .file = name, .ranks = ranks, .created = now(), .topic = topic, .base_sha = base_sha, .sha256 = try std.fmt.allocPrint(arena, "{x}", .{d}) });
    try writeIndex(arena, dir, list.items);
    return name;
}

/// The newest entry dropped from the index (its file stays on disk): the state before that save. Returns the dropped file.
pub fn dropNewest(gpa: Allocator, dir: []const u8) ![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const old = (try listShards(arena, dir)) orelse return error.NothingLearned;
    const ipath = try std.fs.path.join(arena, &.{ dir, index_name });
    if (try readFile(arena, ipath) == null) return error.LivingNoIndex; // a lone v1 file: no index to drop from
    try writeIndex(arena, dir, old[0 .. old.len - 1]);
    return gpa.dupe(u8, old[old.len - 1].file);
}

extern "c" fn time(t: ?*i64) i64;
fn now() i64 {
    return time(null);
}

fn writeIndex(arena: Allocator, dir: []const u8, shards: []const Shard) !void {
    var t: std.ArrayList(u8) = .empty;
    try t.print(arena, "{{\n  \"format\": \"{s}\",\n  \"site\": \"{s}\",\n  \"shards\": [", .{ format2, site });
    for (shards, 0..) |sh, i| {
        try t.print(arena, "{s}\n    {{\"file\": \"{s}\", \"site\": \"{s}\", \"ranks\": {d}, \"created\": {d}, \"topic\": ", .{ if (i > 0) "," else "", sh.file, site, sh.ranks, sh.created });
        try t.appendSlice(arena, try std.json.Stringify.valueAlloc(arena, sh.topic, .{}));
        try t.print(arena, ", \"base_sha\": \"{s}\", \"sha256\": \"{s}\"}}", .{ sh.base_sha, sh.sha256 });
    }
    try t.appendSlice(arena, "\n  ]\n}\n");
    try writeAtomic(arena, try std.fs.path.join(arena, &.{ dir, index_name }), t.items);
}

test "bf16 rounding is nearest-even and survives a round trip" {
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(@bitCast(@as(u32, 0x3f808000)))); // tie, even stays
    try std.testing.expectEqual(@as(u16, 0x3f82), toBf16(@bitCast(@as(u32, 0x3f818000)))); // tie, odd rounds up
    try std.testing.expectEqual(@as(u16, 0x3f81), toBf16(@bitCast(@as(u32, 0x3f808001))));
    var v = [_]f32{ 0.1, -3.3, 1e-20, 65504.5 };
    roundSlice(&v);
    for (v) |x| try std.testing.expectEqual(x, fromBf16(toBf16(x)));
}

test "a sidecar encodes and decodes to the same bf16 values, and refuses what breaks the contract" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const R = 32;
    const a = try arena.alloc(f32, R * in);
    const b = try arena.alloc(f32, R * out);
    var rng = std.Random.DefaultPrng.init(7);
    for (a) |*x| x.* = rng.random().floatNorm(f32) * 0.02;
    for (b) |*x| x.* = rng.random().floatNorm(f32) * 0.002;
    const sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const bytes = try encode(gpa, a, b, R, sha);
    defer gpa.free(bytes);
    const v = try decode(arena, bytes);
    try std.testing.expectEqual(@as(usize, R), v.ranks);
    try std.testing.expectEqualStrings(sha, v.base_sha);
    const a2 = try arena.alloc(f32, R * in);
    v.aF32(a2);
    for (a, a2) |x, y| try std.testing.expectEqual(fromBf16(toBf16(x)), y);
    const b2 = try arena.alloc(f32, R * out);
    v.bF32(b2);
    for (b, b2) |x, y| try std.testing.expectEqual(fromBf16(toBf16(x)), y);
    // the reference reader's view: a plain safetensors file
    const names = try st.parseHeader(arena, bytes[8..][0..std.mem.readInt(u64, bytes[0..8], .little)], bytes.len - 8 - std.mem.readInt(u64, bytes[0..8], .little));
    try std.testing.expectEqual(@as(usize, 2), names.count());
    try std.testing.expectError(error.LivingRanks, encode(gpa, a, b, 8, sha));
    try std.testing.expectError(error.LivingRanks, encode(gpa, a, b, 0, sha));
    const bad = try gpa.dupe(u8, bytes);
    defer gpa.free(bad);
    const at = std.mem.indexOf(u8, bad, "\"scale\":\"10\"").?;
    bad[at + 10] = '9';
    try std.testing.expectError(error.LivingScale, decode(arena, bad));
}

test "the site's sha reads the original weight through the index; files write atomically" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const head = "{\"model.language_model.layers.44.mlp.shared_experts.down_proj.weight\":{\"dtype\":\"U32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"x\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[16,20]}}";
    var shard: [8 + head.len + 20]u8 = undefined;
    std.mem.writeInt(u64, shard[0..8], head.len, .little);
    @memcpy(shard[8..][0..head.len], head);
    for (shard[8 + head.len ..], 0..) |*x, i| x.* = @intCast(i);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00001.safetensors", .data = &shard });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"metadata\":{},\"weight_map\":{\"model.language_model.layers.44.mlp.shared_experts.down_proj.weight\":\"model-00001-of-00001.safetensors\"}}" });
    const dir = try std.fs.path.join(gpa, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer gpa.free(dir);
    const got = try siteSha(gpa, dir);
    var digest: [32]u8 = undefined;
    Sha256.hash(shard[8 + head.len ..][0..16], &digest, .{});
    var want: [64]u8 = undefined;
    _ = try std.fmt.bufPrint(&want, "{x}", .{digest});
    try std.testing.expectEqualStrings(&want, &got);
    const path = try std.fs.path.join(gpa, &.{ dir, file_name });
    defer gpa.free(path);
    try std.testing.expect((try readFile(gpa, path)) == null);
    try writeAtomic(gpa, path, "abc");
    const back = (try readFile(gpa, path)).?;
    defer gpa.free(back);
    try std.testing.expectEqualStrings("abc", back);
}

test "v2 shards: two shards load as one merged shard, the newest dropped is the state before it, topics filter" {
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fs.path.join(gpa, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer gpa.free(dir);
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const R = 32;
    const a = try arena.alloc(f32, R * in);
    const b = try arena.alloc(f32, R * out);
    var rng = std.Random.DefaultPrng.init(3);
    for (a) |*x| x.* = rng.random().floatNorm(f32) * 0.01;
    for (b) |*x| x.* = rng.random().floatNorm(f32) * 0.01;
    const sha = "1111111111111111111111111111111111111111111111111111111111111111";
    try std.testing.expect((try loadAll(arena, dir, sha, null)) == null);
    // v1 file first: it becomes shard 0001 when the first v2 shard is appended
    const v1 = try encode(gpa, a, b, 16, sha);
    defer gpa.free(v1);
    try writeAtomic(gpa, try std.fs.path.join(arena, &.{ dir, file_name }), v1);
    const one = (try loadAll(arena, dir, sha, null)).?;
    try std.testing.expectEqual(@as(usize, 16), one.ranks);
    const n2 = try appendShard(gpa, dir, a[16 * in ..], b[16 * out ..], 16, "family", sha);
    defer gpa.free(n2);
    try std.testing.expectEqualStrings("living-0002.safetensors", n2);
    const two = (try loadAll(arena, dir, sha, null)).?;
    try std.testing.expectEqual(@as(usize, 2), two.shards);
    // == one merged 32-rank shard
    const merged = try encode(gpa, a, b, R, sha);
    defer gpa.free(merged);
    const mv = try decode(arena, merged);
    const ma = try arena.alloc(f32, R * in);
    const mb = try arena.alloc(f32, R * out);
    mv.aF32(ma);
    mv.bF32(mb);
    try std.testing.expectEqualSlices(f32, ma, two.a);
    try std.testing.expectEqualSlices(f32, mb, two.b);
    // topics: only "family" → the second shard alone
    const fam = (try loadAll(arena, dir, sha, "identity,family")).?;
    try std.testing.expectEqual(@as(usize, 16), fam.ranks);
    try std.testing.expectEqualSlices(f32, two.a[16 * in ..], fam.a);
    // a changed shard file is refused
    // drop the newest: back to the v1 state exactly, its file kept
    const dropped = try dropNewest(gpa, dir);
    defer gpa.free(dropped);
    try std.testing.expectEqualStrings("living-0002.safetensors", dropped);
    const back = (try loadAll(arena, dir, sha, null)).?;
    try std.testing.expectEqual(@as(usize, 16), back.ranks);
    try std.testing.expectEqualSlices(f32, one.a, back.a);
    try std.testing.expectEqualSlices(f32, one.b, back.b);
    try std.testing.expect((try readFile(arena, try std.fs.path.join(arena, &.{ dir, "living-0002.safetensors" }))) != null);
    // the next save never reuses the dropped file's number
    const n3 = try appendShard(gpa, dir, a[16 * in ..], b[16 * out ..], 16, "", sha);
    defer gpa.free(n3);
    try std.testing.expectEqualStrings("living-0003.safetensors", n3);
    try std.testing.expectError(error.LivingBaseMismatch, loadAll(arena, dir, "2222222222222222222222222222222222222222222222222222222222222222", null));
}
