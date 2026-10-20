//! Immutable, identity-bound proposal IDs and measured coverage; no serving policy or GPU allocation.
const std = @import("std");
pub const max_bytes = 16 << 20;
pub const Binding = struct { name: []const u8, revision: []const u8, sha256: []const u8 };
pub const Identity = struct {
    target: Binding,
    drafter: Binding,
    head: Binding,
    tokenizer: Binding,
    recipe: Binding,
    vocab: u32,
    hidden: u32,
    format: []const u8,
    group: u32,
    scale_recipe: []const u8,
};
pub const Split = enum { calibration, selection, heldout };
pub const Coverage = struct { split: Split, bucket: []const u8, tokens: u64, covered: u64 };
pub const Provenance = struct {
    corpus_sha256: []const u8,
    prompts_sha256: []const u8,
    responses_sha256: []const u8,
    calibration_sha256: []const u8,
    selection_sha256: []const u8,
    heldout_sha256: []const u8,
    settings_sha256: []const u8,
    collector: Binding,
    mandatory_source: []const u8,
    policy_version: u32,
    source_kind: enum { committed_response_ids, legacy_table },
};
pub const Document = struct {
    version: u32,
    identity: Identity,
    ids: []const u32,
    ids_sha256: []const u8,
    candidate_size: u32,
    mandatory_ids: []const u32,
    provenance: Provenance,
    coverage: []const Coverage,
};
pub const Artifact = struct {
    owned: std.json.Parsed(Document),
    fingerprint: [32]u8,

    pub fn deinit(a: *Artifact) void {
        a.owned.deinit();
        a.* = undefined;
    }
    pub fn isLegacy(a: *const Artifact) bool {
        return a.owned.value.provenance.source_kind == .legacy_table;
    }
    pub fn ids(a: *const Artifact) []const u32 {
        return a.owned.value.ids;
    }
    pub fn matches(a: *const Artifact, expected: Identity) bool {
        return sameIdentity(a.identity(), expected);
    }
    pub fn identity(a: *const Artifact) Identity {
        return a.owned.value.identity;
    }
    /// Only logical columns map to global IDs; a backend's padded columns never denote tokens.
    pub fn global(a: *const Artifact, index: usize) ?u32 {
        return if (index < a.ids().len) a.ids()[index] else null;
    }
    pub fn column(a: *const Artifact, id: u32) ?usize {
        var lo: usize = 0;
        var hi = a.ids().len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (a.ids()[mid] < id) lo = mid + 1 else hi = mid;
        }
        return if (lo < a.ids().len and a.ids()[lo] == id) lo else null;
    }
    pub fn physicalCount(a: *const Artifact, tile: u32) !u32 {
        if (tile == 0) return error.InvalidTile;
        const n: u32 = @intCast(a.ids().len);
        const tiles = n / tile + @intFromBool(n % tile != 0);
        return try std.math.mul(u32, tiles, tile);
    }
    pub fn countCovered(a: *const Artifact, committed: []const u32) usize {
        var count: usize = 0;
        for (committed) |id| count += @intFromBool(a.column(id) != null);
        return count;
    }
};

/// The caller supplies actual checkpoint/tokenizer/head/recipe identity and its complete mandatory ID set.
pub fn load(gpa: std.mem.Allocator, bytes: []const u8, expected: Identity, mandatory: []const u32) !Artifact {
    if (bytes.len == 0 or bytes.len > max_bytes) return error.ArtifactSize;
    const raw = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{ .allocate = .alloc_always, .parse_numbers = false });
    errdefer raw.deinit();
    try strict(Document, raw.value);
    const d = try std.json.parseFromValueLeaky(Document, raw.arena.allocator(), raw.value, .{});
    const owned: std.json.Parsed(Document) = .{ .arena = raw.arena, .value = d };
    if (d.version != 1 or d.provenance.policy_version != 1) return error.UnsupportedVersion;
    if (d.provenance.source_kind != .committed_response_ids) return error.InvalidProvenance;
    try validateIdentity(d.identity);
    try validateIdentity(expected);
    if (!sameIdentity(d.identity, expected)) return error.IdentityMismatch;
    if (d.ids.len == 0 or d.ids.len > d.identity.vocab or d.candidate_size != d.ids.len) return error.InvalidIds;
    try ordered(d.ids, d.identity.vocab);
    try ordered(d.mandatory_ids, d.identity.vocab);
    if (!std.mem.eql(u32, d.mandatory_ids, mandatory)) return error.MandatoryMismatch;
    var a: Artifact = .{ .owned = owned, .fingerprint = undefined };
    for (mandatory) |id| if (a.column(id) == null) return error.MandatoryMissing;
    if (!digest(d.ids_sha256) or !std.mem.eql(u8, &idsHash(d.ids), d.ids_sha256)) return error.ChecksumMismatch;
    const p = d.provenance;
    inline for (.{ p.corpus_sha256, p.prompts_sha256, p.responses_sha256, p.calibration_sha256, p.selection_sha256, p.heldout_sha256, p.settings_sha256 }) |hash|
        if (!digest(hash)) return error.InvalidProvenance;
    try binding(p.collector);
    if (!std.mem.eql(u8, p.mandatory_source, d.identity.recipe.name)) return error.InvalidProvenance;
    try coverage(d.coverage);
    std.crypto.hash.sha2.Sha256.hash(bytes, &a.fingerprint, .{});
    return a;
}

pub const legacy = @import("proposal_legacy.zig");

/// SHA256 of the ordered logical IDs as little-endian u32 values, with no padding or text encoding.
pub fn idsHash(ids: []const u32) [64]u8 {
    var sha = std.crypto.hash.sha2.Sha256.init(.{});
    for (ids) |id| {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &bytes, id, .little);
        sha.update(&bytes);
    }
    return std.fmt.bytesToHex(sha.finalResult(), .lower);
}
fn strict(comptime T: type, v: std.json.Value) anyerror!void {
    switch (@typeInfo(T)) {
        .int => {
            if (v != .number_string or v.number_string.len == 0) return error.InvalidInteger;
            for (v.number_string) |c| if (c < '0' or c > '9') return error.InvalidInteger;
        },
        .@"struct" => |info| {
            if (v != .object) return error.UnexpectedToken;
            inline for (info.field_names, info.field_types) |name, Field|
                try strict(Field, v.object.get(name) orelse return error.MissingField);
        },
        .pointer => |info| {
            if (info.child == u8) {
                if (v != .string) return error.UnexpectedToken;
            } else {
                if (v != .array) return error.UnexpectedToken;
                for (v.array.items) |item| try strict(info.child, item);
            }
        },
        .@"enum" => if (v != .string) return error.UnexpectedToken,
        else => @compileError("unsupported artifact field"),
    }
}
fn digest(s: []const u8) bool {
    if (s.len != 64) return false;
    for (s) |c| if (!(c >= '0' and c <= '9') and !(c >= 'a' and c <= 'f')) return false;
    return true;
}
fn text(s: []const u8) bool {
    return s.len > 0 and s.len <= 4096 and std.mem.indexOfScalar(u8, s, 0) == null;
}
fn binding(b: Binding) !void {
    if (!text(b.name) or !text(b.revision) or !digest(b.sha256)) return error.InvalidIdentity;
}
fn validateIdentity(i: Identity) !void {
    inline for (.{ i.target, i.drafter, i.head, i.tokenizer, i.recipe }) |b| try binding(b);
    if (i.vocab == 0 or i.vocab > 1 << 24 or i.hidden == 0 or i.hidden > 1 << 20 or
        !text(i.format) or !text(i.scale_recipe) or i.group > 1 << 20) return error.InvalidIdentity;
}
fn sameBinding(a: Binding, b: Binding) bool {
    return std.mem.eql(u8, a.name, b.name) and std.mem.eql(u8, a.revision, b.revision) and std.mem.eql(u8, a.sha256, b.sha256);
}
fn sameIdentity(a: Identity, b: Identity) bool {
    return sameBinding(a.target, b.target) and sameBinding(a.drafter, b.drafter) and sameBinding(a.head, b.head) and
        sameBinding(a.tokenizer, b.tokenizer) and sameBinding(a.recipe, b.recipe) and a.vocab == b.vocab and
        a.hidden == b.hidden and std.mem.eql(u8, a.format, b.format) and a.group == b.group and
        std.mem.eql(u8, a.scale_recipe, b.scale_recipe);
}
fn ordered(ids: []const u32, vocab: u32) !void {
    for (ids, 0..) |id, n| if (id >= vocab or (n > 0 and ids[n - 1] >= id)) return error.InvalidIds;
}
fn coverage(rows: []const Coverage) !void {
    if (rows.len < 6 or rows.len > 4096) return error.InvalidCoverage;
    var totals: [3]u64 = @splat(0);
    var covered: [3]u64 = @splat(0);
    var overall: [3]?Coverage = .{ null, null, null };
    for (rows, 0..) |row, n| {
        if (!text(row.bucket) or row.tokens == 0 or row.covered > row.tokens) return error.InvalidCoverage;
        for (rows[0..n]) |prior| if (prior.split == row.split and std.mem.eql(u8, prior.bucket, row.bucket)) return error.InvalidCoverage;
        const split = @backingInt(row.split);
        if (std.mem.eql(u8, row.bucket, "*")) overall[split] = row else {
            totals[split] = std.math.add(u64, totals[split], row.tokens) catch return error.InvalidCoverage;
            covered[split] = std.math.add(u64, covered[split], row.covered) catch return error.InvalidCoverage;
        }
    }
    for (overall, 0..) |maybe, n| {
        const row = maybe orelse return error.InvalidCoverage;
        if (row.tokens != totals[n] or row.covered != covered[n]) return error.InvalidCoverage;
    }
}
test {
    _ = @import("proposal_vocab_test.zig");
}
