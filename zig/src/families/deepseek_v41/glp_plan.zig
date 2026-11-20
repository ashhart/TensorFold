//! Immutable CUDA GLPv3 additive plan: the validated, owned directions of one loaded vector. Format and the
//! DeepSeek paired-site concept: Matt Suiche/weightless (spec b181a07d3980fc48abe39d95ddb044459dd0b868);
//! directional-steering research (Arditi et al., NeurIPS 2024) is background only: this plan adds, it never
//! projects; the additive convention is llama.cpp's; the reader and startup convention is TensorFold #1835.

const std = @import("std");
const glp_ops = @import("dsv41_kernels").glp; // the kernels module's own, never a second copy of its file

/// The persistent role holding every steered layer's direction, [layers, width] fp32 (uploaded once at load).
pub const dirs_role = "s.glp.dirs";

/// The paired site's hook point and the spec version that defines it (core's reader names both; a vector at any
/// other site or version never becomes a plan).
pub const paired_site = "hc_fold_pre_engram";
pub const spec_version: u32 = 3;

/// The domain of the stored-session identity (`bindPolicy`).
const identity_domain = "tf-glp-v3 hc_fold_pre_engram policy";

pub const Error = error{ BadGlpStrength, BadGlpVersion, BadGlpSite, BadGlpWidth, BadGlpLayers, BadGlpValue, GlpRank, GlpUnsupported, GlpMismatch } || std.mem.Allocator.Error;

pub const Layer = struct { id: u32, dir: []const f32 };

pub const Plan = struct {
    /// the effective strength: finite and above 0 (0 never makes a plan)
    alpha: f32,
    width: u32,
    /// the SHA-256 of the vector file's exact bytes: what ranks and stored sessions compare
    file_sha256: [32]u8,
    /// ascending by id, each id at least 1 (layer 0 is never steered); each `dir` is a row of `dirs`
    layers: []Layer,
    /// the plan's own copy of every direction, [layers, width] fp32: what is uploaded, and what outlives the source
    dirs: []f32,

    /// Validate the structural GLP source and copy its directions. A valid vector at strength 0 (either sign)
    /// returns no plan and allocates nothing; an invalid one is refused at any strength.
    pub fn init(gpa: std.mem.Allocator, source: anytype) Error!?Plan {
        const alpha: f32 = source.alpha;
        if (!std.math.isFinite(alpha) or alpha < 0) return error.BadGlpStrength;
        const vector = source.vector;
        if (vector.spec_version != spec_version) return error.BadGlpVersion;
        if (!std.mem.eql(u8, vector.hook_point, paired_site)) return error.BadGlpSite;
        const width: usize = vector.width;
        if (width == 0 or width % 8 != 0 or width > glp_ops.max_width) return error.BadGlpWidth;
        const src = vector.layers;
        if (src.len == 0) return error.BadGlpLayers;
        for (src, 0..) |l, i| {
            if (l.id == 0 or (i > 0 and l.id <= src[i - 1].id)) return error.BadGlpLayers;
            if (l.dir.len != width) return error.BadGlpWidth;
            for (l.dir) |x| if (!std.math.isFinite(x)) return error.BadGlpValue;
        }
        if (alpha == 0) return null;
        const layers = try gpa.alloc(Layer, src.len);
        errdefer gpa.free(layers);
        const dirs = try gpa.alloc(f32, src.len * width);
        errdefer gpa.free(dirs);
        for (src, layers, 0..) |l, *out, i| {
            const row = dirs[i * width ..][0..width];
            @memcpy(row, l.dir);
            out.* = .{ .id = l.id, .dir = row };
        }
        return .{ .alpha = alpha, .width = @intCast(width), .file_sha256 = vector.file_sha256, .layers = layers, .dirs = dirs };
    }

    pub fn deinit(p: *Plan, gpa: std.mem.Allocator) void {
        gpa.free(p.layers);
        gpa.free(p.dirs);
        p.* = undefined;
    }

    /// The model's geometry: the vector is as wide as the stream and steers backbone blocks only.
    pub fn admit(p: *const Plan, backbone_layers: u32, hidden: u32) Error!void {
        if (p.width != hidden) return error.BadGlpWidth;
        for (p.layers) |l| if (l.id >= backbone_layers) return error.BadGlpLayers;
    }

    /// The direction table's role sized to the whole table in the buffer plan `bp` (its views reach only so far).
    pub fn reserve(p: *const Plan, bp: anytype) !void {
        const g = try bp.sizes.getOrPut(bp.a, dirs_role);
        if (!g.found_existing) {
            g.key_ptr.* = try bp.a.dupe(u8, dirs_role);
            g.value_ptr.* = 0;
        }
        g.value_ptr.* = @max(g.value_ptr.*, p.tableBytes());
    }

    /// A copy owned by the arena `a` (the layers and their directions): it outlives what it was built from.
    pub fn clone(p: *const Plan, a: std.mem.Allocator) std.mem.Allocator.Error!Plan {
        const layers = try a.alloc(Layer, p.layers.len);
        const dirs = try a.dupe(f32, p.dirs);
        for (p.layers, layers, 0..) |l, *out, i| out.* = .{ .id = l.id, .dir = dirs[i * p.width ..][0..p.width] };
        var out = p.*;
        out.layers = layers;
        out.dirs = dirs;
        return out;
    }

    /// The row of `id` in the direction table, null when the vector leaves the layer alone.
    pub fn slot(p: *const Plan, id: u32) ?u32 {
        var lo: usize = 0;
        var hi: usize = p.layers.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const at = p.layers[mid].id;
            if (at == id) return @intCast(mid);
            if (at < id) lo = mid + 1 else hi = mid;
        }
        return null;
    }

    pub fn has(p: *const Plan, id: u32) bool {
        return p.slot(id) != null;
    }

    /// The direction table's bytes: every layer's row, fp32 little endian, in `layers` order.
    pub fn tableBytes(p: *const Plan) u64 {
        return @as(u64, p.layers.len) * p.width * 4;
    }

    /// The table to upload at load (one owned copy).
    pub fn table(p: *const Plan, gpa: std.mem.Allocator) ![]u8 {
        return gpa.dupe(u8, std.mem.sliceAsBytes(p.dirs));
    }

    /// This plan's words (see `policyWords`, which the ranks agree on); null: none.
    pub fn words(p: ?*const Plan) [8]i64 {
        const x = p orelse return @splat(0);
        return pack(x.file_sha256, spec_version, x.alpha, x.layers.len);
    }

    /// `checkpoint` bound to this plan's policy: a stored session never restores into another steering. A null plan
    /// is no policy and leaves the checkpoint as it was; a loaded vector at strength 0 has no plan, so its
    /// identity binds through `bindPolicy` on its `policyWords`.
    pub fn bind(checkpoint: [32]u8, p: ?*const Plan) [32]u8 {
        return bindPolicy(checkpoint, words(p));
    }
};

/// `checkpoint` bound to the loaded vector's whole policy, strength 0 included: unchanged exactly when no vector
/// is loaded (word 0 is 0), otherwise a domain-separated hash of the checkpoint and all eight words.
pub fn bindPolicy(checkpoint: [32]u8, policy: [8]i64) [32]u8 {
    if (policy[0] == 0) return checkpoint;
    var h: std.crypto.hash.sha2.Sha256 = .init(.{});
    h.update(identity_domain);
    h.update(&checkpoint);
    var bytes: [64]u8 = undefined;
    for (policy, 0..) |w, i| std.mem.writeInt(i64, bytes[8 * i ..][0..8], w, .little);
    h.update(&bytes);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

fn pack(sha: [32]u8, version: u32, alpha: f32, layers: usize) [8]i64 {
    var w: [8]i64 = undefined;
    w[0] = 1; // a vector is loaded
    for (0..4) |i| w[1 + i] = std.mem.readInt(i64, sha[8 * i ..][0..8], .little);
    w[5] = version;
    w[6] = @as(i64, @as(u32, @bitCast(if (alpha == 0) @as(f32, 0) else alpha))); // -0 is 0, as in the identity
    w[7] = @intCast(layers);
    return w;
}

/// The loaded-policy words the ranks agree on, strength 0 included where no GPU plan exists: presence, the whole
/// file hash (four words), the spec version, the effective strength's FP32 bits (-0 as 0) and the layer count.
/// All zero: no vector loaded, so a loaded vector at strength 0 stays distinct from none.
pub fn policyWords(source: anytype) [8]i64 {
    const s = source orelse return @splat(0);
    return pack(s.vector.file_sha256, s.vector.spec_version, s.alpha, s.vector.layers.len);
}

const testing = std.testing;

const FakeLayer = struct { id: u32, dir: []const f32 };
const FakeVector = struct { spec_version: u32 = spec_version, hook_point: []const u8 = paired_site, width: usize, layers: []const FakeLayer, file_sha256: [32]u8 };
const Fake = struct { alpha: f32, vector: FakeVector };
fn fake(alpha: f32, layers: []const FakeLayer) Fake {
    return .{ .alpha = alpha, .vector = .{ .width = 8, .layers = layers, .file_sha256 = @splat(7) } };
}

test "glp plan: alpha 0 makes no plan and allocates nothing, a source's layers keep their order, a ragged one is refused" {
    const d = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
    const ok = [_]FakeLayer{ .{ .id = 3, .dir = &d }, .{ .id = 9, .dir = &d } };
    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    try testing.expect((try Plan.init(failing.allocator(), fake(0, &ok))) == null);
    try testing.expect((try Plan.init(failing.allocator(), fake(-0.0, &ok))) == null);
    try testing.expectEqual(@as(usize, 0), failing.allocations);
    var p = (try Plan.init(testing.allocator, fake(0.5, &ok))).?;
    defer p.deinit(testing.allocator);
    try testing.expectEqual(@as(?u32, 1), p.slot(9));
    try testing.expectEqual(@as(?u32, null), p.slot(4));
    try testing.expect(p.has(3) and !p.has(0));
    try testing.expectError(error.BadGlpLayers, p.admit(9, 8));
    try testing.expectError(error.BadGlpWidth, p.admit(40, 16));
    try p.admit(10, 8);
    const dup = [_]FakeLayer{ .{ .id = 3, .dir = &d }, .{ .id = 3, .dir = &d } };
    try testing.expectError(error.BadGlpLayers, Plan.init(testing.allocator, fake(1, &dup)));
    const down = [_]FakeLayer{ .{ .id = 9, .dir = &d }, .{ .id = 3, .dir = &d } };
    try testing.expectError(error.BadGlpLayers, Plan.init(testing.allocator, fake(1, &down)));
    const zero = [_]FakeLayer{.{ .id = 0, .dir = &d }}; // layer 0 is never steered
    try testing.expectError(error.BadGlpLayers, Plan.init(testing.allocator, fake(1, &zero)));
    try testing.expectError(error.BadGlpLayers, Plan.init(testing.allocator, fake(1, &.{})));
    const short = [_]FakeLayer{.{ .id = 1, .dir = d[0..4] }};
    try testing.expectError(error.BadGlpWidth, Plan.init(testing.allocator, fake(1, &short)));
    try testing.expectError(error.BadGlpStrength, Plan.init(testing.allocator, fake(-1, &ok)));
    try testing.expectError(error.BadGlpStrength, Plan.init(testing.allocator, fake(std.math.nan(f32), &ok)));
    try testing.expectError(error.BadGlpStrength, Plan.init(testing.allocator, fake(std.math.inf(f32), &ok)));
}

test "glp plan: a zero strength still validates the source (version, site, width, layers, values)" {
    const d = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
    const ok = [_]FakeLayer{.{ .id = 2, .dir = &d }};
    var f = fake(0, &ok);
    f.vector.spec_version = 1; // a legacy residual file is not a paired one
    try testing.expectError(error.BadGlpVersion, Plan.init(testing.allocator, f));
    f.vector.spec_version = 2;
    try testing.expectError(error.BadGlpVersion, Plan.init(testing.allocator, f));
    f = fake(0, &ok);
    f.vector.hook_point = "residual_stream_post_layer"; // the Qwen site
    try testing.expectError(error.BadGlpSite, Plan.init(testing.allocator, f));
    f.vector.hook_point = "";
    try testing.expectError(error.BadGlpSite, Plan.init(testing.allocator, f));
    f = fake(0, &ok);
    f.vector.width = 0;
    try testing.expectError(error.BadGlpWidth, Plan.init(testing.allocator, f));
    f.vector.width = 12;
    try testing.expectError(error.BadGlpWidth, Plan.init(testing.allocator, f));
    f.vector.width = 5128; // whole groups of eight, past the served width
    try testing.expectError(error.BadGlpWidth, Plan.init(testing.allocator, f));
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| {
        var e = d;
        e[5] = bad;
        const ls = [_]FakeLayer{.{ .id = 2, .dir = &e }};
        try testing.expectError(error.BadGlpValue, Plan.init(testing.allocator, fake(0, &ls)));
        try testing.expectError(error.BadGlpValue, Plan.init(testing.allocator, fake(1, &ls)));
    }
    var wide: [5128]f32 = @splat(0);
    const wide_layers = [_]FakeLayer{.{ .id = 1, .dir = &wide }};
    var g = fake(1, &wide_layers);
    g.vector.width = 5128;
    try testing.expectError(error.BadGlpWidth, Plan.init(testing.allocator, g));
    var top: [5120]f32 = @splat(0);
    const top_layers = [_]FakeLayer{.{ .id = 1, .dir = &top }};
    g = fake(1, &top_layers);
    g.vector.width = 5120;
    var plan = (try Plan.init(testing.allocator, g)).?;
    plan.deinit(testing.allocator);
}

test "glp plan: it owns its directions, and a clone owns its own" {
    var d = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const ls = [_]FakeLayer{.{ .id = 2, .dir = &d }};
    var p = (try Plan.init(testing.allocator, fake(1, &ls))).?;
    defer p.deinit(testing.allocator);
    d[0] = 99; // the source is released or rewritten after the plan is made
    try testing.expectEqual(@as(f32, 1), p.layers[0].dir[0]);
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const c = try p.clone(arena.allocator());
    p.dirs[1] = -1;
    try testing.expectEqual(@as(f32, 2), c.layers[0].dir[1]);
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, c.layers[0].dir);
    try testing.expectEqual(p.alpha, c.alpha);
    try testing.expectEqual(p.file_sha256, c.file_sha256);
    try testing.expectEqual(@as(?u32, 0), c.slot(2));
}

test "glp plan: the table holds each layer's row, and the words and the stored identity separate hash, version, strength and layers" {
    const d0 = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const d1 = [_]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };
    const ls = [_]FakeLayer{ .{ .id = 2, .dir = &d0 }, .{ .id = 5, .dir = &d1 } };
    var a = (try Plan.init(testing.allocator, fake(1, &ls))).?;
    defer a.deinit(testing.allocator);
    var c = (try Plan.init(testing.allocator, fake(2, &ls))).?;
    defer c.deinit(testing.allocator);
    var other = fake(1, &ls);
    other.vector.file_sha256[31] ^= 1; // one bit of the file hash
    var h = (try Plan.init(testing.allocator, other)).?;
    defer h.deinit(testing.allocator);
    var fewer = (try Plan.init(testing.allocator, fake(1, ls[0..1]))).?;
    defer fewer.deinit(testing.allocator);
    const t = try a.table(testing.allocator);
    defer testing.allocator.free(t);
    try testing.expectEqual(@as(usize, 2 * 8 * 4), t.len);
    try testing.expectEqual(@as(u64, t.len), a.tableBytes());
    try testing.expectEqual(@as(f32, 8), std.mem.bytesAsSlice(f32, t)[8]);
    const base: [32]u8 = @splat(1);
    for ([_]*Plan{ &c, &h, &fewer }) |x| {
        try testing.expect(!std.mem.eql(i64, &Plan.words(&a), &Plan.words(x)));
        try testing.expect(!std.mem.eql(u8, &Plan.bind(base, &a), &Plan.bind(base, x)));
    }
    try testing.expect(!std.mem.eql(i64, &Plan.words(&a), &Plan.words(null)));
    try testing.expectEqual(base, Plan.bind(base, null)); // no file: the checkpoint's own identity, exactly
    try testing.expect(!std.mem.eql(u8, &base, &Plan.bind(base, &a)));
    const w = Plan.words(&a);
    try testing.expectEqual(@as(i64, 1), w[0]);
    try testing.expectEqual(@as(i64, spec_version), w[5]);
    try testing.expectEqual(@as(i64, @as(u32, @bitCast(@as(f32, 1)))), w[6]);
    try testing.expectEqual(@as(i64, 2), w[7]);
    try testing.expectEqual(std.mem.readInt(i64, @as([32]u8, @splat(7))[0..8], .little), w[1]);
    try testing.expectEqual(std.mem.readInt(i64, @as([32]u8, @splat(7))[24..32], .little), w[4]);
}

test "glp plan: a loaded vector is a policy at strength 0 too, distinct from no vector, and binds the identity" {
    const d = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
    const ls = [_]FakeLayer{.{ .id = 2, .dir = &d }};
    var p = (try Plan.init(testing.allocator, fake(0.5, &ls))).?;
    defer p.deinit(testing.allocator);
    const Opt = ?Fake;
    try testing.expectEqual(Plan.words(&p), policyWords(@as(Opt, fake(0.5, &ls))));
    const none = policyWords(@as(Opt, null));
    try testing.expectEqual(@as([8]i64, @splat(0)), none);
    const zero = policyWords(@as(Opt, fake(0, &ls)));
    try testing.expect(!std.mem.eql(i64, &zero, &none));
    try testing.expectEqual(zero, policyWords(@as(Opt, fake(-0.0, &ls))));
    try testing.expect(!std.mem.eql(i64, &zero, &Plan.words(&p)));
    try testing.expectEqual(@as(i64, 0), zero[6]); // -0 and 0 are one strength
    const base: [32]u8 = @splat(9);
    try testing.expectEqual(base, bindPolicy(base, none)); // absent: unchanged
    try testing.expect(!std.mem.eql(u8, &base, &bindPolicy(base, zero))); // loaded at strength 0: bound
    try testing.expect(!std.mem.eql(u8, &bindPolicy(base, zero), &bindPolicy(base, Plan.words(&p))));
    try testing.expectEqual(bindPolicy(base, Plan.words(&p)), Plan.bind(base, &p));
    try testing.expectEqual(bindPolicy(base, zero), bindPolicy(base, policyWords(@as(Opt, fake(-0.0, &ls)))));
    // every word changes the identity
    const loaded = Plan.words(&p);
    for (1..8) |i| {
        var w = loaded;
        w[i] ^= 1;
        try testing.expect(!std.mem.eql(u8, &bindPolicy(base, w), &bindPolicy(base, loaded)));
    }
    var present = loaded;
    present[0] = 2;
    try testing.expect(!std.mem.eql(u8, &bindPolicy(base, present), &bindPolicy(base, loaded)));
    try testing.expectEqual(base, bindPolicy(base, .{ 0, 5, 5, 5, 5, 3, 7, 7 })); // word 0 decides absence
    const other_base: [32]u8 = @splat(8);
    try testing.expect(!std.mem.eql(u8, &bindPolicy(base, loaded), &bindPolicy(other_base, loaded)));
}

test "glp plan: the direction table's role is reserved at the whole table" {
    const d = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
    const ls = [_]FakeLayer{ .{ .id = 2, .dir = &d }, .{ .id = 6, .dir = &d } };
    var p = (try Plan.init(testing.allocator, fake(1, &ls))).?;
    defer p.deinit(testing.allocator);
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var bp: struct { a: std.mem.Allocator, sizes: std.StringArrayHashMapUnmanaged(u64) = .empty } = .{ .a = arena.allocator() };
    try bp.sizes.put(bp.a, dirs_role, 8); // a view that reached one row only
    try p.reserve(&bp);
    try testing.expectEqual(p.tableBytes(), bp.sizes.get(dirs_role).?);
    var fresh: struct { a: std.mem.Allocator, sizes: std.StringArrayHashMapUnmanaged(u64) = .empty } = .{ .a = arena.allocator() };
    try p.reserve(&fresh);
    try p.reserve(&fresh);
    try testing.expectEqual(@as(usize, 1), fresh.sizes.count());
    try testing.expectEqual(@as(u64, 2 * 8 * 4), fresh.sizes.get(dirs_role).?);
}
