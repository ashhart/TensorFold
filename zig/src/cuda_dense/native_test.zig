//! Native(R)'s host contract with a tiny recipe: the decls it emits and omits, its refusals' words, a compiled path.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const native = @import("native.zig");
const recipe = @import("recipe.zig");
const fixtures = @import("tiny_test.zig");

const tiny = fixtures.tiny;
const Tiny = fixtures.Tiny;
const Fake = fixtures.Fake;

const N = native.Native(Tiny);
const V = native.Native(fixtures.TinyTower);

/// The decl set of a text-only dense family: the host switches paths on by the decls present.
const text_decls = .{ "model_type", "formats", "default_context", "max_segments", "prompt_rows", "window_rows", "explains_open", "takes_drafter", "drafter_refusal", "keeps_no_prompts", "reasoning", "Room", "Options", "Loaded", "open", "textWeightBytes", "attach", "fits", "ledgerPool", "explain" };

/// Whether `F`'s public decls are exactly `want` and `more`.
fn exactly(comptime F: type, comptime want: anytype, comptime more: anytype) bool {
    const decls = @typeInfo(F).@"struct".decl_names;
    if (decls.len != want.len + more.len) return false;
    inline for (want) |d| if (!@hasDecl(F, d)) return false;
    inline for (more) |d| if (!@hasDecl(F, d)) return false;
    return true;
}

test "Native emits the host decls a dense family takes and no others" {
    inline for (.{ "model_type", "formats", "default_context", "max_segments", "prompt_rows", "window_rows", "explains_open", "takes_drafter", "drafter_refusal", "Room", "Options", "Loaded", "open", "textWeightBytes", "attach", "fits", "ledgerPool", "explain", "keeps_no_prompts", "reasoning" }) |d| {
        if (!@hasDecl(N, d)) return error.MissingDecl;
    }
    // each of these switches a host path on (prompt cache, KV sharing, learning, own planning, drafters, ranks)
    inline for (.{ "snap_save", "snap_restore", "snap_drop", "snap_bytes", "snap_slots", "snap_grid", "snap_lookahead", "share_attach", "share_pool", "share_states", "kvPool", "slide", "learnModel", "linked", "contextCeiling", "pagedLayout", "rankRoom", "request_starts", "request_starts_switch", "openDrafter", "warm_quiet" }) |d| {
        if (@hasDecl(N, d)) return error.UnexpectedDecl;
    }
    inline for (.{ "context", "drafts", "segments", "slide", "lanes", "lanes_fixed", "room" }) |f| try std.testing.expect(@hasField(N.Options, f));
    inline for (.{ "vision", "precision", "kv8", "prompt_int8", "loops", "rank", "drafter" }) |f| try std.testing.expect(!@hasField(N.Options, f));
    inline for (.{ "layout", "engine_paged", "free_streams", "target" }) |f| try std.testing.expect(@hasField(N.Loaded, f));
    try std.testing.expectEqualStrings("tiny_dense", N.model_type);
    try std.testing.expect(N.keeps_no_prompts and !N.reasoning and !N.takes_drafter);
    try std.testing.expectEqual(@as(u32, 16), N.window_rows);
    try std.testing.expectEqual(@as(u32, 1024), N.prompt_rows);
    try std.testing.expectEqual(@as(u32, 1), N.max_segments);
    try std.testing.expectStringStartsWith(N.drafter_refusal, "--drafter: Tiny on CUDA");
    try std.testing.expect(comptime exactly(N, text_decls, .{}));
    try std.testing.expect(comptime !exactly(N, text_decls, .{"visionBytes"})); // red control: one decl more is seen
}

test "a recipe with a tower adds exactly the vision decls and fields the host reads, the rest as text-only" {
    try std.testing.expect(comptime exactly(V, text_decls, .{ "default_image_tokens", "visionBytes" }));
    try std.testing.expect(@FieldType(V.Options, "vision") == ?Fake.Limits);
    try std.testing.expect(@FieldType(V.Loaded, "vision") == ?*Fake);
    try std.testing.expectEqual(@as(u32, 64), V.default_image_tokens);
    inline for (.{ "context", "drafts", "segments", "slide", "lanes", "lanes_fixed", "room" }) |f| try std.testing.expect(@hasField(V.Options, f));
    inline for (.{ "precision", "kv8", "prompt_int8", "loops", "rank", "drafter" }) |f| try std.testing.expect(!@hasField(V.Options, f));
    inline for (.{ "layout", "engine_paged", "free_streams", "target" }) |f| try std.testing.expect(@hasField(V.Loaded, f));
    try std.testing.expect(!@hasDecl(N, "default_image_tokens") and !@hasDecl(N, "visionBytes") and !@hasField(N.Loaded, "vision"));
    try std.testing.expectEqual(@as(u64, (5 << 20) + (3 << 16)), try V.visionBytes(std.testing.allocator, std.testing.io, ""));
    try std.testing.expectEqual(@as(?u64, 1 << 20), try V.textWeightBytes(std.testing.allocator, std.testing.io, ""));
}

test "the tower contract: the fake tower passes, and each decl of another shape is refused by name" {
    try std.testing.expect(comptime recipe.towerProblem(Fake) == null);
    comptime recipe.checkTower(Fake);
    const named = .{ .{ .limits, "Limits has max_images" }, .{ .width, "field width" }, .{ .marks, "marks has close" }, .{ .encode, "declares encode" }, .{ .prepare, "pixels, gh, gw, tokens and hash" }, .{ .explain, "declares explain" }, .{ .sure_encode, "declares encode" }, .{ .fallible_explain, "declares explain" } };
    inline for (named) |c| {
        const why = comptime recipe.towerProblem(fixtures.FakeTower(c[0])) orelse return error.TowerAccepted;
        try std.testing.expect(std.mem.indexOf(u8, why, c[1]) != null);
    }
    try std.testing.expect(comptime recipe.towerProblem(u32) != null);
    try std.testing.expect(comptime recipe.towerOf(Tiny) == null and recipe.towerOf(fixtures.TinyTower).? == Fake);
}

test "the tower's refusals follow the engine's and the recipe's words; a text-only recipe leaves them to the server" {
    try std.testing.expectStringStartsWith(V.explain(null, error.FakeImage).?, "the fake tower");
    try std.testing.expect(N.explain(null, error.FakeImage) == null);
    try std.testing.expectStringStartsWith(V.explain(null, error.TinyConfig).?, "this Tiny checkpoint");
    try std.testing.expectStringStartsWith(V.explain(null, error.DenseServedWithoutVision).?, "Tiny on CUDA was served without --vision");
    try std.testing.expect(V.explain(null, error.DenseImageSpan) != null and V.explain(null, error.DenseTowerWidth) != null);
    try std.testing.expect(std.mem.indexOf(u8, V.explain(null, error.DenseImageSpan).?, "(--vision-image-tokens)") != null);
}

/// The host's view of an engine of recipe `R` at a 4,096-token window, tower `v` (void for text-only).
fn loadedOf(comptime R: type, comptime L: type, v: native.Core(R).Dense.Vision) L {
    const C = native.Core(R);
    var own: C.Owned = undefined;
    own.m = .{ .gpa = std.testing.allocator, .d = undefined, .g = tiny, .context = 4096, .acc_width = 512, .part_bytes = 0 };
    own.v = v;
    own.b.tower = v;
    return C.loaded(L, &own);
}

test "each lane's image rows join every stream's bytes and fixed part with --vision, none without" {
    const pages = @import("pages.zig");
    const rows: u64 = 64 * 256 * 2; // the request's image tokens at the decoder's width, bf16
    const plain = pages.streamBytes(tiny, 4096);
    const fixed = pages.layout(tiny, 4096).fixed_bytes;
    var t: Fake = .{};
    const with = loadedOf(fixtures.TinyTower, V.Loaded, &t);
    try std.testing.expectEqual(plain + rows, with.stream_bytes);
    try std.testing.expectEqual(fixed + rows, with.layout.?.fixed_bytes);
    try std.testing.expect(with.vision.? == &t);
    const without = loadedOf(fixtures.TinyTower, V.Loaded, null);
    try std.testing.expectEqual(plain, without.stream_bytes);
    try std.testing.expectEqual(fixed, without.layout.?.fixed_bytes);
    try std.testing.expect(without.vision == null);
    const text = loadedOf(Tiny, N.Loaded, {});
    try std.testing.expectEqual(plain, text.stream_bytes);
    try std.testing.expectEqual(fixed, text.layout.?.fixed_bytes);
    // the page planner charges each extra lane its rows: an automatic count shrinks in a room that fits it without
    const l = text.layout.?;
    const budget: kv.Budget = .{ .room = 4 * (fixed + l.pageBytes()), .engine = 0, .lanes = 5 };
    try std.testing.expectEqual(@as(u32, 5), (try kv.plan(l, budget)).lanes);
    try std.testing.expectEqual(@as(u32, 4), (try kv.plan(with.layout.?, budget)).lanes);
    var fixed_count = budget;
    fixed_count.fixed = true; // a fixed count whose rows do not fit is refused, not shrunk
    try std.testing.expectError(error.TooMany, kv.plan(with.layout.?, fixed_count));
}

test "every engine refusal is worded, the recipe's own after it, the rest left to the server" {
    for (native.engine_errors) |e| try std.testing.expect(V.explain(null, e) != null);
    for (native.engine_errors) |e| try std.testing.expect(N.explain(null, e) != null);
    try std.testing.expectStringStartsWith(N.explain(null, error.TinyConfig).?, "this Tiny checkpoint");
    try std.testing.expectStringStartsWith(N.explain(null, error.DenseTextOnly).?, "Tiny on CUDA reads text prompts only");
    try std.testing.expect(N.explain(null, error.OutOfMemory) == null);
}

test "open refuses captured kernels and a fixed --parallel past a round before it reads the checkpoint" {
    const ctx: *const cuda.Context = undefined;
    const base: N.Options = .{ .context = 4096, .drafts = true };
    try std.testing.expectError(error.CapturedKernels, N.open(std.testing.allocator, undefined, ctx, "", "captured", base));
    var wide = base;
    wide.lanes = 129;
    wide.lanes_fixed = true;
    try std.testing.expectError(error.ParallelPastRound, N.open(std.testing.allocator, undefined, ctx, "", null, wide));
}

test "a recipe's whole path compiles against the engine: open, the lane backend, the forward" {
    const hooks = .{ &N.open, &N.attach, &N.fits, &N.ledgerPool, &N.textWeightBytes, &N.explain };
    inline for (hooks) |h| try std.testing.expect(@intFromPtr(h) != 0);
    const tower = .{ &V.open, &V.attach, &V.fits, &V.ledgerPool, &V.textWeightBytes, &V.explain, &V.visionBytes };
    inline for (tower) |h| try std.testing.expect(@intFromPtr(h) != 0);
}

test "open with a tower refuses captured kernels and a fixed --parallel past a round before it reads the checkpoint" {
    const ctx: *const cuda.Context = undefined;
    const base: V.Options = .{ .context = 4096, .drafts = true, .vision = .{ .image_tokens = 64, .max_images = 4 } };
    try std.testing.expectError(error.CapturedKernels, V.open(std.testing.allocator, undefined, ctx, "", "captured", base));
    var wide = base;
    wide.lanes = 129;
    wide.lanes_fixed = true;
    try std.testing.expectError(error.ParallelPastRound, V.open(std.testing.allocator, undefined, ctx, "", null, wide));
}
