//! Native(R)'s host contract with a tiny recipe: the decls it emits and omits, its refusals' words, a compiled path.
const std = @import("std");
const cuda = @import("cuda");
const native = @import("native.zig");
const Pass = @import("pass.zig").Pass;
const Geometry = @import("recipe.zig").Geometry;
const Allocator = std.mem.Allocator;

const tiny: Geometry = .{ .hidden = 256, .layers = 2, .heads = 4, .kv_heads = 2, .head_dim = 64, .vocab = 512, .eps = 1e-6, .context = 4096, .acc = 512, .wide = 1024, .mid = 512, .theta = 1e4 };

/// A recipe of the dots shape on tiny sizes: biased q|k|v sums, attention, o, SwiGLU, an untied head.
const Tiny = struct {
    pub const name = "Tiny";
    pub const model_type = "tiny_dense";
    pub const formats: []const []const u8 = &.{"bf16"};
    pub const default_context: i64 = 4096;
    pub const reasoning = false;
    pub const post_embed: ?fn (*const Pass(Tiny)) anyerror!void = null;
    pub const Weights = struct {
        embed: u64 = 0,
        norm: u64 = 0,
        bias: u64 = 0,
        proj: cuda.qmmf.Weight = cuda.modelopt.bf16Weight(0, 512, 256),
        pub fn deinit(_: *Weights) void {}
    };
    pub const Kernels = struct {
        pub fn load(_: *const cuda.Context) !Kernels {
            return .{};
        }
        pub fn deinit(_: *Kernels) void {}
    };
    pub fn geometry(_: Allocator, _: std.Io, _: []const u8) !Geometry {
        return tiny;
    }
    pub fn load(_: Allocator, _: std.Io, _: *const cuda.Driver, _: cuda.Stream, _: []const u8) !Weights {
        return .{};
    }
    pub fn weightBytes(_: Allocator, _: std.Io, _: []const u8) !u64 {
        return 1 << 20;
    }
    pub fn embed(p: *const Pass(Tiny)) !void {
        try p.k.embed.bf16(p.s, p.m.s.ids, p.w.embed, p.m.s.h, p.rows, p.g.hidden);
    }
    pub fn layer(p: *const Pass(Tiny), i: usize) !void {
        const x = p.m.s;
        try p.norm(if (i == 0) null else x.r, p.w.norm);
        try p.projectBias(p.w.proj, p.w.bias, x.x, x.wide);
        try p.projectF32(p.w.proj, x.x, x.acc);
        try p.attend(i, x.q, x.kv, x.y);
        try p.project(p.w.proj, x.y, x.r);
        try p.siluMul(x.wide, x.y, p.g.mid);
    }
    pub fn final(p: *const Pass(Tiny)) !void {
        try p.norm(p.m.s.r, p.w.norm);
    }
    pub fn head(w: *const Weights) cuda.qmmf.Weight {
        return w.proj;
    }
    pub fn explain(err: anyerror) ?[]const u8 {
        return if (err == error.TinyConfig) "this Tiny checkpoint is not the one the recipe serves" else null;
    }
};

const N = native.Native(Tiny);

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
}

test "every engine refusal is worded, the recipe's own after it, the rest left to the server" {
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
}
