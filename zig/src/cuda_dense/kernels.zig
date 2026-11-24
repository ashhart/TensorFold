//! The dense engine's shared launches: the lane matmul, norms, embedding rows, bias, draws, paged attention, SwiGLU.
const std = @import("std");
const cuda = @import("cuda");
const recipe = @import("recipe.zig");
const pages = @import("pages.zig");

const Args = cuda.Args;
const Function = cuda.Function;
const Stream = cuda.Stream;
const Geometry = recipe.Geometry;

/// Streams one shared-window attention launch takes (paged_attn.cuh's MAXSEG).
pub const max_segs = 8;

/// paged_attn.cuh's AttnSegs: each stream's caches, position, page table and first row.
pub const AttnSegs = extern struct { kc: [max_segs]u64, vc: [max_segs]u64, meta: [max_segs]u64, pt: [max_segs]u64, ps: i64, row0: [max_segs]c_int, n: c_int };

const images = .{ "sample", "tree_round", "prefill_attention", "nemotron_attention", "lfm2" };

fn int(x: usize) c_int {
    return @intCast(x);
}

pub const Kernels = struct {
    mods: [images.len]cuda.Module,
    g: Geometry,
    shape: recipe.Shape,
    lane: cuda.qmmf.Lane,
    norms: cuda.norms.Norms,
    embed: cuda.embed_rows.EmbedRows,
    vit: cuda.vit.Kernels,
    draws: cuda.sample.Draws,
    round: cuda.tree_round.Kernels,
    pattn: Function,
    kv_write: Function,
    chunk: Function,
    merge: Function,
    silu_mul: Function,

    /// The shared modules, and the attention launches `g`'s shape names; a geometry the kernels lack refuses by name.
    pub fn load(ctx: *const cuda.Context, g: Geometry) !Kernels {
        const d = ctx.d;
        const shape = try recipe.attentionShape(g);
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.g = g;
        k.shape = shape;
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        inline for (images, 0..) |name, i| {
            k.mods[i] = try cuda.Module.load(d, @field(cuda.kernels, name));
            loaded += 1;
        }
        const m = &k.mods;
        k.draws = try cuda.sample.Draws.resolve(m[0]);
        k.round = try cuda.tree_round.Kernels.resolve(m[1]);
        k.pattn = try m[2].function(shape.pattn);
        try k.pattn.allowDynamicShared(shape.shared);
        k.kv_write = try m[3].function("tf_nemo_kv_write_segs");
        k.chunk = try m[3].function(shape.chunk);
        k.merge = try m[3].function(shape.merge);
        k.silu_mul = try m[4].function("tf_lfm2_silu_mul");
        k.lane = try cuda.qmmf.Lane.load(d, (try ctx.capability()) / 10);
        errdefer k.lane.unload();
        try loadPromptEnabled(&k.lane, d);
        k.norms = try cuda.norms.Norms.load(d);
        errdefer k.norms.unload();
        k.embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer k.embed.unload();
        k.vit = try cuda.vit.Kernels.load(d);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.vit.unload();
        k.embed.unload();
        k.norms.unload();
        k.lane.unload();
        for (&k.mods) |*m| m.unload();
    }

    fn go(f: Function, s: Stream, grid: [3]usize, block: usize, smem: u32, a: *Args) !void {
        const g: cuda.Dim3 = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) };
        try cuda.launch.launch(f, .{ .grid = g, .block = .{ .x = @intCast(block) }, .shared = smem }, s, a);
    }

    /// out [rows, f] = bf16(bf16(silu(gate)) * up) from rows gate | up [rows, 2 f] (lfm2.cu).
    pub fn siluMul(k: *const Kernels, s: Stream, gu: u64, out: u64, rows: usize, f: usize) !void {
        var a: Args = .{};
        a.add(gu);
        a.add(out);
        a.add(int(f));
        try go(k.silu_mul, s, .{ rows, (f + 255) / 256, 1 }, 256, 0, &a);
    }

    /// Rows k, v (kv [rows, 2 kv width]) into each stream's pages at its position on.
    pub fn kvWrite(k: *const Kernels, s: Stream, sg: AttnSegs, kv: u64, rows: usize) !void {
        const kvw = k.g.kvWidth();
        var a: Args = .{};
        a.add(sg);
        a.add(kv);
        a.add(int(2 * kvw));
        a.add(@as(c_int, 0));
        a.add(int(kvw));
        try go(k.kv_write, s, .{ rows, 1, 1 }, 256, 0, &a);
    }

    fn scale(k: *const Kernels) f32 {
        return 1.0 / @sqrt(@as(f32, @floatFromInt(k.g.head_dim)));
    }

    /// Decode rows: q [rows, heads, dim] over each stream's pages in 512-key chunks, merged in order into out and xs.
    pub fn attend(k: *const Kernels, s: Stream, sg: AttnSegs, q: u64, po: u64, pm: u64, pl: u64, out: u64, xs: u64, rows: usize, nch: usize, z: usize) !void {
        const g = k.g;
        var a: Args = .{};
        a.add(sg);
        a.add(q);
        for ([_]u64{ po, pm, pl }) |v| a.add(v);
        for ([_]usize{ g.qWidth(), g.heads, g.kv_heads, g.group(), pages.page_rows, nch }) |v| a.add(int(v));
        a.add(k.scale());
        try go(k.chunk, s, .{ rows, g.kv_heads, z }, 256, 0, &a);
        var b: Args = .{};
        b.add(sg);
        for ([_]u64{ po, pm, pl, out, xs }) |v| b.add(v);
        for ([_]usize{ g.heads, g.kv_heads, g.group(), pages.page_rows, nch }) |v| b.add(int(v));
        try go(k.merge, s, .{ rows, g.kv_heads, 1 }, 256, 0, &b);
    }

    /// A prompt chunk's rows at positions p0.. against its stream's pages (written through them): out [rows, H, dim].
    pub fn prompt(k: *const Kernels, s: Stream, q: u64, kc: u64, vc: u64, out: u64, p0: usize, rows: usize, pt: u64, ps: usize) !void {
        const g = k.g;
        var a: Args = .{};
        for ([_]u64{ q, kc, vc, out }) |v| a.add(v);
        for ([_]usize{ p0, rows, g.heads, g.kv_heads, g.group() }) |v| a.add(int(v));
        a.add(k.scale());
        a.add(pt);
        a.add(@as(i64, @intCast(ps)));
        try go(k.pattn, s, .{ k.shape.blocks(rows), k.shape.grid_y, 1 }, 256, k.shape.shared, &a);
    }
};

/// The loader's BF16 prompt enablement: on unless `TF_DENSE_PROMPT_BF16=0`; the host test drives both branches on a stub driver.
pub fn loadPromptEnabled(lane: *cuda.qmmf.Lane, d: *const cuda.Driver) !void {
    if (try @import("sys").environment.read(bool, "TF_DENSE_PROMPT_BF16", true)) try lane.loadPrompt(d);
}

test "AttnSegs is paged_attn.cuh's struct, padded to 8 bytes: 304" {
    try std.testing.expectEqual(@as(usize, 304), @sizeOf(AttnSegs));
    try std.testing.expectEqual(@as(usize, 256), @offsetOf(AttnSegs, "ps"));
    try std.testing.expectEqual(@as(usize, 296), @offsetOf(AttnSegs, "n"));
}

test "the loader's prompt enablement loads the tile by default and TF_DENSE_PROMPT_BF16=0 leaves it empty" {
    const Fake = struct {
        var fns: usize = 0;
        fn getFunction(out: *cuda.abi.Function, _: cuda.abi.Module, _: [*:0]const u8) callconv(.c) c_int {
            fns += 1;
            out.* = @ptrFromInt(fns);
            return 0;
        }
        fn setAttribute(_: cuda.abi.Function, _: cuda.abi.FunctionAttribute, _: c_int) callconv(.c) c_int {
            return 0;
        }
        fn ctxDevice(out: *cuda.abi.Device) callconv(.c) c_int {
            out.* = 0;
            return 0;
        }
        fn deviceAttribute(out: *c_int, a: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) c_int {
            out.* = switch (a) {
                .compute_capability_major => 12,
                .compute_capability_minor => 0,
                .multiprocessor_count => 48,
                else => 0,
            };
            return 0;
        }
        fn driver() cuda.Driver {
            fns = 0;
            var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
            d.api.cuModuleGetFunction = getFunction;
            d.api.cuFuncSetAttribute = setAttribute;
            d.api.cuCtxGetDevice = ctxDevice;
            d.api.cuDeviceGetAttribute = deviceAttribute;
            return d;
        }
    };
    const env = @import("sys").env;
    const name = "TF_DENSE_PROMPT_BF16";
    _ = env.unsetenv(name);
    defer _ = env.unsetenv(name);
    const d = Fake.driver();
    var lane: cuda.qmmf.Lane = .{
        .mod = .{ .d = &d, .handle = @ptrFromInt(1) },
        .fns = undefined,
        .f32_fns = undefined,
        .reduce = undefined,
        .half = undefined,
        .prompt = null,
        .prompt_k32 = null,
        .clusters = true,
    };
    // unset: the default loads the prompt tile through the real loadPrompt on the stub driver
    try loadPromptEnabled(&lane, &d);
    try std.testing.expect(lane.prompt != null);
    try std.testing.expect(lane.prompt_k32 != null);
    try std.testing.expectEqual(@as(u32, 48), lane.prompt_sms);
    // 0: the lane stays decode-only
    try std.testing.expectEqual(@as(c_int, 0), env.setenv(name, "0", 1));
    var off: cuda.qmmf.Lane = .{
        .mod = .{ .d = &d, .handle = @ptrFromInt(1) },
        .fns = undefined,
        .f32_fns = undefined,
        .reduce = undefined,
        .half = undefined,
        .prompt = null,
        .prompt_k32 = null,
        .clusters = true,
    };
    try loadPromptEnabled(&off, &d);
    try std.testing.expect(off.prompt == null and off.prompt_k32 == null);
}
