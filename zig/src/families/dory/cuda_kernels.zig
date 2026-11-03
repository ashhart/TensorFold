//! Dory's launches: the shared operators of the forward, plus fp32 argmax, bf16 casts and keyed draws for serving.
const std = @import("std");
const cuda = @import("cuda");

pub const Kernels = struct {
    lane: cuda.qmmf.Lane,
    prompt: cuda.nvfp4.Prompt,
    embed: cuda.embed_rows.EmbedRows,
    norms: cuda.norms.Norms,
    qk: cuda.qk_rows.QkRows,
    act: cuda.act_rows.ActRows,
    dense: cuda.dense_rows.DenseRows,
    win: cuda.window_attention.WindowAttention,
    mods: [3]cuda.Module = undefined, // sample, torch_argmax, torch_pointwise (null when built without kernels)
    served: bool = false,
    argmax_fn: cuda.Function = undefined,
    to_bf16_fn: cuda.Function = undefined,
    draws: cuda.sample.Draws = undefined,

    /// `paged`: also window_pages.cu (the paged caches' attention); else that module is not loaded.
    pub fn load(ctx: *const cuda.Context, paged: bool) !Kernels {
        const d = ctx.d;
        var k: Kernels = undefined;
        k.served = false;
        k.lane = try cuda.qmmf.Lane.load(d, (try ctx.capability()) / 10);
        errdefer k.lane.unload();
        k.prompt = try cuda.nvfp4.Prompt.load(d);
        errdefer k.prompt.unload();
        try k.prompt.enableBf16();
        k.embed = try cuda.embed_rows.EmbedRows.load(d);
        errdefer k.embed.unload();
        k.norms = try cuda.norms.Norms.load(d);
        errdefer k.norms.unload();
        k.qk = try cuda.qk_rows.QkRows.load(d);
        errdefer k.qk.unload();
        k.act = try cuda.act_rows.ActRows.load(d);
        errdefer k.act.unload();
        k.dense = try cuda.dense_rows.DenseRows.load(d);
        errdefer k.dense.unload();
        k.win = try cuda.window_attention.WindowAttention.load(d);
        errdefer k.win.unload();
        if (paged) try k.win.loadPages(d);
        if (!cuda.kernels.available) return k;
        const kk = cuda.kernels;
        const images = [_][]const u8{ kk.sample, kk.torch_argmax, kk.torch_pointwise };
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            k.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        k.draws = try cuda.sample.Draws.resolve(k.mods[0]);
        k.argmax_fn = try k.mods[1].function("tf_argmax_rows_i32_kernel");
        k.to_bf16_fn = try k.mods[2].function("tf_f32_to_bf16_kernel");
        k.served = true;
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        if (k.served) for (&k.mods) |*m| m.unload();
        k.win.unload();
        k.dense.unload();
        k.act.unload();
        k.qk.unload();
        k.norms.unload();
        k.embed.unload();
        k.prompt.unload();
        k.lane.unload();
    }

    pub fn linear(k: *const Kernels) cuda.qlinear.Linear {
        return .{ .lane = &k.lane, .nvfp4_prompt = &k.prompt, .bf16_prompt = true };
    }

    /// Each row's argmax over `v` fp32 logits, the first maximum on ties (torch.argmax), into `out` (int32 a row).
    pub fn argmax(k: *const Kernels, s: cuda.Stream, logits: u64, rows: usize, v: usize, out: u64) !void {
        if (!k.served) return error.BuiltWithoutKernels;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(out);
        for ([_]usize{ rows, v, v }) |x| a.add(@as(i64, @intCast(x)));
        a.add(@as(i32, 1)); // dtype 1: fp32
        try cuda.launch.launch(k.argmax_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }

    /// bf16 copies of `n` fp32 logits (round to nearest even), as sample.cu's draws read them.
    pub fn toBf16(k: *const Kernels, s: cuda.Stream, x: u64, y: u64, n: usize) !void {
        if (!k.served) return error.BuiltWithoutKernels;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(@as(u64, n));
        const grid: u32 = @intCast(@max(1, @min((n + 255) / 256, 65535)));
        try cuda.launch.launch(k.to_bf16_fn, .{ .grid = .{ .x = grid }, .block = .{ .x = 256 } }, s, &a);
    }
};

test "the serving launches type-check against the shared operators" {
    std.testing.refAllDecls(Kernels);
}

test "BF16 prompt port: Dory observes the threshold, cached switch and decode path" {
    const Mock = struct {
        var selected: usize = 0;

        fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            selected = @intFromPtr(f);
            return cuda.abi.success;
        }

        fn function(d: *const cuda.Driver, id: usize) cuda.Function {
            return .{ .d = d, .handle = @ptrFromInt(id) };
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Mock.launch;
    const s: cuda.Stream = .{ .d = &d, .handle = null };
    const lane_fn = Mock.function(&d, 1);
    const prompt_fn = Mock.function(&d, 2);
    var k: Kernels = undefined;
    k.lane = .{ .mod = .{ .d = &d, .handle = null }, .fns = @splat(@splat(lane_fn)), .f32_fns = @splat(@splat(lane_fn)), .reduce = @splat(lane_fn), .half = @splat(@splat(@splat(lane_fn))), .clusters = false };
    k.prompt = .{ .mod = .{ .d = &d, .handle = null }, .fp4 = prompt_fn, .fp8 = prompt_fn, .fp4_f32 = prompt_fn, .lean = prompt_fn, .lean_f32 = prompt_fn, .sms = 16, .bf16_serving = true, .bf16 = Mock.function(&d, 3), .bf16_f32 = Mock.function(&d, 4) };
    const linear = k.linear();
    const w: cuda.qmmf.Weight = .{ .mode = .bf16, .codes = 0x2000, .scales = 0, .n = 48, .k = 5120, .npad = 128 };
    for ([_]usize{ 1023, 1024, 2048 }) |rows| {
        try linear.prompt(s, .{ .bf16 = w }, .{ .x = 0x1000 }, 0x3000, rows);
        try std.testing.expectEqual(@as(usize, if (rows >= 1024) 3 else 1), Mock.selected);
    }
    k.prompt.bf16_serving = false;
    try linear.prompt(s, .{ .bf16 = w }, .{ .x = 0x1000 }, 0x3000, 2048);
    try std.testing.expectEqual(@as(usize, 1), Mock.selected);
    k.prompt.bf16_serving = true;
    try linear.decode(s, .{ .bf16 = w }, .{ .x = 0x1000, .part = 0x4000 }, 0x3000, 1);
    try std.testing.expectEqual(@as(usize, 1), Mock.selected);
}
