//! The kernels Ripple's CUDA forward loads together: glue, prep, attention, the lane and prompt matmuls, argmax.
const cuda = @import("cuda");
const glue = @import("cuda_kernels.zig");
const rp = @import("cuda_prep.zig");
const attn = @import("cuda_attn.zig");

const wa = cuda.window_attention;
const Stream = cuda.Stream;

pub const Kernels = struct {
    glue: glue.Kernels,
    prep: rp.Kernels,
    win: wa.WindowAttention, // sliding decode (tf_window_decode_tc); owns the prompt attention's module
    attn: attn.Kernels,
    lane: cuda.qmmf.Lane,
    prompt: cuda.nvfp4.Prompt,
    argmax_mod: cuda.Module,
    argmax: cuda.Function, // torch.argmax's first maximum, int32 a row

    pub fn load(ctx: *const cuda.Context) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const d = ctx.d;
        var k: Kernels = undefined;
        k.glue = try glue.Kernels.load(d);
        errdefer k.glue.unload();
        k.prep = try rp.Kernels.load(d);
        errdefer k.prep.unload();
        k.win = try wa.WindowAttention.load(d);
        errdefer k.win.unload();
        k.attn = try attn.Kernels.init(&k.win);
        k.lane = try cuda.qmmf.Lane.load(d, (try ctx.capability()) / 10);
        errdefer k.lane.unload();
        k.prompt = try cuda.nvfp4.Prompt.load(d);
        errdefer k.prompt.unload();
        k.argmax_mod = try cuda.Module.load(d, cuda.kernels.torch_argmax);
        errdefer k.argmax_mod.unload();
        k.argmax = try k.argmax_mod.function("tf_argmax_rows_i32_kernel");
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.argmax_mod.unload();
        k.prompt.unload();
        k.lane.unload();
        k.win.unload();
        k.prep.unload();
        k.glue.unload();
        k.* = undefined;
    }

    /// int32 argmax of `rows` bf16 rows of `vocab` logits (`stride` apart) into `out`.
    pub fn argmaxRows(k: *const Kernels, s: Stream, logits: u64, rows: usize, vocab: usize, stride: usize, out: u64) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(out);
        for ([_]usize{ rows, vocab, stride }) |x| a.add(@as(i64, @intCast(x)));
        a.add(@as(i32, 0)); // dtype 0: bf16
        try cuda.launch.launch(k.argmax, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 1024 }, .shared = 0 }, s, &a);
    }
};
