//! GLM glue kernels: the Python engine's Triton helpers (glue.py) and greedy sampling (decode.py).
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

fn grid1(x: usize) launch_.Dim3 {
    return .{ .x = @intCast(x), .y = 1, .z = 1 };
}

fn grid2(x: usize, y: usize) launch_.Dim3 {
    return .{ .x = @intCast(x), .y = @intCast(y), .z = 1 };
}

fn block(x: u32) launch_.Dim3 {
    return .{ .x = x, .y = 1, .z = 1 };
}

/// The loaded fatbin: embedding, norms, hyper-connections, MLP glue, router top-k, combine, greedy.
pub const Glue = struct {
    mod: Module,
    embed_b16_fn: Function,
    rmsnorm_fn: Function,
    hc_partial_fn: Function,
    hc_finish_fn: Function,
    hc_post_fn: Function,
    residual_add_fn: Function,
    stream_mean_fn: Function,
    swiglu_fn: Function,
    topk_fn: Function,
    combine_fn: Function,
    greedy_fn: Function,
    greedy_pack_fn: Function,

    pub fn load(d: *const driver.Driver) !Glue {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var g: Glue = .{
            .mod = try Module.load(d, kernels.glm_glue),
            .embed_b16_fn = undefined,
            .rmsnorm_fn = undefined,
            .hc_partial_fn = undefined,
            .hc_finish_fn = undefined,
            .hc_post_fn = undefined,
            .residual_add_fn = undefined,
            .stream_mean_fn = undefined,
            .swiglu_fn = undefined,
            .topk_fn = undefined,
            .combine_fn = undefined,
            .greedy_fn = undefined,
            .greedy_pack_fn = undefined,
        };
        errdefer g.mod.unload();
        g.embed_b16_fn = try g.mod.function("tf_glm_embed_b16");
        g.rmsnorm_fn = try g.mod.function("tf_glm_rmsnorm");
        g.hc_partial_fn = try g.mod.function("tf_glm_hc_partial");
        g.hc_finish_fn = try g.mod.function("tf_glm_hc_finish");
        g.hc_post_fn = try g.mod.function("tf_glm_hc_post");
        g.residual_add_fn = try g.mod.function("tf_glm_residual_add");
        g.stream_mean_fn = try g.mod.function("tf_glm_stream_mean");
        g.swiglu_fn = try g.mod.function("tf_glm_swiglu");
        g.topk_fn = try g.mod.function("tf_glm_topk");
        g.combine_fn = try g.mod.function("tf_glm_combine");
        g.greedy_fn = try g.mod.function("tf_glm_greedy");
        g.greedy_pack_fn = try g.mod.function("tf_glm_greedy_pack");
        return g;
    }

    pub fn unload(g: *Glue) void {
        g.mod.unload();
    }

    /// glue.embed_b16: out [rows, copies, d] bf16 = embedding rows `ids` copied `copies` times.
    pub fn embedB16(g: *const Glue, s: Stream, ids: u64, w: u64, out: u64, rows: usize, d: usize, copies: usize) !void {
        if (rows == 0 or d == 0 or d % 64 != 0 or copies == 0) return error.GlmGlueShape;
        var r: launch_.Args = .{};
        r.add(ids);
        r.add(w);
        r.add(out);
        r.add(@as(u32, @intCast(copies))); // the kernel reads rows and d from its grid
        try launch_.launch(g.embed_b16_fn, .{ .grid = grid2(rows, d / 64), .block = block(32), .shared = 0 }, s, &r);
    }

    /// glue.rmsnorm: out/xs [rows, d] and per-64 group sums; d must be 4096 (the served shape).
    pub fn rmsnorm(g: *const Glue, s: Stream, x: u64, xs: i64, w: u64, out: u64, os: i64, sums: u64, eps: f32, rows: usize, d: usize) !void {
        if (rows == 0 or d != 4096) return error.GlmGlueShape;
        var r: launch_.Args = .{};
        r.add(x);
        r.add(xs);
        r.add(w);
        r.add(out);
        r.add(os);
        r.add(sums);
        r.add(eps);
        r.add(@as(u32, @intCast(rows)));
        try launch_.launch(g.rmsnorm_fn, .{ .grid = grid1(rows), .block = block(256), .shared = 0 }, s, &r);
    }

    /// glue.hc_partial: part [rows, 16, 32] fp32 (24 fn dots + ss at slot 24); x [rows, 16384] bf16.
    pub fn hcPartial(g: *const Glue, s: Stream, x: u64, fn_w: u64, part: u64, rows: usize) !void {
        if (rows == 0) return error.GlmGlueShape;
        var r: launch_.Args = .{};
        r.add(x);
        r.add(fn_w);
        r.add(part);
        r.add(@as(u32, @intCast(rows)));
        try launch_.launch(g.hc_partial_fn, .{ .grid = .{ .x = @intCast(rows), .y = 16, .z = 6 }, .block = block(128), .shared = 0 }, s, &r);
    }

    /// glue.hc_finish: mixed/normed output, group sums, post gates [rows, 4] and comb [rows, 16].
    pub fn hcFinish(g: *const Glue, s: Stream, x: u64, part: u64, base: u64, scale: u64, nw: u64, out: u64, xs: u64, post: u64, comb: u64, eps_norm: f32, hc_eps: f32, iters: u32, rows: usize, d: usize) !void {
        if (rows == 0 or d != 4096) return error.GlmGlueShape;
        var r: launch_.Args = .{};
        r.add(x);
        r.add(part);
        r.add(base);
        r.add(scale);
        r.add(nw);
        r.add(out);
        r.add(xs);
        r.add(post);
        r.add(comb);
        r.add(eps_norm);
        r.add(hc_eps);
        r.add(@as(i32, @intCast(iters)));
        r.add(@as(u32, @intCast(rows)));
        try launch_.launch(g.hc_finish_fn, .{ .grid = grid1(rows), .block = block(256), .shared = 0 }, s, &r);
    }

    /// glue.hc_post: xout [rows, s, d] = gates mixing s streams; rank-1 partials sit at element stride rs.
    pub fn hcPost(g: *const Glue, s: Stream, x: u64, xout: u64, gr: u64, post: u64, comb: u64, rs: i64, world: usize, rows: usize, d: usize, n: usize) !void {
        if (rows == 0 or d == 0 or d % 1024 != 0 or n == 0) return error.GlmGlueShape;
        var r: launch_.Args = .{};
        r.add(x);
        r.add(xout);
        r.add(gr);
        r.add(post);
        r.add(comb);
        r.add(rs);
        r.add(@as(u32, @intCast(world)));
        r.add(@as(u32, @intCast(n)));
        try launch_.launch(g.hc_post_fn, .{ .grid = grid2(rows, d / 1024), .block = block(128), .shared = 0 }, s, &r);
    }

    /// glue.residual_add: xout = x + bf16(sum of the world's fp32 partials at stride rs).
    pub fn residualAdd(g: *const Glue, s: Stream, x: u64, xout: u64, gr: u64, rs: i64, world: usize, rows: usize, d: usize) !void {
        if (rows == 0 or d == 0 or d % 1024 != 0) return error.GlmGlueShape;
        var r: launch_.Args = .{};
        r.add(x);
        r.add(xout);
        r.add(gr);
        r.add(rs);
        r.add(@as(u32, @intCast(world)));
        r.add(@as(u32, @intCast(rows)));
        r.add(@as(u32, @intCast(d)));
        try launch_.launch(g.residual_add_fn, .{ .grid = grid2(rows, d / 1024), .block = block(128), .shared = 0 }, s, &r);
    }

    /// glue.stream_mean: out [rows, d] = bf16(mean of the s input streams).
    pub fn streamMean(g: *const Glue, s: Stream, x: u64, out: u64, rows: usize, d: usize, n: usize) !void {
        if (rows == 0 or d == 0 or d % 1024 != 0 or n == 0) return error.GlmGlueShape;
        var r: launch_.Args = .{};
        r.add(x);
        r.add(out);
        r.add(@as(u32, @intCast(n)));
        try launch_.launch(g.stream_mean_fn, .{ .grid = grid2(rows, d / 1024), .block = block(128), .shared = 0 }, s, &r);
    }

    /// glue.swiglu: out [rows, w] bf16 and group sums xs from the fused gate/up bf16 pair [rows, 2w].
    pub fn swiglu(g: *const Glue, s: Stream, gu: u64, out: u64, xs: u64, lim: f32, rows: usize, w: usize) !void {
        if (rows == 0 or w == 0 or w % 512 != 0) return error.GlmGlueShape;
        var r: launch_.Args = .{};
        r.add(gu);
        r.add(out);
        r.add(xs);
        r.add(lim);
        r.add(@as(u32, @intCast(rows)));
        r.add(@as(u32, @intCast(w)));
        try launch_.launch(g.swiglu_fn, .{ .grid = grid2(rows, w / 512), .block = block(128), .shared = 0 }, s, &r);
    }

    /// glue.topk: picks/wts [rows, topk+1]; the last slot is the shared-expert one-hot (ne, 1.0).
    pub fn topk(g: *const Glue, s: Stream, lg: u64, bias: u64, pick: u64, wts: u64, scale: f32, rows: usize, ne: usize, tk: usize) !void {
        if (rows == 0 or ne == 0 or ne > 511 or tk == 0 or tk > 16) return error.GlmGlueShape;
        var r: launch_.Args = .{};
        r.add(lg);
        r.add(bias);
        r.add(pick);
        r.add(wts);
        r.add(scale);
        r.add(@as(i32, @intCast(ne)));
        r.add(@as(i32, @intCast(tk)));
        try launch_.launch(g.topk_fn, .{ .grid = grid1(rows), .block = block(128), .shared = 0 }, s, &r);
    }

    /// glue.combine: out [rows, d] fp32 = sum over slots of wts[k] * y[r, k, :].
    pub fn combine(g: *const Glue, s: Stream, y: u64, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
        if (rows == 0 or d == 0 or d % 1024 != 0 or slots == 0) return error.GlmGlueShape;
        var r: launch_.Args = .{};
        r.add(y);
        r.add(wts);
        r.add(out);
        r.add(@as(u32, @intCast(slots)));
        try launch_.launch(g.combine_fn, .{ .grid = grid2(rows, d / 1024), .block = block(128), .shared = 0 }, s, &r);
    }

    /// decode greedy: tok [rows] int32 argmax of bf16 logits, ties to the lowest token id.
    pub fn greedy(g: *const Glue, s: Stream, lg: u64, tok: u64, rows: usize, n: usize) !void {
        if (rows == 0 or n == 0) return error.GlmGlueShape;
        var r: launch_.Args = .{};
        r.add(lg);
        r.add(tok);
        r.add(@as(u32, @intCast(n)));
        try launch_.launch(g.greedy_fn, .{ .grid = grid1(rows), .block = block(256), .shared = 0 }, s, &r);
    }

    /// sample_rows' greedy candidate per row: packed [rows, 2] f32 of the top value and its vocab id's bits (+offset).
    pub fn greedyPack(g: *const Glue, s: Stream, lg: u64, packed_: u64, rows: usize, n: usize, offset: u32) !void {
        if (rows == 0 or n == 0) return error.GlmGlueShape;
        var r: launch_.Args = .{};
        r.add(lg);
        r.add(packed_);
        r.add(@as(u32, @intCast(n)));
        r.add(offset);
        try launch_.launch(g.greedy_pack_fn, .{ .grid = grid1(rows), .block = block(256), .shared = 0 }, s, &r);
    }
};
