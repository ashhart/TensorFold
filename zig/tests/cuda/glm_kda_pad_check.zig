//! glm-kda-pad-split (GPU): TF_GLM_KDA_PAD pads the bf16 KDA stack and the split stack's bf16 rest at TP=3.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm = @import("glm_cuda");

const dims = 64; // the stack's k: any width shows the rows

pub fn run(gpu: check.Gpu) !void {
    const gpa = gpu.gpa;
    const kb = glm.kda_block;
    const dk = glm.kda.dk;
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    const c = glm.config.Config{};
    const world = 3;
    const max_heads = (c.kda_heads + world - 1) / world;
    kb.pad_to = std.mem.alignForward(usize, 3 * max_heads * dk + 2 * dk + max_heads, 8);
    defer kb.pad_to = 0;
    var cases: usize = 0;
    for (0..world) |rank| {
        const heads = (try glm.shard.Share.of(&c, rank, world)).kda.len();
        for ([_]bool{ false, true }) |split| {
            const fa: usize = if (split) 3 * heads * dk else 0;
            const n = 3 * heads * dk + 2 * dk + heads - fa; // the bf16 rows the checkpoint holds
            const m = kb.pad_to - fa; // the rows the padded width reads
            const host = try gpa.alloc(u16, n * dims);
            defer gpa.free(host);
            for (host, 0..) |*v, i| v.* = @intCast((i * 7 + 1) % 0x7f00 + 1); // no zero, so a pad row cannot pass as one
            var orig = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
            defer orig.free();
            const in: kb.In = if (split) .{ .split = .{ .qkv = undefined, .rest = orig.ptr } } else .{ .bf16 = orig.ptr };
            var layers = [1]glm.weights.Layer{undefined};
            layers[0].mixer = .{ .kda = .{ .proj = in, .fb = 0, .gb = 0, .o = .{ .bf16 = 0 }, .conv_w = 0, .a_log = 0, .dt_bias = 0, .norm_w = 0, .eps = 1e-6, .lower = -5, .heads = heads, .dims = dims } };
            var w: glm.weights.Weights = undefined;
            w.gpa = gpa;
            w.layers = &layers;
            w.mtp = null;
            w.buffers = .empty;
            defer {
                for (w.buffers.items) |*b| b.free();
                w.buffers.deinit(gpa);
            }
            try glm.tile.padKdaInputs(&w, gpu.d, s);
            const got = switch (layers[0].mixer.kda.proj) {
                .bf16 => |p| p,
                .split => |sp| sp.rest,
            };
            const label = if (split) "split stack's bf16 rest" else "bf16 stack";
            try check.expect(got != orig.ptr, "glm-kda-pad-split: rank {d} ({d} heads) {s} kept its unpadded buffer", .{ rank, heads, label });
            try check.expect(w.buffers.items.len == 1 and w.buffers.items[0].ptr == got and w.buffers.items[0].len == m * dims * 2, "glm-kda-pad-split: rank {d} {s}: the padded buffer is not the weights' own {d} rows", .{ rank, label, m });
            const back = try gpa.alloc(u16, m * dims);
            defer gpa.free(back);
            try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = got, .len = m * dims * 2, .borrowed = true }).download(0, std.mem.sliceAsBytes(back));
            try check.expect(std.mem.eql(u16, back[0 .. n * dims], host), "glm-kda-pad-split: rank {d} {s}: the true rows moved", .{ rank, label });
            for (back[n * dims ..]) |v| try check.expect(v == 0, "glm-kda-pad-split: rank {d} {s}: a pad row is not zero", .{ rank, label });
            std.debug.print("rank {d}, {d} heads, {s}: {d} true rows padded to {d}\n", .{ rank, heads, label, n, m });
            cases += 1;
        }
    }
    check.pass("glm-kda-pad-split: {d} stacks at TP=3 padded to {d} columns, true rows then zeros", .{ cases, kb.pad_to });
}
