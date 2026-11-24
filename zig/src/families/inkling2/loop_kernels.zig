//! The CUDA kernels owned by one Mini model, including its format-specific expert and dense projections.
const cuda = @import("cuda");
const ops = @import("ops.zig");
const attention = @import("kernels.zig");
const modelopt = @import("modelopt.zig");
const loop_math = @import("loop_math_cuda.zig");
const exl3_experts = @import("experts.zig");
const weights = @import("loop_weights.zig");

pub const Kernels = struct {
    glue: ops.Kernels,
    attn: attention.Kernels,
    moe: union(enum) { modelopt: modelopt.Kernels, exl3: exl3_experts.Kernels },
    dense: ?cuda.exl3.Dense,
    loop: loop_math.Kernels,
    sample_mod: cuda.Module,
    draws: cuda.sample.Draws,

    pub fn load(ctx: *const cuda.Context, exl3: bool) !Kernels {
        const d = ctx.d;
        var glue = try ops.Kernels.load(d);
        errdefer glue.unload();
        var attn = try attention.Kernels.load(d);
        errdefer attn.unload();
        var moe: @FieldType(Kernels, "moe") = if (exl3)
            .{ .exl3 = try exl3_experts.Kernels.load(d) }
        else
            .{ .modelopt = try modelopt.Kernels.load(d, @intCast(try ctx.attribute(.multiprocessor_count))) };
        errdefer switch (moe) {
            inline else => |*m| m.unload(),
        };
        var dense = if (exl3) try cuda.exl3.Dense.load(d) else null;
        errdefer if (dense) |*m| m.unload();
        var lm = try loop_math.Kernels.load(d);
        errdefer lm.unload();
        var sample_mod = try cuda.Module.load(d, cuda.kernels.sample);
        errdefer sample_mod.unload();
        return .{
            .glue = glue,
            .attn = attn,
            .moe = moe,
            .dense = dense,
            .loop = lm,
            .sample_mod = sample_mod,
            .draws = try cuda.sample.Draws.resolve(sample_mod),
        };
    }

    pub fn reserveDense(k: *Kernels, s: cuda.Stream, layers: []const weights.Layer) !void {
        const dense = if (k.dense) |*d| d else return;
        for (layers) |layer| {
            const a = layer.attn.exl3.?;
            for ([_]cuda.exl3.Weight{ a.q, a.k, a.v, a.o, a.gate }) |w| try dense.reserve(s, w);
            try dense.reserveGroup(s, &.{ a.q, a.k, a.v });
        }
    }

    pub fn project(d: *cuda.exl3.Dense, s: cuda.Stream, w: cuda.exl3.Weight, x: u64, rows: usize, out: u64) !void {
        if (rows <= 16) return d.decode(s, w, x, .bf16, rows, out, .bf16);
        return d.prompt(s, w, x, .bf16, rows, out, .bf16);
    }

    pub fn deinit(k: *Kernels) void {
        k.sample_mod.unload();
        k.loop.unload();
        if (k.dense) |*d| d.unload();
        switch (k.moe) {
            inline else => |*m| m.unload(),
        }
        k.attn.unload();
        k.glue.unload();
        k.* = undefined;
    }
};
