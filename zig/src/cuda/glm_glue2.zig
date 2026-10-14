//! glue.py's 1536/512 rmsnorms and the fp32-y combine: glm_glue.cu is at the file cap (glm_glue2.cu).
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

/// The loaded fatbin: the two narrow rmsnorms and the fp32 combine.
pub const Glue2 = struct {
    mod: Module,
    rmsnorm_512_fn: Function,
    rmsnorm_1536_fn: Function,
    combine_f32_fn: Function,

    pub fn load(d: *const driver.Driver) !Glue2 {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var g: Glue2 = .{
            .mod = try Module.load(d, kernels.glm_glue2),
            .rmsnorm_512_fn = undefined,
            .rmsnorm_1536_fn = undefined,
            .combine_f32_fn = undefined,
        };
        errdefer g.mod.unload();
        g.rmsnorm_512_fn = try g.mod.function("tf_glm_rmsnorm_512");
        g.rmsnorm_1536_fn = try g.mod.function("tf_glm_rmsnorm_1536");
        g.combine_f32_fn = try g.mod.function("tf_glm_combine_f32");
        return g;
    }

    pub fn unload(g: *Glue2) void {
        g.mod.unload();
    }

    /// glue.rmsnorm at d 512 or 1536: row-strided x/out ([rows, 2048] slices), sums [rows, d / 64].
    pub fn rmsnorm2(g: *const Glue2, s: Stream, x: u64, xs: i64, w: u64, out: u64, os: i64, sums: u64, eps: f32, rows: usize, d: usize) !void {
        if (rows == 0) return error.GlmGlueShape;
        const fn_ = switch (d) {
            512 => g.rmsnorm_512_fn,
            1536 => g.rmsnorm_1536_fn,
            else => return error.GlmGlueShape,
        };
        var r: launch_.Args = .{};
        r.add(x);
        r.add(xs);
        r.add(w);
        r.add(out);
        r.add(os);
        r.add(sums);
        r.add(eps);
        r.add(@as(u32, @intCast(rows)));
        try launch_.launch(fn_, .{ .grid = grid1(rows), .block = block(128), .shared = 0 }, s, &r);
    }

    /// glue.combine over fp32 per-slot outputs: out [rows, d] fp32 = sum over slots of wts[k] * y[r, k, :].
    pub fn combineF32(g: *const Glue2, s: Stream, y: u64, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
        if (rows == 0 or d == 0 or d % 1024 != 0 or slots == 0) return error.GlmGlueShape;
        var r: launch_.Args = .{};
        r.add(y);
        r.add(wts);
        r.add(out);
        r.add(@as(u32, @intCast(slots)));
        try launch_.launch(g.combine_f32_fn, .{ .grid = grid2(rows, d / 1024), .block = block(128), .shared = 0 }, s, &r);
    }
};

test {
    _ = Glue2;
}
