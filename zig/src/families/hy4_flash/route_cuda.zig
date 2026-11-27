//! Hy4-flash LatentMoE bitonic top-16 sigmoid router (`hy4_route.cu`), bit-equal to `moe_glue`.
const std = @import("std");
const cuda = @import("cuda");

const Stream = cuda.Stream;

pub const experts: usize = 512;
pub const top_k: usize = 16;

/// The exact shape served by `tf_hy4_route_m4`: 512 experts, top 16, normalized, no shared slot.
pub fn serves(n_experts: usize, k: usize, norm_topk: bool) bool {
    return n_experts == experts and k == top_k and norm_topk;
}

pub const Route = struct {
    mod: cuda.Module,
    m4: cuda.Function,
    m4_plan1: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Route {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.hy4_route);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .m4 = try mod.function("tf_hy4_route_m4"),
            .m4_plan1 = try mod.function("tf_hy4_route_m4_plan1"),
        };
    }

    pub fn unload(r: *Route) void {
        r.mod.unload();
        r.* = undefined;
    }

    /// fp32 logits [rows, 512] + fp32 bias [512] into int32 picks and fp32 weights [rows, 16].
    pub fn topk(r: *const Route, s: Stream, logits: u64, bias: u64, scale: f32, pick: u64, wts: u64, rows: usize) !void {
        try launch(r.m4, s, logits, bias, rows, scale, pick, wts);
    }

    /// Fused single-row top-16 + grouped MoE plan (`rows == 1`, `tile == 16`).
    pub fn topkPlan1(r: *const Route, s: Stream, logits: u64, bias: u64, scale: f32, pick: u64, wts: u64, members: u64, items: u64, counts: u64) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(bias);
        a.add(scale);
        a.add(pick);
        a.add(wts);
        a.add(members);
        a.add(items);
        a.add(counts);
        try cuda.launch.launch(r.m4_plan1, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &a);
    }
};

/// One block of 128 threads a row; `f` is `tf_hy4_route_m4` or a planted control.
pub fn launch(f: cuda.Function, s: Stream, logits: u64, bias: u64, rows: usize, scale: f32, pick: u64, wts: u64) !void {
    if (rows == 0) return;
    if (rows > std.math.maxInt(i32)) return error.RouteOutOfRange;
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(bias);
    a.add(@as(i32, @intCast(rows)));
    a.add(scale);
    a.add(pick);
    a.add(wts);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 128 } }, s, &a);
}

test "route_cuda: serves only 512 experts, top 16, normalized" {
    try std.testing.expect(serves(512, 16, true));
    try std.testing.expect(!serves(496, 16, true));
    try std.testing.expect(!serves(512, 8, true));
    try std.testing.expect(!serves(512, 16, false));
}
