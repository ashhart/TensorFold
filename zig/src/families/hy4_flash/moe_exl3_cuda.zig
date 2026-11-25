//! Standard MUL1 routed experts for the Hy4 latent branch.
const std = @import("std");
const cuda = @import("cuda");
const weights = @import("exl3_weights.zig");
const exl3 = cuda.exl3_experts;
const upload = cuda.exl3_upload;

pub const Route = struct {
    hidden: u64,
    width: usize,
    router: u64,
    bias: u64,
    logits: u64,
    picks: u64,
    probabilities: u64,
    scale: f32,
    output: u64,
};

pub const Routed = struct {
    layer: exl3.Layer,
    scratch: exl3.Scratch,

    pub fn init(
        owner: *upload.Uploader,
        experts: []const weights.Expert,
        latent: usize,
        intermediate: usize,
        rows: usize,
        slots: usize,
        limit: f32,
    ) !Routed {
        if (experts.len == 0 or experts.len > 512 or rows == 0 or slots == 0 or slots > 32 or
            rows > 16384 / slots or !std.math.isFinite(limit) or limit <= 0)
            return error.BadHy4ExpertShape;
        _ = try exl3.plan(latent, intermediate, true);
        _ = try exl3.plan(intermediate, latent, false);
        const a = owner.gpa;
        const projections = try a.alloc(upload.Proj, 3 * experts.len);
        defer a.free(projections);
        const gate = projections[0..experts.len];
        const up = projections[experts.len..][0..experts.len];
        const down = projections[2 * experts.len ..];
        for (experts, gate, up, down) |e, *g, *u, *d| {
            g.* = try projection(e.gate, latent, intermediate);
            u.* = try projection(e.up, latent, intermediate);
            d.* = try projection(e.down, intermediate, latent);
        }
        const mark = owner.bufs.items.len;
        errdefer owner.rollback(mark);
        const layer = try upload.layer(owner, a, latent, intermediate, .mul1, limit, 0, gate, up, down);
        const size = try exl3.scratchBytes(layer, rows, slots);
        const space = try a.alloc(u8, size);
        defer a.free(space);
        @memset(space, 0);
        const base = try owner.dev(space);
        return .{ .layer = layer, .scratch = try exl3.scratch(base, layer, rows, slots) };
    }

    pub fn forward(
        self: Routed,
        kernels: *const exl3.Experts,
        router: *const cuda.route_rows.RouteRows,
        glue: *const cuda.moe_glue.MoeGlue,
        stream: cuda.Stream,
        latent: u64,
        rows: usize,
        route: Route,
    ) !void {
        const slots = self.scratch.slots;
        if (rows == 0 or rows > self.scratch.rows or slots > self.layer.experts or
            route.width == 0 or !std.math.isFinite(route.scale) or route.scale <= 0)
            return error.BadHy4Route;
        try router.logits(stream, route.hidden, route.width, route.router, route.logits, rows, route.width, self.layer.experts);
        try glue.topk(stream, route.logits, route.bias, .{
            .experts = self.layer.experts,
            .topk = slots,
            .score = true,
            .shared = false,
            .normalize = true,
            .scale = route.scale,
        }, route.picks, route.probabilities, rows);
        const output = try self.run(kernels, stream, latent, route.picks, rows);
        try glue.combine(stream, output, .f32, route.probabilities, route.output, .f32, rows, slots, self.layer.dims);
    }

    pub fn run(self: Routed, kernels: *const exl3.Experts, stream: cuda.Stream, x: u64, picks: u64, rows: usize) !u64 {
        if (rows == 0 or rows > self.scratch.rows) return error.BadHy4ExpertRows;
        try kernels.runSlots(stream, self.layer, x, self.layer.dims, .bf16, picks, rows, self.scratch);
        return self.scratch.y;
    }
};

fn projection(p: weights.Projection, k: usize, n: usize) !upload.Proj {
    if (p.geometry.k() != k or p.geometry.n() != n) return error.BadHy4ExpertShape;
    return .{
        .trellis = p.parts.trellis.bytes,
        .suh = p.parts.suh.bytes,
        .svh = p.parts.svh.bytes,
        .k2 = p.geometry.k2(),
    };
}
