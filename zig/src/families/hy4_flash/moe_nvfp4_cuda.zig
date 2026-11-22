//! NVFP4 routed experts for the Hy4 latent branch: ModelOpt's per-expert tensors on core's grouped kernels, one rank.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const slice = core.nvfp4_slice;
const fx = cuda.experts.Nvfp4Experts;

pub const Route = @import("moe_exl3_cuda.zig").Route;

/// Decode's item tile: an item holds at most 16 of one expert's pairs.
const tile = 16;

pub const Kernels = struct {
    module: cuda.Module,
    plan: cuda.grouped.Router,
    experts: fx,

    pub fn load(d: *const cuda.Driver, sms: usize) !Kernels {
        var module = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer module.unload();
        const plan = try cuda.grouped.Router.resolve(module);
        return .{ .module = module, .plan = plan, .experts = try fx.load(d, sms) };
    }

    pub fn unload(k: *Kernels) void {
        k.experts.unload();
        k.module.unload();
    }
};

pub const Routed = struct {
    buffers: [9]cuda.DeviceBuffer, // up, down, up and down global scales, members, items, counts, rank, hist
    act: cuda.DeviceBuffer, // bf16 [pairs][intermediate]
    y: cuda.DeviceBuffer, // fp32 [pairs][latent]
    layer: cuda.experts.Layer,
    rows: usize,
    slots: usize,

    /// Every expert of `prefix` ("model.layers.N.mlp"): gate and up [intermediate, latent], down the other way.
    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, latent: usize, intermediate: usize, experts: usize, rows: usize, slots: usize, limit: f32) !Routed {
        if (experts == 0 or experts > cuda.grouped.max_experts or latent == 0 or latent % 32 != 0 or
            intermediate == 0 or intermediate % 32 != 0 or rows == 0 or slots == 0 or slots > experts or
            !std.math.isFinite(limit) or limit <= 0)
            return error.BadHy4ExpertShape;
        const pairs = std.math.mul(usize, rows, slots) catch return error.BadHy4ExpertShape;
        const up_bytes = fx.bytes(1, intermediate, latent, 2);
        const down_bytes = fx.bytes(1, latent, intermediate, 1);
        var made: usize = 0;
        var buffers: [9]cuda.DeviceBuffer = undefined;
        errdefer for (buffers[0..made]) |*b| b.free();
        for ([_]usize{ experts * up_bytes, experts * down_bytes }) |bytes| {
            buffers[made] = try cuda.DeviceBuffer.alloc(d, bytes);
            made += 1;
        }
        const up_globals = try a.alloc(f32, experts * 2);
        defer a.free(up_globals);
        const down_globals = try a.alloc(f32, experts);
        defer a.free(down_globals);
        const packed_up = try a.alloc(u32, up_bytes / 4);
        defer a.free(packed_up);
        const packed_down = try a.alloc(u32, down_bytes / 4);
        defer a.free(packed_down);
        for (0..experts) |e| {
            var gate = try read(a, store, prefix, e, "gate_proj", intermediate, latent);
            defer gate.deinit(a);
            var up = try read(a, store, prefix, e, "up_proj", intermediate, latent);
            defer up.deinit(a);
            fx.packGateUp(packed_up, .{ gate.codes, gate.scales }, .{ up.codes, up.scales }, intermediate, latent);
            try buffers[0].upload(e * up_bytes, std.mem.sliceAsBytes(packed_up));
            up_globals[e * 2] = gate.global;
            up_globals[e * 2 + 1] = up.global;
            var down = try read(a, store, prefix, e, "down_proj", latent, intermediate);
            defer down.deinit(a);
            fx.packOne(packed_down, down.codes, down.scales, latent, intermediate);
            try buffers[1].upload(e * down_bytes, std.mem.sliceAsBytes(packed_down));
            down_globals[e] = down.global;
        }
        buffers[2] = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(up_globals));
        made += 1;
        buffers[3] = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(down_globals));
        made += 1;
        // the plan's scratch, as grouped.zig's kernels write it at `pairs` pairs over `experts` experts
        const plan_bytes = [5]usize{ pairs * 4, cuda.grouped.maxItems(pairs, experts, tile) * 12, 8, pairs * 4, (pairs + 1023) / 1024 * experts * 4 };
        for (plan_bytes) |bytes| {
            buffers[made] = try cuda.DeviceBuffer.alloc(d, bytes);
            made += 1;
        }
        var act = try cuda.DeviceBuffer.alloc(d, pairs * intermediate * 2);
        errdefer act.free();
        const y = try cuda.DeviceBuffer.alloc(d, pairs * latent * 4);
        return .{
            .buffers = buffers,
            .act = act,
            .y = y,
            .layer = .{ .format = .nvfp4, .up = buffers[0].ptr, .down = buffers[1].ptr, .up_scale = buffers[2].ptr, .down_scale = buffers[3].ptr, .width = intermediate, .dims = latent, .experts = experts, .limit = limit },
            .rows = rows,
            .slots = slots,
        };
    }

    pub fn free(self: *Routed) void {
        for (&self.buffers) |*b| b.free();
        self.act.free();
        self.y.free();
        self.* = undefined;
    }

    fn plan(self: *const Routed) cuda.grouped.Plan {
        const b = &self.buffers;
        return .{ .members = b[4].ptr, .items = b[5].ptr, .counts = b[6].ptr, .rank = b[7].ptr, .hist = b[8].ptr };
    }

    /// The router, its top-k, the experts and the weighted sum into `route.output` (fp32 [rows][latent]).
    pub fn forward(self: *const Routed, k: *const Kernels, router: *const cuda.route_rows.RouteRows, glue: *const cuda.moe_glue.MoeGlue, stream: cuda.Stream, latent: u64, rows: usize, route: Route) !void {
        if (rows == 0 or rows > self.rows or route.width == 0 or !std.math.isFinite(route.scale) or route.scale <= 0)
            return error.BadHy4Route;
        try router.logits(stream, route.hidden, route.width, route.router, route.logits, rows, route.width, self.layer.experts);
        try glue.topk(stream, route.logits, route.bias, .{
            .experts = self.layer.experts,
            .topk = self.slots,
            .score = true,
            .shared = false,
            .normalize = true,
            .scale = route.scale,
        }, route.picks, route.probabilities, rows);
        const output = try self.run(k, stream, latent, route.picks, rows);
        try glue.combine(stream, output, .f32, route.probabilities, route.output, .f32, rows, self.slots, self.layer.dims);
    }

    /// bf16 latent rows `x` [rows][latent] through each row's `slots` picked experts: fp32 [rows * slots][latent].
    pub fn run(self: *const Routed, k: *const Kernels, stream: cuda.Stream, x: u64, picks: u64, rows: usize) !u64 {
        if (rows == 0 or rows > self.rows) return error.BadHy4ExpertRows;
        const pairs = rows * self.slots;
        const p = self.plan();
        try k.plan.route(stream, picks, pairs, self.layer.experts, tile, p);
        const items = cuda.grouped.maxItems(pairs, self.layer.experts, tile);
        try k.experts.run(stream, .up, .{ .x = x, .stride = self.layer.dims, .slots = self.slots }, self.layer, p, self.act.ptr, items, -1);
        try k.experts.run(stream, .down_f32, .{ .x = self.act.ptr, .stride = self.layer.width }, self.layer, p, self.y.ptr, items, -1);
        return self.y.ptr;
    }
};

/// One projection of expert `e`, refused unless stored as ModelOpt NVFP4 at [n, k].
fn read(a: std.mem.Allocator, store: anytype, prefix: []const u8, e: usize, projection: []const u8, n: usize, k: usize) !slice.Linear {
    defer if (comptime @hasDecl(@typeInfo(@TypeOf(store)).pointer.child, "release")) store.release();
    var buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}.experts.{d}.{s}", .{ prefix, e, projection });
    var weight_name: [264]u8 = undefined;
    const w = try store.get(try std.fmt.bufPrint(&weight_name, "{s}.weight", .{name}));
    if (k % 32 != 0 or !w.is(.u8, &.{ n, k / 2 })) return error.BadHy4ExpertShape; // never a slice of a wider tensor
    var linear = try slice.read(store, a, name, .{ .from = 0, .to = n }, .{ .from = 0, .to = k });
    errdefer linear.deinit(a);
    if (linear.n != n or linear.k != k or !std.math.isFinite(linear.global) or linear.global <= 0) return error.BadHy4ExpertShape;
    return linear;
}
