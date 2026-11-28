//! Ling bf16 routed experts using the existing grouped router, row MLPs and Kimi weighted combine.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("ling_checkpoint");
const kernels = @import("kimi_kernels");
const Rows = @import("ling_rows").Rows;
const Ref = kernels.Ref;
const RouteArgs = extern struct { rows: i32, ne: i32, topk: i32, ng: i32, keep: i32, scale: f32 };

pub const Moe = struct {
    library: mtl.Library,
    route: mtl.Pipeline,
    logits: mtl.Buffer,
    ids: mtl.Buffer,
    weights: mtl.Buffer,
    where: mtl.Buffer,
    experts: mtl.Buffer,
    routed: mtl.Buffer,
    shared: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    activation: mtl.Buffer,
    hidden: u32,
    inter: u32,

    pub fn init(rows: *Rows, hidden: u32, inter: u32) !Moe {
        const library = try mtl.Library.fromSource(rows.device, @import("ling_sources").route, mtl.CompileOptions.mlx());
        errdefer library.deinit();
        const route = try mtl.Pipeline.init(rows.device, library, "tf_ling_route", false);
        errdefer route.deinit();
        const r: usize = rows.max_rows;
        return .{
            .library = library,
            .route = route,
            .hidden = hidden,
            .inter = inter,
            .logits = try rows.buffer(r * 128 * 4),
            .ids = try rows.buffer(r * 8 * 4),
            .weights = try rows.buffer(r * 8 * 4),
            .where = try rows.buffer(r * 8 * 4),
            .experts = try rows.buffer(r * 8 * hidden * 2),
            .routed = try rows.buffer(r * hidden * 2),
            .shared = try rows.buffer(r * hidden * 2),
            .gate = try rows.buffer(r * inter * 2),
            .up = try rows.buffer(r * inter * 2),
            .activation = try rows.buffer(r * inter * 2),
        };
    }

    pub fn forward(self: *Moe, rows: *Rows, model: *ck.Checkpoint, prefix: []const u8, x: Ref, out: Ref, n: u32) !void {
        if (n == 0 or n > rows.max_rows) return error.RowCapacity;
        const router = try std.mem.concat(rows.allocator, u8, &.{ prefix, "gate.weight" });
        defer rows.allocator.free(router);
        try rows.linear(model, router, x, .{ .buf = self.logits }, n, self.hidden, 128, true);
        const bias_name = try std.mem.concat(rows.allocator, u8, &.{ prefix, "gate.expert_bias" });
        defer rows.allocator.free(bias_name);
        const bias = try model.get(bias_name, .f32, &.{128});
        const cb = rows.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(self.route);
        e.setBuffer(self.logits, 0, 0);
        e.setBuffer(bias.buffer, bias.offset, 1);
        e.setValue(RouteArgs{ .rows = @intCast(n), .ne = 128, .topk = 8, .ng = 8, .keep = 4, .scale = 2.5 }, 2);
        e.setBuffer(self.ids, 0, 3);
        e.setBuffer(self.weights, 0, 4);
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(32, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
        for (0..n) |t| for (0..8) |j| {
            const at = t * 8 + j;
            const id = self.ids.slice(i32, n * 8)[at];
            if (id < 0 or id >= 128) return error.InvalidExpert;
            const expert = try std.fmt.allocPrint(rows.allocator, "{s}experts.{d}.", .{ prefix, id });
            defer rows.allocator.free(expert);
            try rows.mlp(model, expert, x.at(t * self.hidden * 2), .{ .buf = self.experts, .off = at * self.hidden * 2 }, .{ .buf = self.gate }, .{ .buf = self.up }, .{ .buf = self.activation }, 1, self.hidden, self.inter);
            self.where.slice(u32, n * 8)[at] = @intCast(at);
        };
        const shared = try std.mem.concat(rows.allocator, u8, &.{ prefix, "shared_experts." });
        defer rows.allocator.free(shared);
        try rows.mlp(model, shared, x, .{ .buf = self.shared }, .{ .buf = self.gate }, .{ .buf = self.up }, .{ .buf = self.activation }, n, self.hidden, self.inter);
        const sum_cb = rows.queue.commandBuffer();
        const sum_e = sum_cb.compute(.serial);
        rows.k.combine(sum_e, .{ .buf = self.experts }, .{ .buf = self.where }, .{ .buf = self.weights }, .{ .buf = self.routed }, .{ .buf = self.ids }, n, false, 128, 0, 1, .{ .K = self.inter, .N = self.hidden, .topk = 8, .rows_max = rows.max_rows, .beta = 0, .lin = 0 });
        sum_e.end();
        sum_cb.commit();
        sum_cb.wait();
        if (sum_cb.failure() != null) return error.GpuFailed;
        try rows.add(.{ .buf = self.routed }, .{ .buf = self.shared }, out, n * self.hidden);
    }

    pub fn deinit(self: *Moe) void {
        self.route.deinit();
        self.library.deinit();
    }
};
