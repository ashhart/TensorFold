//! Ling KDA state and unchanged k3_kda dispatch, with bf16 checkpoint vectors widened exactly to fp32.
const std = @import("std");
const mtl = @import("metal");
const kernels = @import("kimi_kernels");
const ck = @import("ling_checkpoint");
const Rows = @import("ling_rows").Rows;
const Ref = kernels.Ref;

pub const Kda = struct {
    state: mtl.Buffer,
    conv: mtl.Buffer,
    segment: mtl.Buffer,
    dummy: mtl.Buffer,
    weights: kernels.KdaWeights,
    heads: u32,
    readers: [6]Ref,
    proj: mtl.Buffer,
    f: mtl.Buffer,
    b: mtl.Buffer,
    g: mtl.Buffer,
    y: mtl.Buffer,

    fn vector(rows: *Rows, model: *ck.Checkpoint, prefix: []const u8, name: []const u8, shape: []const usize, bf16: bool) !Ref {
        const full = try std.mem.concat(rows.allocator, u8, &.{ prefix, name });
        defer rows.allocator.free(full);
        const t = try model.get(full, if (bf16) .bf16 else .f32, shape);
        if (!bf16) return .{ .buf = t.buffer, .off = t.offset };
        var count: usize = 1;
        for (shape) |dim| count *= dim;
        const out = try rows.buffer(count * 4);
        const data = std.mem.bytesAsSlice(u16, t.buffer.contents()[t.offset..][0 .. count * 2]);
        for (data, out.slice(f32, count)) |h, *f| f.* = @bitCast(@as(u32, h) << 16);
        return .{ .buf = out };
    }

    pub fn init(rows: *Rows, model: *ck.Checkpoint, prefix: []const u8, heads: u32) !Kda {
        const width = heads * 128;
        const state = try rows.buffer(@as(usize, width) * 128 * 4);
        const conv = try rows.buffer(9 * width * 2);
        @memset(state.slice(f32, width * 128), 0);
        @memset(conv.slice(u16, 9 * width), 0);
        const readers = [6]Ref{
            try vector(rows, model, prefix, "q_conv1d.weight", &.{ width, 1, 4 }, true),
            try vector(rows, model, prefix, "k_conv1d.weight", &.{ width, 1, 4 }, true),
            try vector(rows, model, prefix, "v_conv1d.weight", &.{ width, 1, 4 }, true),
            try vector(rows, model, prefix, "A_log", &.{heads}, false),
            try vector(rows, model, prefix, "dt_bias", &.{width}, false),
            try vector(rows, model, prefix, "o_norm.weight", &.{128}, true),
        };
        return .{
            .state = state,
            .conv = conv,
            .segment = try rows.buffer(@sizeOf(kernels.KdaSeg)),
            .dummy = try rows.buffer(16),
            .heads = heads,
            .proj = try rows.buffer(@as(usize, rows.max_rows) * 3 * width * 2),
            .f = try rows.buffer(@as(usize, rows.max_rows) * width * 2),
            .b = try rows.buffer(@as(usize, rows.max_rows) * heads * 2),
            .g = try rows.buffer(@as(usize, rows.max_rows) * width * 2),
            .y = try rows.buffer(@as(usize, rows.max_rows) * width * 2),
            .readers = readers,
            .weights = .{ .conv_q = kernels.addr(readers[0]), .conv_k = kernels.addr(readers[1]), .conv_v = kernels.addr(readers[2]), .A_log = kernels.addr(readers[3]), .dt_bias = kernels.addr(readers[4]), .o_norm = kernels.addr(readers[5]) },
        };
    }

    pub fn forward(self: *Kda, rows: *Rows, model: *ck.Checkpoint, prefix: []const u8, x: Ref, out: Ref, n: u32, hidden: u32, lower_bound: f32, eps: f32) !void {
        const width = self.heads * 128;
        for ([_][]const u8{ "q_proj.weight", "k_proj.weight", "v_proj.weight" }, 0..) |part, j| {
            const name = try std.mem.concat(rows.allocator, u8, &.{ prefix, part });
            defer rows.allocator.free(name);
            try rows.linearInto(model, name, x, .{ .buf = self.proj, .off = j * width * 2 }, n, hidden, width, hidden, 3 * width, false);
        }
        for ([_]struct { name: []const u8, buffer: mtl.Buffer, width: u32 }{
            .{ .name = "f_proj.weight", .buffer = self.f, .width = width },
            .{ .name = "b_proj.weight", .buffer = self.b, .width = self.heads },
            .{ .name = "g_proj.weight", .buffer = self.g, .width = width },
        }) |p| {
            const name = try std.mem.concat(rows.allocator, u8, &.{ prefix, p.name });
            defer rows.allocator.free(name);
            try rows.linear(model, name, x, .{ .buf = p.buffer }, n, hidden, p.width, false);
        }
        try self.mix(rows, .{ .buf = self.proj }, .{ .buf = self.f }, .{ .buf = self.b }, .{ .buf = self.g }, .{ .buf = self.y }, n, lower_bound, eps);
        const name = try std.mem.concat(rows.allocator, u8, &.{ prefix, "o_proj.weight" });
        defer rows.allocator.free(name);
        try rows.linear(model, name, .{ .buf = self.y }, out, n, width, hidden, false);
    }

    pub fn mix(self: *Kda, rows: *Rows, qkv: Ref, f: Ref, b: Ref, g: Ref, out: Ref, n: u32, lower_bound: f32, eps: f32) !void {
        if (n == 0 or n > rows.max_rows) return error.RowCapacity;
        self.segment.slice(kernels.KdaSeg, 1)[0] = .{ .first = 0, .rows = n, .kept = 0, .flags = kernels.seg_commit, .slot = 0 };
        const state = kernels.KdaState{
            .S = self.state.gpuAddress(),
            .conv = self.conv.gpuAddress(),
            .log_a = self.dummy.gpuAddress(),
            .log_k = self.dummy.gpuAddress(),
            .log_u = self.dummy.gpuAddress(),
            .log_x = self.dummy.gpuAddress(),
        };
        const width = self.heads * 128;
        const cb = rows.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.useResource(self.state, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        e.useResource(self.conv, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
        e.useResource(self.dummy, mtl.ResourceUsage.read);
        for (self.readers) |r| e.useResource(r.buf, mtl.ResourceUsage.read);
        rows.k.kdaRound(e, qkv, f, b, g, out, .{ .buf = self.segment }, 1, state, self.weights, .{
            .heads = self.heads,
            .head0 = 0,
            .proj_stride = 3 * width,
            .gate_stride = width,
            .out_stride = width,
            .log_rows = 1,
            .lower_bound = lower_bound,
            .eps = eps,
        });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
    }
};
