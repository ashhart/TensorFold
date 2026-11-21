//! Ling's bf16 row operations over existing Kimi kernels, retaining checkpoint weights in mapped buffers.
const std = @import("std");
const mtl = @import("metal");
const kernels = @import("kimi_kernels");
const ck = @import("ling_checkpoint");
const Ref = kernels.Ref;

pub const Rows = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    library: mtl.Library,
    glue: mtl.Library,
    swiglu: mtl.Pipeline,
    k: kernels.Kernels,
    scratch: mtl.Buffer,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    weights: std.StringHashMapUnmanaged(mtl.Buffer) = .empty,
    max_rows: u32,
    fail_buffer_at: ?usize = null,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, max_rows: u32) !Rows {
        const library = try mtl.Library.fromBytes(device, @import("kimi_metallib").bytes);
        errdefer library.deinit();
        var k = try kernels.Kernels.init(device, library);
        errdefer k.deinit();
        const queue = try device.queue();
        errdefer queue.deinit();
        const glue = try mtl.Library.fromSource(device, @import("kernel_sources").lfm3, mtl.CompileOptions.mlx());
        errdefer glue.deinit();
        const swiglu = try mtl.Pipeline.init(device, glue, "lfm3_swiglu", false);
        errdefer swiglu.deinit();
        const scratch = try device.buffer(kernels.partFloats(max_rows) * 4, mtl.ResourceOptions.shared);
        return .{ .allocator = a, .device = device, .queue = queue, .library = library, .glue = glue, .swiglu = swiglu, .k = k, .scratch = scratch, .max_rows = max_rows };
    }

    pub fn buffer(self: *Rows, bytes: usize) !mtl.Buffer {
        if (self.fail_buffer_at) |at| if (self.owned.items.len >= at) return error.OutOfMemory;
        const b = try self.device.buffer(@max(bytes, 16), mtl.ResourceOptions.shared);
        errdefer b.deinit();
        try self.owned.append(self.allocator, b);
        return b;
    }

    pub fn rollbackBuffers(self: *Rows, mark: usize) void {
        for (self.owned.items[mark..]) |b| b.deinit();
        self.owned.shrinkRetainingCapacity(mark);
    }

    fn finish(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !void {
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
    }

    pub fn weight(self: *Rows, model: *ck.Checkpoint, name: []const u8, n: u32, width: u32) !mtl.Buffer {
        if (width % 64 != 0) return error.UnsupportedProjectionWidth;
        const source = try model.get(name, .bf16, &.{ n, width });
        if (self.weights.get(name)) |w| return w;
        const w = try self.buffer(@as(usize, n) * width * 2);
        const cb = self.queue.commandBuffer();
        const e = cb.compute(.serial);
        self.k.interleaved(e, .{ .buf = source.buffer, .off = source.offset }, .{ .buf = w }, n, width);
        try finish(cb, e);
        const key = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(key);
        try self.weights.put(self.allocator, key, w);
        return w;
    }

    pub fn linear(self: *Rows, model: *ck.Checkpoint, name: []const u8, x: Ref, y: Ref, rows: u32, width: u32, n: u32, fp32: bool) !void {
        return self.linearInto(model, name, x, y, rows, width, n, width, n, fp32);
    }

    pub fn linearInto(self: *Rows, model: *ck.Checkpoint, name: []const u8, x: Ref, y: Ref, rows: u32, width: u32, n: u32, x_stride: u32, y_stride: u32, fp32: bool) !void {
        if (rows == 0 or rows > self.max_rows) return error.RowCapacity;
        const w = try self.weight(model, name, n, width);
        const cb = self.queue.commandBuffer();
        const e = cb.compute(.serial);
        self.k.rows(e, x, .{ .buf = w }, y, .{ .buf = self.scratch }, fp32, .{ .K = width, .N = n, .rows = rows, .x_stride = x_stride, .y_stride = y_stride });
        try finish(cb, e);
    }

    pub fn norm(self: *Rows, model: *ck.Checkpoint, name: []const u8, x: Ref, y: Ref, rows: u32, width: u32, eps: f32) !void {
        if (rows == 0 or rows > self.max_rows) return error.RowCapacity;
        const w = try model.get(name, .bf16, &.{width});
        const cb = self.queue.commandBuffer();
        const e = cb.compute(.serial);
        self.k.rmsNorm(e, x, .{ .buf = w.buffer, .off = w.offset }, y, rows, .{ .D = width, .x_stride = width, .y_stride = width, .eps = eps });
        try finish(cb, e);
    }

    pub fn add(self: *Rows, x: Ref, delta: Ref, out: Ref, count: u32) !void {
        const cb = self.queue.commandBuffer();
        const e = cb.compute(.serial);
        self.k.add(e, x, delta, out, count);
        try finish(cb, e);
    }

    pub fn mlp(self: *Rows, model: *ck.Checkpoint, prefix: []const u8, x: Ref, out: Ref, gate: Ref, up: Ref, activation: Ref, rows: u32, hidden: u32, inter: u32) !void {
        const g = try std.fmt.allocPrint(self.allocator, "{s}gate_proj.weight", .{prefix});
        defer self.allocator.free(g);
        const u = try std.fmt.allocPrint(self.allocator, "{s}up_proj.weight", .{prefix});
        defer self.allocator.free(u);
        const d = try std.fmt.allocPrint(self.allocator, "{s}down_proj.weight", .{prefix});
        defer self.allocator.free(d);
        try self.linear(model, g, x, gate, rows, hidden, inter, false);
        try self.linear(model, u, x, up, rows, hidden, inter, false);
        try self.siluMul(gate, up, activation, rows * inter);
        try self.linear(model, d, activation, out, rows, inter, hidden, false);
    }

    pub fn siluMul(self: *Rows, gate: Ref, up: Ref, out: Ref, count: u32) !void {
        const cb = self.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(self.swiglu);
        e.setBuffer(gate.buf, gate.off, 0);
        e.setBuffer(up.buf, up.off, 1);
        e.setBuffer(out.buf, out.off, 2);
        e.setValue(count, 3);
        e.dispatchThreads(mtl.Size.of((count + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
        try finish(cb, e);
    }

    pub fn deinit(self: *Rows) void {
        var keys = self.weights.keyIterator();
        while (keys.next()) |key| self.allocator.free(key.*);
        self.weights.deinit(self.allocator);
        for (self.owned.items) |b| b.deinit();
        self.owned.deinit(self.allocator);
        self.scratch.deinit();
        self.k.deinit();
        self.swiglu.deinit();
        self.glue.deinit();
        self.library.deinit();
        self.queue.deinit();
    }
};
