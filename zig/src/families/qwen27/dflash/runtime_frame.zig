//! A draft frame owns reusable buffers until its serial command buffers complete; parts commit in order behind a fence.
const std = @import("std");
const mtl = @import("metal");
const op = @import("operators.zig");
pub const Ref = @import("core").draft_ops.Ref;
const Entry = struct { ref: Ref, view: op.View };
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
pub const Frame = struct {
    allocator: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    pool: std.ArrayList(mtl.Buffer) = .empty,
    entries: std.ArrayList(Entry) = .empty,
    cursor: usize = 0,
    epoch: u64 = 0,
    active: bool = false,
    completed: bool = false,
    cb: mtl.CommandBuffer = undefined,
    encoder: mtl.ComputeEncoder = undefined,
    external: ?Entry = null,
    external_id: u64 = 0,
    external_epoch: u64 = 0,
    fence: ?mtl.objc.Id = null,
    parts: [8]mtl.CommandBuffer = undefined,
    part_count: usize = 0,
    ops: u32 = 0,
    pub fn deinit(f: *Frame) void {
        f.release();
        if (f.fence) |fence| mtl.objc.release(fence);
        for (f.pool.items) |buffer| buffer.deinit();
        f.pool.deinit(f.allocator);
        f.entries.deinit(f.allocator);
    }
    pub fn begin(f: *Frame) !u64 {
        if (f.active or f.epoch >= 0x7fffffff) return error.DraftFrameBusy;
        f.epoch += 1;
        if (f.external_epoch != f.epoch) f.external = null;
        f.cursor = 0;
        f.entries.clearRetainingCapacity();
        f.cb = f.queue.commandBuffer();
        f.encoder = f.cb.compute(.serial);
        f.active = true;
        f.completed = false;
        f.part_count = 0;
        f.ops = 0;
        return f.epoch;
    }
    /// Commits the work so far and opens the next buffer behind a fence, so the GPU starts early; between ops only.
    pub fn part(f: *Frame) !void {
        if (!f.active or f.completed or f.part_count == f.parts.len) return;
        if (f.fence == null) f.fence = mtl.objc.msg(?mtl.objc.Id, f.device.id, "newFence", .{}) orelse return error.NoFence;
        mtl.objc.msg(void, f.encoder.id, "updateFence:", .{f.fence.?});
        f.encoder.end();
        f.cb.commit();
        f.parts[f.part_count] = f.cb;
        f.part_count += 1;
        f.cb = f.queue.commandBuffer();
        f.encoder = f.cb.compute(.serial);
        mtl.objc.msg(void, f.encoder.id, "waitForFence:", .{f.fence.?});
    }
    pub fn finish(f: *Frame) !void {
        if (!f.active or f.completed) return error.DraftFrameNotActive;
        f.encoder.end();
        f.cb.commit();
        f.cb.wait();
        f.completed = true;
        for (f.parts[0..f.part_count]) |cb| {
            cb.wait();
            if (cb.failure() != null) return error.DraftGpuFailure;
        }
        if (f.cb.failure() != null) return error.DraftGpuFailure;
    }
    pub fn release(f: *Frame) void {
        if (f.active and !f.completed) {
            f.encoder.end();
            for (f.parts[0..f.part_count]) |cb| cb.wait(); // committed parts still use the pool
        }
        f.active = false;
        f.external = null;
    }
    pub fn require(f: *const Frame, epoch: u64) !void {
        if (!f.active or f.completed or f.epoch != epoch) return error.DraftFrameNotActive;
    }
    pub fn add(f: *Frame, r: Ref, rows: u32, width: u32, stride: u32, dtype: @import("../checkpoint.zig").DType) !op.View {
        const view = op.View{ .handle = (f.epoch << 32) | (f.entries.items.len + 1), .rows = rows, .width = width, .stride = stride, .dtype = dtype };
        try view.check(rows, width, dtype);
        const bytes = try std.math.mul(usize, try std.math.add(usize, try std.math.mul(usize, rows - 1, stride), width), dtype.size());
        if (r.off > r.buf.length() or bytes > r.buf.length() - r.off) return error.DraftOperatorShape;
        try f.entries.append(f.allocator, .{ .ref = r, .view = view });
        return view;
    }
    pub fn alloc(f: *Frame, rows: u32, width: u32, dtype: @import("../checkpoint.zig").DType) !op.View {
        if (!f.active or f.completed or rows == 0 or width == 0) return error.DraftFrameNotActive;
        const bytes = try std.math.mul(usize, try std.math.mul(usize, rows, width), dtype.size());
        if (f.cursor == f.pool.items.len) {
            const buffer = try f.device.buffer(bytes, options);
            f.pool.append(f.allocator, buffer) catch |err| {
                buffer.deinit();
                return err;
            };
        } else if (f.pool.items[f.cursor].length() < bytes) {
            const buffer = try f.device.buffer(bytes, options);
            f.pool.items[f.cursor].deinit();
            f.pool.items[f.cursor] = buffer;
        }
        const buffer = f.pool.items[f.cursor];
        f.cursor += 1;
        return f.add(.{ .buf = buffer }, rows, width, width, dtype);
    }
    pub fn ref(f: *const Frame, v: op.View) !Ref {
        if (v.handle & (@as(u64, 1) << 63) != 0) {
            if (!f.active or f.external_epoch != f.epoch) return error.StaleDraftView;
            const entry = f.external orelse return error.StaleDraftView;
            if (!std.meta.eql(v, entry.view)) return error.StaleDraftView;
            return entry.ref;
        }
        const index: usize = @intCast(v.handle & 0xffffffff);
        if (!f.active or v.handle >> 32 != f.epoch or index == 0 or index > f.entries.items.len) return error.StaleDraftView;
        const entry = f.entries.items[index - 1];
        if (!std.meta.eql(v, entry.view)) return error.StaleDraftView;
        return entry.ref;
    }
    pub fn borrow(f: *Frame, r: Ref, rows: u32, width: u32) !op.View {
        if (f.active or rows == 0 or width == 0 or r.off > r.buf.length() or @as(usize, rows) * width * 2 > r.buf.length() - r.off) return error.DraftFrameBusy;
        f.external_id += 1;
        if (f.external_id >= (@as(u64, 1) << 63)) return error.DraftFrameBusy;
        const view = op.View{ .handle = (@as(u64, 1) << 63) | f.external_id, .rows = rows, .width = width, .stride = width, .dtype = .bf16 };
        f.external = .{ .ref = r, .view = view };
        f.external_epoch = f.epoch + 1;
        return view;
    }
};
