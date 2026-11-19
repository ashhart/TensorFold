//! Metal 4 for one model's passes: its own queue, command allocator, argument table, constants ring and residency set.
const std = @import("std");
const objc = @import("objc.zig");
const types = @import("types.zig");
const command = @import("command.zig");
const Buffer = @import("device.zig").Buffer;
const Device = @import("device.zig").Device;

const Id = objc.Id;

/// Apple's unfair lock; off Apple an empty stand-in, since other builds still analyze this file and never run it.
const darwin = @import("builtin").os.tag.isDarwin();
const Lock = if (darwin) std.c.os_unfair_lock else struct {};
fn hold(lock: *Lock) void {
    if (darwin) std.c.os_unfair_lock_lock(lock);
}
fn drop(lock: *Lock) void {
    if (darwin) std.c.os_unfair_lock_unlock(lock);
}

/// Command buffers a context reuses; a 64-layer pass committing every 4 layers takes 16.
pub const max_buffers = 32;
/// The ring setBytes copies into, since Metal 4 has no setBytes: a pass waits for the last, so the ring may wrap.
const ring_bytes: usize = 8 << 20;
const stage_dispatch: usize = 1 << 27; // MTLStageDispatch
const visibility_device: usize = 1; // MTL4VisibilityOptionDevice: a barrier flushes to the device's coherence point
const wait_ms: u64 = 120_000;
const block_is_global: c_int = 1 << 28; // BLOCK_IS_GLOBAL: copying or releasing a handler leaves it where it is

extern "c" var _NSConcreteGlobalBlock: [32]?*anyopaque;

const Descriptor = extern struct { reserved: c_ulong = 0, size: c_ulong };

/// A block literal: Metal calls `invoke` with a commit's feedback once its work ends, on a thread of its own.
const Handler = extern struct {
    isa: *anyopaque,
    flags: c_int = block_is_global,
    reserved: c_int = 0,
    invoke: *const fn (*Handler, Id) callconv(.c) void,
    descriptor: *const Descriptor,
    context: *Context,
    slot: usize,
};

const handler_descriptor = Descriptor{ .size = @sizeOf(Handler) };

const State = enum { free, open, committed };

pub const Context = struct {
    gpa: std.mem.Allocator,
    device: Device,
    queue: Id,
    allocator: Id,
    table: Id,
    event: Id,
    set: Id,
    ring: Buffer,
    ring_address: u64,
    ring_bytes: [*]u8,
    bump: usize = 0,
    buffers: [max_buffers]Id = undefined,
    handlers: [max_buffers]Handler = undefined,
    state: [max_buffers]State = @splat(.free),
    ticks: [max_buffers]u64 = @splat(0),
    done: [max_buffers]std.atomic.Value(u64) = @splat(.init(0)),
    failed: [max_buffers]bool = @splat(false),
    start: [max_buffers]f64 = @splat(0),
    end: [max_buffers]f64 = @splat(0),
    used: usize = 0,
    tick: u64 = 0,
    /// A wait timed out: the GPU may still read this context's buffers, so passes go back to Metal 3.
    broken: bool = false,
    /// Each buffer in the set by its id, with its GPU address.
    known: std.AutoHashMapUnmanaged(usize, u64) = .empty,
    dirty: bool = false,
    /// A dispatch since the encoder's last barrier: without one a barrier orders nothing new.
    pending: bool = false,
    lock: Lock = .{},
    freed: std.ArrayList(Id) = .empty,
    frees: std.atomic.Value(usize) = .init(0),

    /// A context on `device`, or error.NoMetal4 where this macOS or GPU makes no Metal 4 queue.
    pub fn create(gpa: std.mem.Allocator, device: Device) !*Context {
        if (!device.responds("newMTL4CommandQueue")) return error.NoMetal4;
        const queue = objc.msg(?Id, device.id, "newMTL4CommandQueue", .{}) orelse return error.NoMetal4;
        errdefer objc.release(queue);
        const allocator = objc.msg(?Id, device.id, "newCommandAllocator", .{}) orelse return error.NoMetal4;
        errdefer objc.release(allocator);
        const d = objc.msg(Id, objc.msg(Id, objc.class("MTL4ArgumentTableDescriptor"), "alloc", .{}), "init", .{});
        defer objc.release(d);
        objc.msg(void, d, "setMaxBufferBindCount:", .{@as(usize, 31)});
        var err: ?Id = null;
        const table = objc.msg(?Id, device.id, "newArgumentTableWithDescriptor:error:", .{ d, &err }) orelse return error.NoMetal4;
        errdefer objc.release(table);
        const event = objc.msg(?Id, device.id, "newSharedEvent", .{}) orelse return error.NoMetal4;
        errdefer objc.release(event);
        const set = (try device.residencySet(1024)).id;
        errdefer objc.release(set);
        const ring = try device.buffer(ring_bytes, types.ResourceOptions.shared | types.ResourceOptions.untracked);
        errdefer ring.deinit();
        const c = try gpa.create(Context);
        errdefer gpa.destroy(c);
        c.* = .{ .gpa = gpa, .device = device, .queue = queue, .allocator = allocator, .table = table, .event = event, .set = set, .ring = ring, .ring_address = ring.gpuAddress(), .ring_bytes = ring.contents() };
        var made: usize = 0;
        errdefer for (c.buffers[0..made]) |cb| objc.release(cb);
        while (made < max_buffers) : (made += 1) {
            c.buffers[made] = objc.msg(?Id, device.id, "newCommandBuffer", .{}) orelse return error.NoMetal4;
            c.handlers[made] = .{ .isa = @ptrCast(&_NSConcreteGlobalBlock), .invoke = feedback, .descriptor = &handler_descriptor, .context = c, .slot = made };
        }
        try register(c);
        objc.msg(void, set, "addAllocation:", .{ring.id});
        objc.msg(void, set, "commit", .{});
        objc.msg(void, queue, "addResidencySet:", .{set});
        return c;
    }

    /// Waits for every commit, then releases the context and what its set held.
    pub fn destroy(c: *Context) void {
        c.waitAll();
        unregister(c);
        for (c.buffers, c.state) |cb, state| {
            if (state == .open) objc.msg(void, cb, "endCommandBuffer", .{});
            objc.release(cb);
        }
        objc.msg(void, c.queue, "removeResidencySet:", .{c.set});
        objc.release(c.set);
        c.ring.deinit();
        objc.release(c.event);
        objc.release(c.table);
        objc.release(c.allocator);
        objc.release(c.queue);
        c.known.deinit(c.gpa);
        c.freed.deinit(c.gpa);
        c.gpa.destroy(c);
    }

    /// The context as a queue: its command buffers and encoders take this file's paths.
    pub fn asQueue(c: *Context) command.Queue {
        return .{ .id = c.queue, .m4 = c };
    }

    /// The next command buffer, begun; with nothing in flight, freed buffers leave the set and the allocator restarts.
    pub fn commandBuffer(c: *Context) command.CommandBuffer {
        if (c.used == max_buffers) c.waitAll();
        if (c.idle()) {
            c.drainFrees();
            c.reset();
        }
        if (c.used == max_buffers) @panic("metal 4: every command buffer of the context is open");
        const s = c.used;
        c.used += 1;
        objc.msg(void, c.buffers[s], "beginCommandBufferWithAllocator:", .{c.allocator});
        c.state[s] = .open;
        return .{ .id = c.buffers[s], .m4 = c, .slot = @intCast(s) };
    }

    fn idle(c: *const Context) bool {
        for (c.state[0..c.used], c.ticks[0..c.used], c.done[0..c.used]) |state, tick, *done| {
            if (state == .committed and done.load(.acquire) < tick) return false;
        }
        return true;
    }

    /// One command buffer encodes at a time, so an open one here was dropped by an encoding error and never commits.
    fn reset(c: *Context) void {
        for (c.buffers[0..c.used], c.state[0..c.used]) |cb, *state| {
            if (state.* == .open) objc.msg(void, cb, "endCommandBuffer", .{});
            state.* = .free;
        }
        objc.msg(void, c.allocator, "reset", .{});
        c.used = 0;
    }

    pub fn commit(c: *Context, s: u8) void {
        if (c.dirty) {
            objc.msg(void, c.set, "commit", .{});
            c.dirty = false;
        }
        const cb = c.buffers[s];
        objc.msg(void, cb, "endCommandBuffer", .{});
        c.tick += 1;
        c.ticks[s] = c.tick;
        c.failed[s] = false;
        c.state[s] = .committed;
        // Metal calls an options object's handlers for one commit only, so each commit takes new options.
        const options = objc.msg(Id, objc.msg(Id, objc.class("MTL4CommitOptions"), "alloc", .{}), "init", .{});
        defer objc.release(options);
        objc.msg(void, options, "addFeedbackHandler:", .{@as(*anyopaque, @ptrCast(&c.handlers[s]))});
        const list = [1]Id{cb};
        objc.msg(void, c.queue, "commit:count:options:", .{ @as([*]const Id, &list), @as(usize, 1), options });
        objc.msg(void, c.queue, "signalEvent:value:", .{ c.event, c.tick });
    }

    /// Blocks until slot `s`'s commit ended and its feedback arrived; a timeout breaks the context.
    pub fn wait(c: *Context, s: u8) void {
        if (c.state[s] != .committed or c.broken) return;
        const tick = c.ticks[s];
        if (c.done[s].load(.acquire) >= tick) return;
        if (!objc.msg(bool, c.event, "waitUntilSignaledValue:timeoutMS:", .{ tick, wait_ms })) return c.timeOut();
        var tries: usize = 0;
        while (c.done[s].load(.acquire) < tick) : (tries += 1) {
            if (tries == 50_000_000) return c.timeOut(); // the feedback trails the event by microseconds
            std.Thread.yield() catch {};
        }
    }

    pub fn waitAll(c: *Context) void {
        for (0..c.used) |s| c.wait(@intCast(s));
    }

    fn timeOut(c: *Context) void {
        c.broken = true;
        std.log.err("metal 4: a command buffer did not finish in {d} s; passes go back to Metal 3", .{wait_ms / 1000});
    }

    pub fn status(c: *const Context, s: u8) types.CommandBufferStatus {
        if (c.state[s] != .committed) return .not_enqueued;
        if (c.broken) return .@"error";
        if (c.done[s].load(.acquire) < c.ticks[s]) return .committed;
        return if (c.failed[s]) .@"error" else .completed;
    }

    /// Valid once waited: Metal's own error was logged when its feedback arrived.
    pub fn failure(c: *const Context, s: u8) ?[*:0]const u8 {
        if (c.state[s] != .committed) return null;
        if (c.broken) return "a Metal 4 command buffer did not finish";
        return if (c.failed[s]) "a Metal 4 command buffer failed (Metal's error is logged above)" else null;
    }

    /// `buffer`'s GPU address; it joins the set on first use, since a Metal 4 queue reaches only what sets hold.
    pub fn resident(c: *Context, buffer: Buffer) u64 {
        const key = @intFromPtr(buffer.id);
        if (c.known.get(key)) |address| return address;
        const address = buffer.gpuAddress();
        hold(&c.lock);
        c.known.put(c.gpa, key, address) catch {}; // without room it joins again on its next use, which changes nothing
        drop(&c.lock);
        objc.msg(void, c.set, "addAllocation:", .{buffer.id});
        c.dirty = true;
        return address;
    }

    /// Any thread, as a buffer is released: the set's reference keeps it alive until an idle command buffer drops it.
    fn forget(c: *Context, id: Id) void {
        hold(&c.lock);
        defer drop(&c.lock);
        if (!c.known.contains(@intFromPtr(id))) return;
        c.freed.append(c.gpa, id) catch return; // without room the set keeps it until the context goes
        _ = c.frees.fetchAdd(1, .release);
    }

    fn drainFrees(c: *Context) void {
        if (c.frees.load(.acquire) == 0) return;
        hold(&c.lock);
        defer drop(&c.lock);
        for (c.freed.items) |id| {
            objc.msg(void, c.set, "removeAllocation:", .{id});
            _ = c.known.remove(@intFromPtr(id));
        }
        c.freed.clearRetainingCapacity();
        c.frees.store(0, .release);
        objc.msg(void, c.set, "commit", .{});
    }

    /// An encoder on `cb` whose dispatches read their buffers from the context's argument table.
    pub fn encoder(c: *Context, cb: Id) Id {
        const e = objc.msg(Id, cb, "computeCommandEncoder", .{});
        objc.msg(void, e, "setArgumentTable:", .{c.table});
        c.pending = false;
        return e;
    }

    pub fn bind(c: *Context, buffer: Buffer, offset: usize, index: usize) void {
        objc.msg(void, c.table, "setAddress:atIndex:", .{ c.resident(buffer) + offset, index });
    }

    /// setBytes: the bytes go into the ring and the table takes their address.
    pub fn constant(c: *Context, bytes: []const u8, index: usize) void {
        if (c.bump + bytes.len > ring_bytes) c.bump = 0;
        @memcpy(c.ring_bytes[c.bump..][0..bytes.len], bytes);
        objc.msg(void, c.table, "setAddress:atIndex:", .{ c.ring_address + c.bump, index });
        c.bump += (bytes.len + 63) & ~@as(usize, 63);
    }

    pub fn dispatchIndirect(c: *Context, e: Id, args: Buffer, offset: usize, threads: types.Size) void {
        objc.msg(void, e, "dispatchThreadgroupsWithIndirectBuffer:threadsPerThreadgroup:", .{ c.resident(args) + offset, threads });
    }

    /// Later dispatches see every write before this point, as memoryBarrierWithScope does on Metal 3.
    pub fn barrier(c: *Context, e: Id) void {
        if (!c.pending) return;
        c.pending = false;
        objc.msg(void, e, "barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:", .{ stage_dispatch, stage_dispatch, visibility_device });
    }
};

fn feedback(h: *Handler, report: Id) callconv(.c) void {
    const c = h.context;
    const s = h.slot;
    if (objc.msg(?Id, report, "error", .{})) |err| {
        c.failed[s] = true;
        std.log.err("metal 4: a command buffer failed on the GPU: {s}", .{objc.errorText(err)});
    }
    c.start[s] = objc.msg(f64, report, "GPUStartTime", .{});
    c.end[s] = objc.msg(f64, report, "GPUEndTime", .{});
    c.done[s].store(c.ticks[s], .release);
}

pub fn waitForFence(e: Id, fence: Id) void {
    objc.msg(void, e, "waitForFence:beforeEncoderStages:", .{ fence, stage_dispatch });
}

pub fn updateFence(e: Id, fence: Id) void {
    objc.msg(void, e, "updateFence:afterEncoderStages:", .{ fence, stage_dispatch });
}

var registry_lock: Lock = .{};
var registry: [8]?*Context = @splat(null);
var live = std.atomic.Value(usize).init(0);

fn register(c: *Context) !void {
    hold(&registry_lock);
    defer drop(&registry_lock);
    for (&registry) |*entry| if (entry.* == null) {
        entry.* = c;
        _ = live.fetchAdd(1, .release);
        return;
    };
    return error.TooManyMetal4Contexts;
}

fn unregister(c: *Context) void {
    hold(&registry_lock);
    defer drop(&registry_lock);
    for (&registry) |*entry| if (entry.* == c) {
        entry.* = null;
        _ = live.fetchSub(1, .release);
    };
}

/// Buffer.deinit's hook: a buffer about to be released leaves each context's set once that context is idle.
pub fn released(id: Id) void {
    if (live.load(.acquire) == 0) return;
    hold(&registry_lock);
    defer drop(&registry_lock);
    for (registry) |entry| if (entry) |c| c.forget(id);
}

test "a Metal 4 context runs commits back to back, reports each one's feedback and drops freed buffers" {
    const library = @import("library.zig");
    const device = Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const c = Context.create(std.testing.allocator, device) catch return error.SkipZigTest;
    defer c.destroy();
    const source = "#include <metal_stdlib>\nusing namespace metal;\nkernel void add(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant float& k [[buffer(2)]], uint i [[thread_position_in_grid]]) { y[i] = x[i] + k; }";
    const lib = try library.Library.fromSource(device, source, library.CompileOptions.mlx());
    defer lib.deinit();
    const pipeline = try library.Pipeline.init(device, lib, "add", false);
    defer pipeline.deinit();
    const options = types.ResourceOptions.shared | types.ResourceOptions.untracked;
    const x = try device.buffer(1024 * 4, options);
    defer x.deinit();
    for (x.slice(f32, 1024), 0..) |*v, i| v.* = @floatFromInt(i);
    const y = try device.buffer(1024 * 4, options);
    for (0..3) |round| { // each commit takes new commit options, or the second one never reports back
        const cb = c.asQueue().commandBuffer();
        const e = cb.compute(.concurrent);
        e.setPipeline(pipeline);
        e.setBuffer(x, 0, 0);
        e.setBuffer(y, 0, 1);
        e.setValue(@as(f32, @floatFromInt(round)), 2);
        e.dispatchThreads(types.Size.of(1024, 1, 1), types.Size.of(256, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null and cb.status() == .completed);
        try std.testing.expect(cb.gpuSeconds() > 0);
        try std.testing.expectEqual(@as(f32, @floatFromInt(1023 + round)), y.slice(f32, 1024)[1023]);
    }
    const freed = @intFromPtr(y.id);
    y.deinit();
    try std.testing.expect(c.known.contains(freed));
    _ = c.asQueue().commandBuffer(); // nothing in flight: the freed buffer leaves the set
    try std.testing.expect(!c.known.contains(freed) and c.known.contains(@intFromPtr(x.id)));
}
