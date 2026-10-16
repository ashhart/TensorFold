//! The contract's Metal buffers: inputs copied in, outputs poisoned, every payload between guard bytes.
const std = @import("std");
const mtl = @import("metal");

pub const guard = 64;
const pattern = 0xa5;
const opts = mtl.ResourceOptions.shared;

/// A payload's place: its buffer and the offset past the front guard.
pub const Ref = struct { buf: mtl.Buffer, off: usize = guard };

const Kind = enum { input, output, scratch };
const Slot = struct { buf: mtl.Buffer, len: usize, kind: Kind };

/// One call's buffers on one device: each payload its own buffer, guarded on both sides.
pub const Rig = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    slots: std.ArrayList(Slot) = .empty,
    seen: std.ArrayList(u8) = .empty,
    outs: std.ArrayList(u8) = .empty,

    pub fn deinit(r: *Rig) void {
        r.reset();
        r.slots.deinit(r.a);
        r.seen.deinit(r.a);
        r.outs.deinit(r.a);
    }

    pub fn reset(r: *Rig) void {
        for (r.slots.items) |s| s.buf.deinit();
        r.slots.clearRetainingCapacity();
    }

    fn slot(r: *Rig, len: usize, kind: Kind) !Ref {
        const buf = try r.device.buffer(len + 2 * guard, opts);
        errdefer buf.deinit();
        @memset(buf.contents()[0 .. len + 2 * guard], pattern);
        try r.slots.append(r.a, .{ .buf = buf, .len = len, .kind = kind });
        return .{ .buf = buf };
    }

    /// An input holding `bytes`.
    pub fn input(r: *Rig, data: []const u8) !Ref {
        const ref = try r.slot(data.len, .input);
        @memcpy(ref.buf.contents()[guard..][0..data.len], data);
        return ref;
    }

    /// An output of `len` bytes, all 0xff: an element no launch writes reads as NaN.
    pub fn output(r: *Rig, len: usize) !Ref {
        const ref = try r.slot(len, .output);
        @memset(ref.buf.contents()[guard..][0..len], 0xff);
        return ref;
    }

    /// Working memory a call needs between its launches: neither an input nor an output.
    pub fn scratch(r: *Rig, len: usize) !Ref {
        return r.slot(len, .scratch);
    }

    pub fn bytes(_: *const Rig, ref: Ref, len: usize) []u8 {
        return ref.buf.contents()[ref.off..][0..len];
    }

    fn gather(r: *Rig, kind: Kind, out: *std.ArrayList(u8)) ![]const u8 {
        out.clearRetainingCapacity();
        for (r.slots.items) |s| if (s.kind == kind) try out.appendSlice(r.a, s.buf.contents()[guard..][0..s.len]);
        return out.items;
    }

    pub fn outputBytes(r: *Rig) ![]const u8 {
        return r.gather(.output, &r.outs);
    }

    pub fn inputBytes(r: *Rig) ![]const u8 {
        return r.gather(.input, &r.seen);
    }

    /// Every guard of every buffer still holds its pattern.
    pub fn intact(r: *Rig) !bool {
        for (r.slots.items) |s| {
            const all = s.buf.contents()[0 .. s.len + 2 * guard];
            for (all[0..guard]) |v| if (v != pattern) return false;
            for (all[guard + s.len ..]) |v| if (v != pattern) return false;
        }
        return true;
    }

    /// Each payload has a buffer of its own, so an overlap means one buffer bound as both input and output.
    pub fn overlap(r: *Rig) bool {
        for (r.slots.items, 0..) |s, i| for (r.slots.items[i + 1 ..]) |t| {
            if (s.buf.id == t.buf.id and s.kind != t.kind) return true;
        };
        return false;
    }

    /// A command buffer with one serial compute pass; `finish` commits it and waits.
    pub const Pass = struct { pool: mtl.objc.Pool, cb: mtl.CommandBuffer, e: mtl.ComputeEncoder };

    pub fn begin(r: *Rig) Pass {
        const pool = mtl.objc.Pool.push();
        const cb = r.queue.commandBuffer();
        return .{ .pool = pool, .cb = cb, .e = cb.compute(.serial) };
    }

    pub fn finish(_: *Rig, p: Pass) !void {
        defer p.pool.pop();
        p.e.end();
        p.cb.commit();
        p.cb.wait();
        if (p.cb.failure()) |why| {
            std.debug.print("GPU failed: {s}\n", .{why});
            return error.GpuFailed;
        }
    }
};

/// A pipeline's bytes of static threadgroup memory, from Metal's own reflection.
pub fn tgMemory(p: mtl.Pipeline) usize {
    return mtl.objc.msg(usize, p.id, "staticThreadgroupMemoryLength", .{});
}

/// One kernel of a source compiled as the engine compiles (MLX's options).
pub fn pipeline(device: mtl.Device, source: []const u8, name: []const u8) !mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
}
