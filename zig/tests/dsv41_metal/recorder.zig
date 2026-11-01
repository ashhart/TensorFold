//! A stand-in compute encoder (a runtime ObjC class) that records a launch's calls, and their replays on the GPU.
const std = @import("std");
const mtl = @import("metal");

const objc = mtl.objc;
const Size = mtl.Size;
const Id = objc.Id;

extern "c" fn objc_allocateClassPair(super: objc.Class, name: [*:0]const u8, extra: usize) ?objc.Class;
extern "c" fn objc_registerClassPair(cls: objc.Class) void;
extern "c" fn class_addMethod(cls: objc.Class, name: objc.Sel, imp: *const anyopaque, types: [*:0]const u8) bool;

/// One recorded call: state (pipeline, buffer, bytes, threadgroup memory, resource) or a dispatch.
pub const Op = union(enum) {
    pipe: Id,
    buf: struct { b: ?Id, off: usize, at: usize },
    bytes: struct { from: usize, len: usize, at: usize },
    tgmem: struct { len: usize, at: usize },
    use: struct { r: Id, usage: usize },
    groups: [2]Size,
    threads: [2]Size,
    barrier,
};

pub const Recording = struct {
    a: std.mem.Allocator,
    ops: std.ArrayList(Op) = .empty,
    data: std.ArrayList(u8) = .empty,

    pub fn deinit(r: *Recording) void {
        r.ops.deinit(r.a);
        r.data.deinit(r.a);
    }

    pub fn clear(r: *Recording) void {
        r.ops.clearRetainingCapacity();
        r.data.clearRetainingCapacity();
    }

    fn add(r: *Recording, op: Op) void {
        r.ops.append(r.a, op) catch @panic("recorder: out of memory");
    }

    pub fn dispatches(r: *const Recording) usize {
        var n: usize = 0;
        for (r.ops.items) |op| n += @intFromBool(op == .groups or op == .threads);
        return n;
    }

    /// The whole recording into `e` as it was encoded.
    pub fn replay(r: *const Recording, e: mtl.ComputeEncoder) void {
        for (r.ops.items) |op| apply(r, e, op);
    }

    fn apply(r: *const Recording, e: mtl.ComputeEncoder, op: Op) void {
        switch (op) {
            .pipe => |p| objc.msg(void, e.id, "setComputePipelineState:", .{p}),
            .buf => |b| objc.msg(void, e.id, "setBuffer:offset:atIndex:", .{ b.b, b.off, b.at }),
            .bytes => |b| e.setBytes(r.data.items[b.from..][0..b.len], b.at),
            .tgmem => |t| e.setThreadgroupMemory(t.len, t.at),
            .use => |u| objc.msg(void, e.id, "useResource:usage:", .{ u.r, u.usage }),
            .groups => |g| e.dispatchGroups(g[0], g[1]),
            .threads => |g| e.dispatchThreads(g[0], g[1]),
            .barrier => e.barrier(),
        }
    }

    /// Dispatch `i` alone in `e`: every state call before it replayed first (later ones win, as on one encoder).
    pub fn replayOne(r: *const Recording, e: mtl.ComputeEncoder, i: usize) ?Id {
        var n: usize = 0;
        var pipe: ?Id = null;
        for (r.ops.items) |op| {
            switch (op) {
                .groups, .threads => {
                    if (n == i) {
                        apply(r, e, op);
                        return pipe;
                    }
                    n += 1;
                },
                .barrier => {},
                .pipe => |p| {
                    pipe = p;
                    apply(r, e, op);
                },
                else => apply(r, e, op),
            }
        }
        return null;
    }
};

var active: ?*Recording = null;

fn rec() *Recording {
    return active orelse @panic("recorder: no active recording");
}

fn setPipe(_: *anyopaque, _: objc.Sel, p: Id) callconv(.c) void {
    rec().add(.{ .pipe = p });
}

fn setBuf(_: *anyopaque, _: objc.Sel, b: ?Id, off: usize, at: usize) callconv(.c) void {
    rec().add(.{ .buf = .{ .b = b, .off = off, .at = at } });
}

fn setBytes(_: *anyopaque, _: objc.Sel, ptr: [*]const u8, len: usize, at: usize) callconv(.c) void {
    const r = rec();
    const from = r.data.items.len;
    r.data.appendSlice(r.a, ptr[0..len]) catch @panic("recorder: out of memory");
    r.add(.{ .bytes = .{ .from = from, .len = len, .at = at } });
}

fn setTg(_: *anyopaque, _: objc.Sel, len: usize, at: usize) callconv(.c) void {
    rec().add(.{ .tgmem = .{ .len = len, .at = at } });
}

fn useRes(_: *anyopaque, _: objc.Sel, res: Id, usage: usize) callconv(.c) void {
    rec().add(.{ .use = .{ .r = res, .usage = usage } });
}

fn groups(_: *anyopaque, _: objc.Sel, g: Size, t: Size) callconv(.c) void {
    rec().add(.{ .groups = .{ g, t } });
}

fn threads(_: *anyopaque, _: objc.Sel, g: Size, t: Size) callconv(.c) void {
    rec().add(.{ .threads = .{ g, t } });
}

fn barrier(_: *anyopaque, _: objc.Sel, _: usize) callconv(.c) void {
    rec().add(.barrier);
}

fn end(_: *anyopaque, _: objc.Sel) callconv(.c) void {}

var class_: ?objc.Class = null;

fn recorderClass() objc.Class {
    if (class_) |c| return c;
    const c = objc_allocateClassPair(objc.class("NSObject"), "TfDsv41RecordingEncoder", 0) orelse @panic("recorder: class");
    const M = struct { name: [:0]const u8, imp: *const anyopaque };
    const methods = [_]M{
        .{ .name = "setComputePipelineState:", .imp = @ptrCast(&setPipe) },
        .{ .name = "setBuffer:offset:atIndex:", .imp = @ptrCast(&setBuf) },
        .{ .name = "setBytes:length:atIndex:", .imp = @ptrCast(&setBytes) },
        .{ .name = "setThreadgroupMemoryLength:atIndex:", .imp = @ptrCast(&setTg) },
        .{ .name = "useResource:usage:", .imp = @ptrCast(&useRes) },
        .{ .name = "dispatchThreadgroups:threadsPerThreadgroup:", .imp = @ptrCast(&groups) },
        .{ .name = "dispatchThreads:threadsPerThreadgroup:", .imp = @ptrCast(&threads) },
        .{ .name = "memoryBarrierWithScope:", .imp = @ptrCast(&barrier) },
        .{ .name = "endEncoding", .imp = @ptrCast(&end) },
    };
    inline for (methods) |m| _ = class_addMethod(c, objc.sel(m.name), m.imp, "v@:");
    objc_registerClassPair(c);
    class_ = c;
    return c;
}

/// An encoder whose calls land in `r` until `stop`; encode a launch into it as into a real one.
pub fn start(r: *Recording) mtl.ComputeEncoder {
    r.clear();
    active = r;
    return .{ .id = objc.msg(Id, recorderClass(), "new", .{}) };
}

pub fn stop(e: mtl.ComputeEncoder) void {
    objc.release(e.id);
    active = null;
}
