//! Host-memory entry points for the H3 block library, independent of transport.
const std = @import("std");
const cuda = @import("cuda");
const Model = @import("cuda.zig").Model;

/// Every nonzero address is host memory, including the optional attention output.
pub fn forwardHost(self: *Model, in: *const Model.Inputs) !f64 {
    const p = try self.plan(in);
    try self.drain();
    const copy = try stage(self, in, p);
    const took = try self.forward(&copy);
    try readback(self, in, p);
    return took;
}

pub fn stage(self: *Model, in: *const Model.Inputs, p: @import("cuda_validation.zig").Plan) !Model.Inputs {
    const from = [8]u64{ in.x, in.tables, in.line, in.cos, in.sin, in.slot, in.sizes, in.attention_out };
    var copy = in.*;
    inline for (.{ "x", "tables", "line", "cos", "sin", "slot", "sizes", "attention_out" }, 0..) |f, i| {
        const bytes = p.input[i];
        if (bytes == 0) {
            @field(copy, f) = 0;
        } else {
            if (self.inputs[i] == null or self.inputs[i].?.len < bytes) {
                const replacement = try cuda.DeviceBuffer.alloc(&self.driver, bytes);
                if (self.inputs[i]) |*old| old.free();
                self.inputs[i] = replacement;
            }
            if (i < 7) try self.inputs[i].?.upload(0, @as([*]const u8, @ptrFromInt(from[i]))[0..bytes]);
            @field(copy, f) = self.inputs[i].?.ptr;
        }
    }
    return copy;
}
pub fn readback(self: *Model, in: *const Model.Inputs, p: @import("cuda_validation.zig").Plan) !void {
    try self.inputs[0].?.download(0, @as([*]u8, @ptrFromInt(in.x))[0..p.input[0]]);
    if (p.input[7] != 0) try self.inputs[7].?.download(0, @as([*]u8, @ptrFromInt(in.attention_out))[0..p.input[7]]);
}

/// Preserve the descriptor and its byte accounting until a successful drain and matching free.
pub fn hostAlloc(self: *Model, bytes: usize, device: *u64) !*anyopaque {
    try self.ctx.makeCurrent();
    var held = try cuda.HostBuffer.allocMapped(&self.driver, bytes);
    errdefer held.free();
    const dev = try held.device();
    try self.hosts.put(self.gpa, @intFromPtr(held.bytes.ptr), held);
    device.* = dev;
    return held.bytes.ptr;
}
pub fn hostFree(self: *Model, ptr: *anyopaque) void {
    self.drain() catch |e| {
        std.debug.print("H3 host allocation retained: {t}\n", .{e});
        return;
    };
    if (self.hosts.fetchRemove(@intFromPtr(ptr))) |entry| {
        var b = entry.value;
        b.free();
    }
}

/// Requires a successful owner-context drain; no fallible wait follows partial disposal.
pub fn free(self: *Model) void {
    for (&self.inputs) |*held| if (held.*) |*b| {
        b.free();
        held.* = null;
    };
    var it = self.hosts.valueIterator();
    while (it.next()) |b| b.free();
    self.hosts.deinit(self.gpa);
    self.hosts = .empty;
}
