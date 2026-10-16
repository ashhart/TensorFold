//! The idle keepalive's Metal target: one tiny kernel on the engine's queue, touching its residency sets.
const std = @import("std");
const objc = @import("objc.zig");
const command = @import("command.zig");
const residency = @import("residency.zig");
const library = @import("library.zig");
const types = @import("types.zig");
const Device = @import("device.zig").Device;
const Buffer = @import("device.zig").Buffer;

/// A queue and the residency sets one commit keeps wired. The engine owns both; this only reads them.
pub const Target = struct {
    queue: command.Queue,
    sets: []const residency.ResidencySet = &.{},

    /// The core ticker's tick: a one-thread kernel that touches every set; an empty commit lets the GPU go cold.
    pub fn tick(ctx: *anyopaque) void {
        const t: *Target = @ptrCast(@alignCast(ctx));
        const pool = objc.Pool.push();
        defer pool.pop();
        const cmd = t.queue.commandBuffer();
        for (t.sets) |set| cmd.useResidencySet(set);
        if (poke(t.queue)) |p| {
            const e = cmd.compute(.serial);
            e.setPipeline(p.pipeline);
            e.setBuffer(p.buffer, 0, 0);
            e.dispatchThreads(types.Size.of(1, 1, 1), types.Size.of(1, 1, 1));
            e.end();
        }
        cmd.commit();
    }
};

/// The tick's kernel and the word it writes, built once for the process on the first tick's device.
const Poke = struct { pipeline: library.Pipeline, buffer: Buffer };

const source =
    \\#include <metal_stdlib>
    \\kernel void tf_keepalive(device uint* x [[buffer(0)]]) { x[0] += 1; }
;

var made: ?Poke = null;
var tried = false; // only the ticker thread ticks, so these need no lock

fn poke(queue: command.Queue) ?Poke {
    if (tried) return made;
    tried = true;
    const device: Device = .{ .id = objc.msg(objc.Id, queue.id, "device", .{}) };
    const lib = library.Library.fromSource(device, source, .{}) catch |e| return failed(e);
    defer lib.deinit();
    const pipeline = library.Pipeline.init(device, lib, "tf_keepalive", false) catch |e| return failed(e);
    const buffer = device.buffer(16, types.ResourceOptions.private) catch |e| {
        pipeline.deinit();
        return failed(e);
    };
    made = .{ .pipeline = pipeline, .buffer = buffer };
    return made;
}

fn failed(e: anyerror) ?Poke {
    std.log.warn("idle keepalive: no kernel ({s}); ticks commit empty, so the GPU may go cold between requests", .{@errorName(e)});
    return null;
}
