//! Explicit GPU gate: it compiles the serving shaders at runtime and is never part of the default host tests.
const std = @import("std");
const M = @import("metal");
const F = @import("qwen4").family;
const G = F.metal_gpu.Gpu(M);
const B = G.Storage;
const D = F.metal_draw.Drawer(M, @import("lanes"), B);
const Loader = struct {
    pub const reference = F.device_fixture.metal_reference;
    pub fn load(_: @This(), a: std.mem.Allocator, b: *B, c: *const F.Config, p: *F.Problem) !G {
        return G.load(a, b, c, p, .hf);
    }
};
fn read(_: *B, buffer: B.Buffer, out: []u8) !void {
    @memcpy(out, buffer.contents()[0..out.len]);
}
pub fn main(init: std.process.Init) !void {
    const pool = M.objc.Pool.push();
    defer pool.pop();
    const device = try M.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var commands = M.CommandSession.init(init.gpa, queue);
    var commands_closed = false;
    defer if (!commands_closed) commands.deinit() catch {};
    var b: B = .{ .device = device, .queue = queue, .commands = &commands };
    var d = try D.load(init.gpa, &b, 32, 32, 2);
    var drawer_closed = false;
    defer if (!drawer_closed) d.close() catch {};
    const nodes = try F.node_gate.run(G, read, init.gpa, init.io, &b, Loader{}, true);
    if (nodes.different != 0) return error.Qwen4NodeMismatch;
    try F.device_gate.Gate(G, D, @import("lanes"), read).runAll(init.gpa, init.io, &b, Loader{}, &d);
    try d.close();
    drawer_closed = true;
    try commands.deinit();
    commands_closed = true;
}
