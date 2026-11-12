//! Check two actual checkpoint projections against separate production calls on identical saved BF16 rows.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").lfm3;
const Row = @import("tensorfold").row_projection;
fn weights(model: *q.Model, module: []const u8) !Row.Weights {
    var name: [512]u8 = undefined;
    const w = try model.checkpoint.get(try std.fmt.bufPrint(&name, "{s}.weight", .{module}));
    const s = try model.checkpoint.get(try std.fmt.bufPrint(&name, "{s}.scales", .{module}));
    const b = try model.checkpoint.get(try std.fmt.bufPrint(&name, "{s}.biases", .{module}));
    const quant = try model.config.width(module);
    if (w.dtype != .u32 or s.dtype != .bf16 or b.dtype != .bf16 or s.rank != 2) return error.ExpectedAffineProjection;
    return .{ .w = w.buffer, .w_off = w.offset, .scales = s.buffer, .s_off = s.offset, .biases = b.buffer, .b_off = b.offset, .n = s.shape[0], .k = s.shape[1] * quant.group, .group = quant.group, .bits = quant.bits };
}
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 6) return error.ExpectedModelModulesInputRows;
    const rows = try std.fmt.parseInt(usize, args[5], 10);
    if (rows == 0 or rows > 128) return error.InvalidRows;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = try q.Model.load(init.gpa, init.io, args[1]);
    defer model.deinit();
    const first = try weights(model, args[2]);
    const second = try weights(model, args[3]);
    if (!q.dual_projection.compatible(first, second)) return error.IncompatibleProjectionPair;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[4], init.gpa, .limited(4 << 20));
    defer init.gpa.free(bytes);
    if (bytes.len != rows * first.k * 2) return error.InvalidInputRows;
    const x = try model.device.buffer(bytes.len, mtl.ResourceOptions.shared);
    defer x.deinit();
    @memcpy(x.contents()[0..bytes.len], bytes);
    const length = rows * first.n * 2;
    const y = try model.device.buffer(length, mtl.ResourceOptions.shared);
    defer y.deinit();
    const z = try model.device.buffer(length, mtl.ResourceOptions.shared);
    defer z.deinit();
    var scratch = try q.state.Scratch.init(init.gpa, model.device, &model.config.layer_types, rows);
    defer scratch.deinit();
    try q.forward.project(model, &scratch, first, x, y, rows);
    try q.forward.project(model, &scratch, second, x, z, rows);
    const before_y = try init.gpa.dupe(u8, y.contents()[0..length]);
    defer init.gpa.free(before_y);
    const before_z = try init.gpa.dupe(u8, z.contents()[0..length]);
    defer init.gpa.free(before_z);
    @memset(y.contents()[0..length], 0xff);
    @memset(z.contents()[0..length], 0xff);
    var pipes = try q.dual_projection.Pipelines.load(model.device);
    defer pipes.deinit();
    const cb = model.queue.commandBuffer();
    const e = cb.compute(.serial);
    try q.dual_projection.encode(e, &pipes, first, second, x, 0, y, 0, z, 0, rows);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    if (!std.mem.eql(u8, before_y, y.contents()[0..length]) or !std.mem.eql(u8, before_z, z.contents()[0..length])) return error.DualChangedProjectionBytes;
    std.debug.print("DUAL_PROJECT rows={d} k={d} n={d} bits={d} group={d} equal_bytes={d}\n", .{ rows, first.k, first.n, first.bits, first.group, length * 2 });
}
