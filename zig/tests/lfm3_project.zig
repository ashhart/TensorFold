//! One admitted checkpoint projection on saved BF16 input rows through the actual native projection selector.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").lfm3;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const tiled = args.len == 7 and std.mem.eql(u8, args[6], "--tiles");
    const binned = args.len == 7 and std.mem.eql(u8, args[6], "--binned");
    if (args.len != 6 and !tiled and !binned) return error.ExpectedModelModuleInputRowsOutput;
    const rows = try std.fmt.parseInt(usize, args[4], 10);
    if (rows == 0 or rows > 128) return error.ProjectionRowsOutsideFixtureBudget;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = try q.Model.load(init.gpa, init.io, args[1]);
    defer model.deinit();
    if (tiled) try model.enableProjectionCandidate();
    if (binned) try model.enableBinnedProjectionCandidate();
    var name: [512]u8 = undefined;
    const weights = try model.checkpoint.get(try std.fmt.bufPrint(&name, "{s}.weight", .{args[2]}));
    const scales = try model.checkpoint.get(try std.fmt.bufPrint(&name, "{s}.scales", .{args[2]}));
    const biases = try model.checkpoint.get(try std.fmt.bufPrint(&name, "{s}.biases", .{args[2]}));
    const quant = try model.config.width(args[2]);
    if (weights.dtype != .u32 or scales.dtype != .bf16 or biases.dtype != .bf16 or scales.rank != 2)
        return error.ExpectedAffineProjection;
    const w = q.metal_weights.Linear{ .w = weights.buffer, .w_off = weights.offset, .scales = scales.buffer, .s_off = scales.offset, .biases = biases.buffer, .b_off = biases.offset, .n = scales.shape[0], .k = scales.shape[1] * quant.group, .group = quant.group, .bits = quant.bits };
    try w.validate();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], init.gpa, .limited(4 << 20));
    defer init.gpa.free(bytes);
    if (bytes.len != rows * w.k * 2) return error.ExpectedExactBf16InputRows;
    const input = try model.device.buffer(bytes.len, mtl.ResourceOptions.shared);
    defer input.deinit();
    @memcpy(input.contents()[0..bytes.len], bytes);
    const output = try model.device.buffer(rows * w.n * 2, mtl.ResourceOptions.shared);
    defer output.deinit();
    var scratch = try q.state.Scratch.init(init.gpa, model.device, &model.config.layer_types, rows);
    defer scratch.deinit();
    try q.forward.project(model, &scratch, w, input, output, rows);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[5], .data = output.contents()[0..output.length()] });
    std.debug.print("LFM3_PROJECT rows={d} k={d} n={d} bits={d} group={d}\n", .{ rows, w.k, w.n, w.bits, w.group });
}
