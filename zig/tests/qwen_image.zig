//! tf-qwen-image: Qwen-Image-2.1's transformer on a case from tools/zig/qwen_image_case.py, compared and timed.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const qi = tf.qwen_image;

const usage =
    \\tf-qwen-image TRANSFORMER_DIR CASE OUT_DIR
    \\CASE (tools/zig/qwen_image_case.py) holds the prompt's encoder rows, the noise levels, the starting latents and a
    \\reference run: each step's input and velocity. Each step runs on the reference's input and is compared with its
    \\velocity (max |diff|, cosine, worst row); then the whole schedule runs from the starting latents, timed, and
    \\OUT_DIR gets latents.f32 (rows x columns x 64) and run.json.
;

const Match = struct { max_abs: f64, cosine: f64, worst_row: f64 };

/// Largest absolute difference, cosine, and the largest relative L2 error of one row of `width`.
fn match(ours: []const f32, theirs: []const f32, width: usize) Match {
    var dot: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    var most: f64 = 0;
    var worst: f64 = 0;
    var row: usize = 0;
    while (row < ours.len) : (row += width) {
        var rd: f64 = 0;
        var rb: f64 = 0;
        for (ours[row..][0..width], theirs[row..][0..width]) |a, b| {
            const d = @as(f64, a) - b;
            dot += @as(f64, a) * b;
            aa += @as(f64, a) * a;
            bb += @as(f64, b) * b;
            rd += d * d;
            rb += @as(f64, b) * b;
            most = @max(most, @abs(d));
        }
        worst = @max(worst, @sqrt(rd / @max(rb, 1e-30)));
    }
    return .{ .max_abs = most, .cosine = dot / @sqrt(aa * bb), .worst_row = worst };
}

/// The case tensor `name` of `dtype`, copied out as T values.
fn take(comptime T: type, a: std.mem.Allocator, f: *const tf.safetensors.File, name: []const u8, dtype: tf.safetensors.DType) ![]T {
    const t = f.get(name) orelse {
        std.debug.print("the case has no {s}\n", .{name});
        return error.MissingTensor;
    };
    if (t.dtype != dtype) return error.BadCaseTensor;
    const out = try a.alloc(T, t.numel());
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

fn dims(f: *const tf.safetensors.File, name: []const u8) ![]const usize {
    const e = f.names.getPtr(name) orelse return error.MissingTensor;
    return e.shape[0..e.rank];
}

fn print(list: *std.ArrayList(u8), a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    try list.appendSlice(a, try std.fmt.allocPrint(a, fmt, args));
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    var case = try tf.safetensors.File.open(a, io, args[2]);
    defer case.close(io);
    const grid = try dims(&case, "latents");
    if (grid.len != 3 or grid[2] != qi.config.channels) return error.BadCaseTensor;
    const rows = grid[0] * grid[1];
    const count = rows * qi.config.channels;
    const text = (try dims(&case, "prompt"))[0];
    const prompt = try take(u16, a, &case, "prompt", .bf16);
    const sigmas = try take(f32, a, &case, "sigmas", .f32);
    const start = try take(f32, a, &case, "latents", .f32);
    const inputs = try take(f32, a, &case, "inputs", .f32);
    const reference = try take(f32, a, &case, "velocity", .f32);
    const exact: ?[]f32 = if (case.get("velocity_f32") != null) try take(f32, a, &case, "velocity_f32", .f32) else null;
    const steps = sigmas.len - 1;
    if (inputs.len != steps * count or reference.len != steps * count) return error.BadCaseTensor;

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const loading = mtl.clock.seconds();
    const model = try qi.Model.load(std.heap.page_allocator, io, device, args[1], text, grid[1] * 16, grid[0] * 16);
    defer model.deinit();
    const load_s = mtl.clock.seconds() - loading;
    std.debug.print("{s}: {d} image rows after {d} prompt rows, {d} steps; loaded in {d:.1} s\n", .{ device.name(), rows, text, steps, load_s });

    var json: std.ArrayList(u8) = .empty;
    const p = try model.prompt(prompt);
    std.debug.print("prompt rows: {d:.3} s ({d:.3} s on the GPU)\n", .{ p.wall, p.gpu });
    try print(&json, a, "{{\"device\": \"{s}\", \"image_rows\": {d}, \"prompt_rows\": {d}, \"load_s\": {d:.2}, \"prompt_s\": {d:.4}, \"teacher_forced\": [", .{ device.name(), rows, text, load_s, p.wall });

    const velocity = try a.alloc(f32, count);
    const again = try a.alloc(f32, count);
    for (0..steps) |s| {
        _ = try model.forward(inputs[s * count ..][0..count], sigmas[s], velocity);
        const m = match(velocity, reference[s * count ..][0..count], qi.config.channels);
        std.debug.print("step {d}: against the bf16 reference max |diff| {d:.5}, cosine {d:.7}, worst row {d:.5}\n", .{ s + 1, m.max_abs, m.cosine, m.worst_row });
        try print(&json, a, "{s}{{\"step\": {d}, \"sigma\": {d:.6}, \"bf16\": {{\"max_abs\": {d:.6}, \"cosine\": {d:.8}, \"worst_row\": {d:.6}}}", .{ if (s == 0) "" else ", ", s + 1, sigmas[s], m.max_abs, m.cosine, m.worst_row });
        if (exact) |f| if (f.len >= (s + 1) * count) {
            const want = f[s * count ..][0..count];
            const e = match(velocity, want, qi.config.channels);
            const r = match(reference[s * count ..][0..count], want, qi.config.channels);
            std.debug.print("        against the fp32 reference max |diff| {d:.5}, cosine {d:.7}; the bf16 reference's own: {d:.5}, {d:.7}\n", .{ e.max_abs, e.cosine, r.max_abs, r.cosine });
            try print(&json, a, ", \"fp32\": {{\"max_abs\": {d:.6}, \"cosine\": {d:.8}, \"worst_row\": {d:.6}}}, \"bf16_ref_vs_fp32\": {{\"max_abs\": {d:.6}, \"cosine\": {d:.8}}}", .{ e.max_abs, e.cosine, e.worst_row, r.max_abs, r.cosine });
        };
        try json.append(a, '}');
    }
    _ = try model.forward(inputs[0..count], sigmas[0], again);
    _ = try model.forward(inputs[0..count], sigmas[0], velocity);
    const repeat = std.mem.eql(u8, std.mem.sliceAsBytes(velocity), std.mem.sliceAsBytes(again));
    std.debug.print("the same step twice: {s}\n", .{if (repeat) "identical bytes" else "DIFFERENT bytes"});
    try print(&json, a, "], \"repeat_identical\": {}, \"run\": [", .{repeat});

    const x = try a.dupe(f32, start);
    var total: f64 = 0;
    for (0..steps) |s| {
        const t = try model.forward(x, sigmas[s], velocity);
        const by = sigmas[s + 1] - sigmas[s];
        for (x, velocity) |*o, v| o.* += v * by;
        total += t.wall;
        std.debug.print("step {d}/{d}: {d:.3} s ({d:.3} s on the GPU, {d:.3} s modulation on the host)\n", .{ s + 1, steps, t.wall, t.gpu, t.host });
        try print(&json, a, "{s}{{\"wall_s\": {d:.4}, \"gpu_s\": {d:.4}, \"host_s\": {d:.4}}}", .{ if (s == 0) "" else ", ", t.wall, t.gpu, t.host });
    }
    try print(&json, a, "], \"denoise_s\": {d:.3}}}\n", .{total});
    std.debug.print("denoise: {d:.2} s for {d} steps\n", .{ total, steps });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/latents.f32", .{args[3]}), .data = std.mem.sliceAsBytes(x) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/run.json", .{args[3]}), .data = json.items });
    return if (repeat) 0 else 1;
}
