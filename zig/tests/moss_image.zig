//! tf-moss-image: MOSS-Image's transformer on a case from tools/zig/moss_image_case.py, compared and timed.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const mi = tf.moss_image;

const usage =
    \\tf-moss-image TRANSFORMER_DIR CASE OUT_DIR
    \\CASE (tools/zig/moss_image_case.py) holds the prompt's and the empty negative prompt's encoder rows, the timesteps
    \\and noise levels and the starting latents. A `case` also holds a reference run (each step's input and both
    \\branches' velocities): each step then runs both branches on the reference's input and compares them (max |diff|,
    \\cosine, worst row). Then the whole schedule runs from the starting latents with the card's guidance (scale 3, the
    \\negative branch skipping block 22), timed, and OUT_DIR gets latents.f32 (rows x columns x 128) and run.json.
;

/// The card's sampling defaults: the guidance scale, and the block the negative branch skips.
const guidance: f32 = 3.0;
const negative_skip: usize = 22;

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

fn json(list: *std.ArrayList(u8), a: std.mem.Allocator, key: []const u8, m: Match) !void {
    try print(list, a, ", \"{s}\": {{\"max_abs\": {d:.6}, \"cosine\": {d:.8}, \"worst_row\": {d:.6}}}", .{ key, m.max_abs, m.cosine, m.worst_row });
}

/// Each step's two branches on the reference's input against its velocities; true when a repeated forward repeats.
fn teacher(a: std.mem.Allocator, case: *const tf.safetensors.File, model: *mi.Model, branches: *[2]mi.Prompt, timesteps: []const f32, count: usize, out: *std.ArrayList(u8)) !bool {
    const steps = timesteps.len;
    const inputs = try take(f32, a, case, "inputs", .f32);
    const reference = [2][]f32{ try take(f32, a, case, "velocity", .f32), try take(f32, a, case, "velocity_negative", .f32) };
    const exact = [2]?[]f32{
        if (case.get("velocity_f32") != null) try take(f32, a, case, "velocity_f32", .f32) else null,
        if (case.get("velocity_negative_f32") != null) try take(f32, a, case, "velocity_negative_f32", .f32) else null,
    };
    if (inputs.len != steps * count) return error.BadCaseTensor;
    for (reference) |r| if (r.len != steps * count) return error.BadCaseTensor;
    try print(out, a, ", \"teacher_forced\": [", .{});
    const skips = [2]?usize{ null, negative_skip };
    const labels = [2][]const u8{ "prompt", "negative" };
    const velocity = try a.alloc(f32, count);
    const again = try a.alloc(f32, count);
    for (0..steps) |s| {
        try print(out, a, "{s}{{\"step\": {d}, \"timestep\": {d:.4}", .{ if (s == 0) "" else ", ", s + 1, timesteps[s] });
        for (0..2) |b| {
            _ = try model.forward(&branches[b], inputs[s * count ..][0..count], timesteps[s], skips[b], velocity);
            const theirs = reference[b][s * count ..][0..count];
            const m = match(velocity, theirs, mi.config.channels);
            std.debug.print("step {d} {s}: against the bf16 reference max |diff| {d:.5}, cosine {d:.7}, worst row {d:.5}\n", .{ s + 1, labels[b], m.max_abs, m.cosine, m.worst_row });
            try json(out, a, try std.fmt.allocPrint(a, "{s}_bf16", .{labels[b]}), m);
            if (exact[b]) |f| if (f.len >= (s + 1) * count) {
                const want = f[s * count ..][0..count];
                const e = match(velocity, want, mi.config.channels);
                const r = match(theirs, want, mi.config.channels);
                std.debug.print("        against the fp32 reference max |diff| {d:.5}, cosine {d:.7}; the bf16 reference's own: {d:.5}, {d:.7}\n", .{ e.max_abs, e.cosine, r.max_abs, r.cosine });
                try json(out, a, try std.fmt.allocPrint(a, "{s}_fp32", .{labels[b]}), e);
                try json(out, a, try std.fmt.allocPrint(a, "{s}_bf16_ref_vs_fp32", .{labels[b]}), r);
            };
        }
        try out.append(a, '}');
    }
    _ = try model.forward(&branches[0], inputs[0..count], timesteps[0], null, again);
    _ = try model.forward(&branches[1], inputs[0..count], timesteps[1], negative_skip, velocity);
    _ = try model.forward(&branches[0], inputs[0..count], timesteps[0], null, velocity);
    const repeat = std.mem.eql(u8, std.mem.sliceAsBytes(velocity), std.mem.sliceAsBytes(again));
    std.debug.print("the same step twice, another between: {s}\n", .{if (repeat) "identical bytes" else "DIFFERENT bytes"});
    try print(out, a, "], \"repeat_identical\": {}", .{repeat});
    return repeat;
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
    if (grid.len != 3 or grid[2] != mi.config.channels) return error.BadCaseTensor;
    const rows = grid[0] * grid[1];
    const count = rows * mi.config.channels;
    const text = (try dims(&case, "prompt"))[0];
    const negative_rows = (try dims(&case, "negative"))[0];
    const prompt = try take(u16, a, &case, "prompt", .bf16);
    const negative = try take(u16, a, &case, "negative", .bf16);
    const timesteps = try take(f32, a, &case, "timesteps", .f32);
    const sigmas = try take(f32, a, &case, "sigmas", .f32);
    const start = try take(f32, a, &case, "latents", .f32);
    const steps = timesteps.len;
    if (sigmas.len != steps + 1 or start.len != count) return error.BadCaseTensor;

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const loading = mtl.clock.seconds();
    const model = try mi.Model.load(std.heap.page_allocator, io, device, args[1], @max(text, negative_rows), grid[1] * 16, grid[0] * 16);
    defer model.deinit();
    const load_s = mtl.clock.seconds() - loading;
    std.debug.print("{s}: {d} image rows, prompts of {d} and {d} rows, {d} steps; loaded in {d:.1} s\n", .{ device.name(), rows, text, negative_rows, steps, load_s });

    var out: std.ArrayList(u8) = .empty;
    const refining = mtl.clock.seconds();
    var branches: [2]mi.Prompt = undefined;
    branches[0] = try model.prompt(prompt);
    defer branches[0].deinit();
    branches[1] = try model.prompt(negative);
    defer branches[1].deinit();
    const refine_s = mtl.clock.seconds() - refining;
    std.debug.print("both prompts refined: {d:.3} s\n", .{refine_s});
    try print(&out, a, "{{\"device\": \"{s}\", \"image_rows\": {d}, \"prompt_rows\": {d}, \"negative_rows\": {d}, \"load_s\": {d:.2}, \"refine_s\": {d:.4}", .{ device.name(), rows, text, negative_rows, load_s, refine_s });
    const repeat = if (case.get("inputs") == null) true else try teacher(a, &case, model, &branches, timesteps, count, &out);
    try print(&out, a, ", \"run\": [", .{});

    const x = try a.dupe(f32, start);
    const cond = try a.alloc(f32, count);
    const uncond = try a.alloc(f32, count);
    var total: f64 = 0;
    for (0..steps) |s| {
        const c = try model.forward(&branches[0], x, timesteps[s], null, cond);
        const u = try model.forward(&branches[1], x, timesteps[s], negative_skip, uncond);
        const by = sigmas[s + 1] - sigmas[s];
        for (x, cond, uncond) |*o, pc, pu| o.* += (pu + guidance * (pc - pu)) * by;
        const wall = c.wall + u.wall;
        total += wall;
        std.debug.print("step {d}/{d}: {d:.3} s ({d:.3} s on the GPU, {d:.3} s modulation on the host)\n", .{ s + 1, steps, wall, c.gpu + u.gpu, c.host + u.host });
        try print(&out, a, "{s}{{\"wall_s\": {d:.4}, \"gpu_s\": {d:.4}, \"host_s\": {d:.4}}}", .{ if (s == 0) "" else ", ", wall, c.gpu + u.gpu, c.host + u.host });
    }
    try print(&out, a, "], \"denoise_s\": {d:.3}}}\n", .{total});
    std.debug.print("denoise: {d:.2} s for {d} steps ({d} forwards)\n", .{ total, steps, 2 * steps });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/latents.f32", .{args[3]}), .data = std.mem.sliceAsBytes(x) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/run.json", .{args[3]}), .data = out.items });
    return if (repeat) 0 else 1;
}
