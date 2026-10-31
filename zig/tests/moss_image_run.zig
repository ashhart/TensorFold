//! tf-moss-image's case driver for any backend `B`: its family `mi`, safetensors `st`, clock `seconds` and `Device`.
const std = @import("std");

const usage =
    \\tf-moss-image TRANSFORMER_DIR CASE OUT_DIR [--prompt FILE [--negative FILE]]
    \\CASE (tools/zig/moss_image_case.py) holds the prompt's and the empty negative prompt's encoder rows, the timesteps
    \\and noise levels and the starting latents. A `case` also holds a reference run (each step's input and both
    \\branches' velocities): each step then runs both branches on the reference's input and compares them (max |diff|,
    \\cosine, worst row). Then the whole schedule runs from the starting latents with the card's guidance (scale 3, the
    \\negative branch skipping block 22), timed, and OUT_DIR gets latents.f32 (rows x columns x 128) and run.json. On
    \\Metal the transformer is then freed and the checkpoint's image decoder (the vae/ folder beside TRANSFORMER_DIR)
    \\turns the latents into OUT_DIR's picture.png. With --prompt, the encoder in TRANSFORMER_DIR's checkpoint folder
    \\makes both prompts' rows (the negative from --negative's file, else empty) in place of the case's; only the Metal
    \\build has the encoder.
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
fn take(comptime T: type, comptime st: type, a: std.mem.Allocator, f: *const st.File, name: []const u8, dtype: st.DType) ![]T {
    const t = f.get(name) orelse {
        std.debug.print("the case has no {s}\n", .{name});
        return error.MissingTensor;
    };
    if (t.dtype != dtype) return error.BadCaseTensor;
    const out = try a.alloc(T, t.numel());
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

fn dims(f: anytype, name: []const u8) ![]const usize {
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
fn teacher(comptime B: type, a: std.mem.Allocator, case: *const B.st.File, model: *B.mi.Model, branches: *[2]B.mi.Prompt, timesteps: []const f32, count: usize, out: *std.ArrayList(u8)) !bool {
    const mi = B.mi;
    const steps = timesteps.len;
    const inputs = try take(f32, B.st, a, case, "inputs", .f32);
    const reference = [2][]f32{ try take(f32, B.st, a, case, "velocity", .f32), try take(f32, B.st, a, case, "velocity_negative", .f32) };
    const exact = [2]?[]f32{
        if (case.get("velocity_f32") != null) try take(f32, B.st, a, case, "velocity_f32", .f32) else null,
        if (case.get("velocity_negative_f32") != null) try take(f32, B.st, a, case, "velocity_negative_f32", .f32) else null,
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

/// A prompt file's text, trimmed as the case tool trims it.
fn read(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    return std.mem.trim(u8, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)), " \t\r\n");
}

pub fn run(comptime B: type, init: std.process.Init) !u8 {
    const mi = B.mi;
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var prompt_file: ?[]const u8 = null;
    var negative_file: ?[]const u8 = null;
    var i: usize = 4;
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--prompt")) prompt_file = args[i + 1] else if (std.mem.eql(u8, args[i], "--negative")) negative_file = args[i + 1] else break;
    }
    if (args.len < 4 or i != args.len or (negative_file != null and prompt_file == null)) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    if (prompt_file != null and !@hasDecl(B.Device, "encode")) {
        std.debug.print("--prompt: this build has no prompt encoder (the Metal build does)\n", .{});
        return 2;
    }
    var case = try B.st.File.open(a, io, args[2]);
    defer case.close(io);
    const grid = try dims(&case, "latents");
    if (grid.len != 3 or grid[2] != mi.config.channels) return error.BadCaseTensor;
    const rows = grid[0] * grid[1];
    const count = rows * mi.config.channels;
    const timesteps = try take(f32, B.st, a, &case, "timesteps", .f32);
    const sigmas = try take(f32, B.st, a, &case, "sigmas", .f32);
    const start = try take(f32, B.st, a, &case, "latents", .f32);
    const steps = timesteps.len;
    if (sigmas.len != steps + 1 or start.len != count) return error.BadCaseTensor;

    var gpu: B.Device = undefined;
    try gpu.init();
    defer gpu.deinit();
    const device = gpu.name();
    var encode_s: ?f64 = null;
    var both: [2][]u16 = undefined;
    if (prompt_file) |file| {
        if (comptime @hasDecl(B.Device, "encode")) {
            const texts = [2][]const u8{ try read(a, io, file), if (negative_file) |n| try read(a, io, n) else "" };
            encode_s = try gpu.encode(a, io, std.fs.path.dirname(args[1]) orelse ".", texts, &both);
            std.debug.print("both prompts encoded natively: {d:.3} s\n", .{encode_s.?});
        } else unreachable;
    } else both = .{ try take(u16, B.st, a, &case, "prompt", .bf16), try take(u16, B.st, a, &case, "negative", .bf16) };
    const prompt = both[0];
    const negative = both[1];
    const text = prompt.len / mi.config.context;
    const negative_rows = negative.len / mi.config.context;
    var out: std.ArrayList(u8) = .empty;
    // The transformer is freed when its run ends, before the decoder loads.
    const x, const repeat = denoise: {
        const loading = B.seconds();
        const model = try gpu.load(std.heap.page_allocator, io, args[1], @max(text, negative_rows), grid[1] * 16, grid[0] * 16);
        defer model.deinit();
        const load_s = B.seconds() - loading;
        std.debug.print("{s}: {d} image rows, prompts of {d} and {d} rows, {d} steps; loaded in {d:.1} s\n", .{ device, rows, text, negative_rows, steps, load_s });

        const refining = B.seconds();
        var branches: [2]mi.Prompt = undefined;
        branches[0] = try model.prompt(prompt);
        defer branches[0].deinit();
        branches[1] = try model.prompt(negative);
        defer branches[1].deinit();
        const refine_s = B.seconds() - refining;
        std.debug.print("both prompts refined: {d:.3} s\n", .{refine_s});
        try print(&out, a, "{{\"device\": \"{s}\", \"image_rows\": {d}, \"prompt_rows\": {d}, \"negative_rows\": {d}, \"load_s\": {d:.2}, \"refine_s\": {d:.4}", .{ device, rows, text, negative_rows, load_s, refine_s });
        if (encode_s) |s| try print(&out, a, ", \"native_encoder\": true, \"encode_s\": {d:.4}", .{s});
        const repeat = if (case.get("inputs") == null) true else try teacher(B, a, &case, model, &branches, timesteps, count, &out);
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
        try print(&out, a, "], \"denoise_s\": {d:.3}", .{total});
        std.debug.print("denoise: {d:.2} s for {d} steps ({d} forwards)\n", .{ total, steps, 2 * steps });
        break :denoise .{ x, repeat };
    };
    if (comptime @hasDecl(B.Device, "decode")) {
        const width = grid[1] * 16;
        const height = grid[0] * 16;
        const picture = try gpu.decode(a, io, std.fs.path.dirname(args[1]) orelse ".", x, width, height);
        std.debug.print("decoded {d}x{d}: {d:.3} s ({d:.3} s on the GPU), the decoder loaded in {d:.2} s\n", .{ width, height, picture.wall, picture.gpu, picture.load_s });
        try print(&out, a, ", \"decoder_load_s\": {d:.3}, \"decode_s\": {d:.4}", .{ picture.load_s, picture.wall });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/picture.png", .{args[3]}), .data = picture.png });
    }
    try print(&out, a, "}}\n", .{});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/latents.f32", .{args[3]}), .data = std.mem.sliceAsBytes(x) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/run.json", .{args[3]}), .data = out.items });
    return if (repeat) 0 else 1;
}
