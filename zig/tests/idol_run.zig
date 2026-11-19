//! tf-idol's case driver for a backend `B`: its family `id`, safetensors `st`, `seconds`, `Device` and `stream`.
const std = @import("std");

const usage =
    \\tf-idol SNAPSHOT_DIR CASE_DIR [blocks STEP]
    \\SNAPSHOT_DIR is a speridlabs/idol snapshot (config.json, denoiser.safetensors). CASE_DIR (tools/zig/idol_case.py)
    \\holds case.safetensors: both image conditions, the starting noise and the SDK's run (each step's input and both
    \\branches' velocities), and optionally exact.safetensors (the same steps in float32). Each step runs both branches on
    \\the SDK's input; with exact.safetensors each 1 - cosine against float32 must be within twice the SDK's own, and
    \\so must the worst row over the run. Then the whole guided schedule runs from the noise, timed; CASE_DIR gets tf-idol.json and
    \\sample.f32 (8192 x 16, then the 5 camera values). Exit 1 on a failed gate or a forward that does not repeat.
    \\With `blocks STEP`, runs each joint block alone on the SDK's input rows at that step (blocksSTEP.safetensors)
    \\and gates it against the same block in float32 (blocksSTEP_f32.safetensors), worst row and 1 - cosine.
;

/// The card's sampling defaults.
const guidance: f32 = 3.0;
const rescale: f64 = 3.0;

fn take(a: std.mem.Allocator, f: anytype, name: []const u8, n: usize) ![]f32 {
    const t = f.get(name) orelse {
        std.debug.print("the case has no {s}\n", .{name});
        return error.MissingTensor;
    };
    if (t.dtype != .f32 or t.numel() != n) {
        std.debug.print("{s} is not {d} float32 values\n", .{ name, n });
        return error.BadCaseTensor;
    }
    const out = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

fn print(list: *std.ArrayList(u8), a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    try list.appendSlice(a, try std.fmt.allocPrint(a, fmt, args));
}

fn json(list: *std.ArrayList(u8), a: std.mem.Allocator, key: []const u8, m: anytype) !void {
    try print(list, a, ", \"{s}\": {{\"max_abs\": {d:.6}, \"cosine\": {d:.9}, \"worst_row\": {d:.6}}}", .{ key, m.max_abs, m.cosine, m.worst_row });
}

const Case = struct {
    steps: usize,
    timesteps: []f32,
    inputs: []f32,
    inputs_camera: []f32,
    velocity: [2][]f32,
    velocity_camera: [2][]f32,
    exact: ?[2][]f32,
};

const labels = [2][]const u8{ "conditional", "negative" };

/// Each step's two branches on the SDK's input against its velocities; returns false when a gate fails.
fn teacher(comptime B: type, a: std.mem.Allocator, c: Case, model: *B.id.Model, conds: *[2]B.id.Condition, out: *std.ArrayList(u8)) !bool {
    const cfg = B.id.config;
    const ref = B.id.reference;
    const N = cfg.latent_rows * cfg.latent_channels;
    var pass = true;
    var worst = [2]f64{ 0, 0 };
    const v = try a.alloc(f32, N);
    var vc: [cfg.camera_channels]f32 = undefined;
    try print(out, a, ", \"teacher_forced\": [", .{});
    for (0..c.steps) |s| {
        try print(out, a, "{s}{{\"step\": {d}, \"time\": {d:.4}", .{ if (s == 0) "" else ", ", s + 1, c.timesteps[s] });
        const cam: *const [cfg.camera_channels]f32 = c.inputs_camera[s * cfg.camera_channels ..][0..cfg.camera_channels];
        for (0..2) |b| {
            _ = try model.forward(&conds[b], c.inputs[s * N ..][0..N], cam, c.timesteps[s], v, &vc);
            const theirs = c.velocity[b][s * N ..][0..N];
            const m = ref.match(v, theirs, cfg.latent_channels);
            const mc = ref.match(&vc, c.velocity_camera[b][s * cfg.camera_channels ..][0..cfg.camera_channels], cfg.camera_channels);
            std.debug.print("step {d} {s}: against the SDK max |diff| {d:.5}, cosine {d:.8}, worst row {d:.5}; camera cosine {d:.7}\n", .{ s + 1, labels[b], m.max_abs, m.cosine, m.worst_row, mc.cosine });
            try json(out, a, try std.fmt.allocPrint(a, "{s}_sdk", .{labels[b]}), m);
            try json(out, a, try std.fmt.allocPrint(a, "{s}_camera_sdk", .{labels[b]}), mc);
            if (c.exact) |ex| if (ex[b].len >= (s + 1) * N) {
                const want = ex[b][s * N ..][0..N];
                const ours = ref.match(v, want, cfg.latent_channels);
                const sdk = ref.match(theirs, want, cfg.latent_channels);
                const ok = ref.withinCosine(ours, sdk);
                pass = pass and ok;
                worst[0] = @max(worst[0], ours.worst_row);
                worst[1] = @max(worst[1], sdk.worst_row);
                std.debug.print("        against float32: worst row {d:.5}, cosine {d:.8}; the SDK's own {d:.5}, {d:.8}: {s}\n", .{ ours.worst_row, ours.cosine, sdk.worst_row, sdk.cosine, if (ok) "pass" else "FAIL" });
                try json(out, a, try std.fmt.allocPrint(a, "{s}_fp32", .{labels[b]}), ours);
                try json(out, a, try std.fmt.allocPrint(a, "{s}_sdk_vs_fp32", .{labels[b]}), sdk);
            };
        }
        try out.append(a, '}');
    }
    if (c.exact != null) {
        const ok = worst[0] <= ref.factor * worst[1];
        std.debug.print("worst row over the run against float32: {d:.5}; the SDK's own {d:.5}: {s}\n", .{ worst[0], worst[1], if (ok) "pass" else "FAIL" });
        pass = pass and ok;
    }
    try print(out, a, "], \"worst_row_fp32\": {d:.6}, \"sdk_worst_row_fp32\": {d:.6}", .{ worst[0], worst[1] });
    return pass;
}

/// The same step twice with the other branch between: identical bytes or not.
fn repeats(comptime B: type, a: std.mem.Allocator, c: Case, model: *B.id.Model, conds: *[2]B.id.Condition) !bool {
    const cfg = B.id.config;
    const N = cfg.latent_rows * cfg.latent_channels;
    const first = try a.alloc(f32, N);
    const again = try a.alloc(f32, N);
    var vc: [cfg.camera_channels]f32 = undefined;
    const cam: *const [cfg.camera_channels]f32 = c.inputs_camera[0..cfg.camera_channels];
    _ = try model.forward(&conds[0], c.inputs[0..N], cam, c.timesteps[0], first, &vc);
    _ = try model.forward(&conds[1], c.inputs[0..N], cam, c.timesteps[@min(1, c.steps - 1)], again, &vc);
    _ = try model.forward(&conds[0], c.inputs[0..N], cam, c.timesteps[0], again, &vc);
    return std.mem.eql(u8, std.mem.sliceAsBytes(first), std.mem.sliceAsBytes(again));
}

/// The case tensor `name`, fp16 or fp32 rows, as f32.
fn rows(a: std.mem.Allocator, f: anytype, name: []const u8) ![]f32 {
    const t = f.get(name) orelse return error.MissingTensor;
    const out = try a.alloc(f32, t.numel());
    switch (t.dtype) {
        .f16 => for (out, @as([*]align(1) const f16, @ptrCast(t.bytes.ptr))[0..out.len]) |*o, v| {
            o.* = v;
        },
        .f32 => @memcpy(std.mem.sliceAsBytes(out), t.bytes),
        else => return error.BadCaseTensor,
    }
    return out;
}

/// Each joint block alone on the SDK's own input rows at step `step`, against that block in float32; false on a fail.
fn blockGates(comptime B: type, a: std.mem.Allocator, io: std.Io, dir: []const u8, step: usize, c: Case, model: *B.id.Model) !bool {
    const cfg = B.id.config;
    const ref = B.id.reference;
    var f = try B.st.File.open(a, io, try std.fmt.allocPrint(a, "{s}/blocks{d}.safetensors", .{ dir, step }));
    defer f.close(io);
    var g = try B.st.File.open(a, io, try std.fmt.allocPrint(a, "{s}/blocks{d}_f32.safetensors", .{ dir, step }));
    defer g.close(io);
    const H = cfg.hidden;
    var pass = true;
    var name: [32]u8 = undefined;
    for (1..cfg.layers) |i| {
        const input = try rows(a, &f, try std.fmt.bufPrint(&name, "block.{d}", .{i - 1}));
        const in16 = try a.alloc(f16, input.len);
        for (in16, input) |*o, v| o.* = @floatCast(v);
        try model.blockAt(cfg.refiners + i, in16, c.timesteps[step]);
        const ours = try a.alloc(f32, input.len);
        for (ours, try B.stream(model, cfg.joint_rows)) |*o, v| o.* = v;
        const label = try std.fmt.bufPrint(&name, "block.{d}", .{i});
        const want = try rows(a, &g, label);
        const m = ref.match(ours, want, H);
        const r = ref.match(try rows(a, &f, label), want, H);
        const ok = ref.within(m, r);
        pass = pass and ok;
        std.debug.print("step {d} block {d} against float32: worst row {d:.5}, cosine {d:.9}; the SDK's own {d:.5}, {d:.9}: {s}\n", .{ step + 1, i, m.worst_row, m.cosine, r.worst_row, r.cosine, if (ok) "pass" else "FAIL" });
    }
    return pass;
}

pub fn run(comptime B: type, init: std.process.Init) !u8 {
    const id = B.id;
    const cfg = id.config;
    const ref = id.reference;
    const N = cfg.latent_rows * cfg.latent_channels;
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3 and args.len != 5) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    var case = try B.st.File.open(a, io, try std.fmt.allocPrint(a, "{s}/case.safetensors", .{args[2]}));
    defer case.close(io);
    const steps = (case.get("timesteps") orelse return error.MissingTensor).numel();
    const S = steps;
    const cam_n = cfg.camera_channels;
    var c: Case = .{ .steps = S, .timesteps = try take(a, &case, "timesteps", S), .inputs = try take(a, &case, "inputs", S * N), .inputs_camera = try take(a, &case, "inputs_camera", S * cam_n), .velocity = .{ try take(a, &case, "velocity", S * N), try take(a, &case, "velocity_negative", S * N) }, .velocity_camera = .{ try take(a, &case, "velocity_camera", S * cam_n), try take(a, &case, "velocity_negative_camera", S * cam_n) }, .exact = null };
    const exact_path = try std.fmt.allocPrint(a, "{s}/exact.safetensors", .{args[2]});
    var exact_file: ?B.st.File = B.st.File.open(a, io, exact_path) catch null;
    defer if (exact_file) |*f| f.close(io);
    if (exact_file) |*f| {
        const n = (f.get("velocity_f32") orelse return error.MissingTensor).numel();
        c.exact = .{ try take(a, f, "velocity_f32", n), try take(a, f, "velocity_negative_f32", n) };
    }
    const f1 = try take(a, &case, "feature1", cfg.cond_rows * cfg.cond_channels);
    const f2 = try take(a, &case, "feature2", cfg.cond_rows * cfg.cond2_channels);
    const noise = try take(a, &case, "latent", N);
    const noise_cam = try take(a, &case, "camera", cam_n);

    var gpu: B.Device = undefined;
    try gpu.init();
    defer gpu.deinit();
    const device = gpu.name();
    const loading = B.seconds();
    const model = try gpu.load(std.heap.page_allocator, io, args[1]);
    defer model.deinit();
    const load_s = B.seconds() - loading;
    std.debug.print("{s}: {d} steps; loaded in {d:.1} s\n", .{ device, S, load_s });

    var out: std.ArrayList(u8) = .empty;
    const conditioning = B.seconds();
    var conds: [2]id.Condition = undefined;
    conds[0] = try model.condition(f1, f2);
    defer conds[0].deinit();
    @memset(f1, 0);
    @memset(f2, 0);
    conds[1] = try model.condition(f1, f2);
    defer conds[1].deinit();
    const condition_s = B.seconds() - conditioning;
    if (args.len == 5) return if (try blockGates(B, a, io, args[2], try std.fmt.parseInt(usize, args[4], 10), c, model)) 0 else 1;
    try print(&out, a, "{{\"device\": \"{s}\", \"steps\": {d}, \"load_s\": {d:.2}, \"condition_s\": {d:.4}", .{ device, S, load_s, condition_s });
    const pass = try teacher(B, a, c, model, &conds, &out);
    const repeat = try repeats(B, a, c, model, &conds);
    std.debug.print("the same step twice, another between: {s}\n", .{if (repeat) "identical bytes" else "DIFFERENT bytes"});
    try print(&out, a, ", \"gates_pass\": {}, \"repeat_identical\": {}, \"run\": [", .{ pass and c.exact != null, repeat });

    const ts = try a.alloc(f64, S + 1);
    id.host.schedule(ts, rescale);
    const x = try a.alloc(f32, N + cam_n);
    @memcpy(x[0..N], noise);
    @memcpy(x[N..], noise_cam);
    const vc = try a.alloc(f32, N);
    const vu = try a.alloc(f32, N);
    var cc: [cam_n]f32 = undefined;
    var cu: [cam_n]f32 = undefined;
    var total: f64 = 0;
    for (0..S) |s| {
        const time: f32 = @floatCast(1000 * ts[s]);
        const cam: *const [cam_n]f32 = x[N..][0..cam_n];
        const p = try model.forward(&conds[0], x[0..N], cam, time, vc, &cc);
        const q = try model.forward(&conds[1], x[0..N], cam, time, vu, &cu);
        const by: f32 = @floatCast(ts[s] - ts[s + 1]);
        for (x[0..N], vc, vu) |*o, pc, pu| o.* -= ((1 + guidance) * pc - guidance * pu) * by;
        for (x[N..], cc, cu) |*o, pc, pu| o.* -= ((1 + guidance) * pc - guidance * pu) * by;
        const wall = p.wall + q.wall;
        total += wall;
        std.debug.print("step {d}/{d}: {d:.3} s ({d:.3} s on the GPU)\n", .{ s + 1, S, wall, p.gpu + q.gpu });
        try print(&out, a, "{s}{{\"wall_s\": {d:.4}, \"gpu_s\": {d:.4}}}", .{ if (s == 0) "" else ", ", wall, p.gpu + q.gpu });
    }
    if (case.get("sample") != null) {
        const sample = try take(a, &case, "sample", N);
        const m = ref.match(x[0..N], sample, cfg.latent_channels);
        std.debug.print("final sample against the SDK's: cosine {d:.6}, worst row {d:.4}\n", .{ m.cosine, m.worst_row });
        try print(&out, a, "]", .{});
        try json(&out, a, "sample_sdk", m);
        try print(&out, a, ", \"denoise_s\": {d:.3}}}\n", .{total});
    } else try print(&out, a, "], \"denoise_s\": {d:.3}}}\n", .{total});
    std.debug.print("denoise: {d:.2} s for {d} steps ({d} forwards)\n", .{ total, S, 2 * S });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/sample.f32", .{args[2]}), .data = std.mem.sliceAsBytes(x) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/tf-idol.json", .{args[2]}), .data = out.items });
    return if (repeat and (c.exact == null or pass)) 0 else 1;
}
