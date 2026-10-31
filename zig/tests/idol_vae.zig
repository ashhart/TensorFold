//! tf-idol-vae: Idol's image encoder on a case from tools/zig/idol_case.py's vae, gated against float64 and timed.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const vae = tf.idol.vae;
const ref = tf.idol.reference;

const usage =
    \\tf-idol-vae VAE_DIR CASE_DIR [FEATURES]
    \\VAE_DIR is FLUX.2-klein-4B's vae/ folder (config.json, diffusion_pytorch_model.safetensors). CASE_DIR (idol_case.py vae)
    \\holds vae.safetensors: the picture [1024, 1024, 3] in [0, 1], the SDK's posterior noise [128, 128, 32], the SDK's
    \\condition rows (`sdk`, [4101, 128]) and float64's on the same picture and noise (`ruler`). Our rows' worst row and
    \\1 - cosine against float64 must be within twice the SDK's own, and two more encodes must repeat them byte for byte.
    \\With vae_stages.safetensors, each stage is compared with float64's too. FEATURES, a safetensors file holding
    \\`feature2` [1, 4101, 128] (the SDK's own run), is compared as well. CASE_DIR gets tf-idol-vae.json and rows.f32.
;

/// Encodes timed after the gated one.
const timed = 5;

fn take(a: std.mem.Allocator, f: *const tf.safetensors.File, name: []const u8, n: usize) ![]f32 {
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

fn json(list: *std.ArrayList(u8), a: std.mem.Allocator, key: []const u8, m: ref.Match) !void {
    try print(list, a, ", \"{s}\": {{\"max_abs\": {d:.6}, \"cosine\": {d:.12}, \"worst_row\": {d:.8}}}", .{ key, m.max_abs, m.cosine, m.worst_row });
}

const stage_names = [_][]const u8{ "down.0", "down.1", "down.2", "down.3", "mid", "moments" };
const stage_sizes = [_]usize{ 512 * 512 * 128, 256 * 256 * 256, 128 * 128 * 512, 128 * 128 * 512, 128 * 128 * 512, 128 * 128 * 64 };
const stage_widths = [_]usize{ 128, 256, 512, 512, 512, 64 };

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3 and args.len != 4) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    var case = try tf.safetensors.File.open(a, io, try std.fmt.allocPrint(a, "{s}/vae.safetensors", .{args[2]}));
    defer case.close(io);
    const n = vae.rows * vae.channels;
    const image = try take(a, &case, "image", vae.size * vae.size * 3);
    const noise = try take(a, &case, "noise", vae.latent * vae.latent * 32);
    const sdk = try take(a, &case, "sdk", n);
    const ruler = try take(a, &case, "ruler", n);
    var stages_file: ?tf.safetensors.File = tf.safetensors.File.open(a, io, try std.fmt.allocPrint(a, "{s}/vae_stages.safetensors", .{args[2]})) catch null;
    defer if (stages_file) |*f| f.close(io);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const loading = mtl.clock.seconds();
    const enc = try vae.Encoder.load(std.heap.page_allocator, io, device, args[1]);
    defer enc.deinit();
    const load_s = mtl.clock.seconds() - loading;
    std.debug.print("{s}: encoder loaded in {d:.2} s\n", .{ device.name(), load_s });
    var out: std.ArrayList(u8) = .empty;
    try print(&out, a, "{{\"device\": \"{s}\", \"load_s\": {d:.3}", .{ device.name(), load_s });

    if (stages_file) |*f| {
        var stages: vae.Stages = .{};
        var mine: [stage_names.len][]f32 = undefined;
        for (&mine, stage_sizes) |*m, s| m.* = try a.alloc(f32, s);
        for (0..4) |i| stages.down[i] = mine[i];
        stages.mid = mine[4];
        stages.moments = mine[5];
        const scratch = try a.alloc(f32, n);
        _ = try enc.encode(image, noise, scratch, stages);
        try print(&out, a, ", \"stages\": {{\"_\": 0", .{});
        for (stage_names, mine, stage_widths) |name, m, w| {
            const want = try take(a, f, name, m.len);
            const r = ref.match(m, want, w);
            std.debug.print("stage {s} against float64: worst pixel {e:.3}, 1 - cosine {e:.3}, max |diff| {e:.3}\n", .{ name, r.worst_row, 1 - r.cosine, r.max_abs });
            try json(&out, a, name, r);
        }
        try out.append(a, '}');
    }

    const rows = try a.alloc(f32, n);
    const first = try enc.encode(image, noise, rows, .{});
    const v = vae.gate(rows, sdk, ruler);
    std.debug.print("against float64: ours worst row {e:.3}, 1 - cosine {e:.3}; the SDK's {e:.3}, {e:.3}: {s}\n", .{ v.ours.worst_row, 1 - v.ours.cosine, v.sdk.worst_row, 1 - v.sdk.cosine, if (v.pass) "pass" else "FAIL" });
    const vs_sdk = ref.match(rows, sdk, vae.channels);
    std.debug.print("against the SDK: worst row {d:.4}, cosine {d:.8}, max |diff| {d:.4}\n", .{ vs_sdk.worst_row, vs_sdk.cosine, vs_sdk.max_abs });
    try json(&out, a, "ours_vs_float64", v.ours);
    try json(&out, a, "sdk_vs_float64", v.sdk);
    try json(&out, a, "ours_vs_sdk", vs_sdk);
    if (args.len == 4) {
        var feats = try tf.safetensors.File.open(a, io, try std.fmt.allocPrint(a, "{s}", .{args[3]}));
        defer feats.close(io);
        const theirs = try take(a, &feats, "feature2", n);
        const m = ref.match(rows, theirs, vae.channels);
        const same = std.mem.eql(u8, std.mem.sliceAsBytes(theirs), std.mem.sliceAsBytes(sdk));
        std.debug.print("against {s}: worst row {d:.4}, cosine {d:.8} (the case's SDK rows {s} it)\n", .{ args[3], m.worst_row, m.cosine, if (same) "equal" else "differ from" });
        try json(&out, a, "ours_vs_features", m);
        try print(&out, a, ", \"case_sdk_equals_features\": {}", .{same});
    }

    const again = try a.alloc(f32, n);
    var repeat = true;
    var times: [timed]f64 = undefined;
    var gpus: [timed]f64 = undefined;
    for (&times, &gpus) |*t, *gp| {
        const tm = try enc.encode(image, noise, again, .{});
        t.* = tm.wall;
        gp.* = tm.gpu;
        repeat = repeat and std.mem.eql(u8, std.mem.sliceAsBytes(rows), std.mem.sliceAsBytes(again));
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    std.mem.sort(f64, &gpus, {}, std.sort.asc(f64));
    std.debug.print("{d} more encodes: {s}; first {d:.3} s, median {d:.4} s ({d:.4} s on the GPU)\n", .{ timed, if (repeat) "identical bytes" else "DIFFERENT bytes", first.wall, times[timed / 2], gpus[timed / 2] });
    try print(&out, a, ", \"gate_pass\": {}, \"repeat_identical\": {}, \"first_s\": {d:.4}, \"median_s\": {d:.4}, \"median_gpu_s\": {d:.4}}}\n", .{ v.pass, repeat, first.wall, times[timed / 2], gpus[timed / 2] });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/rows.f32", .{args[2]}), .data = std.mem.sliceAsBytes(rows) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/tf-idol-vae.json", .{args[2]}), .data = out.items });
    return if (v.pass and repeat) 0 else 1;
}
