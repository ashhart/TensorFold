//! FastH3's Metal transformer against a case file: the first forward's velocities, then every step, timed.
const std = @import("std");
const mtl = @import("metal");
const h3 = @import("h3");

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, count: usize, file: *anyopaque) usize;
extern "c" fn fclose(file: *anyopaque) c_int;

const Match = struct { cosine: f64, relative: f64, worst_row: f64 };

/// Cosine and relative error over everything, and the largest relative error of one row of `width`.
fn match(ours: []const f32, theirs: []const f32, width: usize) Match {
    var dot: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    var dd: f64 = 0;
    var worst: f64 = 0;
    var row: usize = 0;
    while (row < ours.len) : (row += width) {
        var rd: f64 = 0;
        var rb: f64 = 0;
        for (ours[row..][0..width], theirs[row..][0..width]) |a, b| {
            dot += @as(f64, a) * b;
            aa += @as(f64, a) * a;
            bb += @as(f64, b) * b;
            rd += (@as(f64, a) - b) * (@as(f64, a) - b);
            rb += @as(f64, b) * b;
        }
        dd += rd;
        worst = @max(worst, @sqrt(rd / @max(rb, 1e-30)));
    }
    return .{ .cosine = dot / @sqrt(aa * bb), .relative = @sqrt(dd / bb), .worst_row = worst };
}

fn write(gpa: std.mem.Allocator, prefix: []const u8, comptime suffix: []const u8, values: []const f32) !void {
    const path = try std.fmt.allocPrint(gpa, "{s}" ++ suffix ++ "\x00", .{prefix});
    defer gpa.free(path);
    const file = fopen(@ptrCast(path.ptr), "wb") orelse return error.CannotWrite;
    defer _ = fclose(file);
    if (fwrite(values.ptr, 4, values.len, file) != values.len) return error.CannotWrite;
}

/// One Euler step as the reference takes it: x0 from the sigma the model saw, then the grid's ratio.
fn advance(x: []f32, velocity: []const f32, seen: f32, ratio: f32) void {
    for (x, velocity) |*o, v| o.* = ratio * o.* + (1.0 - ratio) * (o.* + seen * v);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5 and args.len != 6) {
        std.debug.print("usage: tf-h3-dit transformer_dir shards case.safetensors out_prefix [dense | profile | tables | steps=N | revoice=N | from=K | once | DEFINE, ...]\n", .{});
        return error.BadArguments;
    }
    const gpa = init.gpa;
    const device = try mtl.Device.init();
    var case = try h3.Tensors.open(gpa, device, args[3]);
    defer case.deinit();
    const loading = mtl.clock.seconds();
    // Last: dense, profile, tables, steps=N, revoice=N, from=K (case holds step K), once (one step), a define.
    var options = h3.Options{};
    var defines: [256]u8 = undefined;
    var used: usize = 0;
    var dense = false;
    var profile = false;
    var native = false;
    var count: usize = 0;
    var from: usize = 0;
    var revoice: usize = 0;
    var once = false;
    if (args.len == 6) {
        var it = std.mem.splitScalar(u8, args[5], ',');
        while (it.next()) |word| {
            if (std.mem.eql(u8, word, "dense")) dense = true else if (std.mem.eql(u8, word, "profile")) profile = true else if (std.mem.eql(u8, word, "tables")) native = true else if (std.mem.startsWith(u8, word, "steps=")) {
                native = true;
                count = try std.fmt.parseInt(usize, word[6..], 10);
            } else if (std.mem.eql(u8, word, "once")) {
                once = true;
            } else if (std.mem.startsWith(u8, word, "revoice=")) {
                revoice = try std.fmt.parseInt(usize, word[8..], 10);
            } else if (std.mem.startsWith(u8, word, "from=")) {
                from = try std.fmt.parseInt(usize, word[5..], 10);
            } else {
                const text = try std.fmt.bufPrint(defines[used..], "#define {s}\n", .{word});
                used += text.len;
            }
        }
    }
    options.defines = defines[0..used];
    const model = try h3.Model.load(gpa, args[1], try std.fmt.parseInt(usize, args[2], 10), &case, options);
    model.dense = dense or model.base;
    var steps = (case.names.get("video_step") orelse return error.MissingTensor).shape[0];
    const nv = model.video * model.video_in;
    const na = model.audio * model.audio_in;
    const video = try case.copy(f32, gpa, try case.at("video", .f32, nv), nv);
    const audio = try case.copy(f32, gpa, try case.at("audio", .f32, na), na);
    const video_start = try case.copy(f32, gpa, try case.at("video", .f32, nv), nv);
    const audio_start = try case.copy(f32, gpa, try case.at("audio", .f32, na), na);
    const expect_video = try case.copy(f32, gpa, try case.at("video_velocity", .f32, nv), nv);
    const expect_audio = try case.copy(f32, gpa, try case.at("audio_velocity", .f32, na), na);
    var video_step = try case.copy(f32, gpa, try case.at("video_step", .f32, 2 * steps), 2 * steps);
    var audio_step = try case.copy(f32, gpa, try case.at("audio_step", .f32, 2 * steps), 2 * steps);
    if (revoice > 0) {
        const made = try @import("h3_steps.zig").revoice(init, model, &case, revoice);
        steps = made.video.len / 2;
        video_step = made.video;
        audio_step = made.audio;
    } else if (native) {
        const made = try @import("h3_steps.zig").reschedule(init, model, &case, args[1], if (count == 0) steps else count);
        steps = made.video.len / 2;
        video_step = made.video;
        audio_step = made.audio;
    }
    std.debug.print("{s}: {d} rows (text {d}, keyframe {d}, audio {d} with {d} held, video {d}), {d} tiles ({d} prefix, {d} video kept a query), {d} blocks of {d}/{d}/{d}, {d} steps, loaded in {d:.1} s\n", .{ device.name(), model.rows, model.text, model.condition, model.audio, model.held_audio, model.video, model.tiles, model.prefix_tiles, model.keep, model.blocks.len, model.hidden, model.inner, model.mlp, steps, mtl.clock.seconds() - loading });

    const vv = try gpa.alloc(f32, nv);
    const av = try gpa.alloc(f32, na);
    // a knock-out or a probe only wants the profile
    const quick = args.len == 6 and (std.mem.indexOf(u8, args[5], "H3_KO") != null or std.mem.indexOf(u8, args[5], "H3_PROBE") != null);
    try h3.geometry.stepRange(from, steps);
    for (from..if (quick) from else if (once) from + 1 else steps) |step| {
        const took = try model.forward(video, audio, step, vv, av);
        if (step == from) {
            const m = match(vv, expect_video, model.video_in);
            const n = match(av, expect_audio, model.audio_in);
            std.debug.print("first video velocity against the reference: cosine {d:.6}, relative error {d:.5}, worst row {d:.5}\n", .{ m.cosine, m.relative, m.worst_row });
            std.debug.print("first audio velocity against the reference: cosine {d:.6}, relative error {d:.5}, worst row {d:.5}\n", .{ n.cosine, n.relative, n.worst_row });
            try write(gpa, args[4], ".first-video.f32", vv);
            try write(gpa, args[4], ".first-audio.f32", av);
        }
        advance(video, vv, video_step[2 * step], video_step[2 * step + 1]);
        // reference audio rows are clean and never stepped
        const held = model.held_audio * model.audio_in;
        advance(audio[held..], av[held..], audio_step[2 * step], audio_step[2 * step + 1]);
        std.debug.print("step {d}/{d}: {d:.3} s wall, {d:.3} s on the GPU\n", .{ step + 1, steps, took.wall, took.gpu });
    }
    try write(gpa, args[4], ".video.f32", video);
    try write(gpa, args[4], ".audio.f32", audio);
    if (!profile and !quick) return;
    model.profile = true;
    _ = try model.forward(video_start, audio_start, 0, vv, av);
    std.debug.print("one forward by stage, each in its own command buffer:\n", .{});
    for (h3.stages, model.spent) |stage, seconds| std.debug.print("  {s}: {d:.3} s\n", .{ stage, seconds });
}
