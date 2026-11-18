//! tf-h3-dit-cuda: tf-h3-dit's denoising loop over the CUDA block library, a case file's steps to float32 rows.
const std = @import("std");
const loop = @import("h3_cuda_loop");

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, count: usize, file: *anyopaque) usize;
extern "c" fn fclose(file: *anyopaque) c_int;

fn write(gpa: std.mem.Allocator, prefix: []const u8, comptime suffix: []const u8, values: []const f32) !void {
    const path = try std.fmt.allocPrint(gpa, "{s}" ++ suffix ++ "\x00", .{prefix});
    defer gpa.free(path);
    const file = fopen(@ptrCast(path.ptr), "wb") orelse return error.CannotWrite;
    defer _ = fclose(file);
    if (fwrite(values.ptr, 4, values.len, file) != values.len) return error.CannotWrite;
}

/// Every block of the stream through the library's host entry (tf_h3_forward_host).
fn forwardHost(ctx: *anyopaque, in: *const loop.Inputs) anyerror!void {
    _ = try loop.host.forwardHost(@ptrCast(@alignCast(ctx)), in);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4 and args.len != 5) {
        std.debug.print("usage: tf-h3-dit-cuda transformer.safetensors case.safetensors out_prefix [dense | from=K | once, ...]\n", .{});
        return error.BadArguments;
    }
    const gpa = init.gpa;
    const io = init.io;
    var dense = false;
    var from: usize = 0;
    var once = false;
    if (args.len == 5) {
        var it = std.mem.splitScalar(u8, args[4], ',');
        while (it.next()) |word| {
            if (std.mem.eql(u8, word, "dense")) dense = true else if (std.mem.eql(u8, word, "once")) once = true else if (std.mem.startsWith(u8, word, "from=")) {
                from = try std.fmt.parseInt(usize, word[5..], 10);
            } else {
                // the Metal tool's tables, steps=N and revoice=N remake the schedule on the GPU: not here
                std.debug.print("tf-h3-dit-cuda: unknown option '{s}' (dense, from=K, once)\n", .{word});
                return error.BadArguments;
            }
        }
    }
    // the transformer's shapes for the case's checks; the library reads its weights itself
    var ck = try loop.st.File.open(gpa, io, args[1]);
    defer ck.close(io);
    const q = ck.names.get("blocks.0.attn.qkv_proj.weight") orelse return error.MissingTensor;
    const fc2 = ck.names.get("blocks.0.mlp.fc2.weight") orelse return error.MissingTensor;
    var layers: usize = 0;
    var name: [64]u8 = undefined;
    while (ck.names.get(try std.fmt.bufPrint(&name, "blocks.{d}.norm1.weight", .{layers})) != null) layers += 1;
    var file = try loop.st.File.open(gpa, io, args[2]);
    defer file.close(io);
    const view: loop.geometry.Case = .{ .names = file.names, .data = file.map.memory[file.data..] };
    const g = try loop.geometry.read(gpa, view, q, fc2, layers, true);
    var c = try loop.Case.read(gpa, view, g, layers, dense);
    defer c.deinit();
    const model = try loop.Model.load(gpa, args[1]);
    defer model.deinit();
    if (model.hidden != g.hidden or model.blocks.len != layers) return error.H3CaseModelMismatch;
    var l = try loop.Loop.init(gpa, &c, .{ .ctx = model, .run = forwardHost });
    defer l.deinit(gpa);
    const r = g.layout;
    const video = try loop.take(f32, gpa, view, "video", r.video * g.video_in);
    defer gpa.free(video);
    const audio = try loop.take(f32, gpa, view, "audio", r.audio * g.audio_in);
    defer gpa.free(audio);
    const to = if (once) from + 1 else g.steps;
    std.debug.print("{d} rows (text {d}, keyframe {d}, audio {d} with {d} held, video {d}), {d} tiles, {d} blocks of {d}, steps {d} to {d}\n", .{ r.rows, r.text, r.condition, r.audio, r.held_audio, r.video, r.tiles, layers, g.hidden, from, to });
    // the first step alone, so its velocities can be written as the Metal tool writes them
    try l.run(video, audio, from, from + 1);
    try write(gpa, args[3], ".first-video.f32", l.vv);
    try write(gpa, args[3], ".first-audio.f32", l.av);
    if (to > from + 1) try l.run(video, audio, from + 1, to);
    try write(gpa, args[3], ".video.f32", video);
    try write(gpa, args[3], ".audio.f32", audio);
}
