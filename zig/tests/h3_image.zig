//! MiniMax-H3-Image's transformer on Metal from a picture case: the six distilled steps, each first on the
//! reference's own input (teacher-forced) and gated, then all six from the noise with our own rows.
const std = @import("std");
const mtl = @import("metal");
const st = @import("safetensors");
const h3 = @import("h3");
const fixture = @import("checkpoint_fixture");

const picture = h3.picture;

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, count: usize, file: *anyopaque) usize;
extern "c" fn fclose(file: *anyopaque) c_int;

const usage =
    \\usage: tf-h3-image MODEL_DIR picture.safetensors OUT_PREFIX [once | free]
    \\  MODEL_DIR    the MiniMax-H3-Image folder: model_index.json, transformer/, pdd_config.json and its heads
    \\  picture      the case: canvas, text (the token refiner's rows), noise, and optionally text_tags,
    \\               references and anchors, the reference's step_in, velocity and latent, and its positions,
    \\               timesteps, deltas and rotary_cos / rotary_sin for the exact checks
    \\  once         the first step alone, teacher-forced; free: the free run alone
    \\Exit 0: every gate passed; 1: a gate failed; 2: no reference to gate against (inconclusive).
    \\
;

fn read(init: std.process.Init, a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    return std.Io.Dir.cwd().readFileAlloc(init.io, path, a, .limited(64 << 20)) catch |err| {
        std.debug.print("cannot read {s}: {s}\n", .{ path, @errorName(err) });
        return err;
    };
}

fn write(a: std.mem.Allocator, prefix: []const u8, suffix: []const u8, values: []const f32) !void {
    const path = try std.fmt.allocPrintSentinel(a, "{s}{s}", .{ prefix, suffix }, 0);
    const file = fopen(path.ptr, "wb") orelse return error.CannotWrite;
    defer _ = fclose(file);
    if (fwrite(values.ptr, 4, values.len, file) != values.len) return error.CannotWrite;
}

/// A tensor of the checkpoint's in its own bytes, checked for its type and size.
fn raw(ck: *const h3.Checkpoint, name: []const u8, dtype: st.DType, count: usize) ![]const u8 {
    for (ck.shards) |*shard| if (shard.names.get(name)) |e| {
        if (e.dtype != dtype or e.end - e.begin != count * dtype.size()) {
            std.debug.print("{s}: unexpected type or size\n", .{name});
            return error.BadTensor;
        }
        return shard.map.bytes[shard.data + e.begin ..][0 .. e.end - e.begin];
    };
    std.debug.print("no tensor {s}\n", .{name});
    return error.MissingTensor;
}

/// The case's optional tensor `name` of `dtype` and `count` values, else null.
fn optional(case: *const h3.Tensors, comptime T: type, name: []const u8, dtype: st.DType, count: usize) !?[]const T {
    if (case.names.get(name) == null) return null;
    return case.values(T, try case.at(name, dtype, count), count);
}

fn bytes(comptime T: type, values: []const T) []const u8 {
    return std.mem.sliceAsBytes(values);
}

/// A safetensors image whose tensors all start aligned: the header padded so the data starts on 16 bytes, the
/// four-byte tensors first, then the two-byte ones.
fn image(a: std.mem.Allocator, entries: []fixture.Entry) ![]u8 {
    std.mem.sort(fixture.Entry, entries, {}, struct {
        fn wide(_: void, x: fixture.Entry, y: fixture.Entry) bool {
            return !std.mem.eql(u8, x.dtype, "BF16") and std.mem.eql(u8, y.dtype, "BF16");
        }
    }.wide);
    const plain = try fixture.image(a, entries);
    const header: usize = @intCast(std.mem.readInt(u64, plain[0..8], .little));
    const pad = (16 - (8 + header) % 16) % 16;
    const out = try a.alloc(u8, plain.len + pad);
    std.mem.writeInt(u64, out[0..8], header + pad, .little);
    @memcpy(out[8..][0..header], plain[8..][0..header]);
    @memset(out[8 + header ..][0..pad], ' ');
    @memcpy(out[8 + header + pad ..], plain[8 + header ..]);
    return out;
}

const Gate = struct { failed: bool = false, gated: bool = false };

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4 and args.len != 5) {
        std.debug.print("{s}", .{usage});
        return error.BadArguments;
    }
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const dir = args[1];
    const mode: enum { all, once, free } = if (args.len == 4) .all else if (std.mem.eql(u8, args[4], "once")) .once else if (std.mem.eql(u8, args[4], "free")) .free else {
        std.debug.print("{s}", .{usage});
        return error.BadArguments;
    };

    // the folder: the image pipeline, an audio-free transformer, the distilled schedule and its heads
    const heads_name = try picture.checkIndex(a, try read(init, a, dir, "model_index.json"));
    const transformer_dir = try std.fs.path.join(a, &.{ dir, "transformer" });
    try picture.checkTransformer(a, try read(init, a, transformer_dir, "config.json"));
    const shards = try picture.shardCount(a, try read(init, a, transformer_dir, "model.safetensors.index.json"));
    const pdd = try picture.pdd(a, try read(init, a, std.fs.path.dirname(try std.fs.path.join(a, &.{ dir, heads_name })) orelse dir, "pdd_config.json"));
    const steps = pdd.steps();

    const device = try mtl.Device.init();
    const ck = try h3.Checkpoint.open(gpa, device, transformer_dir, shards);
    var heads = try h3.Tensors.open(gpa, device, try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, heads_name }, 0));
    defer heads.deinit();
    var pic = try h3.Tensors.open(gpa, device, args[2]);
    defer pic.deinit();

    // the picture's rows: the canvas's target grid, the text, each reference's grid
    const width = picture.row_width;
    const hidden_entry = ck.entry("final_layer.norm.weight") orelse return error.MissingTensor;
    const hidden = hidden_entry.shape[0];
    const canvas_v = pic.values(i32, try pic.at("canvas", .i32, 2), 2);
    const canvas = picture.Size{ .width = @intCast(@max(canvas_v[0], 0)), .height = @intCast(@max(canvas_v[1], 0)) };
    const target_grid = try picture.gridOf(canvas);
    const target = try picture.gridRows(target_grid);
    const text_entry = pic.names.get("text") orelse return error.MissingTensor;
    if (text_entry.rank != 2 or text_entry.shape[1] != hidden) return error.BadTensor;
    const text = text_entry.shape[0];
    var refs: []picture.Grid = &.{};
    if (pic.names.get("references")) |e| {
        if (e.rank != 2 or e.shape[1] != 2) return error.BadTensor;
        const v = pic.values(i32, try pic.at("references", .i32, 2 * e.shape[0]), 2 * e.shape[0]);
        refs = try a.alloc(picture.Grid, e.shape[0]);
        for (refs, 0..) |*g, k| g.* = .{ .rows = @intCast(@max(v[2 * k], 0)), .columns = @intCast(@max(v[2 * k + 1], 0)) };
    }
    const layout = picture.Layout{ .text = text, .references = refs, .target = target_grid };
    const held = try layout.referenceRows();
    const rows = try layout.rows();
    const tags = try optional(&pic, i32, "text_tags", .i32, text);
    if (tags) |t| for (t) |tag| if (tag != 0 and tag != 1) return error.BadTensor;

    // positions and the rotary table from the checkpoint's frequencies, the tile map of a dense forward
    const pos = try picture.positions(a, layout);
    const inv_freq = try ck.floats(a, "rope.inv_freq");
    const rot = try picture.rotary(a, pos, inv_freq);
    const map = try picture.tiles(a, text + held, target);
    var gate = Gate{};
    exact: {
        const theirs = (try optional(&pic, f32, "positions", .f32, rows * 3)) orelse break :exact;
        var same: usize = 0;
        for (pos, 0..) |q, i| for (q, 0..) |v, ax| {
            same += @intFromBool(@as(u32, @bitCast(v)) == @as(u32, @bitCast(theirs[3 * i + ax])));
        };
        std.debug.print("positions against the reference's: {d} of {d} bit-identical\n", .{ same, rows * 3 });
        gate.failed = gate.failed or same != rows * 3;
    }
    exact: {
        const c = (try optional(&pic, u16, "rotary_cos", .bf16, rows * picture.angles)) orelse break :exact;
        const s = (try optional(&pic, u16, "rotary_sin", .bf16, rows * picture.angles)) orelse break :exact;
        var same: usize = 0;
        var worst: u32 = 0;
        for (0..rows) |i| for (0..picture.angles) |j| {
            const ours = [2]u16{ picture.bf16(rot.cos[i * picture.rotated + j]), picture.bf16(rot.sin[i * picture.rotated + j]) };
            const want = [2]u16{ c[i * picture.angles + j], s[i * picture.angles + j] };
            for (ours, want) |x, y| {
                same += @intFromBool(x == y);
                worst = @max(worst, picture.ulps(x, y));
            }
        };
        std.debug.print("rotary table against the reference's: {d} of {d} bf16 values identical, at most {d} ULP apart\n", .{ same, 2 * rows * picture.angles, worst });
        gate.failed = gate.failed or worst > picture.gates.rotary_ulps;
    }
    inline for (.{ .{ "timesteps", pdd.timesteps }, .{ "deltas", pdd.deltas } }) |kv| exact: {
        const theirs = (try optional(&pic, f32, kv[0], .f32, steps)) orelse break :exact;
        const same = std.mem.eql(u32, @as([]const u32, @ptrCast(kv[1])), @as([]const u32, @ptrCast(theirs)));
        std.debug.print("{s} against the reference's: {s}\n", .{ kv[0], if (same) "bit-identical" else "DIFFERENT" });
        gate.failed = gate.failed or !same;
    }

    // the case the transformer reads: these rows, the checkpoint's float projections and the step heads, and
    // placeholder tables that the distilled schedule's replace below
    const noise = pic.values(f32, try pic.at("noise", .f32, target * width), target * width);
    const reference = try optional(&pic, f32, "velocity", .f32, steps * target * width);
    const step_in = try optional(&pic, f32, "step_in", .f32, steps * target * width);
    const latent = try optional(&pic, f32, "latent", .f32, target * width);
    const anchors: ?[]const f32 = if (held > 0) pic.values(f32, try pic.at("anchors", .f32, held * width), held * width) else null;
    const geometry = [_]i32{ @intCast(map.prefix), @intCast(map.target), @intCast(map.target), @intCast(text), 0, @intCast(target), @intCast(held) };
    const adaln = try a.alloc(i32, steps * rows);
    for (0..steps) |k| for (0..rows) |r| {
        adaln[k * rows + r] = if (r < text) (if (tags) |t| t[r] else 1) else 0;
    };
    const times = try a.alloc(i32, steps * rows);
    @memset(times, 0);
    const video_step = try a.alloc(f32, 2 * steps);
    for (0..steps) |k| video_step[2 * k ..][0..2].* = .{ pdd.sigmas[k], pdd.deltas[k] };
    const zero_velocity = try a.alloc(f32, target * width);
    @memset(zero_velocity, 0);
    const tables = try a.alloc(u16, 6 * 3 * hidden);
    @memset(tables, 0);
    const final = try a.alloc(u16, 2 * hidden);
    @memset(final, 0);
    const text_bytes = pic.map.bytes[pic.data + text_entry.begin ..][0 .. text * hidden * 2];
    var entries: std.ArrayList(fixture.Entry) = .empty;
    const shape = struct {
        fn of(al: std.mem.Allocator, dims: anytype) ![]const usize {
            return al.dupe(usize, &dims);
        }
    };
    try entries.appendSlice(a, &.{
        .{ .name = "geometry", .dtype = "I32", .shape = try shape.of(a, [_]usize{7}), .bytes = bytes(i32, &geometry) },
        .{ .name = "text", .dtype = "BF16", .shape = try shape.of(a, [_]usize{ text, hidden }), .bytes = text_bytes },
        .{ .name = "final", .dtype = "BF16", .shape = try shape.of(a, [_]usize{ 1, 2 * hidden }), .bytes = bytes(u16, final) },
        .{ .name = "final_norm.weight", .dtype = "BF16", .shape = try shape.of(a, [_]usize{hidden}), .bytes = try raw(&ck, "final_layer.norm.weight", .bf16, hidden) },
        .{ .name = "video_in.weight", .dtype = "F32", .shape = try shape.of(a, [_]usize{ hidden, width }), .bytes = try raw(&ck, "video_patch_proj.weight", .f32, hidden * width) },
        .{ .name = "video_in.bias", .dtype = "F32", .shape = try shape.of(a, [_]usize{hidden}), .bytes = try raw(&ck, "video_patch_proj.bias", .f32, hidden) },
        .{ .name = "video_out.weight", .dtype = "F32", .shape = try shape.of(a, [_]usize{ steps, width, hidden }), .bytes = bytes(f32, heads.values(f32, try heads.at("video_out.weight", .f32, steps * width * hidden), steps * width * hidden)) },
        .{ .name = "video_out.bias", .dtype = "F32", .shape = try shape.of(a, [_]usize{ steps, width }), .bytes = bytes(f32, heads.values(f32, try heads.at("video_out.bias", .f32, steps * width), steps * width)) },
        .{ .name = "video", .dtype = "F32", .shape = try shape.of(a, [_]usize{ target, width }), .bytes = bytes(f32, noise) },
        .{ .name = "video_velocity", .dtype = "F32", .shape = try shape.of(a, [_]usize{ target, width }), .bytes = bytes(f32, if (reference) |v| v[0 .. target * width] else zero_velocity) },
        .{ .name = "video_step", .dtype = "F32", .shape = try shape.of(a, [_]usize{ steps, 2 }), .bytes = bytes(f32, video_step) },
        .{ .name = "cos", .dtype = "F32", .shape = try shape.of(a, [_]usize{ rows, picture.rotated }), .bytes = bytes(f32, rot.cos) },
        .{ .name = "sin", .dtype = "F32", .shape = try shape.of(a, [_]usize{ rows, picture.rotated }), .bytes = bytes(f32, rot.sin) },
        .{ .name = "adaln", .dtype = "I32", .shape = try shape.of(a, [_]usize{ steps, rows }), .bytes = bytes(i32, adaln) },
        .{ .name = "times", .dtype = "I32", .shape = try shape.of(a, [_]usize{ steps, rows }), .bytes = bytes(i32, times) },
        .{ .name = "tile_slot", .dtype = "I32", .shape = try shape.of(a, [_]usize{rows}), .bytes = bytes(i32, map.slot) },
        .{ .name = "tile_sizes", .dtype = "I32", .shape = try shape.of(a, [_]usize{map.sizes.len}), .bytes = bytes(i32, map.sizes) },
    });
    if (anchors) |v| try entries.append(a, .{ .name = "condition", .dtype = "F32", .shape = try shape.of(a, [_]usize{ held, width }), .bytes = bytes(f32, v) });
    var name: [32]u8 = undefined;
    var layers: usize = 0;
    while (ck.entry(try std.fmt.bufPrint(&name, "blocks.{d}.norm1.weight", .{layers})) != null) layers += 1;
    for (0..layers) |i| try entries.append(a, .{ .name = try std.fmt.allocPrint(a, "tables.{d}", .{i}), .dtype = "BF16", .shape = try shape.of(a, [_]usize{ 6, 3, hidden }), .bytes = bytes(u16, tables) });
    const case_path = try std.fmt.allocPrintSentinel(a, "{s}.h3case.safetensors", .{args[3]}, 0);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = case_path, .data = try image(a, entries.items) });
    var case = try h3.Tensors.open(gpa, device, case_path);
    defer case.deinit();

    const loading = mtl.clock.seconds();
    const model = try h3.Model.load(gpa, transformer_dir, shards, &case, .{});
    if (!model.base or model.step_heads != steps or model.audio != 0) return error.H3NotPictureModel;
    model.dense = true;
    const schedule = h3.schedule.Schedule{ .sigmas = pdd.sigmas, .timesteps = pdd.timesteps };
    _ = try h3.modulation.reschedule(model, a, schedule, schedule, false);
    std.debug.print("{s}: {d}x{d} canvas, {d} rows (text {d}, references {d}, target {d}), {d} tiles, {d} blocks, {d} steps, {d} timesteps, loaded in {d:.1} s\n", .{ device.name(), canvas.width, canvas.height, rows, text, held, target, map.prefix + map.target, model.blocks.len, steps, model.lines / h3.schedule.modalities, mtl.clock.seconds() - loading });

    const velocity = try a.alloc(f32, target * width);
    const x = try a.alloc(f32, target * width);
    // teacher-forced: every step on the reference's own input, its target velocity gated
    if (mode != .free) teacher: {
        const ins = step_in orelse {
            std.debug.print("no step_in in the case: the teacher-forced gate is inconclusive\n", .{});
            break :teacher;
        };
        const want = reference orelse break :teacher;
        for (0..if (mode == .once) 1 else steps) |k| {
            @memcpy(x, ins[k * target * width ..][0 .. target * width]);
            const took = try model.forward(x, &.{}, k, velocity, &.{});
            const d = try picture.distance(velocity, want[k * target * width ..][0 .. target * width], width);
            const ok = picture.stepPasses(d);
            gate.gated = true;
            gate.failed = gate.failed or !ok;
            std.debug.print("step {d}/{d} teacher-forced: cosine {d:.6}, relative error {d:.5}, worst row {d:.5} ({s}), {d:.2} s wall, {d:.2} s GPU\n", .{ k + 1, steps, d.cosine, d.relative, d.worst_row, if (ok) "pass" else "FAIL", took.wall, took.gpu });
            if (k == 0) try write(a, args[3], ".first-velocity.f32", velocity);
        }
    }
    // free-running: our own rows from the noise through every step
    if (mode != .once) {
        @memcpy(x, noise);
        for (0..steps) |k| {
            const took = try model.forward(x, &.{}, k, velocity, &.{});
            picture.euler(x, velocity, pdd.deltas[k]);
            std.debug.print("step {d}/{d}: {d:.2} s wall, {d:.2} s GPU\n", .{ k + 1, steps, took.wall, took.gpu });
        }
        try write(a, args[3], ".rows.f32", x);
        const out = try a.alloc(f32, x.len);
        try picture.unpatchify(x, target_grid, out);
        try write(a, args[3], ".latent.f32", out);
        if (latent) |want| {
            const d = try picture.distance(x, want, width);
            const ok = d.cosine >= picture.gates.latent_cosine;
            gate.gated = true;
            gate.failed = gate.failed or !ok;
            std.debug.print("final latent against the reference's: cosine {d:.6}, relative error {d:.5}, worst row {d:.5} ({s})\n", .{ d.cosine, d.relative, d.worst_row, if (ok) "pass" else "FAIL" });
        }
        std.debug.print("wrote {s}.latent.f32: [{d}, {d}, {d}] normalised float32\n", .{ args[3], picture.channels, target_grid.rows, target_grid.columns });
    }
    if (gate.failed) std.process.exit(1);
    if (!gate.gated) std.process.exit(2);
}
