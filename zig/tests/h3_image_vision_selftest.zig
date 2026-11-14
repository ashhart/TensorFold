//! tf-h3-image-vision selftest: a tiny Qwen3-VL tower with DeepStack, loaded as a run loads one, against float64.
const std = @import("std");
const mtl = @import("metal");
const vl = @import("qwen_vl");
const fixture = @import("checkpoint_fixture");
const bf16_round = @import("bf16_round");
const tool = @import("h3_image_vision.zig");
const files = @import("h3_image_files.zig");
const check = @import("h3_image_match.zig");

const ref = vl.reference;

/// The selftest's gates, fixed before any run.
pub const gates = struct {
    /// Each output against float64: 1 - cosine and relative error at most twice a float32-sum host run's, cosine 0.999.
    pub const noise_factor: f64 = 2;
    pub const cosine_floor: f64 = 0.999;
};

/// Qwen3-VL's tower, small: 4 blocks of 144 (2 heads of 72, MLP 272), a 2 x 2 merge to 200, DeepStack at 0, 2, 3.
const config_json =
    \\{"model_type": "qwen3_vl", "image_token_id": 151655, "vision_start_token_id": 151652, "vision_end_token_id": 151653,
    \\ "vision_config": {"deepstack_visual_indexes": [0, 2, 3], "depth": 4, "hidden_act": "gelu_pytorch_tanh",
    \\  "hidden_size": 144, "in_channels": 3, "intermediate_size": 272, "num_heads": 2, "num_position_embeddings": 64,
    \\  "out_hidden_size": 200, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2}}
;
/// Qwen3-VL's processor, with pixel bounds a 288 x 192 picture sits inside unresized: 12 x 18 patches, 54 tokens.
const processor_json =
    \\{"size": {"longest_edge": 262144, "shortest_edge": 4096}, "patch_size": 16, "temporal_patch_size": 2,
    \\ "merge_size": 2, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
;
const width = 288;
const height = 192;
/// The HF checkpoint's vision prefix, found by the loader as a run finds it.
const prefix = "model.visual.";

/// A synthetic checkpoint's tensors: bf16 drawn uniformly, kept as fp32 for the reference.
const Synth = struct {
    a: std.mem.Allocator,
    rand: std.Random,
    entries: std.ArrayList(fixture.Entry) = .empty,

    fn tensor(s: *Synth, name: []const u8, shape: []const usize, center: f32, spread: f32) ![]const f32 {
        var n: usize = 1;
        for (shape) |d| n *= d;
        const vals = try s.a.alloc(f32, n);
        const bits = try s.a.alloc(u16, n);
        for (vals, bits) |*v, *b| {
            b.* = bf16_round.bits(center + spread * (2 * s.rand.float(f32) - 1));
            v.* = bf16_round.value(b.*);
        }
        try s.entries.append(s.a, .{ .name = name, .dtype = "BF16", .shape = try s.a.dupe(usize, shape), .bytes = std.mem.sliceAsBytes(bits) });
        return vals;
    }

    /// A tensor by its field: biases small, norm weights near 1, projections of unit gain over their inputs.
    fn weight(s: *Synth, comptime field: []const u8, name: []const u8, shape: []const usize) ![]const f32 {
        if (comptime std.mem.endsWith(u8, field, "_b")) return s.tensor(name, shape, 0, 0.1);
        if (comptime std.mem.startsWith(u8, field, "norm")) return s.tensor(name, shape, 1, 0.2);
        var k: usize = 1;
        for (shape[1..]) |d| k *= d;
        return s.tensor(name, shape, 0, @sqrt(3.0 / @as(f32, @floatFromInt(k))));
    }

    /// A merger under `at`, its norm `norm` wide: the main one per patch, a DeepStack one over the merged width.
    fn merger(s: *Synth, c: vl.Config, at: []const u8, norm: usize) !ref.Merger {
        const m: usize = c.merged();
        const o: usize = c.out_hidden;
        var out: ref.Merger = undefined;
        const fields = .{ .{ "norm_w", "norm.weight", &[_]usize{norm} }, .{ "norm_b", "norm.bias", &[_]usize{norm} }, .{ "fc1_w", "linear_fc1.weight", &[_]usize{ m, m } }, .{ "fc1_b", "linear_fc1.bias", &[_]usize{m} }, .{ "fc2_w", "linear_fc2.weight", &[_]usize{ o, m } }, .{ "fc2_b", "linear_fc2.bias", &[_]usize{o} } };
        inline for (fields) |f| @field(out, f[0]) = try s.weight(f[0], try std.fmt.allocPrint(s.a, "{s}{s}", .{ at, f[1] }), f[2]);
        return out;
    }

    /// Every tower tensor under `at` with HF's names and layouts (the patch kernel [O, C, T, Y, X]), DeepStack's too.
    fn tower(s: *Synth, c: vl.Config, at: []const u8) !ref.Weights {
        const a = s.a;
        const h: usize = c.hidden;
        const i: usize = c.intermediate;
        var w: ref.Weights = undefined;
        w.patch_w = try s.weight("patch_w", try std.fmt.allocPrint(a, "{s}patch_embed.proj.weight", .{at}), &.{ h, 3, c.temporal, c.patch, c.patch });
        w.patch_b = try s.weight("patch_b", try std.fmt.allocPrint(a, "{s}patch_embed.proj.bias", .{at}), &.{h});
        w.pos = try s.tensor(try std.fmt.allocPrint(a, "{s}pos_embed.weight", .{at}), &.{ c.positions, h }, 0, 0.5);
        const blocks = try a.alloc(ref.Block, c.depth);
        for (blocks, 0..) |*b, bi| {
            const fields = .{ .{ "norm1_w", "norm1.weight", &[_]usize{h} }, .{ "norm1_b", "norm1.bias", &[_]usize{h} }, .{ "qkv_w", "attn.qkv.weight", &[_]usize{ 3 * h, h } }, .{ "qkv_b", "attn.qkv.bias", &[_]usize{3 * h} }, .{ "proj_w", "attn.proj.weight", &[_]usize{ h, h } }, .{ "proj_b", "attn.proj.bias", &[_]usize{h} }, .{ "norm2_w", "norm2.weight", &[_]usize{h} }, .{ "norm2_b", "norm2.bias", &[_]usize{h} }, .{ "fc1_w", "mlp.linear_fc1.weight", &[_]usize{ i, h } }, .{ "fc1_b", "mlp.linear_fc1.bias", &[_]usize{i} }, .{ "fc2_w", "mlp.linear_fc2.weight", &[_]usize{ h, i } }, .{ "fc2_b", "mlp.linear_fc2.bias", &[_]usize{h} } };
            inline for (fields) |f| @field(b, f[0]) = try s.weight(f[0], try std.fmt.allocPrint(a, "{s}blocks.{d}.{s}", .{ at, bi, f[1] }), f[2]);
        }
        w.blocks = blocks;
        w.merger = try s.merger(c, try std.fmt.allocPrint(a, "{s}merger.", .{at}), h);
        const deep = try a.alloc(ref.Merger, c.taps().len);
        for (deep, 0..) |*d, j| d.* = try s.merger(c, try std.fmt.allocPrint(a, "{s}deepstack_merger_list.{d}.", .{ at, j }), c.merged());
        w.deep = deep;
        return w;
    }
};

/// The same tower as MLX holds a Qwen3.5 one: under `vision_tower.`, the patch kernel [O, T, Y, X, C], no DeepStack.
fn mlx(a: std.mem.Allocator, entries: []const fixture.Entry, c: vl.Config) ![]fixture.Entry {
    var out: std.ArrayList(fixture.Entry) = .empty;
    for (entries) |e| {
        if (!std.mem.startsWith(u8, e.name, prefix)) continue;
        const rest = e.name[prefix.len..];
        if (std.mem.startsWith(u8, rest, "deepstack_merger_list.")) continue;
        var moved = e;
        moved.name = try std.fmt.allocPrint(a, "vision_tower.{s}", .{rest});
        if (std.mem.eql(u8, rest, "patch_embed.proj.weight")) {
            const src = @as([*]const u16, @ptrCast(@alignCast(e.bytes.ptr)))[0 .. e.bytes.len / 2];
            const dst = try a.alloc(u16, src.len);
            const plane: usize = c.patch * c.patch;
            const T: usize = c.temporal;
            for (0..c.hidden) |o| for (0..3) |ch| for (0..T) |t| for (0..plane) |yx| {
                dst[((o * T + t) * plane + yx) * 3 + ch] = src[((o * 3 + ch) * T + t) * plane + yx];
            };
            moved.bytes = std.mem.sliceAsBytes(dst);
            moved.shape = try a.dupe(usize, &.{ c.hidden, T, c.patch, c.patch, 3 });
        }
        try out.append(a, moved);
    }
    return out.items;
}

fn writeAt(io: std.Io, a: std.mem.Allocator, parts: []const []const u8, data: []const u8) !void {
    const path = try std.fs.path.join(a, parts);
    try std.Io.Dir.cwd().createDirPath(io, std.fs.path.dirname(path).?);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
}

pub fn run(init: std.process.Init, a: std.mem.Allocator, dir: []const u8) !u8 {
    const io = init.io;
    const gpa = init.gpa;
    var prng = std.Random.DefaultPrng.init(0x5135);
    const rand = prng.random();
    const text_dir = try std.fs.path.join(a, &.{ dir, "text_encoder" });
    const model_dir = try std.fs.path.join(a, &.{ dir, "model" });
    const mlx_dir = try std.fs.path.join(a, &.{ dir, "qwen35" });
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    try writeAt(io, a, &.{ text_dir, "config.json" }, config_json);
    try writeAt(io, a, &.{ model_dir, "processor", "preprocessor_config.json" }, processor_json);
    const c = try tool.settings(io, a, text_dir, model_dir);
    var synth = Synth{ .a = a, .rand = rand };
    _ = try synth.tensor("model.language_model.layers.0.input_layernorm.weight", &.{64}, 1, 0.2); // left alone
    const w = try synth.tower(c, prefix);
    try files.checkpoint(io, a, text_dir, synth.entries.items);
    try files.checkpoint(io, a, mlx_dir, try mlx(a, synth.entries.items, c));

    // a picture through the processor's patches, as a run cuts its Qwen copy
    const px = try a.alloc(u8, width * height * 3);
    for (0..height) |y| for (0..width) |x| {
        const at = (y * width + x) * 3;
        px[at..][0..3].* = .{ @intCast(x * 255 / (width - 1)), @intCast(y * 255 / (height - 1)), rand.int(u8) };
    };
    const p = try tool.patches(a, c, .{ .width = width, .height = height, .pixels = px });
    if (p.gh != 12 or p.gw != 18 or p.tokens != 54) return error.SelftestGeometry;
    const n = @as(usize, p.tokens) * c.out_hidden;

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const queue = try device.queue();
    const found = try vl.shards.findPrefix(a, io, text_dir);
    var failed = !std.mem.eql(u8, found, prefix);
    const tower = try vl.tower.Tower.loadAt(gpa, io, device, text_dir, found, c, p.tokens, device);
    defer tower.deinit();
    std.debug.print("{s}: synthetic Qwen3-VL tower under {s}, {d} blocks of {d}, DeepStack after {any}; {d} x {d} patches, {d} tokens\n", .{ device.name(), found, c.depth, c.hidden, c.taps(), p.gh, p.gw, p.tokens });
    const started = mtl.clock.seconds();
    const first = try tool.pass(device, queue, tower, a, p, 3);
    const took = mtl.clock.seconds() - started;
    const again = try tool.pass(device, queue, tower, a, p, 3);
    const alone = try tool.pass(device, queue, tower, a, p, 0);

    const exact = try a.alloc(f32, 4 * n);
    const sum32 = try a.alloc(f32, 4 * n);
    inline for (.{ .{ f64, exact }, .{ f32, sum32 } }) |run_| {
        const o = run_[1];
        try ref.image(run_[0], a, c, w, p.pixels, p.gh, p.gw, o[0..n], &.{ o[n..][0..n], o[2 * n ..][0..n], o[3 * n ..][0..n] });
    }
    const noise_bits = try check.bf16s(a, sum32);
    for (0..4) |j| {
        var label: [64]u8 = undefined;
        const name = if (j == 0) "merged rows" else try std.fmt.bufPrint(&label, "DeepStack after block {d}", .{c.taps()[j - 1]});
        const noise = check.match(noise_bits[j * n ..][0..n], exact[j * n ..][0..n], c.out_hidden);
        const m = check.match(first[j * n ..][0..n], exact[j * n ..][0..n], c.out_hidden);
        const pass = check.within(m, noise, gates.noise_factor, gates.cosine_floor);
        check.report(name, m, pass);
        std.debug.print("  the float32-sum host run: cosine {d:.7}, relative error {e:.2}, {d:.4} bit-equal\n", .{ noise.cosine, noise.relative, noise.equal });
        failed = failed or !pass;
    }
    const same = std.mem.eql(u16, first, again);
    const untapped = std.mem.eql(u16, first[0..n], alone);
    std.debug.print("tower pass in {d:.3} s; again, byte for byte the same: {}; without taps, the same rows: {}\n", .{ took, same, untapped });
    failed = failed or !same or !untapped;

    // the engine's Qwen3.5 path: MLX's names and patch layout, no taps, through Tower.load and image()
    var plain = c;
    plain.deepstack_len = 0;
    const qwen35 = try vl.tower.Tower.load(gpa, io, device, mlx_dir, plain, p.tokens);
    defer qwen35.deinit();
    const buf = try device.buffer(n * 2, mtl.ResourceOptions.shared);
    defer buf.deinit();
    try qwen35.image(queue, p.pixels, p.gh, p.gw, .{ .buf = buf });
    const kept = std.mem.eql(u16, first[0..n], buf.slice(u16, n));
    std.debug.print("the same tower as vision_tower.* in MLX's layout, untapped, through Tower.load: the same rows: {}\n", .{kept});
    failed = failed or !kept;
    std.debug.print("{s}\n", .{if (failed) "FAIL" else "PASS every gate"});
    return @intFromBool(failed);
}
