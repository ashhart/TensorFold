//! tf-h3-image-prompt: MiniMax-H3-Image's prompt to the refiner's text rows on Metal, or its synthetic selftest.
const std = @import("std");
const mtl = @import("metal");
const hp = @import("h3_prompt");
const tokenizer = @import("tokenizer");
const fixture = @import("core").checkpoint_fixture;
const bf16_round = @import("bf16_round");
const torch_noise = @import("moss_noise");
const check = @import("h3_image_prompt_check.zig");

const rules = hp.rules;
const ref = hp.reference;
const weights = hp.weights; // its fixture test runs here

const usage =
    \\usage: tf-h3-image-prompt TEXT_ENCODER_DIR MODEL_DIR prompt.txt out.safetensors [WIDTHxHEIGHT [SEED [MIN_AREA]]]
    \\       tf-h3-image-prompt selftest [WORK_DIR]
    \\  TEXT_ENCODER_DIR  MiniMax-H3's text_encoder/ (Qwen3-VL): config.json, its index and shards
    \\  MODEL_DIR         the MiniMax-H3-Image folder: model_index.json, tokenizer/, transformer/
    \\  prompt.txt        the prompt, verbatim but for one trailing line ending
    \\  out               text BF16 [rows, 5376] and text_tags I32 [rows], as tf-h3-image's picture case reads them,
    \\                    and ids I32 [rows]
    \\  WIDTHxHEIGHT      also a text-to-image case: the canvas (grown to MIN_AREA, default 2048 x 2048, at the
    \\                    size's ratio, sides snapped to 32), its starting rows from SEED (default 0) as torch's
    \\                    CPU randn draws them, and the requested size
    \\  selftest          the GPU kernels on a tiny synthetic checkpoint against a float64 host reference, gated;
    \\                    WORK_DIR (default under TMPDIR) holds the checkpoint while it runs
    \\Exit 0: done or every gate passed; 1: a gate failed.
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len >= 2 and std.mem.eql(u8, args[1], "selftest") and args.len <= 3) {
        const tmp = init.environ_map.get("TMPDIR") orelse "/tmp";
        const dir = if (args.len == 3) args[2] else try std.fs.path.join(a, &.{ tmp, "tf-h3-image-prompt-selftest" });
        return selftest(init, a, dir);
    }
    if (args.len < 5 or args.len > 8) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const request: ?Request = if (args.len >= 6) .{
        .size = try size(args[5]),
        .seed = if (args.len >= 7) try std.fmt.parseInt(u64, args[6], 10) else 0,
        .min_area = if (args.len == 8) try std.fmt.parseInt(u64, args[7], 10) else hp.picture.default_min_area,
    } else null;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var encoder = try Encoder.load(init, a, args[1], args[2]);
    defer encoder.deinit(init.gpa);
    var prompt: []const u8 = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], a, .limited(16 << 20));
    if (std.mem.endsWith(u8, prompt, "\n")) prompt = prompt[0 .. prompt.len - 1];
    if (std.mem.endsWith(u8, prompt, "\r")) prompt = prompt[0 .. prompt.len - 1];
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[4], .data = try encoder.case(a, prompt, request) });
    std.debug.print("wrote {s}: text [{d}, {d}] bf16, text_tags and ids\n", .{ args[4], encoder.text, encoder.front.hidden });
    return 0;
}

/// A text-to-image request: the size asked for, the seed and the smallest area drawn.
pub const Request = struct { size: hp.picture.Size, seed: u64, min_area: u64 };

pub fn size(text: []const u8) !hp.picture.Size {
    const x = std.mem.indexOfScalar(u8, text, 'x') orelse return error.BadSize;
    return .{ .width = try std.fmt.parseInt(usize, text[0..x], 10), .height = try std.fmt.parseInt(usize, text[x + 1 ..], 10) };
}

fn read(io: std.Io, a: std.mem.Allocator, parts: []const []const u8) ![]u8 {
    const path = try std.fs.path.join(a, parts);
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch |err| {
        std.debug.print("cannot read {s}: {s}\n", .{ path, @errorName(err) });
        return err;
    };
}

/// A safetensors image whose data starts on 16 bytes: the header padded with spaces.
fn image(a: std.mem.Allocator, entries: []const fixture.Entry) ![]u8 {
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

/// The prompt path loaded once: the configs, the tokenizer, the text model's and the refiner's weights and kernels.
pub const Encoder = struct {
    enc: rules.Encoder,
    front: rules.Refiner,
    tok: tokenizer.Tokenizer,
    device: mtl.Device,
    text_w: hp.weights.Text,
    front_w: hp.weights.Refiner,
    runner: hp.Runner,
    /// The last case's token count.
    text: usize = 0,

    pub fn load(init: std.process.Init, a: std.mem.Allocator, text_dir: []const u8, model_dir: []const u8) !Encoder {
        const io = init.io;
        const gpa = init.gpa;
        const enc = try rules.checkEncoder(a, try read(io, a, &.{ text_dir, "config.json" }));
        _ = try hp.picture.checkIndex(a, try read(io, a, &.{ model_dir, "model_index.json" }));
        const transformer = try std.fs.path.join(a, &.{ model_dir, "transformer" });
        const front = try rules.checkRefiner(a, try read(io, a, &.{ transformer, "config.json" }));
        var tok = try tokenizer.loadTokenizer(io, a, try std.fs.path.join(a, &.{ model_dir, "tokenizer" }));
        errdefer tok.deinit();
        const device = try mtl.Device.init();
        const loading = mtl.clock.seconds();
        var text_w = try hp.weights.loadText(gpa, io, device, text_dir, enc.text);
        errdefer text_w.deinit(gpa);
        var front_w = try hp.weights.loadRefiner(gpa, io, device, transformer, front);
        errdefer front_w.deinit(gpa);
        var runner = try hp.Runner.init(device);
        runner.give_way = .{ .io = io };
        std.debug.print("{s}: prompt weights mapped and kernels built in {d:.1} s\n", .{ device.name(), mtl.clock.seconds() - loading });
        return .{ .enc = enc, .front = front, .tok = tok, .device = device, .text_w = text_w, .front_w = front_w, .runner = runner };
    }

    pub fn deinit(e: *Encoder, gpa: std.mem.Allocator) void {
        e.runner.deinit();
        e.front_w.deinit(gpa);
        e.text_w.deinit(gpa);
        e.tok.deinit();
    }

    /// `prompt` as tf-h3-image's picture case: canvas, noise, text_tags, ids and the refiner's text rows.
    pub fn case(e: *Encoder, a: std.mem.Allocator, prompt: []const u8, request: ?Request) ![]u8 {
        const p = try rules.present(a, &e.tok, e.enc.marks, &.{}, prompt);
        const pos = try rules.positions(a, p, e.enc.marks, &.{});
        const T = p.ids.len;
        e.text = T;

        const rows = try a.alloc(u16, T * e.enc.text.hidden);
        var started = mtl.clock.seconds();
        try e.runner.encode(&e.text_w, p.ids, pos, null, rows);
        std.debug.print("{d} tokens; text encoder, {d} layers: {d:.2} s wall, {d:.2} s GPU, {d:.2} s waited for speech\n", .{ T, e.enc.text.layers, mtl.clock.seconds() - started, e.runner.gpu, e.runner.paused });
        const text = try a.alloc(u16, T * e.front.hidden);
        started = mtl.clock.seconds();
        try e.runner.refine(&e.front_w, rows, text);
        std.debug.print("condition_proj and the token refiner: {d:.2} s wall, {d:.2} s GPU, {d:.2} s waited for speech\n", .{ mtl.clock.seconds() - started, e.runner.gpu, e.runner.paused });
        const ids = try a.alloc(i32, T);
        for (ids, p.ids) |*o, id| o.* = @intCast(id);
        var entries: std.ArrayList(fixture.Entry) = .empty;
        if (request) |r| {
            // the canvas and the seed's starting rows, ahead of the two-byte text so every tensor stays aligned
            const canvas = try hp.picture.canvas(r.size.width, r.size.height, r.min_area);
            const grid = try hp.picture.gridOf(canvas);
            const start = try hp.picture.startRows(a, grid, r.seed, &torch_noise.randn);
            const dims = try a.dupe(i32, &.{ @intCast(canvas.width), @intCast(canvas.height), @intCast(r.size.width), @intCast(r.size.height) });
            try entries.appendSlice(a, &.{
                .{ .name = "canvas", .dtype = "I32", .shape = &.{2}, .bytes = std.mem.sliceAsBytes(dims[0..2]) },
                .{ .name = "requested", .dtype = "I32", .shape = &.{2}, .bytes = std.mem.sliceAsBytes(dims[2..4]) },
                .{ .name = "noise", .dtype = "F32", .shape = try a.dupe(usize, &.{ start.len / hp.picture.row_width, hp.picture.row_width }), .bytes = std.mem.sliceAsBytes(start) },
            });
            std.debug.print("{d}x{d} drawn on a {d}x{d} canvas ({d} x {d} cells), seed {d}\n", .{ r.size.width, r.size.height, canvas.width, canvas.height, grid.rows, grid.columns, r.seed });
        }
        try entries.appendSlice(a, &.{
            .{ .name = "text_tags", .dtype = "I32", .shape = &.{T}, .bytes = std.mem.sliceAsBytes(p.tags) },
            .{ .name = "ids", .dtype = "I32", .shape = &.{T}, .bytes = std.mem.sliceAsBytes(ids) },
            .{ .name = "text", .dtype = "BF16", .shape = &.{ T, e.front.hidden }, .bytes = std.mem.sliceAsBytes(text) },
        });
        return image(a, entries.items);
    }
};

/// The selftest's geometry: two layers of a 256-wide text model (4 query and 2 KV heads), a 384-wide refiner.
pub const tiny_text: rules.Text = .{ .hidden = 256, .heads = 4, .kv_heads = 2, .ffn = 512, .vocab = 96, .layers = 2, .eps = 1e-6, .theta = 1e4, .sections = .{ 24, 20, 20 } };
pub const tiny_front: rules.Refiner = .{ .text = 256, .hidden = 384, .heads = 2, .mlp = 512, .blocks = 2, .eps = 1e-5, .qk_eps = 1e-5, .final_eps = 1e-5 };
/// Rows a run: past one 64-row tile, not a multiple of it.
const tiny_rows = 70;

/// A synthetic checkpoint folder: bf16 tensors drawn uniformly, one shard and its index.
const Synth = struct {
    a: std.mem.Allocator,
    rand: std.Random,
    entries: std.ArrayList(fixture.Entry) = .empty,

    /// A tensor `name` of `shape` drawn in center +- spread, kept in the shard; its values as fp32.
    fn tensor(s: *Synth, name: []const u8, shape: rules.Shape, center: f32, spread: f32) ![]const f32 {
        const n = shape[0] * @max(shape[1], 1);
        const vals = try s.a.alloc(f32, n);
        const bits = try s.a.alloc(u16, n);
        for (vals, bits) |*v, *b| {
            b.* = bf16_round.bits(center + spread * (2 * s.rand.float(f32) - 1));
            v.* = bf16_round.value(b.*);
        }
        const dims: []const usize = if (shape[1] == 0) try s.a.dupe(usize, shape[0..1]) else try s.a.dupe(usize, &shape);
        try s.entries.append(s.a, .{ .name = try s.a.dupe(u8, name), .dtype = "BF16", .shape = dims, .bytes = std.mem.sliceAsBytes(bits) });
        return vals;
    }

    /// A weight by its field: norms near 1, projections of unit gain, biases small.
    fn weight(s: *Synth, name: []const u8, field: []const u8, shape: rules.Shape) ![]const f32 {
        if (std.mem.indexOf(u8, field, "norm") != null) return s.tensor(name, shape, 1, 0.2);
        if (shape[1] == 0) return s.tensor(name, shape, 0, 0.1);
        return s.tensor(name, shape, 0, @sqrt(3.0 / @as(f32, @floatFromInt(shape[1]))));
    }

    fn write(s: *Synth, io: std.Io, dir: []const u8) !void {
        try std.Io.Dir.cwd().createDirPath(io, dir);
        const file = "model-00001-of-00001.safetensors";
        var index: std.ArrayList(u8) = .empty;
        try index.appendSlice(s.a, "{\"weight_map\": {");
        for (s.entries.items, 0..) |e, i| try index.print(s.a, "{s}\"{s}\": \"{s}\"", .{ if (i == 0) "" else ", ", e.name, file });
        try index.appendSlice(s.a, "}}");
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(s.a, &.{ dir, "model.safetensors.index.json" }), .data = index.items });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(s.a, &.{ dir, file }), .data = try image(s.a, s.entries.items) });
    }
};

/// The tiny text model and refiner written under `dir` (text_encoder/, transformer/), and their host copies.
pub const Tiny = struct {
    text_dir: []const u8,
    front_dir: []const u8,
    host: ref.Text,
    front: ref.Refiner,

    pub fn write(io: std.Io, a: std.mem.Allocator, rand: std.Random, dir: []const u8) !Tiny {
        const g = tiny_text;
        const f = tiny_front;
        // the text model: three layers written, two run, beside a vision tensor the loader must leave alone
        var text = Synth{ .a = a, .rand = rand };
        const prefix = "model.language_model.";
        const embed = try text.tensor(prefix ++ "embed_tokens.weight", .{ g.vocab, g.hidden }, 0, 1);
        _ = try text.tensor("model.visual.blocks.0.norm1.weight", .{ 128, 0 }, 1, 0.2);
        const layers = try a.alloc(ref.Layer, g.layers + 1);
        for (layers, 0..) |*l, i| inline for (comptime std.meta.fieldNames(ref.Layer), rules.layerTensors(g)) |field, s| {
            @field(l, field) = try text.weight(try std.fmt.allocPrint(a, "{s}layers.{d}.{s}", .{ prefix, i, s[1] }), field, s[2]);
        };
        var front = Synth{ .a = a, .rand = rand };
        var host_front: ref.Refiner = undefined;
        inline for (.{ "cond_w", "cond_b", "final_norm" }, rules.frontTensors(f)) |field, s| @field(host_front, field) = try front.weight(s[1], field, s[2]);
        const blocks = try a.alloc(ref.Block, f.blocks);
        for (blocks, 0..) |*b, i| inline for (comptime std.meta.fieldNames(ref.Block), rules.blockTensors(f)) |field, s| {
            @field(b, field) = try front.weight(try std.fmt.allocPrint(a, "token_refiner.blocks.{d}.{s}", .{ i, s[1] }), field, s[2]);
        };
        host_front.blocks = blocks;
        const out: Tiny = .{
            .text_dir = try std.fs.path.join(a, &.{ dir, "text_encoder" }),
            .front_dir = try std.fs.path.join(a, &.{ dir, "transformer" }),
            .host = .{ .embed = embed, .layers = layers },
            .front = host_front,
        };
        try text.write(io, out.text_dir);
        try front.write(io, out.front_dir);
        return out;
    }
};

fn selftest(init: std.process.Init, a: std.mem.Allocator, dir: []const u8) !u8 {
    const io = init.io;
    const gpa = init.gpa;
    var prng = std.Random.DefaultPrng.init(0x5133);
    const rand = prng.random();
    const g = tiny_text;
    const f = tiny_front;
    defer std.Io.Dir.cwd().deleteTree(io, dir) catch {};
    const tiny = try Tiny.write(io, a, rand, dir);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    var runner = try hp.Runner.init(device);
    defer runner.deinit();
    var text_w = try hp.weights.loadText(gpa, io, device, tiny.text_dir, g);
    defer text_w.deinit(gpa);
    var front_w = try hp.weights.loadRefiner(gpa, io, device, tiny.front_dir, f);
    defer front_w.deinit(gpa);
    std.debug.print("{s}: synthetic text model {d} wide ({d} layers run of {d}), refiner {d} wide, {d} rows\n", .{ device.name(), g.hidden, g.layers, g.layers + 1, f.hidden, tiny_rows });
    var failed = !try check.kernelsAlone(&runner, a, rand);

    const T = tiny_rows;
    const ids = try a.alloc(u32, T);
    for (ids) |*id| id.* = rand.uintLessThan(u32, @intCast(g.vocab));
    const pos = try a.alloc([3]i32, T);
    for (pos, 0..) |*p, i| p.* = @splat(@intCast(i));
    const enc = try a.alloc(u16, T * g.hidden);
    const started = mtl.clock.seconds();
    try runner.encode(&text_w, ids, pos, null, enc);
    const enc_s = mtl.clock.seconds() - started;
    const out = try a.alloc(u16, T * f.hidden);
    try runner.refine(&front_w, enc, out);
    const cos = try a.alloc(f32, T * 64);
    const sin = try a.alloc(f32, T * 64);
    rules.rotary(cos, sin, pos, g.theta, g.sections);
    const host = tiny.host;
    const enc_ref = try a.alloc(f32, T * g.hidden);
    try ref.encode(f64, a, g, host, ids, cos, sin, null, enc_ref);
    const enc_32 = try a.alloc(f32, T * g.hidden);
    try ref.encode(f32, a, g, host, ids, cos, sin, null, enc_32);
    failed = !try whole(a, "text encoder rows, a text prompt", enc, enc_32, enc_ref, g.hidden) or failed;
    const enc_f = try a.alloc(f32, enc.len);
    for (enc_f, enc) |*o, v| o.* = bf16_round.value(v);
    const out_ref = try a.alloc(f32, T * f.hidden);
    try ref.refine(f64, a, f, tiny.front, enc_f, out_ref);
    const out_32 = try a.alloc(f32, T * f.hidden);
    try ref.refine(f32, a, f, tiny.front, enc_f, out_32);
    failed = !try whole(a, "condition_proj, refiner and final norm on the GPU's encoder rows", out, out_32, out_ref, f.hidden) or failed;
    const again = try a.alloc(u16, enc.len);
    try runner.encode(&text_w, ids, pos, null, again);
    const out_again = try a.alloc(u16, out.len);
    try runner.refine(&front_w, again, out_again);
    const same = std.mem.eql(u16, enc, again) and std.mem.eql(u16, out, out_again);
    std.debug.print("encoded in {d:.3} s; again, byte for byte the same: {}\n", .{ enc_s, same });
    failed = failed or !same;
    failed = !try withPicture(&runner, &text_w, a, rand, host) or failed;
    std.debug.print("{s}\n", .{if (failed) "FAIL" else "PASS every gate"});
    return @intFromBool(failed);
}

/// A whole run's rows against float64, gated by a float32-sum host run's distance from it.
fn whole(a: std.mem.Allocator, label: []const u8, ours: []const u16, sum32: []const f32, exact: []const f32, width: usize) !bool {
    const noise = check.match(try check.bf16s(a, sum32), exact, width);
    const m = check.match(ours, exact, width);
    const pass = check.rowsPass(m, noise);
    check.report(label, m, pass);
    std.debug.print("  the float32-sum host run: cosine {d:.7}, relative error {e:.2}, {d:.4} bit-equal\n", .{ noise.cosine, noise.relative, noise.equal });
    return pass;
}

/// The selftest's tokenizer: one id per byte, below the vision marks.
const Bytes = struct {
    pub fn encode(_: Bytes, a: std.mem.Allocator, text: []const u8) ![]u32 {
        const out = try a.alloc(u32, text.len);
        for (out, text) |*o, c| o.* = c % 80;
        return out;
    }
};

/// An edit's presentation: a label, a 4 x 6 patch grid's 6 pads with vision and DeepStack rows, at M-RoPE positions.
fn withPicture(runner: *hp.Runner, w: *const hp.weights.Text, a: std.mem.Allocator, rand: std.Random, host: ref.Text) !bool {
    const g = w.g;
    const H = g.hidden;
    const marks: rules.Marks = .{ .start = 90, .image = 91, .end = 92 };
    const p = try rules.present(a, Bytes{}, marks, &.{6}, "a picture's prompt, long enough to pass sixty rows.");
    const pos = try rules.positions(a, p, marks, &.{.{ 4, 6 }});
    const pads = try rules.pads(a, p, marks, &.{6});
    const T = p.ids.len;
    var blocks: [4][]f32 = undefined;
    var bits: [4][]u16 = undefined;
    for (&blocks, &bits) |*v, *h| {
        v.* = try a.alloc(f32, pads.len * H);
        h.* = try a.alloc(u16, pads.len * H);
        for (v.*, h.*) |*x, *b| {
            b.* = bf16_round.bits(2 * rand.float(f32) - 1);
            x.* = bf16_round.value(b.*);
        }
    }
    const enc = try a.alloc(u16, T * H);
    try runner.encode(w, p.ids, pos, .{ .pads = pads, .rows = bits[0], .deepstack = bits[1..4].* }, enc);
    const cos = try a.alloc(f32, T * 64);
    const sin = try a.alloc(f32, T * 64);
    rules.rotary(cos, sin, pos, g.theta, g.sections);
    const vision: ref.Vision = .{ .pads = pads, .rows = blocks[0], .deepstack = blocks[1..4].* };
    const exact = try a.alloc(f32, T * H);
    try ref.encode(f64, a, g, host, p.ids, cos, sin, vision, exact);
    const sum32 = try a.alloc(f32, T * H);
    try ref.encode(f32, a, g, host, p.ids, cos, sin, vision, sum32);
    return whole(a, "text encoder rows, a reference's vision and DeepStack rows at M-RoPE positions", enc, sum32, exact, H);
}

/// `TextNames`' rule, for the read-ahead fixture: `prefix`'s embedding and its layers below `n`.
const wanted = struct {
    prefix: []const u8,
    layers: usize,

    pub fn accepts(w: @This(), name: []const u8) bool {
        if (!std.mem.startsWith(u8, name, w.prefix)) return false;
        const rest = name[w.prefix.len..];
        if (std.mem.eql(u8, rest, "embed_tokens.weight")) return true;
        if (!std.mem.startsWith(u8, rest, "layers.")) return false;
        const dot = std.mem.indexOfScalar(u8, rest["layers.".len..], '.') orelse return false;
        const i = std.fmt.parseInt(usize, rest["layers.".len..][0..dot], 10) catch return false;
        return i < w.layers;
    }
};

test "h3 prompt weights: only the accepted tensors' bytes are read ahead, prefetch on and off" {
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // one shard with accepted tensors beside a vision tensor and an unused layer's, one rejected shard
    const header =
        \\{"model.embed_tokens.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]}
        \\,"model.layers.0.input_layernorm.weight":{"dtype":"BF16","shape":[4],"data_offsets":[16,24]}
        \\,"model.layers.0.self_attn.q_proj.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[24,40]}
        \\,"model.layers.0.mlp.gate_proj.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[40,56]}
        \\,"model.vision_tower.patch_embed.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[56,72]}
        \\,"model.layers.10.self_attn.q_proj.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[72,88]}
    ;
    var json: std.ArrayList(u8) = .empty;
    try json.appendSlice(a, header);
    while ((8 + json.items.len + 1) % 8 != 0) try json.append(a, ' ');
    try json.append(a, '}');
    var file: std.ArrayList(u8) = .empty;
    try file.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u64, json.items.len)));
    try file.appendSlice(a, json.items);
    for (0..88) |i| try file.append(a, @truncate(i));
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "a.safetensors", .data = file.items });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "b.safetensors", .data = file.items });
    const index = try std.fmt.allocPrint(a,
        \\{{"weight_map":{{"model.embed_tokens.weight":"a.safetensors","model.layers.0.input_layernorm.weight":"a.safetensors","model.layers.0.self_attn.q_proj.weight":"a.safetensors","model.layers.0.mlp.gate_proj.weight":"a.safetensors","model.vision_tower.patch_embed.weight":"a.safetensors","model.layers.10.self_attn.q_proj.weight":"a.safetensors","model.vision_tower.blocks.0.weight":"b.safetensors","model.layers.10.mlp.gate_proj.weight":"b.safetensors"}}}}
    , .{});
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model.safetensors.index.json", .data = index });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var shards = try weights.open(testing.allocator, testing.io, try mtl.Device.init(), dir, wanted{ .prefix = "model.", .layers = 2 });
    defer shards.deinit(testing.allocator);
    const readers = try @import("core").checkpoint_read_ahead.readers();
    try testing.expectEqual(if (readers == 0) @as(u64, 0) else 56, shards.ahead);
    try testing.expectEqual(@as(usize, 1), shards.files.len);
    try testing.expectEqualSlices(u8, file.items[8 + json.items.len ..][0..16], weights.Shards.bytes(try shards.take("model.embed_tokens.weight", .{ 2, 4 }), 16));
}

test "h3 prompt weights: an extra refiner block's bytes are not read ahead, prefetch on and off" {
    const testing = std.testing;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // shard a holds the front tensors and blocks 0-1 beside an extra block 2's tensor; shard b only block 2's
    const header =
        \\{"condition_proj.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]}
        \\,"condition_proj.bias":{"dtype":"BF16","shape":[2],"data_offsets":[16,20]}
        \\,"token_refiner.final_norm.weight":{"dtype":"BF16","shape":[4],"data_offsets":[24,32]}
        \\,"token_refiner.blocks.0.norm1.weight":{"dtype":"BF16","shape":[4],"data_offsets":[32,40]}
        \\,"token_refiner.blocks.1.attn.qkv_proj.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[40,56]}
        \\,"token_refiner.blocks.2.norm1.weight":{"dtype":"BF16","shape":[4],"data_offsets":[56,64]}
    ;
    var json: std.ArrayList(u8) = .empty;
    try json.appendSlice(a, header);
    while ((8 + json.items.len + 1) % 8 != 0) try json.append(a, ' ');
    try json.append(a, '}');
    var file: std.ArrayList(u8) = .empty;
    try file.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u64, json.items.len)));
    try file.appendSlice(a, json.items);
    for (0..64) |i| try file.append(a, @truncate(i));
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "a.safetensors", .data = file.items });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "b.safetensors", .data = file.items });
    const index = try std.fmt.allocPrint(a,
        \\{{"weight_map":{{"condition_proj.weight":"a.safetensors","condition_proj.bias":"a.safetensors","token_refiner.final_norm.weight":"a.safetensors","token_refiner.blocks.0.norm1.weight":"a.safetensors","token_refiner.blocks.1.attn.qkv_proj.weight":"a.safetensors","token_refiner.blocks.2.norm1.weight":"a.safetensors","token_refiner.blocks.2.attn.out_proj.weight":"b.safetensors"}}}}
    , .{});
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model.safetensors.index.json", .data = index });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    var shards = try weights.open(testing.allocator, testing.io, try mtl.Device.init(), dir, weights.FrontNames{ .blocks = 2 });
    defer shards.deinit(testing.allocator);
    const readers = try @import("core").checkpoint_read_ahead.readers();
    try testing.expectEqual(if (readers == 0) @as(u64, 0) else 52, shards.ahead);
    try testing.expectEqual(@as(usize, 1), shards.files.len);
    try testing.expectEqualSlices(u8, file.items[8 + json.items.len ..][40..56], weights.Shards.bytes(try shards.take("token_refiner.blocks.1.attn.qkv_proj.weight", .{ 2, 4 }), 16));
}
