//! tf-h3-image-prompt-host: H3-Image's prompt stage on the CPU, the tf-h3-image-prompt case on the host reference.
const std = @import("std");
const hp = @import("h3_prompt");
const ck = @import("h3_checkpoint");
const tokenizer = @import("tokenizer");
const noise = @import("moss_noise");
const bf16_round = @import("bf16_round");

const rules = hp.rules;
const picture = hp.picture;
const ref = hp.reference;
const stage = hp.case;

const usage =
    \\usage: tf-h3-image-prompt-host TEXT_ENCODER_DIR MODEL_DIR prompt.txt out.safetensors [WIDTHxHEIGHT [SEED [MIN_AREA]]]
    \\  TEXT_ENCODER_DIR  MiniMax-H3's text_encoder/ (Qwen3-VL): config.json, its index and shards
    \\  MODEL_DIR         the MiniMax-H3-Image folder: model_index.json, tokenizer/, transformer/
    \\  prompt.txt        the prompt, verbatim but for one trailing line ending
    \\  out               text BF16 [rows, 5376] and text_tags I32 [rows], as tf-h3-image's picture case reads them,
    \\                    and ids I32 [rows]
    \\  WIDTHxHEIGHT      a text-to-image case: the canvas (grown to MIN_AREA, default 2048 x 2048, at the size's
    \\                    ratio, sides snapped to 32), its starting rows from SEED (default 0) as torch's CPU randn
    \\                    draws them, and the requested size
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5 or args.len > 8) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const request: ?stage.Request = if (args.len >= 6) .{
        .size = try stage.size(args[5]),
        .seed = if (args.len >= 7) try std.fmt.parseInt(u64, args[6], 10) else 0,
        .min_area = if (args.len == 8) try std.fmt.parseInt(u64, args[7], 10) else picture.default_min_area,
    } else null;
    const started = seconds();

    // the folder and its tokenizer, checked as the Metal prompt path checks them
    const io = init.io;
    const enc = try rules.checkEncoder(a, try read(io, a, &.{ args[1], "config.json" }));
    _ = try picture.checkIndex(a, try read(io, a, &.{ args[2], "model_index.json" }));
    const transformer = try std.fs.path.join(a, &.{ args[2], "transformer" });
    const front = try rules.checkRefiner(a, try read(io, a, &.{ transformer, "config.json" }));
    var tok = try tokenizer.loadTokenizer(io, a, try std.fs.path.join(a, &.{ args[2], "tokenizer" }));
    defer tok.deinit();
    var prompt: []const u8 = try std.Io.Dir.cwd().readFileAlloc(io, args[3], a, .limited(16 << 20));
    if (std.mem.endsWith(u8, prompt, "\n")) prompt = prompt[0 .. prompt.len - 1];
    if (std.mem.endsWith(u8, prompt, "\r")) prompt = prompt[0 .. prompt.len - 1];

    const p = try rules.present(a, &tok, enc.marks, &.{}, prompt);
    const pos = try rules.positions(a, p, enc.marks, &.{});
    const T = p.ids.len;
    const cos = try a.alloc(f32, T * 64);
    const sin = try a.alloc(f32, T * 64);
    rules.rotary(cos, sin, pos, enc.text.theta, enc.text.sections);

    // the text model, one layer's weights in memory at a time: its layers do not fit beside the transformer's
    var shards = try Shards.open(io, a, gpa, args[1]);
    defer shards.deinit(gpa);
    const prefix = try rules.textPrefix(shards.where.keys());
    const x = try a.alloc(f32, T * enc.text.hidden);
    try shards.embed(enc.text, prefix, p.ids, x);
    const scratch = try ref.Scratch.init(a, enc.text, T);
    var held = std.heap.ArenaAllocator.init(gpa);
    defer held.deinit();
    for (0..enc.text.layers) |li| {
        const layer = try shards.layer(held.allocator(), enc.text, prefix, li);
        try ref.encodeLayer(f32, gpa, enc.text, layer, li, x, cos, sin, null, scratch);
        _ = held.reset(.free_all);
    }

    // condition_proj and the two-block refiner, their weights together
    var front_shards = try Shards.open(io, a, gpa, transformer);
    defer front_shards.deinit(gpa);
    const text = try a.alloc(u16, T * front.hidden);
    {
        const w = try front_shards.refiner(held.allocator(), front);
        const rows = try a.alloc(f32, T * front.hidden);
        try ref.refine(f32, gpa, front, w, x, rows);
        for (text, rows) |*o, v| o.* = bf16_round.bits(v); // every value is bf16-exact, so this only re-codes it
    }
    _ = held.reset(.free_all);

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[4], .data = try stage.write(a, front.hidden, p, text, request, &noise.randn, &.{}) });
    std.debug.print("wrote {s}: text [{d}, {d}] bf16, text_tags and ids\n", .{ args[4], T, front.hidden });
    std.debug.print("prompt: {d:.2} s\n", .{seconds() - started});
    return 0;
}

fn read(io: std.Io, a: std.mem.Allocator, parts: []const []const u8) ![]u8 {
    const path = try std.fs.path.join(a, parts);
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch |err| {
        std.debug.print("cannot read {s}: {s}\n", .{ path, @errorName(err) });
        return err;
    };
}

/// Wall seconds on the monotonic clock: mtl.clock.seconds without Metal.
fn seconds() f64 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t);
    return @as(f64, @floatFromInt(t.sec)) + @as(f64, @floatFromInt(t.nsec)) / 1e9;
}

/// One bf16 pair as the fp32 value it holds exactly.
fn value(pair: []const u8) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, pair[0..2], .little)) << 16);
}

/// A host checkpoint folder: the index's name-to-shard map, shards mapped on first use; callers free what they convert.
const Shards = struct {
    gpa: std.mem.Allocator,
    /// Tensor name to shard file, both in the caller's arena.
    where: std.StringArrayHashMapUnmanaged([]const u8) = .empty,
    folder: ck.Folder,

    fn open(io: std.Io, a: std.mem.Allocator, gpa: std.mem.Allocator, dir: []const u8) !Shards {
        const text = try read(io, a, &.{ dir, "model.safetensors.index.json" });
        const doc = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return error.H3BadShardIndex;
        if (doc != .object) return error.H3BadShardIndex;
        const map = doc.object.get("weight_map") orelse return error.H3BadShardIndex;
        if (map != .object or map.object.count() == 0) return error.H3BadShardIndex;
        var s: Shards = .{ .gpa = gpa, .folder = .{ .dir = try a.dupe(u8, dir) } };
        var it = map.object.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .string or std.mem.indexOf(u8, e.value_ptr.string, "/") != null) return error.H3BadShardIndex;
            try s.where.put(a, e.key_ptr.*, e.value_ptr.string);
        }
        return s;
    }

    fn deinit(s: *Shards, gpa: std.mem.Allocator) void {
        s.folder.deinit(gpa);
    }

    /// The shard holding `name`.
    fn shard(s: *Shards, name: []const u8) !*ck.Checkpoint {
        const file = s.where.get(name) orelse {
            std.debug.print("no tensor {s}\n", .{name});
            return error.MissingTensor;
        };
        return s.folder.shard(s.gpa, file);
    }

    /// The bf16 tensor `name` at `shape` (columns 0: a vector of `rows` values).
    fn tensor(s: *Shards, name: []const u8, shape: rules.Shape) !ck.Checkpoint.Entry {
        const e = try (try s.shard(name)).bf16(name);
        const rows: usize = if (shape[1] == 0) 1 else shape[0];
        const cols: usize = if (shape[1] == 0) shape[0] else shape[1];
        if (e.rows != rows or e.cols != cols) {
            std.debug.print("{s} is [{d}, {d}]; the prompt path reads [{d}, {d}]\n", .{ name, e.rows, e.cols, rows, cols });
            return error.H3UnexpectedTensor;
        }
        return e;
    }

    /// The bf16 tensor `name` at `shape` as fp32 values, exactly bf16's.
    fn floats(s: *Shards, a: std.mem.Allocator, name: []const u8, shape: rules.Shape) ![]f32 {
        const e = try s.tensor(name, shape);
        const out = try a.alloc(f32, e.rows * e.cols);
        for (out, 0..) |*o, i| o.* = value(e.bytes[2 * i ..]);
        return out;
    }

    /// The embedding rows of `ids`, converted from the mapped bf16 table: the table is never held in fp32.
    fn embed(s: *Shards, g: rules.Text, prefix: []const u8, ids: []const u32, x: []f32) !void {
        var name: [160]u8 = undefined;
        const table = try s.tensor(try std.fmt.bufPrint(&name, "{s}embed_tokens.weight", .{prefix}), .{ g.vocab, g.hidden });
        for (ids, 0..) |id, r| {
            if (id >= g.vocab) return error.H3TokenOutOfRange;
            const row = table.bytes[@as(usize, id) * g.hidden * 2 ..][0 .. g.hidden * 2];
            for (x[r * g.hidden ..][0..g.hidden], 0..) |*o, i| o.* = value(row[2 * i ..]);
        }
    }

    /// Text layer `i`'s weights as fp32.
    fn layer(s: *Shards, a: std.mem.Allocator, g: rules.Text, prefix: []const u8, i: usize) !ref.Layer {
        var l: ref.Layer = undefined;
        var name: [176]u8 = undefined;
        inline for (comptime std.meta.fieldNames(ref.Layer), rules.layerTensors(g)) |field, spec| {
            @field(l, field) = try s.floats(a, try std.fmt.bufPrint(&name, "{s}layers.{d}.{s}", .{ prefix, i, spec[1] }), spec[2]);
        }
        return l;
    }

    /// The refiner's weights as fp32: condition_proj, its blocks and the final norm.
    fn refiner(s: *Shards, a: std.mem.Allocator, g: rules.Refiner) !ref.Refiner {
        var w: ref.Refiner = undefined;
        inline for (.{ "cond_w", "cond_b", "final_norm" }, rules.frontTensors(g)) |field, spec| {
            std.debug.assert(std.mem.eql(u8, field, spec[0]));
            @field(w, field) = try s.floats(a, spec[1], spec[2]);
        }
        const blocks = try a.alloc(ref.Block, g.blocks);
        for (blocks, 0..) |*b, i| {
            var name: [96]u8 = undefined;
            inline for (comptime std.meta.fieldNames(ref.Block), rules.blockTensors(g)) |field, spec| {
                @field(b, field) = try s.floats(a, try std.fmt.bufPrint(&name, "token_refiner.blocks.{d}.{s}", .{ i, spec[1] }), spec[2]);
            }
        }
        w.blocks = blocks;
        return w;
    }
};
