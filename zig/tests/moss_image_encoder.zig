//! tf-moss-image-encoder: MOSS-Image's prompt encoder on a case from tools/zig/moss_image_case.py `encode`, compared.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const mi = tf.moss_image;
const tokenizer = @import("tokenizer");

const usage =
    \\tf-moss-image-encoder MODEL_DIR CASE [PROMPT_FILE]
    \\MODEL_DIR is the OpenMOSS-Team/MOSS-Image-Base folder. CASE (moss_image_case.py encode) holds the prompt's token
    \\ids, every layer's rows and the final rows from the publisher's torch path in bf16, and the final rows in float32.
    \\The prompt (PROMPT_FILE, else the empty negative prompt) is tokenized and checked against the ids, then encoded:
    \\each layer's rows and the final ones are compared (max |diff|, cosine, worst row) and the encoding is timed.
;

const Match = struct { max_abs: f64, cosine: f64, worst_row: f64 };

/// Largest absolute difference, cosine, and the largest relative L2 error of one row of `width`.
fn match(ours: []const u16, theirs: []const f32, width: usize) Match {
    var dot: f64 = 0;
    var aa: f64 = 0;
    var bb: f64 = 0;
    var most: f64 = 0;
    var worst: f64 = 0;
    var row: usize = 0;
    while (row < ours.len) : (row += width) {
        var rd: f64 = 0;
        var rb: f64 = 0;
        for (ours[row..][0..width], theirs[row..][0..width]) |h, b| {
            const a: f64 = @as(f32, @bitCast(@as(u32, h) << 16));
            const d = a - b;
            dot += a * b;
            aa += a * a;
            bb += @as(f64, b) * b;
            rd += d * d;
            rb += @as(f64, b) * b;
            most = @max(most, @abs(d));
        }
        worst = @max(worst, @sqrt(rd / @max(rb, 1e-30)));
    }
    return .{ .max_abs = most, .cosine = dot / @sqrt(aa * bb), .worst_row = worst };
}

/// The case tensor `name` as f32 (from bf16 or f32).
fn floats(a: std.mem.Allocator, f: *const tf.safetensors.File, name: []const u8) ![]f32 {
    const t = f.get(name) orelse {
        std.debug.print("the case has no {s}\n", .{name});
        return error.MissingTensor;
    };
    const out = try a.alloc(f32, t.numel());
    switch (t.dtype) {
        .f32 => @memcpy(std.mem.sliceAsBytes(out), t.bytes),
        .bf16 => for (out, std.mem.bytesAsSlice(u16, t.bytes)) |*o, h| {
            o.* = @bitCast(@as(u32, h) << 16);
        },
        else => return error.BadCaseTensor,
    }
    return out;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3 and args.len != 4) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    var case = try tf.safetensors.File.open(a, io, args[2]);
    defer case.close(io);
    const prompt = if (args.len == 4) std.mem.trim(u8, try std.Io.Dir.cwd().readFileAlloc(io, args[3], a, .limited(1 << 20)), " \t\r\n") else "";
    const want_ids = case.get("ids") orelse return error.MissingTensor;
    if (want_ids.dtype != .i32) return error.BadCaseTensor;

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const loading = mtl.clock.seconds();
    const enc = try mi.Encoder.load(std.heap.page_allocator, io, device, args[1]);
    defer enc.deinit();
    const load_s = mtl.clock.seconds() - loading;
    var tok = try tokenizer.loadTokenizer(io, a, try std.fs.path.join(a, &.{ args[1], "tokenizer", "tokenizer.json" }));
    defer tok.deinit();
    const ids = try mi.encoder.ids(&tok, a, prompt);
    const same_ids = std.mem.eql(u8, std.mem.sliceAsBytes(ids), want_ids.bytes);
    std.debug.print("{s}: {d} tokens, the same ids as the tokenizer's: {}; loaded in {d:.1} s\n", .{ device.name(), ids.len, same_ids, load_s });
    if (!same_ids) return 1;

    const H = mi.encoder_config.hidden;
    const L = mi.encoder_config.layers;
    const T = ids.len;
    const out = try a.alloc(u16, T * H);
    const kept = try a.alloc(u16, L * T * H);
    try enc.rows(ids, out, kept);
    var worst: f64 = 1;
    for (0..L) |i| {
        var name: [32]u8 = undefined;
        const ref = try floats(a, &case, try std.fmt.bufPrint(&name, "layer.{d}", .{i}));
        const m = match(kept[i * T * H ..][0 .. T * H], ref, H);
        worst = @min(worst, m.cosine);
        std.debug.print("layer {d:>2} ({s}): max |diff| {d:.4}, cosine {d:.6}, worst row {d:.5}\n", .{ i, if (mi.encoder_config.isFull(i)) "attention" else "delta    ", m.max_abs, m.cosine, m.worst_row });
    }
    const bf = match(out, try floats(a, &case, "final"), H);
    const fp = match(out, try floats(a, &case, "final_fp32"), H);
    std.debug.print("final rows against torch bf16: max |diff| {d:.4}, cosine {d:.6}, worst row {d:.5}\n", .{ bf.max_abs, bf.cosine, bf.worst_row });
    std.debug.print("final rows against float32:    max |diff| {d:.4}, cosine {d:.6}, worst row {d:.5}\n", .{ fp.max_abs, fp.cosine, fp.worst_row });

    const again = try a.alloc(u16, T * H);
    const timing = mtl.clock.seconds();
    try enc.rows(ids, again, null);
    const encode_s = mtl.clock.seconds() - timing;
    const repeat = std.mem.eql(u16, out, again);
    std.debug.print("encoded again in {d:.3} s, the same bytes: {}\n", .{ encode_s, repeat });
    return if (repeat) 0 else 1;
}
