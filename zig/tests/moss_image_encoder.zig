//! tf-moss-image-encoder: MOSS-Image's prompt encoder on a case from tools/zig/moss_image_case.py `encode`, compared.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const mi = tf.moss_image;
const tokenizer = @import("tokenizer");

const usage =
    \\tf-moss-image-encoder MODEL_DIR CASE [PROMPT_FILE]
    \\MODEL_DIR is the OpenMOSS-Team/MOSS-Image-Base folder. CASE (moss_image_case.py encode) holds the prompt's token
    \\ids, every layer's rows and the final rows from the publisher's torch path in bf16 and in float32, and torch
    \\bf16's rows for each step alone on the float32 run's input. The prompt (PROMPT_FILE, else the empty negative
    \\prompt) is tokenized and checked against the ids. Each layer and the final norm, run alone on the same input, must
    \\be as close to float32 as twice torch bf16's (or bf16 rounding's) cosine error and worst row there. The whole
    \\prompt is then encoded twice, compared with both torch runs, and must repeat byte for byte; else it fails.
;

const ref = mi.reference;
const H = mi.encoder_config.hidden;
const L = mi.encoder_config.layers;

/// One comparison's line: against float32 within its gate, beside torch bf16's own distance.
fn report(label: []const u8, bf: ref.Match, v: ref.Verdict) void {
    std.debug.print("{s}: float32 1-cosine {e:.2} of {e:.2}, worst row {d:.5} (row {d}) of {d:.5}, torch bf16 {d:.5} (row {d}): {s}\n", .{ label, v.cosine_err, v.cosine_limit, v.row, v.row_at, v.row_limit, bf.worst_row, bf.worst_at, if (v.pass) "pass" else "FAIL" });
}

/// Ours against float32 rows `fp32_name`, gated by torch bf16's rows `name` on the same input.
fn check(a: std.mem.Allocator, case: *const tf.safetensors.File, ours: []const u16, name: []const u8, fp32_name: []const u8) !struct { ref.Match, ref.Verdict } {
    const bf16 = try floats(a, case, name);
    const exact = try floats(a, case, fp32_name);
    const theirs = ref.match(f32, bf16, exact, H);
    return .{ theirs, ref.gate(ref.match(u16, ours, exact, H), theirs, ref.rounding(exact, H)) };
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

    const T = ids.len;
    const inputs = try a.alloc(u16, (L + 1) * T * H);
    for (0..L + 1) |i| {
        var name: [32]u8 = undefined;
        const rows = try floats(a, &case, if (i == 0) "embed" else try std.fmt.bufPrint(&name, "layer_fp32.{d}", .{i - 1}));
        for (inputs[i * T * H ..][0 .. T * H], rows) |*o, x| o.* = mi.host.bf16(x);
    }
    const alone = try a.alloc(u16, (L + 1) * T * H);
    try enc.steps(T, inputs, alone);
    var verdicts: [L + 1]ref.Verdict = undefined;
    for (0..L + 1) |i| {
        var name: [32]u8 = undefined;
        var fp32_name: [32]u8 = undefined;
        var label: [32]u8 = undefined;
        const last = i == L;
        const bf, const v = try check(a, &case, alone[i * T * H ..][0 .. T * H], if (last) "alone.final" else try std.fmt.bufPrint(&name, "alone.{d}", .{i}), if (last) "final_fp32" else try std.fmt.bufPrint(&fp32_name, "layer_fp32.{d}", .{i}));
        report(if (last) "final norm alone" else try std.fmt.bufPrint(&label, "layer {d:>2} alone ({s})", .{ i, if (mi.encoder_config.isFull(i)) "attention" else "delta    " }), bf, v);
        verdicts[i] = v;
    }

    const out = try a.alloc(u16, T * H);
    const kept = try a.alloc(u16, L * T * H);
    try enc.rows(ids, out, kept);
    var lowest: f64 = 1;
    for (0..L) |i| {
        var name: [32]u8 = undefined;
        lowest = @min(lowest, ref.match(u16, kept[i * T * H ..][0 .. T * H], try floats(a, &case, try std.fmt.bufPrint(&name, "layer.{d}", .{i})), H).cosine);
    }
    const bf, const v = try check(a, &case, out, "final", "final_fp32");
    std.debug.print("whole prompt, not gated: each layer against torch bf16's, cosine {d:.6} or more; the final rows against float32, 1-cosine {e:.2}, worst row {d:.5} (row {d}), torch bf16's {e:.2}, {d:.5} (row {d})\n", .{ lowest, v.cosine_err, v.row, v.row_at, 1 - bf.cosine, bf.worst_row, bf.worst_at });

    const again = try a.alloc(u16, T * H);
    const timing = mtl.clock.seconds();
    try enc.rows(ids, again, null);
    const encode_s = mtl.clock.seconds() - timing;
    const repeat = std.mem.eql(u16, out, again);
    std.debug.print("encoded again in {d:.3} s, the same bytes: {}\n", .{ encode_s, repeat });
    const pass = ref.accept(&verdicts, repeat);
    std.debug.print("{s}\n", .{if (pass) "PASS every step within its gate, repeated byte for byte" else "FAIL"});
    return if (pass) 0 else 1;
}
