//! tf-uocr-decode: a page through Unlimited-OCR-2 end to end against a reference dump: the prompt's ids, the
//! embeddings, every layer's prefill rows gated against float32, the first logits, then the greedy reply.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const uo = tf.uocr;
const cfg = uo.config;
const gate = uo.gate;
const H = cfg.hidden;

const usage =
    \\tf-uocr-decode MODEL_DIR PAGE[,PAGE...] DUMP_BF16 DUMP_F32 [--their-embeds] [--chunk ROWS]
    \\The dumps are ref_dump.py's for the pages (DUMP.json beside each .safetensors holds the prompt ids and the greedy
    \\reply). The page is preprocessed and encoded here (gundam when the dump encoded tiles), the prompt must match
    \\the dump's ids, every layer's prefill rows must be as close to float32 as twice torch bf16's distance, and the
    \\reply is generated greedily for as many tokens as the bf16 reference's, reporting where it first differs from
    \\each reference. --their-embeds feeds the bf16 dump's embeddings to the decoder in place of this encoder's, so
    \\the layers and the reply measure the decoder alone (each layer is then also compared with torch bf16's).
    \\--chunk ROWS prefills again in chunks of ROWS rows and requires the first logits and the whole reply to equal
    \\the one-pass prefill's bit for bit.
;

fn floats(a: std.mem.Allocator, f: *const tf.safetensors.File, name: []const u8) ![]f32 {
    const t = f.get(name) orelse return error.MissingTensor;
    const out = try a.alloc(f32, t.numel());
    switch (t.dtype) {
        .f32 => @memcpy(std.mem.sliceAsBytes(out), t.bytes),
        .bf16 => for (out, std.mem.bytesAsSlice(u16, t.bytes)) |*o, h| {
            o.* = uo.resize.widen(h);
        },
        else => return error.BadDumpTensor,
    }
    return out;
}

fn widen(a: std.mem.Allocator, h: []const u16) ![]f32 {
    const out = try a.alloc(f32, h.len);
    for (out, h) |*o, x| o.* = uo.resize.widen(x);
    return out;
}

const Ref = struct { prompt: []u32, gen: []u32 };

fn reference(a: std.mem.Allocator, io: std.Io, safetensors_path: []const u8) !Ref {
    const json_path = try std.fmt.allocPrint(a, "{s}.json", .{safetensors_path[0 .. safetensors_path.len - ".safetensors".len]});
    const text = try std.Io.Dir.cwd().readFileAlloc(io, json_path, a, .limited(64 << 20));
    const v = try std.json.parseFromSliceLeaky(struct { prompt: []u32, gen: []u32 }, a, text, .{ .ignore_unknown_fields = true });
    return .{ .prompt = v.prompt, .gen = v.gen };
}

fn firstDiff(a: []const u32, b: []const u32) ?usize {
    for (0..@min(a.len, b.len)) |i| if (a[i] != b[i]) return i;
    return if (a.len == b.len) null else @min(a.len, b.len);
}

fn noop(_: void, _: u32) void {}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var their = false;
    var chunk: ?usize = null;
    var ai: usize = 5;
    while (ai < args.len) : (ai += 1) {
        if (std.mem.eql(u8, args[ai], "--their-embeds")) {
            their = true;
        } else if (std.mem.eql(u8, args[ai], "--chunk") and ai + 1 < args.len) {
            ai += 1;
            chunk = std.fmt.parseInt(usize, args[ai], 10) catch null;
            if (chunk == null or chunk.? == 0) break;
        } else break;
    }
    if (args.len < 5 or ai < args.len) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    var bf = try tf.safetensors.File.open(a, io, args[3]);
    var ex = try tf.safetensors.File.open(a, io, args[4]);
    const ref_bf = try reference(a, io, args[3]);
    const ref_ex = try reference(a, io, args[4]);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    if (!device.tensorUnits()) {
        std.debug.print("{s} has no tensor units; this family runs on the M5's alone\n", .{device.name()});
        return 1;
    }
    const queue = try device.queue();
    defer queue.deinit();
    const gpa = std.heap.smp_allocator;
    var w = try uo.weights.load(gpa, io, device, args[1]);
    defer w.deinit();
    var k = try uo.kernels.Kernels.load(gpa, device);
    defer k.deinit();
    var enc = try uo.vision.Encoder.init(gpa, device, queue, &k, &w);
    defer enc.deinit();

    // each page's views, encoded, laid out as its image tokens
    var paths = std.mem.splitScalar(u8, args[2], ',');
    var page_tokens: std.ArrayList(usize) = .empty;
    var all_rows: std.ArrayList(u16) = .empty;
    const gundam = bf.get("pixels1") != null and std.mem.indexOfScalar(u8, args[2], ',') == null;
    var t_enc: f64 = 0;
    while (paths.next()) |path| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
        var page = try uo.preprocess.decode(io, a, bytes);
        const views = try uo.preprocess.views(a, page.rgb, page.width, page.height, gundam);
        const t0 = mtl.clock.seconds();
        const g = cfg.side(cfg.base_size);
        const t = cfg.side(cfg.tile_size);
        const global = try a.alloc(u16, g * g * H);
        try enc.encode(cfg.base_size, views.global, global, .{});
        const tiles = try a.alloc(u16, views.tileCount() * t * t * H);
        for (0..views.tileCount()) |i| try enc.encode(cfg.tile_size, views.tile(i), tiles[i * t * t * H ..][0 .. t * t * H], .{});
        const page_rows = try all_rows.addManyAsSlice(a, views.tokens() * H);
        uo.vision.layout(&w, global, tiles, views.across, views.down, page_rows);
        t_enc += mtl.clock.seconds() - t0;
        try page_tokens.append(a, views.tokens());
        std.debug.print("{s}: {d} x {d}, {s} {d} x {d}\n", .{ path, page.width, page.height, if (gundam) "gundam" else "base", views.across, views.down });
        page.deinit(a);
    }
    const rows = all_rows.items;
    var p = try uo.prompt.build(a, page_tokens.items, &cfg.instruction);
    const ids_same = std.mem.eql(u32, p.ids, ref_bf.prompt);
    std.debug.print("prompt {d} tokens, ids {s}; encoded in {d:.2} s\n", .{ p.len(), if (ids_same) "match" else "DIFFER", t_enc });
    if (!ids_same) return 1;

    const n = p.len();
    const max_new = ref_bf.gen.len + 8;
    var dec = try uo.decoder.Decoder.init(gpa, device, queue, &k, &w, n + max_new + 1, n);
    defer dec.deinit();
    const trace = try a.alloc([]u16, cfg.layers);
    for (trace) |*s| s.* = try a.alloc(u16, n * H);
    dec.trace = trace;
    var parts: [2][][]u16 = undefined;
    for (&parts) |*pt| {
        pt.* = try a.alloc([]u16, cfg.layers);
        for (pt.*) |*sl| sl.* = try a.alloc(u16, n * H);
    }
    dec.trace_parts = parts;

    // the spliced embeddings: text rows exact, image rows gated
    {
        const theirs = try floats(a, &bf, "embeds");
        const exact = try floats(a, &ex, "embeds");
        var text_differ: usize = 0;
        var img: usize = 0;
        const ours = try a.alloc(f32, n * H);
        for (p.ids, 0..) |id, i| {
            for (0..H) |c| ours[i * H + c] = if (id == cfg.image_token) uo.resize.widen(rows[(img * H) + c]) else uo.resize.widen(@as([*]const u16, @ptrCast(@alignCast(w.embed.addr())))[id * H + c]);
            if (id == cfg.image_token) {
                img += 1;
            } else for (0..H) |c| {
                text_differ += @intFromBool(ours[i * H + c] != theirs[i * H + c]);
            }
        }
        const v = gate.verdict(gate.match(ours, exact, H), gate.match(theirs, exact, H), try gate.rounding(a, exact, H));
        std.debug.print("embeds: text values differing {d}; 1-cos {e:.2} of {e:.2}, worst row {d:.4} of {d:.4}: {s}\n", .{ text_differ, v.cosine_err, v.cosine_limit, v.row, v.row_limit, if (v.pass and text_differ == 0) "pass" else "FAIL" });
    }

    if (their) {
        const emb = bf.get("embeds").?;
        const all: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, emb.bytes));
        var img: usize = 0;
        for (p.ids, 0..) |id, i| if (id == cfg.image_token) {
            @memcpy(rows[img * H ..][0..H], all[i * H ..][0..H]);
            img += 1;
        };
    }
    const t1 = mtl.clock.seconds();
    try dec.prefill(p, rows);
    const t_pre = mtl.clock.seconds() - t1;
    dec.trace = null;
    dec.trace_parts = null;
    var all_pass = true;
    for (0..cfg.layers) |l| {
        const name = try std.fmt.allocPrint(a, "layer{d}", .{l});
        const theirs = try floats(a, &bf, name);
        const exact = try floats(a, &ex, name);
        const v = gate.verdict(gate.match(try widen(a, trace[l]), exact, H), gate.match(theirs, exact, H), try gate.rounding(a, exact, H));
        std.debug.print("  layer {d:>2}: 1-cos {e:.2} of {e:.2}, worst row {d:.4} (row {d}) of {d:.4}: {s}\n", .{ l, v.cosine_err, v.cosine_limit, v.row, v.row_at, v.row_limit, if (v.pass) "pass" else "FAIL" });
        if (their) {
            const m = gate.match(try widen(a, trace[l]), theirs, H);
            std.debug.print("      against torch bf16: 1-cos {e:.2}, worst row {d:.4} (row {d})\n", .{ 1 - m.cosine, m.worst_row, m.worst_at });
            for ([_][]const u8{ "attn", "mlp" }, 0..) |part, pi| {
                const pt = try floats(a, &bf, try std.fmt.allocPrint(a, "{s}{d}", .{ part, l }));
                const pm = gate.match(try widen(a, parts[pi][l]), pt, H);
                std.debug.print("        {s}: 1-cos {e:.2}, worst row {d:.4} (row {d})\n", .{ part, 1 - pm.cosine, pm.worst_row, pm.worst_at });
            }
        }
        all_pass = all_pass and v.pass;
    }
    std.debug.print("prefill {d} tokens in {d:.3} s\n", .{ n, t_pre });

    // the first logits, then the reply
    {
        const lg = try dec.logits();
        const theirs = (try floats(a, &bf, "logits"))[0..cfg.vocab];
        const exact = (try floats(a, &ex, "logits"))[0..cfg.vocab];
        const v = gate.verdict(gate.match(try widen(a, lg), exact, cfg.vocab), gate.match(theirs, exact, cfg.vocab), try gate.rounding(a, exact, cfg.vocab));
        std.debug.print("first logits: rel err {d:.4} of {d:.4}: {s}\n", .{ v.row, v.row_limit, if (v.pass) "pass" else "FAIL" });
        all_pass = all_pass and v.pass;
    }
    const first_logits = try a.dupe(u16, try dec.logits());
    var gen: std.ArrayList(u32) = .empty;
    const t2 = mtl.clock.seconds();
    try uo.generate.greedy(&dec, p, ref_bf.gen.len, &gen, a, {}, noop);
    const t_gen = mtl.clock.seconds() - t2;
    std.debug.print("reply: {d} tokens in {d:.2} s ({d:.1} tok/s)\n", .{ gen.items.len, t_gen, @as(f64, @floatFromInt(gen.items.len)) / t_gen });
    for ([_]struct { []const u8, []const u32 }{ .{ "torch bf16", ref_bf.gen }, .{ "torch fp32", ref_ex.gen } }) |r| {
        if (firstDiff(gen.items, r[1][0..@min(r[1].len, gen.items.len)])) |at|
            std.debug.print("  vs {s} ({d} tokens): first differs at token {d}\n", .{ r[0], r[1].len, at })
        else
            std.debug.print("  vs {s} ({d} tokens): identical over {d}\n", .{ r[0], r[1].len, gen.items.len });
    }
    const bf_fp = firstDiff(ref_bf.gen, ref_ex.gen[0..@min(ref_ex.gen.len, ref_bf.gen.len)]);
    std.debug.print("  (torch bf16 vs fp32: first differs at {?d})\n", .{bf_fp});
    std.debug.print("ids {any}\n", .{gen.items});
    if (chunk) |c| {
        dec.deinit();
        dec = try uo.decoder.Decoder.init(gpa, device, queue, &k, &w, n + max_new + 1, @min(c, n));
        const t3 = mtl.clock.seconds();
        try dec.prefill(p, rows);
        const lg = try dec.logits();
        const t_chunked = mtl.clock.seconds() - t3;
        const same_logits = std.mem.eql(u16, lg, first_logits);
        var again: std.ArrayList(u32) = .empty;
        try uo.generate.greedy(&dec, p, ref_bf.gen.len, &again, a, {}, noop);
        const same_reply = std.mem.eql(u32, again.items, gen.items);
        std.debug.print("chunks of {d} rows ({d:.3} s): first logits {s}, reply {s}\n", .{ c, t_chunked, if (same_logits) "bit-identical" else "DIFFER", if (same_reply) "identical" else "DIFFERS" });
        all_pass = all_pass and same_logits and same_reply;
    }
    std.debug.print("{s}\n", .{if (all_pass) "PASS" else "FAIL"});
    return if (all_pass) 0 else 1;
}
