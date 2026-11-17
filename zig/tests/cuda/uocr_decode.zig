//! tf-cuda-test uocr-decode: tf-uocr-decode's reply checks on CUDA, against the same reference dumps and tie gate.
const std = @import("std");
const core = @import("core");
const check = @import("check.zig");
const uo = @import("uocr_cuda");
const cfg = uo.config;
const gate = uo.gate;
const pages = uo.preprocess.With(core.picture);
const H = cfg.hidden;

const Ref = struct { prompt: []u32, gen: []u32, top2: []gate.Top2 };

fn reference(a: std.mem.Allocator, io: std.Io, safetensors_path: []const u8) !Ref {
    if (!std.mem.endsWith(u8, safetensors_path, ".safetensors")) return error.UocrDumpName;
    const json_path = try std.fmt.allocPrint(a, "{s}.json", .{safetensors_path[0 .. safetensors_path.len - ".safetensors".len]});
    const text = try std.Io.Dir.cwd().readFileAlloc(io, json_path, a, .limited(64 << 20));
    const v = try std.json.parseFromSliceLeaky(struct { prompt: []u32, gen: []u32, top2: []gate.Top2 = &.{} }, a, text, .{ .ignore_unknown_fields = true });
    return .{ .prompt = v.prompt, .gen = v.gen, .top2 = v.top2 };
}

fn floats(a: std.mem.Allocator, f: *const core.safetensors.File, name: []const u8) ![]f32 {
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

fn firstDiff(a: []const u32, b: []const u32) ?usize {
    for (0..@min(a.len, b.len)) |i| if (a[i] != b[i]) return i;
    return if (a.len == b.len) null else @min(a.len, b.len);
}

fn seconds(gpu: check.Gpu, since: i96) f64 {
    return @as(f64, @floatFromInt(check.now(gpu.io) - since)) / 1e9;
}

/// The served path's greedy reply: the guard's forced token, else the GPU's argmax; up to `max_new` or EOS.
fn greedy(dec: *uo.decoder.Decoder, p: uo.prompt.Prompt, max_new: usize, out: *std.ArrayList(u32), a: std.mem.Allocator) !void {
    var guard = uo.guard.Guard.init(p);
    var st = uo.prompt.Step.init(p);
    while (out.items.len < max_new) {
        const id = guard.forcedToken() orelse try dec.argmax(guard.barsEos());
        if (id == cfg.eos) break;
        try out.append(a, id);
        guard.take(id);
        if (out.items.len == max_new) break;
        try dec.step(id, st.next(id));
    }
}

/// MODEL PAGE[,PAGE...] DUMP_BF16 DUMP_F32 [--reply-reported] [--chunk ROWS[,ROWS...]], as tf-uocr-decode reads them.
pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const io = gpu.io;
    if (rest.len < 4) return error.UocrDecodeArguments;
    var reply_reported = false;
    var chunks: std.ArrayList(usize) = .empty;
    var ai: usize = 4;
    while (ai < rest.len) : (ai += 1) {
        if (std.mem.eql(u8, rest[ai], "--reply-reported")) {
            reply_reported = true;
        } else if (std.mem.eql(u8, rest[ai], "--chunk") and ai + 1 < rest.len) {
            ai += 1;
            var it = std.mem.splitScalar(u8, rest[ai], ',');
            while (it.next()) |c| {
                const cut = try std.fmt.parseInt(usize, c, 10);
                if (cut == 0) return error.UocrDecodeArguments;
                try chunks.append(a, cut);
            }
        } else return error.UocrDecodeArguments;
    }
    var bf = try core.safetensors.File.open(a, io, rest[2]);
    var ex = try core.safetensors.File.open(a, io, rest[3]);
    const ref_bf = try reference(a, io, rest[2]);
    const ref_ex = try reference(a, io, rest[3]);
    try check.expect(ref_bf.top2.len == ref_bf.gen.len, "uocr-decode: {s}'s top2 has {d} steps for {d} tokens; rerun uocr_reference.py dump", .{ rest[2], ref_bf.top2.len, ref_bf.gen.len });

    // the pages' views on the host first, so the caches fit the prompt; gundam when the dump's pages hold tiles
    var images: usize = 0;
    var markers: usize = 0;
    for (ref_bf.prompt) |id| {
        images += @intFromBool(id == cfg.image_token);
        markers += @intFromBool(id == cfg.page_token);
    }
    const gundam = images > @max(markers, 1) * uo.serve.basePageTokens();
    var views: std.ArrayList(uo.preprocess.Views) = .empty;
    var page_tokens: std.ArrayList(usize) = .empty;
    var paths = std.mem.splitScalar(u8, rest[1], ',');
    while (paths.next()) |path| {
        var page = try pages.decode(io, a, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)));
        const v = try pages.views(a, page.rgb, page.width, page.height, gundam);
        std.debug.print("{s}: {d} x {d}, {s} {d} x {d}\n", .{ path, page.width, page.height, if (gundam) "gundam" else "base", v.across, v.down });
        page.deinit(a);
        try views.append(a, v);
        try page_tokens.append(a, v.tokens());
    }
    const p = try uo.prompt.build(a, page_tokens.items, &cfg.instruction);
    try check.expect(std.mem.eql(u32, p.ids, ref_bf.prompt), "uocr-decode: the prompt's {d} ids differ from the dump's {d}", .{ p.len(), ref_bf.prompt.len });
    const n = p.len();
    const cap = n + ref_bf.gen.len + 8;
    const m = try uo.serve.Model.load(gpu.gpa, io, gpu.d, rest[0], cap);
    defer m.deinit();

    const t0 = check.now(io);
    var rows: std.ArrayList(u16) = .empty;
    const g = cfg.side(cfg.base_size);
    const t = cfg.side(cfg.tile_size);
    for (views.items) |v| {
        const global = try a.alloc(u16, g * g * H);
        try m.encoder.encode(cfg.base_size, v.global, global);
        const tiles = try a.alloc(u16, v.tileCount() * t * t * H);
        for (0..v.tileCount()) |i| try m.encoder.encode(cfg.tile_size, v.tile(i), tiles[i * t * t * H ..][0 .. t * t * H]);
        uo.vision.layout(m.weights, global, tiles, v.across, v.down, try rows.addManyAsSlice(a, v.tokens() * H));
    }
    const t_enc = seconds(gpu, t0);
    const t1 = check.now(io);
    try m.decoder.prefill(p, rows.items);
    const first = try a.alloc(u16, cfg.vocab);
    try m.decoder.logits(first);
    std.debug.print("prompt {d} tokens, ids match; encoded in {d:.2} s, prefilled in {d:.3} s\n", .{ n, t_enc, seconds(gpu, t1) });
    // the first logits from our own image rows: reported against float32, as tf-uocr-decode reports them
    {
        const ours = try a.alloc(f32, cfg.vocab);
        for (ours, first) |*o, h| o.* = uo.resize.widen(h);
        const theirs = (try floats(a, &bf, "logits"))[0..cfg.vocab];
        const exact = (try floats(a, &ex, "logits"))[0..cfg.vocab];
        const v = gate.verdict(gate.match(ours, exact, cfg.vocab), gate.match(theirs, exact, cfg.vocab), try gate.rounding(a, exact, cfg.vocab));
        std.debug.print("first logits: rel err {d:.4} of {d:.4}: {s}\n", .{ v.row, v.row_limit, if (v.pass) "pass" else "over (reported)" });
    }

    var gen: std.ArrayList(u32) = .empty;
    const t2 = check.now(io);
    try greedy(&m.decoder, p, ref_bf.gen.len, &gen, a);
    const t_gen = seconds(gpu, t2);
    std.debug.print("reply: {d} tokens in {d:.2} s ({d:.1} tok/s)\n", .{ gen.items.len, t_gen, @as(f64, @floatFromInt(gen.items.len)) / t_gen });
    for ([_]struct { []const u8, []const u32 }{ .{ "torch bf16", ref_bf.gen }, .{ "torch fp32", ref_ex.gen } }) |r| {
        if (firstDiff(gen.items, r[1][0..@min(r[1].len, gen.items.len)])) |at|
            std.debug.print("  vs {s} ({d} tokens): first differs at token {d}\n", .{ r[0], r[1].len, at })
        else
            std.debug.print("  vs {s} ({d} tokens): identical over {d}\n", .{ r[0], r[1].len, gen.items.len });
    }

    // torch bf16's reply fed back: wherever our pick differs, the same two contenders, both margins within a bf16 step
    try m.decoder.prefill(p, rows.items);
    var guard = uo.guard.Guard.init(p);
    var st = uo.prompt.Step.init(p);
    const lg = try a.alloc(u16, cfg.vocab);
    var flips: usize = 0;
    var ties: usize = 0;
    for (ref_bf.gen, 0..) |want, i| {
        try m.decoder.logits(lg);
        const ours = guard.pick(lg);
        if (ours != want) {
            const d = gate.departure(lg, ours, want, ref_bf.top2[i]);
            flips += 1;
            ties += @intFromBool(d.tie());
            const t2s = ref_bf.top2[i];
            const other: u32 = @intFromFloat(if (@as(u32, @intFromFloat(t2s[0])) == want) t2s[2] else t2s[0]);
            std.debug.print("  fed step {d}: torch bf16 {d} over {d} by {d:.2} steps, ours {d} over {d} by {d:.2}: {s}\n", .{ i, want, other, d.theirs, ours, want, d.ours, if (d.tie()) "a tie" else if (d.same_two) "the same two, but not within a step" else "NOT the same two" });
        }
        guard.take(want);
        if (i + 1 == ref_bf.gen.len) break;
        try m.decoder.step(want, st.next(want));
    }
    const all_ties = ties == flips;
    std.debug.print("fed torch bf16's reply: {d} of {d} steps pick otherwise, {d} of them ties: {s}\n", .{ flips, ref_bf.gen.len, ties, if (all_ties) "pass" else if (reply_reported) "over (reported)" else "FAIL" });

    // prefill again in chunks of ROWS rows: the first logits and the whole reply bit-identical to the first pass's
    var cut_ok = true;
    for (chunks.items) |c| {
        var dec = try uo.decoder.Decoder.init(gpu.gpa, gpu.d, m.s, &m.kernels, &m.prompt_gemms, m.weights, cap, @min(c, n));
        defer dec.deinit();
        try dec.prefill(p, rows.items);
        try dec.logits(lg);
        var again: std.ArrayList(u32) = .empty;
        try greedy(&dec, p, ref_bf.gen.len, &again, a);
        const same_logits = std.mem.eql(u16, lg, first);
        const same_reply = std.mem.eql(u32, again.items, gen.items);
        std.debug.print("chunks of {d} rows: first logits {s}, reply {s}\n", .{ c, if (same_logits) "bit-identical" else "DIFFER", if (same_reply) "identical" else "DIFFERS" });
        cut_ok = cut_ok and same_logits and same_reply;
    }
    try check.expect(cut_ok, "uocr-decode: a prompt cut in chunks changes the run", .{});
    try check.expect(all_ties or reply_reported, "uocr-decode: {d} of {d} departures are not ties", .{ flips - ties, flips });
    check.pass("uocr-decode {s}: prompt ids, reply departures {s}", .{ rest[1], if (flips == 0) "none" else if (all_ties) "all ties" else "reported" });
}
