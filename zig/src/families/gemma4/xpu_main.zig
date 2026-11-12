//! Dev runner of the Gemma 4 E4B XPU forward: greedy tokens with timings; `--teacher`: forced tokens, oracle logits.
const std = @import("std");
const dev = @import("xpu_dev.zig");
const cfg = @import("xpu_config.zig");
const fwd = @import("xpu_forward.zig");
const gemm_test = @import("xpu_gemm_test.zig");
const ple_check = @import("xpu_ple_check.zig");
const geglu_check = @import("xpu_geglu_check.zig");
const alloc = dev.alloc;
const now = dev.now;

fn readIds(path: []const u8) ![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, try dev.mapFile(path), " \n,[]");
    while (it.next()) |w| try ids.append(alloc, try std.fmt.parseInt(u32, w, 10));
    return ids.items;
}

fn jsonIds(v: std.json.Value) ![]u32 {
    if (v != .array) return error.BadReference;
    const out = try alloc.alloc(u32, v.array.items.len);
    for (v.array.items, out) |x, *o| {
        if (x != .integer) return error.BadReference;
        o.* = @intCast(x.integer);
    }
    return out;
}

fn writeNpy(path: []const u8, rows: []const f32, n: usize, width: usize) !void {
    var hdr: [128]u8 = undefined;
    const body = try std.fmt.bufPrint(hdr[10..], "{{'descr': '<f4', 'fortran_order': False, 'shape': ({d}, {d}), }}", .{ n, width });
    var len = body.len + 1;
    len = (len + 10 + 63) / 64 * 64 - 10;
    @memset(hdr[10 + body.len .. 10 + len], ' ');
    hdr[10 + len - 1] = '\n';
    @memcpy(hdr[0..6], "\x93NUMPY");
    hdr[6] = 1;
    hdr[7] = 0;
    std.mem.writeInt(u16, hdr[8..10], @intCast(len), .little);
    try dev.writeFile(path, &.{ hdr[0 .. 10 + len], std.mem.sliceAsBytes(rows) });
}

fn teacher(r: *fwd.Runner, ref_dir: []const u8, out_dir: []const u8, only: ?[]const u8) !void {
    const list_path = try std.fmt.allocPrint(alloc, "{s}/prompts.json", .{ref_dir});
    const prompts = try std.json.parseFromSlice(std.json.Value, alloc, try dev.mapFile(list_path), .{});
    for (prompts.value.array.items) |pr| {
        const name = pr.object.get("name").?.string;
        if (only) |o| {
            var found = false;
            var it = std.mem.tokenizeScalar(u8, o, ',');
            while (it.next()) |n| found = found or std.mem.eql(u8, n, name);
            if (!found) continue;
        }
        const path = try std.fmt.allocPrint(alloc, "{s}/{s}.json", .{ ref_dir, name });
        const meta = try std.json.parseFromSlice(std.json.Value, alloc, try dev.mapFile(path), .{});
        const prompt = try jsonIds(meta.value.object.get("prompt_ids").?);
        const gen = try jsonIds(meta.value.object.get("generated_ids").?);
        const rows = try alloc.alloc(f32, gen.len * cfg.vocab);
        r.reset();
        const t0 = now();
        try r.feed(prompt, true);
        const t1 = now();
        @memcpy(rows[0..cfg.vocab], try r.logits(0));
        for (1..gen.len) |i| {
            try r.feed(gen[i - 1 .. i], true);
            @memcpy(rows[i * cfg.vocab ..][0..cfg.vocab], try r.logits(0));
        }
        const t2 = now();
        const out = try std.fmt.allocPrint(alloc, "{s}/{s}.logits.npy", .{ out_dir, name });
        try writeNpy(out, rows, gen.len, cfg.vocab);
        std.debug.print("{s:<16} prompt {d:5} tokens in {d:6.3} s ({d:6.0} tok/s), {d:3} forced steps in {d:6.3} s\n", .{ name, prompt.len, t1 - t0, @as(f64, @floatFromInt(prompt.len)) / (t1 - t0), gen.len - 1, t2 - t1 });
        alloc.free(rows);
    }
}

/// Windows of 1, 2, 4, 8, 16 rows, then the 16-row cut four more times: the next-token logits must be the same bits.
fn windows(r: *fwd.Runner, ids: []const u32) !void {
    var first: ?u64 = null;
    var all_equal = true;
    for ([_]u32{ 1, 2, 4, 8, 16, 17, 64, 100, 128, 256, 512, 1024, 16, 16 }) |w| {
        r.reset();
        r.window = w;
        try r.feed(ids, true);
        const raw = try alloc.alloc(u8, cfg.vocab * 2);
        defer alloc.free(raw);
        try r.d.read(raw, r.a.logits);
        const h = std.hash.Wyhash.hash(0, raw);
        if (first == null) first = h;
        all_equal = all_equal and h == first.?;
        std.debug.print("window {d:2}: next token {d}, logits hash {x:0>16}{s}\n", .{ w, r.last, h, if (h == first.?) "" else "  DIFFERENT" });
    }
    std.debug.print("{s}\n", .{if (all_equal) "windows == rows: identical logits" else "windows differ"});
}

/// Prompts cut from the ids file at each length: the next-token logits hash of every length, for comparing two builds.
fn lengths(r: *fwd.Runner, ids: []const u32, list: []const u8) !void {
    var it = std.mem.tokenizeScalar(u8, list, ',');
    while (it.next()) |tok| {
        const n = try std.fmt.parseInt(usize, tok, 10);
        if (n == 0 or n > ids.len) return error.Usage;
        r.reset();
        try r.feed(ids[0..n], true);
        const raw = try alloc.alloc(u8, cfg.vocab * 2);
        defer alloc.free(raw);
        try r.d.read(raw, r.a.logits);
        std.debug.print("length {d:5}: next token {d}, logits hash {x:0>16}\n", .{ n, r.last, std.hash.Wyhash.hash(0, raw) });
    }
}

/// The card's top-64 of each step's logits against the host's, over a prompt and 24 greedy steps; and its time.
fn topkTest(r: *fwd.Runner, ids: []const u32) !void {
    r.reset();
    try r.feed(ids, true);
    var bad: usize = 0;
    var t_card: f64 = 0;
    var t_host: f64 = 0;
    for (0..24) |_| {
        var v: [64]f32 = undefined;
        var id: [64]u32 = undefined;
        const a = now();
        try r.topK(0, 64, &v, &id);
        t_card += now() - a;
        const b = now();
        const row = try r.logits(0);
        // host: 64 largest, ties to the lower id
        var hv: [64]f32 = @splat(-std.math.inf(f32));
        var hid: [64]u32 = @splat(0);
        for (row, 0..) |x, i| {
            if (!(x > hv[63])) continue;
            var j: usize = 63;
            while (j > 0 and x > hv[j - 1]) : (j -= 1) {
                hv[j] = hv[j - 1];
                hid[j] = hid[j - 1];
            }
            hv[j] = x;
            hid[j] = @intCast(i);
        }
        t_host += now() - b;
        if (!std.mem.eql(u32, &id, &hid)) bad += 1;
        try r.feed(&.{id[0]}, true);
    }
    std.debug.print("top-64 on the card equals the host's in {d} of 24 steps; card {d:.2} ms a step (with the copy), host scan {d:.2} ms\n", .{ 24 - bad, t_card / 24 * 1e3, t_host / 24 * 1e3 });
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 3) {
        std.debug.print("usage: xpu-gemma4 MODEL_DIR IDS_FILE [STEPS] | MODEL_DIR --teacher REF_DIR OUT_DIR [--only a,b]\n", .{});
        return;
    }
    const ctx: u32 = if (std.c.getenv("XPU_CTX")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 4096;
    const t_load = now();
    const r = try fwd.Runner.init(argv[1], ctx);
    defer r.deinit();
    std.debug.print("loaded in {d:.1} s, {d:.2} GB on the card\n", .{ now() - t_load, @as(f64, @floatFromInt(r.d.allocated)) / 1e9 });
    if (std.mem.eql(u8, argv[2], "--topk-test")) return topkTest(r, try readIds(argv[3]));
    if (std.mem.eql(u8, argv[2], "--windows")) return windows(r, try readIds(argv[3]));
    if (std.mem.eql(u8, argv[2], "--lengths")) return lengths(r, try readIds(argv[3]), argv[4]);
    if (std.mem.eql(u8, argv[2], "--gemm-test")) return gemm_test.gemmTest(r);
    if (std.mem.eql(u8, argv[2], "--geglu-check")) return geglu_check.run(r);
    if (std.mem.eql(u8, argv[2], "--ple-check")) {
        const ids = try alloc.alloc(u32, argv.len - 3);
        for (argv[3..], ids) |a, *o| o.* = try std.fmt.parseInt(u32, a, 10);
        return ple_check.run(r, ids);
    }
    if (std.mem.eql(u8, argv[2], "--teacher")) {
        if (argv.len < 5) return error.Usage;
        return teacher(r, argv[3], argv[4], if (argv.len > 6 and std.mem.eql(u8, argv[5], "--only")) argv[6] else null);
    }
    const steps: usize = if (argv.len > 3) try std.fmt.parseInt(usize, argv[3], 10) else 32;
    const ids = try readIds(argv[2]);
    const t0 = now();
    try r.feed(ids, true);
    const t1 = now();
    r.d.profReport("prompt");
    var out: std.ArrayList(u32) = .empty;
    try out.append(alloc, r.last);
    for (1..steps) |_| {
        try r.feed(&.{r.last}, true);
        try out.append(alloc, r.last);
    }
    const t2 = now();
    r.d.profReport("decode");
    std.debug.print("prompt: {d} tokens in {d:.3} s ({d:.0} tok/s)\n", .{ ids.len, t1 - t0, @as(f64, @floatFromInt(ids.len)) / (t1 - t0) });
    if (std.c.getenv("XPU_TIMING") != null) std.debug.print("host: {d:.2} ms a token queueing a forward, {d:.2} ms a token in forwards\n", .{ r.t_enqueue / @as(f64, @floatFromInt(steps + 1)) * 1e3, r.t_forward / @as(f64, @floatFromInt(steps + 1)) * 1e3 });
    std.debug.print("decode: {d} tokens in {d:.3} s ({d:.1} tok/s)\n", .{ steps - 1, t2 - t1, @as(f64, @floatFromInt(steps - 1)) / (t2 - t1) });
    std.debug.print("tokens:", .{});
    for (out.items) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
}
