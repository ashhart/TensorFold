//! tf-idol-dino's case driver for a backend `B`: its encoder `dn`, safetensors `st`, seconds, Device, stream, hidden.
const std = @import("std");

const usage =
    \\tf-idol-dino DINO_DIR CASE_DIR [RUNS]
    \\DINO_DIR is timm/vit_huge_plus_patch16_dinov3.lvd1689m (config.json, model.safetensors). CASE_DIR (idol_case.py
    \\dino) holds dino.safetensors: the image's pixels, the SDK's feature1 (bf16 on MPS) and the float32 ruler's; and
    \\optionally dino_blocks.safetensors (torch's rows into every block) and dino_blocks_f32.safetensors (each block in
    \\float32 on them). feature1 from the image must sit as close to the ruler as the SDK's within 2x, worst row and
    \\1 - cosine; so must the embedding, each block on torch's own input, and the last norm. Then the encode repeats
    \\byte for byte and is timed RUNS times (default 5); CASE_DIR gets tf-idol-dino.json. Exit 1 on a failed gate.
;

fn tensor(f: anytype, name: []const u8) !@TypeOf(f.get(name).?) {
    return f.get(name) orelse {
        std.debug.print("the case has no {s}\n", .{name});
        return error.MissingTensor;
    };
}

/// bf16 bits as fp32 (dino_host's `value`, backend-free).
fn value(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// The case tensor `name` (f32 or bf16) as f32.
fn floats(a: std.mem.Allocator, f: anytype, name: []const u8, n: usize) ![]f32 {
    const t = try tensor(f, name);
    if (t.numel() != n) {
        std.debug.print("{s} holds {d} values, not {d}\n", .{ name, t.numel(), n });
        return error.BadCaseTensor;
    }
    const out = try a.alloc(f32, n);
    switch (t.dtype) {
        .f32 => @memcpy(std.mem.sliceAsBytes(out), t.bytes),
        .bf16 => for (out, @as([*]align(1) const u16, @ptrCast(t.bytes.ptr))[0..n]) |*o, v| {
            o.* = value(v);
        },
        else => return error.BadCaseTensor,
    }
    return out;
}

/// The case tensor `name` as bf16 bits.
fn bits(a: std.mem.Allocator, f: anytype, name: []const u8, n: usize) ![]u16 {
    const t = try tensor(f, name);
    if (t.dtype != .bf16 or t.numel() != n) {
        std.debug.print("{s} is not {d} bf16 values\n", .{ name, n });
        return error.BadCaseTensor;
    }
    const out = try a.alloc(u16, n);
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

fn widen(a: std.mem.Allocator, src: []const u16) ![]f32 {
    const out = try a.alloc(f32, src.len);
    for (out, src) |*o, v| o.* = value(v);
    return out;
}

fn print(list: *std.ArrayList(u8), a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
    try list.appendSlice(a, try std.fmt.allocPrint(a, fmt, args));
}

fn json(list: *std.ArrayList(u8), a: std.mem.Allocator, key: []const u8, m: anytype) !void {
    try print(list, a, ", \"{s}\": {{\"max_abs\": {d:.6}, \"cosine\": {d:.9}, \"worst_row\": {d:.6}}}", .{ key, m.max_abs, m.cosine, m.worst_row });
}

/// One stage's gate printed and recorded; returns whether it passed.
fn stage(comptime ref: type, out: *std.ArrayList(u8), a: std.mem.Allocator, label: []const u8, ours: []const f32, torch: []const f32, ruler: []const f32) !bool {
    const g = ref.gate(ours, torch, ruler);
    const wo = ref.worstRow(ours, ruler);
    const wt = ref.worstRow(torch, ruler);
    std.debug.print("{s} against float32: worst row {d:.5} (row {d}), 1 - cos {e:.3}; torch's {d:.5} (row {d}), {e:.3}: {s}\n", .{ label, g.ours.worst_row, wo.row, 1 - g.ours.cosine, g.torch.worst_row, wt.row, 1 - g.torch.cosine, if (g.pass) "pass" else "FAIL" });
    try print(out, a, ", \"{s}\": {{\"pass\": {}, \"worst_row_index\": {d}, \"torch_worst_row_index\": {d}", .{ label, g.pass, wo.row, wt.row });
    try json(out, a, "ours_vs_ruler", g.ours);
    try json(out, a, "torch_vs_ruler", g.torch);
    try out.append(a, '}');
    return g.pass;
}

/// The embedding, each block on torch's own input rows, and the last norm, each against its float32 run.
fn blocks(comptime B: type, a: std.mem.Allocator, io: std.Io, dir: []const u8, enc: *B.dn.Encoder, pixels: []const f32, out: *std.ArrayList(u8)) !?bool {
    const cfg = B.dn.config;
    const ref = B.dn.reference;
    const N = cfg.rows * cfg.width;
    var f = B.st.File.open(a, io, try std.fmt.allocPrint(a, "{s}/dino_blocks.safetensors", .{dir})) catch return null;
    defer f.close(io);
    var g = try B.st.File.open(a, io, try std.fmt.allocPrint(a, "{s}/dino_blocks_f32.safetensors", .{dir}));
    defer g.close(io);
    var pass = true;
    var name: [32]u8 = undefined;
    try enc.embedOnly(pixels);
    const torch0 = try floats(a, &f, "block.0", N);
    const rows = try B.stream(enc);
    const embed = try widen(a, rows);
    const same = std.mem.eql(u8, std.mem.sliceAsBytes(rows), std.mem.sliceAsBytes(try bits(a, &f, "block.0", N)));
    std.debug.print("embedding: {s} torch's bf16 rows\n", .{if (same) "the same bytes as" else "differs from"});
    try print(out, a, ", \"embed_bytes_equal_torch\": {}", .{same});
    pass = (try stage(ref, out, a, "embed", embed, torch0, try floats(a, &g, "embed", N))) and pass;
    for (0..cfg.layers) |i| {
        try enc.blockAt(i, try bits(a, &f, try std.fmt.bufPrint(&name, "block.{d}", .{i}), N));
        const label = try std.fmt.bufPrint(&name, "block.{d}", .{i + 1});
        const ours = try widen(a, try B.stream(enc));
        pass = (try stage(ref, out, a, label, ours, try floats(a, &f, label, N), try floats(a, &g, label, N))) and pass;
    }
    const last = try bits(a, &f, "block.32", N);
    const ours = try a.alloc(f32, N);
    try enc.finalAt(last, ours);
    var case = try B.st.File.open(a, io, try std.fmt.allocPrint(a, "{s}/dino.safetensors", .{dir}));
    defer case.close(io);
    pass = (try stage(ref, out, a, "final", ours, try floats(a, &case, "feature1", N), try floats(a, &g, "feature1", N))) and pass;
    return pass;
}

pub fn run(comptime B: type, init: std.process.Init) !u8 {
    const cfg = B.dn.config;
    const ref = B.dn.reference;
    const N = cfg.rows * cfg.width;
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3 and args.len != 4) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    const runs = if (args.len == 4) try std.fmt.parseInt(usize, args[3], 10) else 5;
    var case = try B.st.File.open(a, io, try std.fmt.allocPrint(a, "{s}/dino.safetensors", .{args[2]}));
    defer case.close(io);
    const pixels = try floats(a, &case, "pixels", 3 * cfg.image * cfg.image);
    const sdk = try floats(a, &case, "feature1", N);
    const ruler = try floats(a, &case, "feature1_f32", N);

    var device: B.Device = undefined;
    try device.init();
    defer device.deinit();
    const loading = B.seconds();
    const enc = try device.load(std.heap.page_allocator, io, args[1]);
    defer enc.deinit();
    const load_s = B.seconds() - loading;
    std.debug.print("{s}: loaded in {d:.2} s\n", .{ device.name(), load_s });
    var out: std.ArrayList(u8) = .empty;
    try print(&out, a, "{{\"device\": \"{s}\", \"load_s\": {d:.3}", .{ device.name(), load_s });

    // the RoPE tables against torch's own bf16 cos and sin
    const tables = enc.ropeTables();
    var equal: usize = 0;
    for ([_][]const u8{ "cos", "sin" }, tables) |n, ours| {
        for (ours, try bits(a, &case, n, cfg.patches * cfg.head_dim)) |x, y| equal += @intFromBool(x == y);
    }
    std.debug.print("RoPE tables: {d} of {d} bf16 values equal torch's\n", .{ equal, 2 * cfg.patches * cfg.head_dim });
    try print(&out, a, ", \"rope_equal\": {d}, \"rope_values\": {d}", .{ equal, 2 * cfg.patches * cfg.head_dim });

    const ours = try a.alloc(f32, N);
    const again = try a.alloc(f32, N);
    const first = try enc.encode(pixels, ours);
    var walls = try a.alloc(f64, runs);
    var gpus = try a.alloc(f64, runs);
    var repeat = true;
    for (0..runs) |r| {
        const t = try enc.encode(pixels, again);
        walls[r] = t.wall;
        gpus[r] = t.gpu;
        repeat = repeat and std.mem.eql(u8, std.mem.sliceAsBytes(ours), std.mem.sliceAsBytes(again));
    }
    std.mem.sort(f64, walls, {}, std.sort.asc(f64));
    std.mem.sort(f64, gpus, {}, std.sort.asc(f64));
    std.debug.print("encode: first {d:.4} s; median of {d} {d:.4} s wall, {d:.4} s on the GPU\n", .{ first.wall, runs, walls[runs / 2], gpus[runs / 2] });
    std.debug.print("{d} encodes after the first: {s}\n", .{ runs, if (repeat) "identical bytes" else "DIFFERENT bytes" });
    try print(&out, a, ", \"first_s\": {d:.4}, \"encode_s\": {d:.4}, \"gpu_s\": {d:.4}, \"host_s\": {d:.4}, \"repeat_identical\": {}", .{ first.wall, walls[runs / 2], gpus[runs / 2], first.host, repeat });

    const vs_sdk = ref.match(ours, sdk, cfg.width);
    std.debug.print("feature1 against the SDK's: max |diff| {d:.5}, cosine {d:.8}, worst row {d:.5} (row {d})\n", .{ vs_sdk.max_abs, vs_sdk.cosine, vs_sdk.worst_row, ref.worstRow(ours, sdk).row });
    try json(&out, a, "feature1_vs_sdk", vs_sdk);
    var pass = try stage(ref, &out, a, "feature1", ours, sdk, ruler);
    if (case.get("hidden") != null) {
        const theirs = try bits(a, &case, "hidden", N);
        var same: usize = 0;
        for (try B.hidden(enc), theirs) |x, y| same += @intFromBool(x == y);
        std.debug.print("last_hidden_state: {d:.2}% of bf16 values equal the SDK's\n", .{100 * @as(f64, @floatFromInt(same)) / N});
        try print(&out, a, ", \"hidden_equal_fraction\": {d:.5}", .{@as(f64, @floatFromInt(same)) / N});
    }
    if (try blocks(B, a, io, args[2], enc, pixels, &out)) |ok| {
        pass = pass and ok;
        try print(&out, a, ", \"blocks_pass\": {}", .{ok});
    }
    try print(&out, a, ", \"gates_pass\": {}}}\n", .{pass});
    std.debug.print("gates: {s}\n", .{if (pass) "all pass" else "FAIL"});
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/tf-idol-dino.json", .{args[2]}), .data = out.items });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/feature1.f32", .{args[2]}), .data = std.mem.sliceAsBytes(ours) });
    return if (pass and repeat) 0 else 1;
}
