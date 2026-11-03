//! Tiny dense checkpoints exercise the actual Metal loader, writer and row bindings without shader execution.
const std = @import("std");
const metal = @import("metal");
const core = @import("core");
const living = @import("living");
const pocket = @import("pocket");
const storage = @import("storage");
const Selection = @FieldType(pocket.Pockets, "selected");
const d = 64;
const width = 128;
const layers = 4;
const cap = 64;
const n = cap * d;
const Shards = living.shards.Io(core);

fn family(scoped: bool, wrapped: bool) living.Family {
    var f = living.families.qwen3_5;
    if (!scoped) f.names.layers = "model.layers.";
    if (!wrapped) f.keys.scope = null;
    return f;
}

fn fixture(a: std.mem.Allocator, io: std.Io, dir: []const u8, scoped: bool, wrapped: bool) !void {
    var tensors: std.ArrayList(core.shard_edit.New) = .empty;
    for (0..layers) |l| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, p| {
        const rows: usize = if (p == 2) d else width;
        const cols: usize = if (p == 2) width else d;
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |leaf, kind| {
            const k = if (kind == 0) cols / 8 else cols / 64;
            const bytes = try a.alloc(u8, rows * k * (if (kind == 0) @as(usize, 4) else 2));
            if (kind == 0) {
                for (0..rows * k) |i| std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @as(u32, @intCast(i)) + 0x12345670, .little);
            } else for (0..rows * k) |i| std.mem.writeInt(u16, bytes[i * 2 ..][0..2], if (kind == 1) 0x3d80 else 0xbf00, .little);
            try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}model.layers.{d}.mlp.{s}.{s}", .{ if (scoped) "language_model." else "", l, proj, leaf }), .dtype = if (kind == 0) .u32 else .bf16, .shape = try a.dupe(usize, &.{ rows, k }), .bytes = bytes });
        }
    };
    try core.shard_edit.create(a, try std.fs.path.join(a, &.{ dir, "model.safetensors" }), &.{.{ "format", "mlx" }}, tensors.items);
    const conf =
        \\"model_type":"qwen3_5","hidden_size":64,"intermediate_size":128,"num_hidden_layers":4,
        \\"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"vocab_size":97,
        \\"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":16,"linear_value_head_dim":16,
        \\"linear_conv_kernel_dim":4,"eos_token_id":2
    ;
    const text = try std.fmt.allocPrint(a, "{{\"quantization\":{{\"bits\":4,\"group_size\":64}},{s}{s}{s}}}", .{ if (wrapped) "\"text_config\":{" else "", conf, if (wrapped) "}" else "" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "config.json" }), .data = text });
}

fn layout() living.Layout {
    var all: living.Layers = .empty;
    for (0..layers) |l| all.set(l);
    return .fromExperts(d, width, layers, all);
}

fn vectors(a: std.mem.Allocator) ![3][]f32 {
    var out: [3][]f32 = undefined;
    for (&out) |*v| {
        v.* = try a.alloc(f32, n);
        @memset(v.*, 0);
    }
    out[0][0] = 16;
    out[1][0] = 0.0625;
    out[2][0] = 0.125;
    out[2][1] = -0.125;
    return out;
}

fn covered(r: *const metal.Record) [9]bool {
    var seen: [9]bool = @splat(false);
    for (r.calls[0..r.count]) |c| {
        const first = (c.output - 32) / (d * 2);
        for (first..first + c.rows) |row| seen[row] = true;
    }
    return seen;
}

test "qwen27 living: an empty folder and a failed first pipeline allocate no retained pocket work" {
    const a = std.testing.allocator;
    var r: metal.Record = .{ .a = a };
    var p = try pocket.Pockets.init(a, r.device(), d, layers);
    defer p.deinit();
    const zero: [n]f32 = @splat(0);
    try p.set(0, &zero, &zero, &zero, 0);
    for ([_]usize{ 1, 16, 128, 2048 }) |rows| try p.add(r.encoder(), 0, undefined, undefined, rows);
    try std.testing.expectEqual(@as(usize, 0), r.allocated);
    try std.testing.expectEqual(@as(usize, 0), r.libraries);
    try std.testing.expectEqual(@as(usize, 0), r.count);
    r.fail_pipeline = true;
    try std.testing.expectError(error.PlantedPipelineFailure, p.set(0, &zero, &zero, &zero, 1));
    try std.testing.expect(p.layers[0] == null and p.kernel == null);
    try std.testing.expectEqual(@as(usize, 0), r.live);
}

test "qwen27 living: actual mixed verify, prompt chunks and single rows bind only taught rows" {
    const a = std.testing.allocator;
    var r: metal.Record = .{ .a = a };
    var p = try pocket.Pockets.init(a, r.device(), d, layers);
    defer p.deinit();
    const zero: [n]f32 = @splat(0);
    try p.set(0, &zero, &zero, &zero, 1);
    const x = try r.device().buffer(2048 * d * 2 + 32, 0);
    defer x.deinit();
    const y = try r.device().buffer(2048 * d * 2 + 32, 0);
    defer y.deinit();
    const Span = struct { parents: []const i32 };
    const spans = [_]Span{ .{ .parents = &.{ -1, 0, 1 } }, .{ .parents = &.{ -1, 0, 1, 2 } }, .{ .parents = &.{ -1, 0 } } };
    p.selected = try Selection.windows(&.{ 0, 1, 2 }, &.{ 0, 3, 7 }, &spans, &.{ false, true, false });
    try p.add(r.encoder(), 0, .{ .buffer = x, .offset = 16 }, .{ .buffer = y, .offset = 32 }, 9);
    try std.testing.expectEqual(@as(usize, 2), r.count);
    try std.testing.expectEqual([9]bool{ true, true, true, false, false, false, false, true, true }, covered(&r));
    try std.testing.expectEqual(@as(usize, 16 + 7 * d * 2), r.calls[1].input);
    try std.testing.expectEqual([4]u32{ d, 1, cap, 0 }, r.calls[0].shape);
    const verify = covered(&r);
    r.count = 0;
    p.selected = .{};
    for (verify, 0..) |taught, i| if (taught) try p.add(r.encoder(), 0, .{ .buffer = x, .offset = 16 + i * d * 2 }, .{ .buffer = y, .offset = 32 + i * d * 2 }, 1);
    try std.testing.expectEqual(verify, covered(&r));
    r.count = 0;
    p.selected = .{ .all = false };
    try p.add(r.encoder(), 0, .{ .buffer = x }, .{ .buffer = y }, 128);
    try std.testing.expectEqual(@as(usize, 0), r.count);
    p.selected = .{};
    try p.add(r.encoder(), 0, .{ .buffer = x }, .{ .buffer = y }, 2048);
    try std.testing.expectEqual(@as(usize, 2048), r.calls[0].rows);
    try std.testing.expectError(error.BadPocketRows, p.selected.add(std.math.maxInt(usize), 2));
}

test "qwen27 living: CUDA and Metal saves reopen through the actual Metal loader and common CUDA reader" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    inline for (.{ true, false }) |scoped| inline for (.{ true, false }) |wrapped| {
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        const ar = arena.allocator();
        var temp = std.testing.tmpDir(.{});
        defer temp.cleanup();
        const dir = try temp.dir.realPathFileAlloc(io, ".", ar);
        try fixture(ar, io, dir, scoped, wrapped);
        const f = comptime family(scoped, wrapped);
        var ck = try core.Checkpoint.openModel(a, io, dir);
        const prefix = try std.fmt.allocPrint(ar, "{s}0.mlp.down_proj.", .{f.names.layers});
        const old = try ck.get(try std.fmt.allocPrint(ar, "{s}weight", .{prefix}));
        const base_bytes = try ar.dupe(u8, old.bytes);
        ck.close();
        const v = try vectors(ar);
        _ = try Shards.write(a, io, dir, f, layout(), &.{0}, cap, &.{ v[0], v[1] }, v[2]);
        const config_text = try temp.dir.readFileAlloc(io, "config.json", ar, .limited(1 << 20));
        const c = try pocket.config.parse(a, config_text, null);
        try std.testing.expectEqual(@as(usize, width), c.intermediate);
        var r: metal.Record = .{ .a = a };
        {
            var p = try pocket.Pockets.init(a, r.device(), d, layers);
            defer p.deinit();
            try p.load(io, dir, c);
            const actual = try vectors(ar);
            try std.testing.expectEqual(@as(usize, 1), try p.read(0, actual[0], actual[1], actual[2]));
            for (v, actual) |expected, got| try std.testing.expectEqualSlices(f32, expected, got);
            try std.testing.expect(p.layers[1] == null and p.layers[2] == null and p.layers[3] == null);
            v[2][0] = 0.25;
            try core.shard_edit.save_guard.begin(io, dir);
            _ = try storage.save(a, io, dir, layout(), &.{0}, cap, v[0], v[1], v[2]);
            try std.testing.expectError(error.CheckpointSaveUncertain, core.Checkpoint.openModel(a, io, dir));
            try core.shard_edit.save_guard.finish(io, dir);
            var saved = try core.Checkpoint.openModel(a, io, dir);
            defer saved.close();
            const down = try saved.get(try std.fmt.allocPrint(ar, "{s}weight", .{prefix}));
            for (0..d) |row| try std.testing.expectEqualSlices(u8, base_bytes[row * width / 2 ..][0 .. width / 2], down.bytes[row * (width + cap) / 2 ..][0 .. width / 2]);
            const base_view = try ar.alloc(u8, base_bytes.len);
            try core.dense_pocket.copyBase(down.bytes, d, (width + cap) / 2, width / 2, base_view);
            try std.testing.expectEqualSlices(u8, base_bytes, base_view);
            const read = try ar.alloc(f32, n);
            const base = try std.fmt.allocPrint(ar, "{s}0.", .{f.names.layers});
            try core.dense_pocket.read(&saved, base, "mlp.down_proj", d, width, read);
            try std.testing.expectEqualSlices(f32, v[2], read);
        }
        try std.testing.expectEqual(@as(usize, 0), r.live);
    };
}
