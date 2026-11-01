//! The shared expert's saved pocket through the production shard writer and widened reader.
const std = @import("std");
const living = @import("living");
const core = @import("core");
const f = living.families.qwen3_5_moe;
const d = 2048;
const width = 512;
const cap = 64;
const wide = @import("cuda_wide.zig");

test "MoE learning: shared shards reopen exact base bytes and zero or judged pocket rows" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", scratch);
    const stem = "language_model.model.layers.0.mlp.shared_expert.";
    var ts: std.ArrayList(core.shard_edit.New) = .empty;
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, pi| {
        const rows: usize = if (pi < 2) width else d;
        const cols: usize = if (pi < 2) d else width;
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |part, si| {
            const words = cols / (if (si == 0) @as(usize, 8) else 64);
            const bytes = try scratch.alloc(u8, rows * words * (if (si == 0) @as(usize, 4) else 2));
            if (si == 0) {
                for (bytes, 0..) |*b, i| b.* = @truncate(i *% 37);
            } else {
                const v = living.affine.bf16of(if (si == 1) 0.125 else -0.25);
                for (0..rows * words) |i| std.mem.writeInt(u16, bytes[2 * i ..][0..2], v, .little);
            }
            try ts.append(scratch, .{ .name = try std.fmt.allocPrint(scratch, "{s}{s}.{s}", .{ stem, proj, part }), .dtype = if (si == 0) .u32 else .bf16, .shape = try scratch.dupe(usize, &.{ rows, words }), .bytes = bytes });
        }
    }
    const sentinel: [8]u8 = @splat(13);
    try ts.append(scratch, .{ .name = "language_model.mtp.shared", .dtype = .bf16, .shape = &.{4}, .bytes = &sentinel });
    try core.shard_edit.create(a, try std.fs.path.join(scratch, &.{ dir, "model.safetensors" }), &.{}, ts.items);
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(scratch, "{\"weight_map\":{");
    for (ts.items, 0..) |t, i| try index.print(scratch, "{s}\"{s}\":\"model.safetensors\"", .{ if (i == 0) "" else ",", t.name });
    try index.appendSlice(scratch, "}}");
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"qwen3_5_moe\",\"text_config\":{\"hidden_size\":2048,\"num_hidden_layers\":1,\"moe_intermediate_size\":512,\"shared_expert_intermediate_size\":512}}" });
    var all: living.Layers = .empty;
    all.set(0);
    const lay = living.Layout.fromExperts(d, width, 1, all);
    const gate = try scratch.alloc(f32, cap * d);
    const up = try scratch.alloc(f32, cap * d);
    const down = try scratch.alloc(f32, cap * d);
    const restored = try scratch.alloc(f32, cap * d);
    @memset(gate, 0);
    @memset(up, 0);
    @memset(down, 0);
    for ([_]bool{ false, true }) |taught| {
        if (taught) {
            @memset(gate[0..d], 0.5);
            @memset(up[0..d], 0.125);
            @memset(down[0..d], -0.25);
        }
        _ = try living.shards.Io(core).write(a, io, dir, f, lay, &.{0}, cap, &.{ gate, up }, down);
        try std.testing.expectEqual(@as(?i64, 512), try core.shard_edit.configIntIn(a, io, dir, "text_config", "moe_intermediate_size"));
        try std.testing.expectEqual(@as(?i64, 576), try core.shard_edit.configIntIn(a, io, dir, "text_config", f.keys.width));
        var ck = try core.Checkpoint.openModel(a, io, dir);
        defer ck.close();
        for (ts.items[0..9], 0..) |old, i| {
            const got = try ck.get(old.name);
            const stride = got.bytes.len / got.dim(0);
            const cut = try wide.base(a, got.bytes, old.shape[0], stride, old.bytes.len / old.shape[0]);
            defer a.free(cut);
            try std.testing.expectEqualSlices(u8, old.bytes, cut);
            if (i % 3 != 0) continue;
            const pi = i / 3;
            const name = ([_][]const u8{ "gate_proj", "up_proj", "down_proj" })[pi];
            try wide.read(&ck, stem, name, if (pi < 2) width else d, if (pi < 2) d else width, restored);
            try std.testing.expectEqualSlices(f32, ([_][]f32{ gate, up, down })[pi], restored);
        }
        try std.testing.expectEqualSlices(u8, &sentinel, (try ck.get("language_model.mtp.shared")).bytes);
    }
}
