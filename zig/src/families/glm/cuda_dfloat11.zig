//! DFloat11-coded bf16 weights for the CUDA loader: config.json's coding on its checkpoints, the shapes the files omit.

const std = @import("std");
const core = @import("core");
const cfg = @import("config.zig");

const prefix = "model.language_model.";
const layers = prefix ++ "layers.";

/// The stored shape of a language-model bf16 matrix the loader takes coded, by its full name; null for any other.
pub fn shapeOf(c: *const cfg.Config, name: []const u8) ?[2]usize {
    if (std.mem.eql(u8, name, "lm_head.weight") or std.mem.eql(u8, name, prefix ++ "embed_tokens.weight")) return .{ c.vocab, c.hidden };
    if (!std.mem.startsWith(u8, name, layers)) return null;
    const rest = name[layers.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const i = std.fmt.parseInt(u32, rest[0..dot], 10) catch return null;
    if (i > c.layers) return null;
    const tail = rest[dot + 1 ..];
    if (c.kind(i) == .mla) {
        if (std.mem.eql(u8, tail, "self_attn.kv_b_proj.weight")) return .{ c.mla_heads * (c.nope + c.v_dim), c.kv_lora };
        return null;
    }
    inline for (.{ "q", "k", "v" }) |x| if (std.mem.eql(u8, tail, "self_attn." ++ x ++ "_proj.weight")) return .{ c.kdaWidth(), c.hidden };
    if (std.mem.eql(u8, tail, "self_attn.o_proj.weight")) return .{ c.hidden, c.kdaWidth() };
    return null;
}

fn shape(ctx: *const anyopaque, name: []const u8, buf: *[core.safetensors.max_rank]usize) ?[]const usize {
    const c: *const cfg.Config = @ptrCast(@alignCast(ctx));
    buf[0..2].* = shapeOf(c, name) orelse return null;
    return buf[0..2];
}

/// config.json's dfloat11_config, when it has one, attached to `cks`; `c` must outlive them.
pub fn attach(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const cfg.Config, cks: []const *core.Checkpoint) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const scratch = arena.allocator();
    const path = try std.fs.path.join(scratch, &.{ dir, "config.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, scratch, .limited(1 << 24));
    const coding = (try core.dfloat11_config.parse(scratch, text)) orelse return;
    for (cks) |ck| try core.checkpoint.coded.attach(ck, coding, .{ .ctx = c, .of = shape });
}

const testing = std.testing;

test {
    _ = @import("cuda_dfloat11_test.zig");
}

test "the coded names' shapes are the published bf16 checkpoint's header shapes, rows first" {
    const c: cfg.Config = .{};
    // the published bf16 headers' shapes at the default sizes: layer 5 is KDA, 3 and the MTP layer (45) are MLA
    try testing.expect(c.kind(5) == .kda and c.kind(3) == .mla and c.kind(45) == .mla);
    const stored = [_]struct { []const u8, [2]usize }{
        .{ "lm_head.weight", .{ 154880, 4096 } },
        .{ "model.language_model.embed_tokens.weight", .{ 154880, 4096 } },
        .{ "model.language_model.layers.5.self_attn.q_proj.weight", .{ 8192, 4096 } },
        .{ "model.language_model.layers.5.self_attn.k_proj.weight", .{ 8192, 4096 } },
        .{ "model.language_model.layers.5.self_attn.v_proj.weight", .{ 8192, 4096 } },
        .{ "model.language_model.layers.5.self_attn.o_proj.weight", .{ 4096, 8192 } },
        .{ "model.language_model.layers.3.self_attn.kv_b_proj.weight", .{ 32768, 512 } },
        .{ "model.language_model.layers.45.self_attn.kv_b_proj.weight", .{ 32768, 512 } },
    };
    for (stored) |s| try testing.expectEqual(@as(?[2]usize, s[1]), shapeOf(&c, s[0]));
    // names the loader is not given coded: no shape, so a coded one is refused by name
    for ([_][]const u8{
        "model.language_model.layers.3.self_attn.o_proj.weight",
        "model.language_model.layers.5.self_attn.kv_b_proj.weight",
        "model.language_model.layers.5.self_attn.q_proj",
        "model.language_model.layers.46.self_attn.q_proj.weight",
        "model.language_model.layers.x.self_attn.q_proj.weight",
        "model.language_model.norm.weight",
        "model.visual.blocks.0.attn.qkv.weight",
    }) |name| try testing.expectEqual(@as(?[2]usize, null), shapeOf(&c, name));
}

test "a config with the coding attaches it to every checkpoint given; one without leaves them as they were" {
    const io = testing.io;
    const gpa = testing.allocator;
    const group = core.dfloat11_fixtures.groups[2]; // 1,500 elements, blocks of 4 threads
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const parts = [_]struct { []const u8, []const u8, []const u8, []const usize }{
        .{ "luts", "U8", group.luts, &.{ group.luts.len / 256, 256 } },
        .{ "encoded_exponent", "U8", group.encoded_exponent, &.{group.encoded_exponent.len} },
        .{ "sign_mantissa", "U8", group.sign_mantissa, &.{group.sign_mantissa.len} },
        .{ "output_positions", "U8", group.output_positions, &.{group.output_positions.len} },
        .{ "gaps", "U8", group.gaps, &.{group.gaps.len} },
        .{ "split_positions", "I64", "", &.{0} },
    };
    var entries: [parts.len]core.checkpoint_fixture.Entry = undefined;
    for (parts, &entries) |p, *e| e.* = .{
        .name = try std.fmt.allocPrint(a, "model.language_model.layers.5.self_attn.q_proj.{s}", .{p[0]}),
        .dtype = p[1],
        .shape = p[3],
        .bytes = p[2],
    };
    try core.checkpoint_fixture.write(a, io, tmp.dir, "model.safetensors", &entries);
    var buf: [64]u8 = undefined;
    const dir = try std.fmt.bufPrint(&buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    // a model whose q_proj has the fixture's 1,500 elements: 12 heads of 5 dims, 25 wide
    const c: cfg.Config = .{ .kda_heads = 12, .kda_dim = 5, .hidden = 25 };

    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"glm\"}" });
    {
        var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, prefix);
        defer ck.close();
        try attach(gpa, io, dir, &c, &.{&ck});
        try testing.expectEqual(@as(?*core.checkpoint.coded.Coded, null), ck.coded);
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"dfloat11_config":{"version":"0.2.0","bytes_per_thread":8,"threads_per_block":[4],"pattern_dict":
        \\{"model\\.language_model\\.layers\\.\\d+\\.self_attn\\.q_proj":[]}}}
    });
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, prefix);
    defer ck.close();
    try attach(gpa, io, dir, &c, &.{&ck});
    const q = try ck.get(prefix ++ "layers.5.self_attn.q_proj.weight");
    try testing.expect(q.is(.bf16, &.{ 60, 25 }));
    try testing.expectEqualSlices(u8, group.want, q.bytes);
}
