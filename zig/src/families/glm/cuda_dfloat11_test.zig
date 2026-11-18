//! The loader's cuts over DFloat11-coded weights: the bytes a plain checkpoint gives, through the same formats code.

const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const formats = @import("cuda_formats.zig");
const coded = @import("cuda_dfloat11.zig");

const testing = std.testing;
const Tensor = core.checkpoint.Tensor;
const prefix = "model.language_model.";
const group = core.dfloat11_fixtures.groups[2]; // 1,500 elements, blocks of 4 threads
const names = [_]struct { []const u8, [2]usize }{
    .{ "lm_head", .{ 60, 25 } },
    .{ prefix ++ "layers.5.self_attn.o_proj", .{ 25, 60 } },
    .{ prefix ++ "layers.5.self_attn.q_proj", .{ 60, 25 } },
};
const coding =
    \\{"dfloat11_config":{"version":"0.2.0","bytes_per_thread":8,"threads_per_block":[4],"pattern_dict":{"lm_head":[],
    \\"model\\.language_model\\.layers\\.5\\.self_attn\\.o_proj":[],"model\\.language_model\\.layers\\.5\\.self_attn\\.q_proj":[]}}}
;

/// What formats.cut asks of the loader, with the uploads kept as bytes: a same-dtype cut is a copy of rows or columns.
const Cuts = struct {
    gpa: std.mem.Allocator,
    ck: *core.Checkpoint,
    host: std.ArrayList(u8) = .empty,
    out: std.ArrayList(u8) = .empty,
    /// formats.cut's FP8 branches name the lane; a bf16 cut never reads it.
    tools: struct { lane: *const cuda.qmmf.Lane } = undefined,

    pub fn formatOne(_: *Cuts, ck: *core.Checkpoint, module: []const u8, wants: []const formats.Format) !formats.Format {
        var buf: [224]u8 = undefined;
        const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{module}));
        const f = try formats.of(.{ .mixed = null }, .{ .weight = formats.metaOf(&w) });
        for (wants) |want| if (f == want) return f;
        return error.UnreadableFormat;
    }

    pub fn rows(L: *Cuts, t: Tensor, lo: usize, hi: usize, total: usize, to: core.checkpoint.DType) !void {
        if (t.dtype != to or t.numel() % total != 0) return error.UnexpectedTensor;
        const per = 2 * t.numel() / total;
        try L.host.appendSlice(L.gpa, t.bytes[lo * per .. hi * per]);
    }

    pub fn cols(L: *Cuts, t: Tensor, lo: usize, hi: usize, to: core.checkpoint.DType) !void {
        if (t.dtype != to or t.rank != 2) return error.UnexpectedTensor;
        for (0..t.dim(0)) |r| try L.host.appendSlice(L.gpa, t.bytes[2 * (r * t.dim(1) + lo) .. 2 * (r * t.dim(1) + hi)]);
    }

    pub fn whole(L: *Cuts, t: Tensor, to: core.checkpoint.DType) !void {
        if (t.dtype != to) return error.UnexpectedTensor;
        try L.host.appendSlice(L.gpa, t.bytes);
    }

    pub fn put(L: *Cuts, comptime _: []const u8, _: anytype) !u64 {
        try L.out.appendSlice(L.gpa, L.host.items);
        L.host.clearRetainingCapacity();
        return 1;
    }
};

/// The head's rows 10 to 40, o_proj's columns 5 to 35 and q_proj whole, cut as the loader cuts them from `dir`.
fn cuts(gpa: std.mem.Allocator, dir: []const u8, c: *const cfg.Config) ![]u8 {
    const io = testing.io;
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, prefix);
    defer ck.close();
    var head = try core.Checkpoint.openModelPrefix(gpa, io, dir, "lm_head.");
    defer head.close();
    try coded.attach(gpa, io, dir, c, &.{ &ck, &head });
    var L: Cuts = .{ .gpa = gpa, .ck = &ck };
    defer L.host.deinit(gpa);
    errdefer L.out.deinit(gpa);
    _ = try formats.cut(&L, &head, &.{"lm_head"}, &.{.{ .rel = "lm_head", .rows = .{ 10, 40 } }}, &.{ .bf16, .mxfp8 }, "head", .{});
    _ = try formats.cutProj(&L, prefix, 5, &.{.{ .rel = "self_attn.o_proj", .cols = .{ 5, 35 } }}, &.{ .bf16, .mxfp8 }, "layers.{d}.kda.o");
    ck.release(); // the loader's, between layers: what was decoded goes, and the next cut decodes again
    _ = try formats.cutProj(&L, prefix, 5, &.{.{ .rel = "self_attn.q_proj" }}, &.{.bf16}, "layers.{d}.kda.q");
    return L.out.toOwnedSlice(gpa);
}

test "the loader's cuts of coded weights (head rows, o_proj columns, a whole q_proj) are the plain checkpoint's bytes" {
    const io = testing.io;
    const gpa = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // a model whose three matrices have the fixture's 1,500 elements: 12 heads of 5 dims, 25 wide, 60 tokens
    const c: cfg.Config = .{ .kda_heads = 12, .kda_dim = 5, .hidden = 25, .vocab = 60 };
    var got: [2][]u8 = undefined;
    for (&got, 0..) |*out, as_groups| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        var entries: std.ArrayList(core.checkpoint_fixture.Entry) = .empty;
        for (names) |n| {
            if (as_groups == 0) {
                try entries.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.weight", .{n[0]}), .dtype = "BF16", .shape = try a.dupe(usize, &n[1]), .bytes = group.want });
                continue;
            }
            const parts = [_]struct { []const u8, []const u8, []const u8, []const usize }{
                .{ "luts", "U8", group.luts, &.{ group.luts.len / 256, 256 } },
                .{ "encoded_exponent", "U8", group.encoded_exponent, &.{group.encoded_exponent.len} },
                .{ "sign_mantissa", "U8", group.sign_mantissa, &.{group.sign_mantissa.len} },
                .{ "output_positions", "U8", group.output_positions, &.{group.output_positions.len} },
                .{ "gaps", "U8", group.gaps, &.{group.gaps.len} },
                .{ "split_positions", "I64", "", &.{0} },
            };
            for (parts) |p| try entries.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}.{s}", .{ n[0], p[0] }), .dtype = p[1], .shape = try a.dupe(usize, p[3]), .bytes = p[2] });
        }
        try core.checkpoint_fixture.write(a, io, tmp.dir, "model.safetensors", entries.items);
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = if (as_groups == 0) "{\"model_type\":\"glm\"}" else coding });
        var buf: [64]u8 = undefined;
        out.* = try cuts(gpa, try std.fmt.bufPrint(&buf, ".zig-cache/tmp/{s}", .{tmp.sub_path}), &c);
    }
    defer for (got) |g| gpa.free(g);
    try testing.expectEqual(@as(usize, 2 * (30 * 25 + 25 * 30 + 1500)), got[0].len);
    try testing.expectEqualSlices(u8, group.want[2 * 10 * 25 .. 2 * 40 * 25], got[0][0 .. 2 * 30 * 25]);
    try testing.expectEqualSlices(u8, group.want[2 * 5 .. 2 * 35], got[0][2 * 30 * 25 ..][0 .. 2 * 30]); // o_proj's first row
    try testing.expectEqualSlices(u8, got[0], got[1]);
}
