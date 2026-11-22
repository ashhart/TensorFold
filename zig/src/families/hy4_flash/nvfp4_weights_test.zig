//! The NVFP4 weight map against 344 of the 254,168 header entries of the published build (atakans/Hy4-flash-NVFP4).
const std = @import("std");
const config = @import("config.zig");
const nvfp4 = @import("nvfp4_weights.zig");

const Entry = struct { dtype: nvfp4.DType, shape: [4]usize, rank: u8, seen: bool = false };

/// Answers check() from a sample of a header: a name in the sample must match it, any other name is only counted.
const Sample = struct {
    entries: std.StringHashMapUnmanaged(Entry) = .empty,
    asked: usize = 0,

    fn load(self: *Sample, a: std.mem.Allocator, text: []const u8) !void {
        var lines = std.mem.tokenizeScalar(u8, text, '\n');
        while (lines.next()) |line| {
            var fields = std.mem.tokenizeScalar(u8, line, ' ');
            const name = fields.next().?;
            const dtype_name = fields.next().?;
            const dims = fields.next().?;
            const dtype: nvfp4.DType = inline for (.{ .{ "BF16", .bf16 }, .{ "F32", .f32 }, .{ "I32", .i32 }, .{ "U8", .u8 }, .{ "F8_E4M3", .f8_e4m3 } }) |p| {
                if (std.mem.eql(u8, dtype_name, p[0])) break p[1];
            } else return error.FixtureDType;
            var entry: Entry = .{ .dtype = dtype, .shape = @splat(0), .rank = 0 };
            if (!std.mem.eql(u8, dims, "scalar")) {
                var it = std.mem.tokenizeScalar(u8, dims, 'x');
                while (it.next()) |d| : (entry.rank += 1) entry.shape[entry.rank] = try std.fmt.parseInt(usize, d, 10);
            }
            try self.entries.put(a, name, entry);
        }
    }

    pub fn expect(self: *Sample, name: []const u8, dtype: nvfp4.DType, shape: []const usize) !void {
        self.asked += 1;
        const entry = self.entries.getPtr(name) orelse return;
        if (entry.seen) return error.AskedTwice;
        entry.seen = true;
        if (entry.dtype != dtype or !std.mem.eql(usize, entry.shape[0..entry.rank], shape)) {
            std.debug.print("{s}: the build stores {t} {any}, the map asks {t} {any}\n", .{ name, entry.dtype, entry.shape[0..entry.rank], dtype, shape });
            return error.UnexpectedTensor;
        }
    }
};

test "the NVFP4 map asks for every tensor of the published build, as it is stored" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try config.parse(a, @embedFile("config_fixture.json"));
    var sample: Sample = .{};
    try sample.load(a, @embedFile("nvfp4_header.txt"));
    try std.testing.expectEqual(@as(usize, 344), sample.entries.count());
    try nvfp4.check(&sample, parsed.value);
    var it = sample.entries.iterator();
    while (it.next()) |e| if (!e.value_ptr.seen) {
        std.debug.print("{s} is in the build but not in the map\n", .{e.key_ptr.*});
        return error.MissedTensor;
    };
    try std.testing.expectEqual(@as(usize, 253728), sample.asked);
}

test "a changed tensor refuses by name" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try config.parse(a, @embedFile("config_fixture.json"));
    var sample: Sample = .{};
    try sample.load(a, @embedFile("nvfp4_header.txt"));
    // An expert stored with BF16 block scales, as a build from another tool might store it.
    sample.entries.getPtr("model.layers.41.mlp.experts.511.down_proj.weight_scale").?.dtype = .bf16;
    try std.testing.expectError(error.UnexpectedTensor, nvfp4.check(&sample, parsed.value));
}

const modelopt =
    \\{"producer": {"name": "modelopt", "version": "0.48.0"},
    \\ "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": null, "group_size": 16,
    \\  "exclude_modules": ["lm_head", "model.layers.0.self_attn"]}}
;

test "the quantization block: ModelOpt NVFP4 in groups of 16 only" {
    const a = std.testing.allocator;
    try nvfp4.checkQuant(a, modelopt);
    const other = [_]struct { []const u8, []const u8 }{
        .{ "\"NVFP4\"", "\"FP8\"" },
        .{ "\"group_size\": 16", "\"group_size\": 32" },
        .{ "\"kv_cache_quant_algo\": null", "\"kv_cache_quant_algo\": \"FP8\"" },
        .{ "\"quantization\"", "\"quantisation\"" },
    };
    for (other) |o| {
        const changed = try std.mem.replaceOwned(u8, a, modelopt, o[0], o[1]);
        defer a.free(changed);
        try std.testing.expectError(error.UnsupportedHy4Quant, nvfp4.checkQuant(a, changed));
    }
    try std.testing.expectError(error.UnsupportedHy4Quant, nvfp4.checkQuant(a, "not json"));
}
