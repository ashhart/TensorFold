//! Native attention on pinned independent HIP SDPA fixtures; no libtorch is loaded by this program.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const sys = @import("sys");
const attn = hip.attention_aotriton;
const a = std.testing.allocator;

fn read(dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(16 << 20));
}

test "HIP AOTriton device: vision, grouped causal decoder and decode match independent BF16 outputs" {
    const dir = sys.getenv("TF_GLM_OCR_ATTENTION_FIXTURES") orelse return error.MissingFixtures;
    const source = try read(dir, "cases.json");
    defer a.free(source);
    const Case = struct { name: []const u8, heads: usize, kv_heads: usize, rows: usize, keys: usize, dim: usize, causal: bool };
    const cases = try std.json.parseFromSlice([]const Case, a, source, .{});
    defer cases.deinit();
    var r = try hip.Runtime.open();
    defer r.close();
    if (try r.deviceCount() != 1) return error.SelectOneHipDevice;
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try hip.Stream.init(&r);
    defer stream.deinit();
    var library = try attn.Library.open(stream);
    defer library.close(stream);
    for (cases.value) |cell| {
        var names: [4][128]u8 = undefined;
        var files: [4][]u8 = undefined;
        var count: usize = 0;
        defer for (files[0..count]) |bytes| a.free(bytes);
        for ([_][]const u8{ "q", "k", "v", "out" }, 0..) |part, i| {
            files[i] = try read(dir, try std.fmt.bufPrint(&names[i], "{s}-{s}.npy", .{ cell.name, part }));
            count += 1;
        }
        var arrays: [4]core.npy.Array = undefined;
        for (&arrays, files) |*array, bytes| array.* = try core.npy.parse(bytes);
        var q = try hip.DeviceBuffer.fromHost(&r, arrays[0].data);
        defer q.free();
        var k = try hip.DeviceBuffer.fromHost(&r, arrays[1].data);
        defer k.free();
        var v = try hip.DeviceBuffer.fromHost(&r, arrays[2].data);
        defer v.free();
        var out = try hip.DeviceBuffer.alloc(&r, arrays[3].data.len);
        defer out.free();
        var lse = try hip.DeviceBuffer.alloc(&r, cell.heads * cell.rows * 4);
        defer lse.free();
        var counter = try hip.DeviceBuffer.alloc(&r, 4);
        defer counter.free();
        const got = try a.alloc(u8, out.len);
        defer a.free(got);
        for (0..5) |_| {
            try library.forward(stream, attn.rows(q.base(), cell.heads, cell.rows, cell.dim), attn.rows(k.base(), cell.kv_heads, cell.keys, cell.dim), attn.rows(v.base(), cell.kv_heads, cell.keys, cell.dim), attn.rows(out.base(), cell.heads, cell.rows, cell.dim), lse.base(), counter.base(), cell.causal);
            try stream.synchronize();
            try out.download(0, got);
            var differences: usize = 0;
            for (std.mem.bytesAsSlice(u16, got), std.mem.bytesAsSlice(u16, arrays[3].data)) |x, y| {
                if (x != y) differences += 1;
            }
            std.debug.print("AOTriton {s}: {d} BF16 differences\n", .{ cell.name, differences });
            try std.testing.expectEqual(@as(usize, 0), differences);
        }
    }
}
