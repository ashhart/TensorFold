//! Learned offset groups stay bit-exact through the real affine shard writer, reopening and another save.
const std = @import("std");
const t = std.testing;
const living = @import("living");
const st = @import("core/safetensors.zig");
const se = @import("core/shard_edit.zig");
const Shards = living.shards.Io(struct {
    pub const safetensors = st;
    pub const shard_edit = se;
});

fn check(comptime f: living.Family) !void {
    const gpa = t.allocator;
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    const path = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
    const d = f.quant.group;
    const own = d;
    const n = f.neurons * d;
    const base = try f.expertBase(try a.alloc(u8, 128), 0);
    const projs = try a.alloc([]const u8, f.names.inputs.len + 1);
    @memcpy(projs[0..f.names.inputs.len], f.names.inputs);
    projs[f.names.inputs.len] = f.names.output;
    var tensors: std.ArrayList(se.New) = .empty;
    const words = try a.alloc(u32, own * d / 8);
    const scales = try a.alloc(u16, own * d / d);
    const biases = try a.alloc(u16, scales.len);
    @memset(words, 0x76543210);
    @memset(scales, 0x3b80);
    @memset(biases, 0xbb80);
    for (projs) |proj| {
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}{s}.weight", .{ base, proj }), .dtype = .u32, .shape = &.{ own, d / 8 }, .bytes = std.mem.sliceAsBytes(words) });
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}{s}.scales", .{ base, proj }), .dtype = .bf16, .shape = &.{ own, 1 }, .bytes = std.mem.sliceAsBytes(scales) });
        try tensors.append(a, .{ .name = try std.fmt.allocPrint(a, "{s}{s}.biases", .{ base, proj }), .dtype = .bf16, .shape = &.{ own, 1 }, .bytes = std.mem.sliceAsBytes(biases) });
    }
    try se.create(gpa, path, &.{}, tensors.items);
    const config = try std.fmt.allocPrint(a, "{{\"{s}\":{d}}}", .{ f.keys.width, own });
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = if (f.keys.scope) |scope| try std.fmt.allocPrint(a, "{{\"{s}\":{s}}}", .{ scope, config }) else config });
    const down = try a.alloc(f32, n);
    var inputs: [f.names.inputs.len][]f32 = undefined;
    for (&inputs, 0..) |*x, p| {
        x.* = try a.alloc(f32, n);
        for (x.*, 0..) |*v, i| v.* = (-1499.9329 + 47.0843 * @as(f32, @floatFromInt(i % d)) / (d - 1)) * @as(f32, if (p == 0) 1 else -1);
    }
    for (down, 0..) |*v, i| v.* = 1499.9329 - 47.0843 * @as(f32, @floatFromInt(i / d % d)) / (d - 1);
    try living.pocket_round.layer(f, gpa, d, f.neurons, if (f.neuron.act == .swiglu) inputs[0] else &.{}, inputs[inputs.len - 1], down);
    var expected: [inputs.len][]f32 = undefined;
    for (&expected, inputs) |*x, v| x.* = try a.dupe(f32, v);
    const down0 = try a.dupe(f32, down);
    var has: living.Layers = .empty;
    has.set(0);
    const layout = living.Layout.fromExperts(d, own, 1, has);
    _ = try Shards.write(gpa, io, dir, f, layout, &.{0}, f.neurons, &inputs, down);
    for (inputs, expected) |x, want| try t.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(x));
    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(down0), std.mem.sliceAsBytes(down));
    {
        var file = try st.File.open(gpa, io, path);
        defer file.close(io);
        for (projs, 0..) |proj, p| {
            const w = file.get(try std.fmt.allocPrint(a, "{s}{s}.weight", .{ base, proj })).?;
            const s = file.get(try std.fmt.allocPrint(a, "{s}{s}.scales", .{ base, proj })).?;
            const b = file.get(try std.fmt.allocPrint(a, "{s}{s}.biases", .{ base, proj })).?;
            const input = p < inputs.len;
            const cols = if (input) d else own + f.neurons;
            const rows = if (input) own + f.neurons else d;
            const row = try a.alloc(f32, cols);
            const ws = try a.alloc(u32, cols / 8);
            const ss = try a.alloc(u16, cols / d);
            const bs = try a.alloc(u16, cols / d);
            for (0..rows) |r| {
                @memcpy(std.mem.sliceAsBytes(ws), w.bytes[r * cols / 2 ..][0 .. cols / 2]);
                @memcpy(std.mem.sliceAsBytes(ss), s.bytes[r * cols / d * 2 ..][0 .. cols / d * 2]);
                @memcpy(std.mem.sliceAsBytes(bs), b.bytes[r * cols / d * 2 ..][0 .. cols / d * 2]);
                // The original expert's every word and BF16 pair is preserved.
                if (!input or r < own) {
                    const count = if (input) cols / 8 else own / 8;
                    try t.expectEqualSlices(u32, words[0..count], ws[0..count]);
                    const groups = if (input) cols / d else own / d;
                    try t.expectEqualSlices(u16, scales[0..groups], ss[0..groups]);
                    try t.expectEqualSlices(u16, biases[0..groups], bs[0..groups]);
                }
                living.affine.dequantize(f.quant, ws, ss, bs, row);
                if (input and r >= own) {
                    try t.expectEqualSlices(f32, expected[p][(r - own) * d ..][0..d], row);
                } else if (!input) for (row[own..], 0..) |v, k| try t.expectEqual(down0[k * d + r], v);
            }
        }
    }
    const first = try tmp.dir.readFileAlloc(io, "model.safetensors", a, .limited(1 << 22));
    _ = try Shards.write(gpa, io, dir, f, layout, &.{0}, f.neurons, &inputs, down);
    const again = try tmp.dir.readFileAlloc(io, "model.safetensors", a, .limited(1 << 22));
    try t.expectEqualSlices(u8, first, again);
}

test "learned fixed point: every descriptor's offset groups survive save reopen and save with original bytes intact" {
    inline for (living.families.all) |f| try check(f);
}
