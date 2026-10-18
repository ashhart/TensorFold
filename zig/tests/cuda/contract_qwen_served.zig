//! Served MoE and MTP functions against FP64, with guarded buffers and planted numerical failures.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const ref = @import("contract_qwen_served_ref.zig");
const rig_ = @import("contract_rig.zig");
pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var moe = try cuda.Module.load(gpu.d, cuda.kernels.moe);
    defer moe.unload();
    var mtp = try cuda.Module.load(gpu.d, cuda.kernels.qwen36_mtp);
    defer mtp.unload();
    const fns = try cuda.moe.Fns.resolve(moe);
    for (std.enums.values(ref.Mode)) |mode| {
        const f = try (if (@backingInt(mode) < 3) moe else mtp).function(ref.symbol(mode));
        for ([_]usize{ 1, 15, 16, 17, 64, 128 }) |rows| for ([_]bool{ false, true }) |variant| {
            if (mode == .tree and rows > 64) continue;
            var arena = std.heap.ArenaAllocator.init(gpu.gpa);
            defer arena.deinit();
            const a = arena.allocator();
            const fixture = try ref.make(a, mode, rows, variant);
            var rig = rig_.Rig{ .gpu = gpu, .s = s };
            defer rig.deinit();
            const pointers = try a.alloc(u64, fixture.buffers.items.len);
            for (fixture.buffers.items, pointers) |b, *p| p.* = if (b.output) try rig.output(b.bytes.len) else try rig.input(b.bytes);
            const before = try a.dupe(u8, try rig.inputBytes());
            var args: cuda.Args = .{};
            for (fixture.args.items) |arg| switch (arg) {
                .buffer => |i| args.add(pointers[i]),
                .int => |v| args.add(v),
                .long => |v| args.add(v),
                .float => |v| args.add(v),
            };
            var first: ?[]const u8 = null;
            for (0..5) |_| {
                switch (mode) {
                    .router => try fns.route(s, pointers[0], 2048 + @as(usize, if (variant) 16 else 0), rows, pointers[1], pointers[2], 2048, 257),
                    .topk => try fns.pick(s, pointers[0], pointers[1], pointers[2], rows, 256, 8),
                    .combine => try fns.combine(s, pointers[0], variant, pointers[1], pointers[2], rows, 9, 2048),
                    else => try cuda.launch.launch(f, .{ .grid = .{ .x = fixture.grid[0], .y = fixture.grid[1], .z = fixture.grid[2] }, .block = .{ .x = fixture.threads } }, s, &args),
                }
                try s.synchronize();
                for (fixture.results.items) |result| {
                    const bytes = try rig.bytes(pointers[result.buffer], fixture.buffers.items[result.buffer].bytes.len);
                    try check.expect(result.accepts(bytes), "{s} FP64 mismatch at rows {d}, variant {}", .{ ref.symbol(mode), rows, variant });
                    const corrupt = try a.dupe(u8, bytes);
                    const width: usize = if (result.elem == .bf16) 2 else 4;
                    @memset(corrupt[0..width], 0x40);
                    try check.expect(!result.accepts(corrupt), "{s} missed planted finite corruption", .{ref.symbol(mode)});
                }
                const output = try rig.outputBytes();
                if (first) |bytes| try check.expect(std.mem.eql(u8, bytes, output), "{s} nondeterministic bytes", .{ref.symbol(mode)}) else first = try a.dupe(u8, output);
                try check.expect(try rig.intact() and !rig.overlap() and std.mem.eql(u8, before, try rig.inputBytes()), "{s} changed an input or guard", .{ref.symbol(mode)});
            }
        };
        std.debug.print("PASS {s}: served geometry, FP64, five repeats, guards, inputs; finite corruption RED\n", .{ref.symbol(mode)});
    }
}
