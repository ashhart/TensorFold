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
                    .router, .topk, .combine => try served(fns, s, mode, pointers, rows, variant, 0),
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
            if (@backingInt(mode) < 3) for ([_]usize{ 1, rows / 2 + 1, rows - 1 }) |live| if (live < rows) try liveRows(gpu, s, fns, mode, fixture, rows, variant, live);
        };
        std.debug.print("PASS {s}: served geometry, FP64, five repeats, guards, inputs; finite corruption RED\n", .{ref.symbol(mode)});
    }
    try planRows(gpu, s);
}

fn served(fns: cuda.moe.Fns, s: cuda.Stream, mode: ref.Mode, p: []const u64, rows: usize, variant: bool, params: u64) !void {
    switch (mode) {
        .router => try fns.route(s, p[0], 2048 + @as(usize, if (variant) 16 else 0), rows, p[1], p[2], 2048, 257, params),
        .topk => try fns.pick(s, p[0], p[1], p[2], rows, 256, 8, params),
        .combine => try fns.combine(s, p[0], variant, p[1], p[2], rows, 9, 2048, params),
        else => unreachable,
    }
}

/// A bucket of `rows` with params word 2 at `live`: the live rows meet FP64, the padding rows' outputs stay unwritten.
fn liveRows(gpu: check.Gpu, s: cuda.Stream, fns: cuda.moe.Fns, mode: ref.Mode, fixture: ref.Fixture, rows: usize, variant: bool, live: usize) !void {
    var rig = rig_.Rig{ .gpu = gpu, .s = s };
    defer rig.deinit();
    var pointers: [3]u64 = undefined;
    for (fixture.buffers.items, &pointers) |b, *p| p.* = if (b.output) try rig.output(b.bytes.len) else try rig.input(b.bytes);
    const params = [8]i32{ 0, 0, @intCast(live), 1, 1, 0, 0, 0 };
    try served(fns, s, mode, &pointers, rows, variant, try rig.input(std.mem.sliceAsBytes(&params)));
    try s.synchronize();
    for (fixture.results.items) |result| {
        const bytes = try rig.bytes(pointers[result.buffer], fixture.buffers.items[result.buffer].bytes.len);
        const per = result.want.len / rows;
        const width: usize = if (result.elem == .bf16) 2 else 4;
        const head: ref.Result = .{ .buffer = result.buffer, .elem = result.elem, .want = result.want[0 .. live * per], .bound = result.bound[0 .. live * per] };
        try check.expect(head.accepts(bytes[0 .. live * per * width]), "{s} FP64 mismatch at {d} live rows of {d}", .{ ref.symbol(mode), live, rows });
        for (bytes[live * per * width ..]) |byte| try check.expect(byte == 0xff, "{s} wrote a padding row ({d} live of {d})", .{ ref.symbol(mode), live, rows });
    }
    try check.expect(try rig.intact() and !rig.overlap(), "{s} broke a guard at {d} live rows", .{ ref.symbol(mode), live });
}

/// Live-row plans equal plans over the live pairs, including buckets past the 1,024-pair single-block limit.
fn planRows(gpu: check.Gpu, s: cuda.Stream) !void {
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try cuda.grouped.Router.resolve(mod);
    const per = 9;
    const count = 257;
    for ([_]usize{ 1, 16, 32, 64, 114, 128 }) |bucket| for (1..bucket + 1) |live| {
        var arena = std.heap.ArenaAllocator.init(gpu.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const pairs = bucket * per;
        const picks = try a.alloc(i32, pairs);
        for (picks, 0..) |*v, i| v.* = if (i % per == per - 1) 256 else @intCast((i * 37 + i / per * 11) % 256);
        const params = [8]i32{ 0, 0, @intCast(live), 1, 1, 0, 0, 0 };
        const items = 3 * cuda.grouped.maxItems(pairs, count, 16) * 4;
        var outs: [2][]const u8 = undefined;
        for (0..2) |k| {
            var rig = rig_.Rig{ .gpu = gpu, .s = s };
            defer rig.deinit();
            const dp = try rig.input(std.mem.sliceAsBytes(picks));
            const dlive = try rig.input(std.mem.sliceAsBytes(&params)) + 8;
            const plan: cuda.grouped.Plan = .{ .members = try rig.output(pairs * 4), .items = try rig.output(items), .counts = try rig.output(8), .rank = try rig.scratch(pairs * 4), .hist = try rig.scratch((pairs + 1023) / 1024 * count * 4) };
            var first: ?[]const u8 = null;
            for (0..5) |_| {
                if (k == 0) try router.routeRows(s, dp, pairs, count, 16, plan, dlive, per) else try router.route(s, dp, live * per, count, 16, plan);
                try s.synchronize();
                const out = try rig.outputBytes();
                if (first) |b| try check.expect(std.mem.eql(u8, b, out), "routeRows nondeterministic bytes", .{}) else first = try a.dupe(u8, out);
            }
            try check.expect(try rig.intact() and !rig.overlap(), "routeRows broke a guard", .{});
            outs[k] = first.?;
        }
        try check.expect(std.mem.eql(u8, outs[0], outs[1]), "routeRows at {d} live rows of {d} differs from route over their pairs", .{ live, bucket });
    };
    std.debug.print("PASS plan_rows: buckets 1, 16, 32, 64, 114, 128 at every live count (wide past 1,024 pairs) equal the plan over the live pairs, five repeats, guards\n", .{});
}
