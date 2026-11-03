//! A finite qualification capture from production forwards, for the same-weight FP64 scoring oracle.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const E = q.engine;
const C = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    e: *E.Engine,
    serial: usize = 0,
    fn write(c: *C, suffix: []const u8, bytes: []const u8) !void {
        const name = try std.fmt.allocPrint(c.gpa, "{s}/{d}-{s}", .{ c.dir, c.serial, suffix });
        defer c.gpa.free(name);
        try std.Io.Dir.cwd().writeFile(c.io, .{ .sub_path = name, .data = bytes });
    }
    fn projection(c: *C, phase: []const u8, layer: usize, name: []const u8, matrix: E.Proj) !void {
        const b = matrix.bf;
        const rows: usize = if (std.mem.eql(u8, phase, "prompt")) 2 else 1;
        const input = try c.gpa.alloc(u16, rows * b.k);
        defer c.gpa.free(input);
        for (input, 0..) |*x, i| {
            const value: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 37) % 97)) - 48)) / 32;
            x.* = @intCast(@as(u32, @bitCast(value)) >> 16);
        }
        var xd = try cuda.DeviceBuffer.fromHost(c.e.d, std.mem.sliceAsBytes(input));
        defer xd.free();
        var out = try cuda.DeviceBuffer.alloc(c.e.d, rows * b.n * 2);
        defer out.free();
        errdefer cuda.completion.drain(c.e.d, c.e.stream, "projection reference did not drain");
        try c.e.proj(matrix, xd.ptr, out.ptr, rows, std.mem.eql(u8, phase, "prompt"));
        try c.e.stream.synchronize();
        const result = try c.gpa.alloc(u8, rows * b.n * 2);
        defer c.gpa.free(result);
        try out.download(0, result);
        try c.write("hidden.bf16", std.mem.sliceAsBytes(input));
        try c.write("output.bf16", result);
        const text = try std.fmt.allocPrint(c.gpa, "{{\"phase\":\"{s}\",\"rows\":{d},\"hidden\":{d},\"vocab\":{d},\"projection\":\"model.language_model.layers.{d}.{s}.weight\"}}\n", .{ phase, rows, b.k, b.n, layer, name });
        defer c.gpa.free(text);
        try c.write("meta.json", text);
        c.serial += 1;
    }
    fn capture(c: *C, phase: []const u8, first: usize, picks: []const u32) !void {
        const e = c.e;
        errdefer cuda.completion.drain(e.d, e.stream, "reference capture did not drain");
        const b = e.bf16.?;
        const n = picks.len;
        const x = e.x + first * e.g.hidden * 2;
        @memcpy(b.picks.slice(u32)[0..n], picks);
        try b.kernels.denseF32(e.stream, x, e.head.bf.w, b.logits.ptr, n, e.g.vocab, e.g.hidden);
        try b.kernels.stats(e.stream, b.logits.ptr, e.g.vocab, try b.picks.device(), 20, try b.out.device(), n);
        try e.stream.synchronize();
        const h = try c.gpa.alloc(u8, n * e.g.hidden * 2);
        defer c.gpa.free(h);
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(h.ptr, x, h.len), "reference hidden rows");
        try c.write("hidden.bf16", h);
        const logits = try c.gpa.alloc(u8, n * e.g.vocab * 4);
        defer c.gpa.free(logits);
        try b.logits.download(0, logits);
        try c.write("logits.f32", logits);
        try c.write("stats.u32", b.out.bytes[0 .. n * 43 * 4]);
        const pick_json = try std.json.Stringify.valueAlloc(c.gpa, picks, .{});
        defer c.gpa.free(pick_json);
        const text = try std.fmt.allocPrint(c.gpa, "{{\"phase\":\"{s}\",\"rows\":{d},\"hidden\":{d},\"vocab\":{d},\"picks\":{s}}}\n", .{ phase, n, e.g.hidden, e.g.vocab, pick_json });
        defer c.gpa.free(text);
        try c.write("meta.json", text);
        c.serial += 1;
    }
};
pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) return error.Usage;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(1 << 22));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice([]const []const u32, a, text, .{});
    defer parsed.deinit();
    try std.Io.Dir.cwd().createDirPath(io, args[3]);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const e = try E.Engine.init(a, io, &ctx, args[1], false, null);
    defer e.deinit();
    if (!e.g.bf16) return error.ExpectedPlainBf16;
    var capture: C = .{ .gpa = a, .io = io, .dir = args[3], .e = e };
    for (parsed.value) |ids| {
        if (ids.len < 2 or ids.len > 65530) return error.InvalidFixture;
        var cache = try E.Cache.init(a, e, ids.len + 16);
        defer cache.deinit(a);
        var start: usize = 0;
        while (start < ids.len) {
            const end = @min(ids.len, start + E.max_rows);
            try e.forward(&.{.{ .cache = &cache, .rows = end - start }}, ids[start..end], .last);
            const scored = (end - start) - @intFromBool(end == ids.len);
            if (scored > 0) {
                const n = @min(scored, 2);
                try capture.capture("prompt", scored - n, ids[end - @intFromBool(end == ids.len) - n + 1 ..][0..n]);
            }
            start = end;
        }
        var next: u32 = 0;
        for (cache.logits, 0..) |word, id| if (@as(f32, @bitCast(@as(u32, word) << 16)) > @as(f32, @bitCast(@as(u32, cache.logits[next]) << 16))) {
            next = @intCast(id);
        };
        for (0..4) |_| {
            const draw = E.Draw{ .rule = null, .position = cache.len + 1 };
            try e.forward(&.{.{ .cache = &cache, .rows = 1 }}, &.{next}, .{ .draws = &.{draw} });
            next = e.tokens[0];
            try capture.capture("decode", 0, &.{next});
        }
    }
    for ([_][]const u8{ "prompt", "decode" }) |phase| {
        const delta = e.layers[0];
        const attn = e.layers[3];
        if (delta.mixer != .delta or attn.mixer != .attention) return error.ExpectedQwen27BLayerKinds;
        const mlp = delta.mlp.dense;
        try capture.projection(phase, 0, "mlp.gate_proj", mlp.gate);
        try capture.projection(phase, 0, "mlp.up_proj", mlp.up);
        try capture.projection(phase, 0, "mlp.down_proj", mlp.down);
        const d = delta.mixer.delta;
        inline for (.{ .{ "qkv", "in_proj_qkv" }, .{ "z", "in_proj_z" }, .{ "a", "in_proj_a" }, .{ "b", "in_proj_b" }, .{ "out", "out_proj" } }) |item|
            try capture.projection(phase, 0, "linear_attn." ++ item[1], @field(d, item[0]));
        const attention = attn.mixer.attention;
        inline for (.{ "q", "k", "v", "o" }) |name|
            try capture.projection(phase, 3, "self_attn." ++ name ++ "_proj", @field(attention, name));
    }
}
