//! Finite production-CUDA residual capture; no alternate forward or arithmetic.
const std = @import("std");
const cuda = @import("cuda");
const E = @import("qwen35").engine;
const Capture = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    e: *E.Engine,
    next: usize = 0,
    fn boundary(ctx: *anyopaque, index: usize, ptr: u64, rows: usize) !void {
        const c: *@This() = @ptrCast(@alignCast(ctx));
        if (index != c.next) return error.MissingTraceBoundary;
        try c.e.stream.synchronize();
        const bytes = try c.gpa.alloc(u8, rows * c.e.g.hidden * 2);
        defer c.gpa.free(bytes);
        try c.e.d.check(c.e.d.api.cuMemcpyDtoH_v2(bytes.ptr, ptr, bytes.len), "hidden reference");
        const path = try std.fmt.allocPrint(c.gpa, "{s}/{d}.bf16", .{ c.dir, index });
        defer c.gpa.free(path);
        try std.Io.Dir.cwd().writeFile(c.io, .{ .sub_path = path, .data = bytes });
        c.next += 1;
    }
};

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) return error.Usage;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(1 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice([]const u32, a, text, .{});
    defer parsed.deinit();
    const ids = parsed.value;
    if (ids.len < 2 or ids.len > E.max_rows) return error.InvalidFixture;
    try std.Io.Dir.cwd().createDirPath(io, args[3]);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const e = try E.Engine.init(a, io, &ctx, args[1], false, null);
    defer e.deinit();
    if (!e.g.bf16) return error.ExpectedPlainBf16;
    var cache = try E.Cache.init(a, e, ids.len + 16);
    defer cache.deinit(a);
    var capture: Capture = .{ .gpa = a, .io = io, .dir = args[3], .e = e };
    e.hidden_trace = .{ .ctx = &capture, .capture = Capture.boundary };
    defer e.hidden_trace = null;
    errdefer cuda.completion.drain(e.d, e.stream, "hidden trace did not drain");
    try e.forward(&.{.{ .cache = &cache, .rows = ids.len }}, ids, .last);
    if (capture.next != e.g.layers + 2) return error.IncompleteTrace;
    const logits = try a.dupe(u16, cache.logits);
    defer a.free(logits);
    e.hidden_trace = null;
    var plain = try E.Cache.init(a, e, ids.len + 16);
    defer plain.deinit(a);
    try e.forward(&.{.{ .cache = &plain, .rows = ids.len }}, ids, .last);
    if (!std.mem.eql(u16, logits, plain.logits)) return error.TraceChangedLogits;
    const meta = try std.fmt.allocPrint(a, "{{\"rows\":{d},\"hidden\":{d},\"layers\":{d},\"boundaries\":{d},\"trace_plain_logits_equal\":true}}\n", .{ ids.len, e.g.hidden, e.g.layers, capture.next });
    defer a.free(meta);
    const path = try std.fmt.allocPrint(a, "{s}/meta.json", .{args[3]});
    defer a.free(path);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = meta });
}
