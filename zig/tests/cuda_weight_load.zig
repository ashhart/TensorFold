//! Loads the actual family once; timed receipts and final device-weight digests are separate processes.
const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35");
const nemo = @import("nemotron");
const digest = @import("cuda_weight_digest.zig");
const Span = digest.Span;

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 5) return error.ExpectedFamilyModelModeOutput;
    const family = args[1];
    const q = std.mem.eql(u8, family, "qwen");
    if (!q and !std.mem.eql(u8, family, "nemotron")) return error.BadFamily;
    const mode = args[3];
    const hashing = std.mem.eql(u8, mode, "hash") or std.mem.eql(u8, mode, "control");
    if (!hashing and !std.mem.eql(u8, mode, "time")) return error.BadMode;
    const output = try std.Io.Dir.cwd().createFile(init.io, args[4], .{ .exclusive = true });
    defer output.close(init.io);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const integrated = try ctx.attribute(.integrated) != 0;
    if (!integrated) return error.RequiresIntegratedGpu;
    const capability = try driver.capability(0);
    if (capability != 121) return error.RequiresGb10;
    _ = cuda.usage(true);
    const start = std.Io.Clock.awake.now(init.io).toNanoseconds();
    var qe: ?*qwen.engine.Engine = null;
    var ne: ?*nemo.Engine = null;
    defer if (qe) |e| e.deinit();
    defer if (ne) |e| e.deinit();
    if (q) qe = try qwen.engine.Engine.init(init.gpa, init.io, &ctx, args[2], false, null) else ne = try nemo.Engine.init(init.gpa, init.io, &ctx, args[2], null, .{ .context = 4096, .mtp = false, .graphs = false });
    if (qe) |e| if (e.g.kinds.modelopt or e.g.exl3 or e.g.moe() or e.g.pocket > 0) return error.ReceiptNeedsDenseMlx;
    const elapsed = std.Io.Clock.awake.now(init.io).toNanoseconds() - start;
    const load_s = if (qe) |e| @as(f64, @floatFromInt(e.startup.load)) / 1e9 else ne.?.load_seconds;
    const memory = cuda.usage(false); // measured before any readback allocation or control write
    const hwm = try peakRss(init.io, a);
    const spans = if (qe) |e| blk: {
        const out = try a.alloc(Span, 1);
        out[0] = .{ .name = "qwen.weights", .ptr = e.arena.ptr, .len = e.weight_bytes };
        break :blk out;
    } else blk: {
        const out = try a.alloc(Span, ne.?.w.named.items.len);
        for (out, ne.?.w.named.items) |*o, n| o.* = .{ .name = n.name, .ptr = n.ptr, .len = n.len };
        break :blk out;
    };
    var bytes: usize = 0;
    for (spans) |s| bytes = try std.math.add(usize, bytes, s.len);
    const weights = if (hashing) try digest.collect(a, &driver, if (qe) |e| e.stream else ne.?.stream, spans, std.mem.eql(u8, mode, "control")) else null;
    const json = try std.json.Stringify.valueAlloc(a, .{
        .schema = "cuda-weight-load-v1",
        .family = family,
        .mode = mode,
        .integrated = integrated,
        .sm = capability,
        .initialization_seconds = @as(f64, @floatFromInt(elapsed)) / 1e9,
        .load_seconds = load_s,
        .memory_before_hash = memory,
        .peak_rss_bytes_before_hash = hwm,
        .weight_spans = spans.len,
        .weight_bytes = bytes,
        .weights = weights,
        .hash_readback = "pinned asynchronous DtoH followed by stream completion",
        .nemotron_context = if (q) @as(?u32, null) else 4096,
        .mtp = false,
    }, .{ .whitespace = .indent_2 });
    try output.writeStreamingAll(init.io, json);
}
fn peakRss(io: std.Io, a: std.mem.Allocator) !?u64 {
    if (@import("builtin").os.tag != .linux) return null;
    const file = try std.Io.Dir.cwd().openFile(io, "/proc/self/status", .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var reader = file.readerStreaming(io, &buf);
    const text = try reader.interface.allocRemaining(a, .limited(1 << 20));
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| if (std.mem.startsWith(u8, line, "VmHWM:")) {
        var words = std.mem.tokenizeAny(u8, line[6..], " \t");
        return try std.math.mul(u64, try std.fmt.parseInt(u64, words.next() orelse return error.BadVmHwm, 10), 1024);
    };
    return null;
}
