//! TF_DSV41_HANDOFF_CHECK=synth:TOKENS:FILE.mkvh2[+synth:...]: hand-off streams of synthetic finite rows, to time decode at any context.
const std = @import("std");
const tf = @import("tensorfold");
const ho = tf.deepseek_v41_handoff;
const log = std.log.scoped(.handoff_check);

/// The first token id of the synthetic prompt; ids walk a 9,000 id range, as the check tool's own prompts do.
const first_id: u32 = 1200;

/// `FILE`'s `<id>.tokens.json` sits beside it, as a producer's dump has it.
fn tokensPath(a: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "{s}.tokens.json", .{path[0 .. path.len - ho.file.extension.len]});
}

/// One buffer of random bf16 rows (normal, so finite) that every part's bytes alias: the cost of decode ignores values.
fn rows(a: std.mem.Allocator, len: usize) ![]u8 {
    const buf = try a.alloc(u8, len);
    var prng: std.Random.DefaultPrng = .init(7);
    const r = prng.random();
    for (0..len / 2) |i| {
        const bits: u32 = @bitCast(r.floatNorm(f32) * 0.5);
        std.mem.writeInt(u16, buf[2 * i ..][0..2], @intCast(bits >> 16), .little);
    }
    return buf;
}

/// Writes the stream of `n` positions and its tokens file; returns the bytes written.
fn write(gpa: std.mem.Allocator, io: std.Io, checkpoint: []const u8, n: u64, path: []const u8) !u64 {
    if (!std.mem.endsWith(u8, path, ho.file.extension)) return error.NotAStreamName;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const shape: ho.plan.Shape = .{ .upto = n };
    const specs = try ho.plan.specs(a, shape);
    var most: u64 = 0;
    for (specs) |s| most = @max(most, s.bytes());
    const shared = try rows(a, @intCast(most));
    const parts = try a.alloc(ho.export_cuda.Part, specs.len);
    for (specs, parts) |s, *p| p.* = .{ .spec = s, .bytes = shared[0..@intCast(s.bytes())] };
    const tokens = try a.alloc(u32, @intCast(n));
    for (tokens, 0..) |*t, i| t.* = @intCast(first_id + (i * 37) % 9000);
    const built = try tf.deepseek_v41_handoff_p2.layers(a, parts);
    const exp: tf.fabric.producer.Export = .{ .handoff = @splat(0x5a), .tokens = tokens, .first_token = 0, .tokens_per_row = 1, .layers = built.layers, .sources = built.sources };
    var json: std.Io.Writer.Allocating = .init(a);
    try std.json.Stringify.value(tokens, .{}, &json.writer);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try tokensPath(a, path), .data = json.written() });
    var f = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer f.close(io);
    var buf: [1 << 20]u8 = undefined;
    var w = f.writerStreaming(io, &buf);
    try ho.file.write(a, &w.interface, &exp, checkpoint, 0, 1);
    try w.interface.flush();
    const size = try f.length(io);
    log.info("synthetic hand-off of {d} positions, {d} parts, {d} bytes, written to {s}", .{ n, parts.len, size, path });
    return size;
}

/// Writes every stream of `spec` (specs joined by `+`) and returns true; false when `spec` is not a synth spec.
pub fn run(gpa: std.mem.Allocator, io: std.Io, checkpoint: []const u8, spec: []const u8) !bool {
    if (parse(spec) == null) return false;
    var it = std.mem.splitScalar(u8, spec, '+');
    while (it.next()) |one| {
        const s = parse(one) orelse return error.NotASynthSpec;
        _ = try write(gpa, io, checkpoint, s.n, s.path);
    }
    return true;
}

/// `synth:TOKENS:FILE` -> its two fields; null when `spec` is not one.
pub fn parse(spec: []const u8) ?struct { n: u64, path: []const u8 } {
    const rest = std.mem.cutPrefix(u8, spec, "synth:") orelse return null;
    const colon = std.mem.indexOfScalar(u8, rest, ':') orelse return null;
    const n = std.fmt.parseInt(u64, rest[0..colon], 10) catch return null;
    return .{ .n = n, .path = rest[colon + 1 ..] };
}

test "a synth spec names its tokens and file, and nothing else is one" {
    const s = parse("synth:262144:/tmp/x.mkvh2").?;
    try std.testing.expectEqual(@as(u64, 262144), s.n);
    try std.testing.expectEqualStrings("/tmp/x.mkvh2", s.path);
    try std.testing.expect(parse("roundtrip") == null and parse("synth:x:/y") == null and parse("synth:5") == null);
}
