//! DeepSeek-V4.1 tokenizer throughput over the golden corpus, ids checked: `tf-dsv41-tokbench MODEL GOLDEN [passes]`.
const std = @import("std");
const serve = @import("dsv41_serve");

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: tf-dsv41-tokbench MODEL_DIR GOLDEN_DIR [passes]\n", .{});
        return 2;
    }
    const passes = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 3;
    var t0 = std.Io.Clock.awake.now(io);
    const tok = try serve.tokenizer.Tokenizer.load(gpa, io, args[1]);
    defer tok.deinit();
    const load_s = seconds(io, t0);
    const path = try std.fs.path.join(gpa, &.{ args[2], "tokenizer.golden" });
    defer gpa.free(path);
    const data = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(data);
    var texts: std.ArrayList([]const u8) = .empty;
    defer texts.deinit(gpa);
    var wants: std.ArrayList([]const u8) = .empty;
    defer wants.deinit(gpa);
    var at: usize = 0;
    var bytes: usize = 0;
    var tokens: usize = 0;
    while (at < data.len) {
        const n = std.mem.readInt(u32, data[at..][0..4], .little);
        try texts.append(gpa, data[at + 4 ..][0..n]);
        bytes += n;
        at += 4 + n;
        const k = std.mem.readInt(u32, data[at..][0..4], .little);
        try wants.append(gpa, data[at + 4 ..][0 .. 4 * k]);
        tokens += k;
        at += 4 + 4 * k;
    }
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    try out.ensureTotalCapacity(gpa, 1 << 20);
    std.debug.print("loaded tokenizer.json in {d:.2}s; corpus {d} docs, {d:.2} MB, {d} tokens\n", .{ load_s, texts.items.len, @as(f64, @floatFromInt(bytes)) / 1e6, tokens });
    for (0..passes) |pass| {
        const enc = try serve.tokenizer.Encoder.create(gpa, tok); // a cold cache each pass
        defer enc.destroy();
        var bad: usize = 0;
        t0 = std.Io.Clock.awake.now(io);
        for (texts.items, wants.items) |text, want| {
            out.clearRetainingCapacity();
            try enc.encode(gpa, text, &out);
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(out.items), want)) bad += 1;
        }
        const s = seconds(io, t0);
        std.debug.print("pass {d}: {d:.2} MB/s, {d:.2} M tok/s ({d:.3}s), {d} docs differ\n", .{ pass, @as(f64, @floatFromInt(bytes)) / s / 1e6, @as(f64, @floatFromInt(tokens)) / s / 1e6, s, bad });
        if (bad > 0) return 1;
    }
    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(gpa);
    t0 = std.Io.Clock.awake.now(io);
    for (wants.items) |want| {
        var d: serve.tokenizer.Detokenizer = .{ .tok = tok };
        text.clearRetainingCapacity();
        const ids: []align(1) const u32 = std.mem.bytesAsSlice(u32, want);
        for (ids) |id| try d.push(gpa, id, &text);
        try d.flush(gpa, &text);
    }
    const s = seconds(io, t0);
    std.debug.print("streamed decode: {d:.1} M tok/s\n", .{@as(f64, @floatFromInt(tokens)) / s / 1e6});
    return 0;
}

fn seconds(io: std.Io, since: std.Io.Timestamp) f64 {
    const now = std.Io.Clock.awake.now(io);
    return @as(f64, @floatFromInt(now.toNanoseconds() - since.toNanoseconds())) / 1e9;
}
