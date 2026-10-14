//! `logits <model dir> <ids.npy> <out.npy> [--f32] [--decode]`: every row's logits for tools/truth/score.py to score.

const std = @import("std");
const hip = @import("hip");
const qwen35 = @import("qwen3_5");
const ids_file = @import("ids_file.zig");
const Kind = @import("logit_kind.zig").Kind;

const Engine = qwen35.engine.Engine;

/// Rows a head projection covers at once.
pub const chunk = 32;

pub const Mode = struct { decode: bool = false, wide: bool = false };

/// Takes each block of rows as it is read: rows `first..first + count` of `vocab` logits, `bytes` in the kind's layout.
pub const Sink = struct {
    ctx: *anyopaque,
    put: *const fn (ctx: *anyopaque, first: usize, count: usize, bytes: []const u8) anyerror!void,
};

/// The head's width in logits (the padded vocabulary).
pub fn vocab(e: *const Engine) usize {
    return e.model().spec.vocab;
}

pub fn kindOf(e: *const Engine, mode: Mode) Kind {
    return if (mode.wide) .f32 else if (e.model().act == .f16) .f16 else .bf16;
}

/// Every row's logits of `ids` into `sink`, in blocks of up to `chunk` rows.
pub fn stream(gpa: std.mem.Allocator, e: *Engine, ids: []const u32, mode: Mode, sink: Sink) !void {
    const m = e.model();
    var caches = try e.newCaches(ids.len + 8);
    defer {
        e.drain();
        caches.deinit(gpa);
    }
    e.prompts.reset();
    const o: hip.ops.Ops = e.ops(&e.prompts);
    var ids_dev = try hip.DeviceBuffer.fromHost(&e.driver, std.mem.sliceAsBytes(ids));
    defer ids_dev.free();
    const hidden = try qwen35.forward.span(o, m, &caches, ids_dev.base(), if (mode.decode) 1 else ids.len, 0, null);

    const width = vocab(e) * kindOf(e, mode).size();
    const block = try gpa.alloc(u8, chunk * width);
    defer gpa.free(block);
    if (mode.decode) {
        // row 0 from its one-row prefill, then each token a one-row round at its slot, kept before the next
        const first = try o.project(hidden, m.head, 1, mode.wide);
        try e.stream.synchronize();
        try hip.runtime.check(e.driver.api.hipMemcpyDtoH(block.ptr, @ptrFromInt(first.ptr), width));
        for (1..ids.len) |i| {
            const rows = [1]Engine.Rows{.{ .caches = &caches, .pos = i, .tokens = ids[i..][0..1] }};
            const reqs = [1]qwen35.draw.Request{.{ .sampling = null, .position = i + 1 }};
            var drawn: [1]u32 = undefined;
            const r = try e.verify(&rows, &reqs, &drawn);
            const round: hip.ops.Ops = e.ops(&e.rounds);
            const logits = try round.project(r.hidden, m.head, 1, mode.wide);
            try e.stream.synchronize();
            try hip.runtime.check(e.driver.api.hipMemcpyDtoH(block[(i % chunk) * width ..].ptr, @ptrFromInt(logits.ptr), width));
            e.keep(0, 1);
            try e.flush();
            if (i % chunk == chunk - 1) try sink.put(sink.ctx, i + 1 - chunk, chunk, block);
        }
        const tail = ids.len % chunk;
        if (tail > 0) try sink.put(sink.ctx, ids.len - tail, tail, block[0 .. tail * width]);
        return;
    }
    var row: usize = 0;
    while (row < ids.len) : (row += chunk) {
        const rows = @min(chunk, ids.len - row);
        const at = e.prompts.mark();
        const logits = try o.project(qwen35.forward.at(hidden, row * m.spec.hidden), m.head, rows, mode.wide);
        try e.stream.synchronize();
        try hip.runtime.check(e.driver.api.hipMemcpyDtoH(block.ptr, @ptrFromInt(logits.ptr), rows * width));
        e.prompts.release(at);
        try sink.put(sink.ctx, row, rows, block[0 .. rows * width]);
    }
}

/// The rows of a whole file in memory, written as a .npy at the end.
const File = struct {
    body: []u8,
    width: usize,

    fn put(ctx: *anyopaque, first: usize, _: usize, bytes: []const u8) anyerror!void {
        const f: *File = @ptrCast(@alignCast(ctx));
        @memcpy(f.body[first * f.width ..][0..bytes.len], bytes);
    }
};

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 3) return error.MissingArgument;
    var mode: Mode = .{};
    for (args[3..]) |a| {
        if (std.mem.eql(u8, a, "--f32")) mode.wide = true else if (std.mem.eql(u8, a, "--decode")) mode.decode = true else return error.UnknownOption;
    }
    const ids = try ids_file.load(gpa, io, args[1]);
    defer gpa.free(ids);
    const e = try Engine.open(gpa, io, args[0], .{ .capacity = ids.len + 64, .prompt_rows = ids.len + 64, .streams = 1, .batch_rows = 32, .policy = (try @import("group.zig").resolve("", 0)).policy });
    defer e.deinit();
    const kind = kindOf(e, mode);
    var header_buf: [128]u8 = undefined;
    const dict = try std.fmt.bufPrint(&header_buf, "{{'descr': '{s}', 'fortran_order': False, 'shape': ({d}, {d}), }}", .{ kind.descr(), ids.len, vocab(e) });
    const pad = 64 - (10 + dict.len + 1) % 64;
    const head = 10 + dict.len + pad + 1;
    const width = vocab(e) * kind.size();
    const out = try gpa.alloc(u8, head + ids.len * width);
    defer gpa.free(out);
    @memcpy(out[0..8], "\x93NUMPY\x01\x00");
    std.mem.writeInt(u16, out[8..10], @intCast(dict.len + pad + 1), .little);
    @memcpy(out[10..][0..dict.len], dict);
    @memset(out[10 + dict.len ..][0..pad], ' ');
    out[head - 1] = '\n';
    var file: File = .{ .body = out[head..], .width = width };
    try stream(gpa, e, ids, mode, .{ .ctx = &file, .put = File.put });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[2], .data = out });
    std.debug.print("wrote {d}{s} rows x {d} logits to {s}\n", .{ ids.len, if (mode.decode) " decoded" else "", vocab(e), args[2] });
}
