//! `hidden <model dir> <ids.npy> <out.npy> [--cut N|whole] [--layers]`: final-norm rows piece by piece, or residuals.

const std = @import("std");
const hip = @import("hip");
const qwen35 = @import("qwen3_5");
const ids_file = @import("ids_file.zig");

const Engine = qwen35.engine.Engine;

/// The rows of a prompt in pieces of `cut` (0: whole) through `Engine.spanRows`, into `out` in the activation's bits.
pub fn rows(e: *Engine, ids: []const u32, cut: usize, out: []u8) !void {
    const m = e.model();
    const width = m.spec.hidden * m.act.size();
    var caches = try e.newCaches(ids.len + 8);
    defer {
        e.drain();
        caches.deinit(e.gpa);
    }
    const step = if (cut == 0) ids.len else cut;
    var pos: usize = 0;
    while (pos < ids.len) {
        const end = @min(ids.len, pos + step);
        const span = try e.spanRows(&caches, ids, pos, end, null);
        try hip.runtime.check(e.driver.api.hipMemcpyDtoH(out[pos * width ..].ptr, @ptrFromInt(span.ptr), (end - pos) * width));
        pos = end;
    }
}

/// Each layer's residual rows, downloaded as the forward's trace hands them over.
const Layers = struct {
    e: *Engine,
    out: []u8,

    fn layer(ctx: *anyopaque, index: usize, x: hip.ops.Tensor, count: usize) anyerror!void {
        const l: *Layers = @ptrCast(@alignCast(ctx));
        try l.e.stream.synchronize();
        const n = count * l.e.model().spec.hidden * l.e.model().act.size();
        try hip.runtime.check(l.e.driver.api.hipMemcpyDtoH(l.out[index * n ..].ptr, @ptrFromInt(x.ptr), n));
    }
};

/// The whole prompt in one pass, every layer's residual rows (layers, rows, hidden) into `out`.
fn residuals(e: *Engine, ids: []const u32, out: []u8) !void {
    var caches = try e.newCaches(ids.len + 8);
    defer {
        e.drain();
        caches.deinit(e.gpa);
    }
    e.prompts.reset();
    var ids_dev = try hip.DeviceBuffer.fromHost(&e.driver, std.mem.sliceAsBytes(ids));
    defer ids_dev.free();
    var l: Layers = .{ .e = e, .out = out };
    _ = try qwen35.forward.span(e.ops(&e.prompts), e.model(), &caches, ids_dev.base(), ids.len, 0, .{ .ctx = &l, .layer = Layers.layer });
    try e.stream.synchronize();
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 3) return error.MissingArgument;
    var cut: usize = qwen35.prefix.chunk;
    var layers = false;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--layers")) {
            layers = true;
        } else if (std.mem.eql(u8, args[i], "--cut") and i + 1 < args.len) {
            i += 1;
            cut = if (std.mem.eql(u8, args[i], "whole")) 0 else try std.fmt.parseInt(usize, args[i], 10);
        } else return error.UnknownOption;
    }
    if (layers) cut = 0;
    const ids = try ids_file.load(gpa, io, args[1]);
    defer gpa.free(ids);
    if (ids.len == 0) return error.MissingArgument;
    const piece = if (cut == 0) ids.len else @min(cut, ids.len);
    const e = try Engine.open(gpa, io, args[0], .{ .capacity = ids.len + 64, .prompt_rows = piece, .streams = 1, .batch_rows = 32, .policy = (try @import("group.zig").resolve("", 0)).policy });
    defer e.deinit();
    const m = e.model();
    // numpy has no bf16: its bits as uint16, which the oracle widens
    const descr = if (m.act == .f16) "<f2" else "<u2";
    var header_buf: [128]u8 = undefined;
    const planes: usize = if (layers) m.spec.n_layers else 1;
    const dict = if (layers)
        try std.fmt.bufPrint(&header_buf, "{{'descr': '{s}', 'fortran_order': False, 'shape': ({d}, {d}, {d}), }}", .{ descr, planes, ids.len, m.spec.hidden })
    else
        try std.fmt.bufPrint(&header_buf, "{{'descr': '{s}', 'fortran_order': False, 'shape': ({d}, {d}), }}", .{ descr, ids.len, m.spec.hidden });
    const pad = 64 - (10 + dict.len + 1) % 64;
    const head = 10 + dict.len + pad + 1;
    const out = try gpa.alloc(u8, head + planes * ids.len * m.spec.hidden * m.act.size());
    defer gpa.free(out);
    @memcpy(out[0..8], "\x93NUMPY\x01\x00");
    std.mem.writeInt(u16, out[8..10], @intCast(dict.len + pad + 1), .little);
    @memcpy(out[10..][0..dict.len], dict);
    @memset(out[10 + dict.len ..][0..pad], ' ');
    out[head - 1] = '\n';
    if (layers) try residuals(e, ids, out[head..]) else try rows(e, ids, cut, out[head..]);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[2], .data = out });
    const what = if (layers) "layers' residual rows" else "final-norm rows";
    std.debug.print("wrote {d} {s} x {d} ({s}, pieces of {d}) to {s}\n", .{ ids.len, what, m.spec.hidden, descr, piece, args[2] });
}
