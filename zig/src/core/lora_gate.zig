//! The check of the MLX quantizer against shipped tensors: each dense weight quantized and compared word for word.
const std = @import("std");
const lm = @import("lora_quant.zig");
const lp = @import("lora_shards.zig");
const Allocator = std.mem.Allocator;
const Io = std.Io;

/// The quantizer's output on each module's dense weight against the shipped words, scales and biases.
pub fn quantizerCheck(gpa: Allocator, arena: Allocator, io: Io, w: *Io.Writer, dense_dir: []const u8, shipped_dir: []const u8, modules: []const []const u8) !bool {
    var dense = try lp.Shards.init(gpa, arena, io, w, dense_dir);
    defer dense.deinit();
    var shipped = try lp.Shards.init(gpa, arena, io, w, shipped_dir);
    defer shipped.deinit();
    const quant = try lp.quantConfig(arena, io, w, shipped_dir);
    var ok = true;
    for (modules) |module| {
        const lay = try lp.layoutOf(arena, w, &shipped, quant, module);
        const j: lp.Job = .{ .module = module, .a = "", .b = "", .n = lay.n, .k = lay.k, .group = lay.group, .dtype = lay.dtype };
        const x = try lp.denseOf(gpa, w, &dense, j);
        defer gpa.free(x);
        const old = try lp.quantOf(gpa, &shipped, j);
        defer lp.free(gpa, old);
        const new = try lm.allocQuant(gpa, j.n, j.k, j.group);
        defer lm.freeQuant(gpa, new);
        if (try lm.quantize(j.group, x, new)) |bad| return lp.refuse(w, module, try std.fmt.allocPrint(arena, "group {d}: a non-finite value, range or scale", .{bad}));
        const a = lm.agreement(j.group, lm.viewOf(old), lm.viewOf(new));
        try w.print("{s} [{d} x {d}, group {d}]: scales {d:.4}%, biases {d:.4}%, codes {d:.4}%, whole groups {d:.4}%: {s}\n", .{ module, j.n, j.k, j.group, a.scales * 100, a.biases * 100, a.codes * 100, a.groups * 100, if (a.identical()) "identical" else "DIFFERS" });
        ok = ok and a.identical();
    }
    try w.print("{s}\n", .{if (ok) "quantizer check: byte-identical on every tensor" else "quantizer check: FAILED"});
    return ok;
}
