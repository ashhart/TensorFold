//! Sliding Weights on CUDA: one step's milliseconds in each mode, for a tokenized example, with one block open.

const std = @import("std");
const nemotron = @import("nemotron");
const check = @import("check.zig");

const dims = nemotron.slide_dims;
const reps = 10;

/// MODEL with IDS_FILE's tokens (the answer from START): the median of ten steps in each mode.
pub fn run(gpu: check.Gpu, model: []const u8, ids_path: []const u8, start_text: []const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, ids_path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |w| try ids.append(gpa, try std.fmt.parseInt(u32, w, 10));
    const start = try std.fmt.parseInt(usize, start_text, 10);
    const e = try nemotron.Engine.init(gpa, io, gpu.ctx, model, null, .{ .context = 2048, .mtp = false, .graphs = false, .sampling = null, .segments = 1, .slide = true });
    defer e.deinit();
    const t = try nemotron.train.Trainer.init(gpa, e);
    defer t.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    for (t.sites.list) |*site| {
        for (site.a.slice(f32, dims.block * site.in)) |*v| v.* = (r.float(f32) - 0.5) * 0.02;
        site.gate(0).* = -std.math.inf(f32);
    }
    t.sites.rank = dims.block;
    t.attach(true);
    for ([_]nemotron.train.Mode{ .loss, .grad, .learn, .project, .seek }) |mode| {
        var ms: [reps]f64 = undefined;
        for (&ms) |*m| {
            const t0 = check.now(io);
            _ = try t.step(ids.items, start, mode);
            m.* = @as(f64, @floatFromInt(check.now(io) - t0)) / 1e6;
        }
        std.debug.print("RESULT {t} step, {d} rows: {d:.1} ms (median of {d})\n", .{ mode, ids.items.len - 1, check.median(&ms), reps });
    }
}
