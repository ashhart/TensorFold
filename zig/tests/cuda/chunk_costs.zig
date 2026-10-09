//! chunk-costs: what r rows cost, as the decode window graph (r <= 16) and as a prompt chunk behind a copy gate (any r), fastest of 7.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check = @import("check.zig");

const reps = 7;
const gate_bytes = 256 << 20;

/// MODEL at a 16k window from the prompt in IDS_FILE: each width's window graph and prompt-chunk ms.
pub fn run(gpu: check.Gpu, model: []const u8, ids_path: []const u8, widths_text: []const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, ids_path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |t| try ids.append(gpa, try std.fmt.parseInt(u32, t, 10));
    var widths: std.ArrayList(usize) = .empty;
    defer widths.deinit(gpa);
    var wt = std.mem.tokenizeAny(u8, widths_text, ", ");
    while (wt.next()) |t| try widths.append(gpa, try std.fmt.parseInt(usize, t, 10));
    var most: usize = 0;
    for (widths.items) |w| most = @max(most, w);
    if (ids.items.len < most + 2) return error.PromptTooShort;
    const e = try nemotron.Engine.init(gpa, io, gpu.ctx, model, null, .{ .context = 16384, .mtp = true, .graphs = true, .sampling = null, .segments = 1 });
    defer e.deinit();
    const prompt = ids.items[0 .. ids.items.len - most - 1];
    const cont = ids.items[prompt.len..];
    _ = try e.prefill(prompt, null, null);
    var saved = try e.b.snapshot(e.ops());
    defer saved.free();
    var gate = try cuda.DeviceBuffer.alloc(e.ctx.d, gate_bytes);
    defer gate.free();
    var a = try cuda.Event.init(e.ctx.d, true);
    defer a.deinit();
    var b = try cuda.Event.init(e.ctx.d, true);
    defer b.deinit();
    const host = try gpa.alloc(u32, most);
    defer gpa.free(host);
    for (widths.items) |r| {
        var window: f64 = std.math.inf(f64);
        var chunk: f64 = std.math.inf(f64);
        for (0..reps + 1) |i| {
            if (r <= nemotron.state.max_rows) {
                try restore(e, saved, prompt.len);
                _ = try e.step(cont[0], null);
                try e.stream.synchronize();
                try a.record(e.stream);
                try e.verify(cont[1..][0..r], r, null);
                try b.record(e.stream);
                try b.synchronize();
                if (i > 0) window = @min(window, try cuda.Event.elapsedMs(a, b));
            }
            try restore(e, saved, prompt.len);
            @memcpy(host[0..r], cont[0..r]);
            try e.ops().upload(e.b.p_ids, std.mem.sliceAsBytes(host[0..r]));
            for (0..32) |_| try e.ops().copy(gate.ptr + gate_bytes / 2, gate.ptr, gate_bytes / 2);
            try a.record(e.stream);
            try e.forward(null).chunk(r, prompt.len);
            try b.record(e.stream);
            try b.synchronize();
            if (i > 0) chunk = @min(chunk, try cuda.Event.elapsedMs(a, b));
        }
        if (window == std.math.inf(f64)) {
            std.debug.print("WIDTH {d}: chunk {d:.3} ms\n", .{ r, chunk });
        } else std.debug.print("WIDTH {d}: window graph {d:.3} ms, chunk {d:.3} ms\n", .{ r, window, chunk });
    }
    try e.reset();
}

fn restore(e: *nemotron.Engine, saved: cuda.DeviceBuffer, pos: usize) !void {
    try e.b.restore(e.ops(), saved);
    e.pos = pos;
    e.parity = 0;
    e.prev_keep = 0;
}
