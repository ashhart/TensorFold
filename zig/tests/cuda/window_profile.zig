//! window-profile: a decode window's GPU ms by kernel class, launched eagerly behind a gate so launches never starve it, beside its graph.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check = @import("check.zig");

const reps = 5;
const marks_max = 1024;
const gate_bytes = 256 << 20;
const classes = @typeInfo(nemotron.Class).@"enum".field_names.len;

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return if (xs.len % 2 == 1) xs[xs.len / 2] else (xs[xs.len / 2 - 1] + xs[xs.len / 2]) / 2;
}

/// The prompt's state again, one serial token on, as each timed window starts.
fn setup(e: *nemotron.Engine, saved: cuda.DeviceBuffer, pos: usize, token: u32) !void {
    try e.b.restore(e.ops(), saved);
    e.pos = pos;
    e.parity = 0;
    e.prev_keep = 0;
    _ = try e.step(token, null);
    try e.stream.synchronize();
}

/// MODEL at a 16k window with drafts and graphs, the prompt from IDS_FILE (comma-separated ids), WIDTHS like 1,4,16.
pub fn run(gpu: check.Gpu, model: []const u8, widths_text: []const u8, ids_path: []const u8) !void {
    const gpa = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, ids_path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |t| try ids.append(gpa, try std.fmt.parseInt(u32, t, 10));
    var widths: std.ArrayList(usize) = .empty;
    defer widths.deinit(gpa);
    var wt = std.mem.tokenizeAny(u8, widths_text, ", ");
    while (wt.next()) |t| try widths.append(gpa, try std.fmt.parseInt(usize, t, 10));
    const e = try nemotron.Engine.init(gpa, gpu.io, gpu.ctx, model, null, .{ .context = 16384, .mtp = true, .graphs = true, .sampling = null, .segments = 1 });
    defer e.deinit();
    try profile(gpa, e, ids.items, widths.items);
}

/// Each width's window graph ms, then its eager ms by class with the shared expert forked and on one stream (medians of 5).
fn profile(gpa: std.mem.Allocator, e: *nemotron.Engine, ids: []const u32, widths: []const usize) !void {
    const most = nemotron.state.max_rows;
    if (ids.len < most + 2) return error.PromptTooShort;
    const prompt = ids[0 .. ids.len - most - 1];
    const cont = ids[prompt.len..];
    _ = try e.prefill(prompt, null, null);
    var saved = try e.b.snapshot(e.ops());
    defer saved.free();
    var gate = try cuda.DeviceBuffer.alloc(e.ctx.d, gate_bytes);
    defer gate.free();
    const ev = try gpa.alloc(cuda.Event, marks_max + 1);
    defer gpa.free(ev);
    var made: usize = 0;
    defer for (ev[0..made]) |*x| x.deinit();
    while (made < ev.len) : (made += 1) ev[made] = try cuda.Event.init(e.ctx.d, true);
    const class = try gpa.alloc(nemotron.Class, marks_max);
    defer gpa.free(class);
    for (widths) |r| {
        if (r < 1 or r > most) return error.BadWidth;
        var times: [reps]f64 = undefined;
        for (&times) |*x| {
            try setup(e, saved, prompt.len, cont[0]);
            try ev[0].record(e.stream);
            try e.verify(cont[1..][0..r], r, null);
            try ev[1].record(e.stream);
            try ev[1].synchronize();
            x.* = try cuda.Event.elapsedMs(ev[0], ev[1]);
        }
        const graph = median(&times);
        for (&times) |*x| {
            try setup(e, saved, prompt.len, cont[0]);
            try e.upload(cont[1]);
            try e.ops().upload(e.b.ids, std.mem.sliceAsBytes(cont[1..][0..r]));
            for (0..32) |_| try e.ops().copy(gate.ptr + gate_bytes / 2, gate.ptr, gate_bytes / 2);
            try ev[0].record(e.stream);
            try e.forward(null).window(r);
            try ev[1].record(e.stream);
            try ev[1].synchronize();
            x.* = try cuda.Event.elapsedMs(ev[0], ev[1]);
        }
        const bare = median(&times);
        std.debug.print("WINDOW {d} rows at position {d}: graph {d:.3} ms, eager without marks {d:.3} ms\n", .{ r, prompt.len + 1, graph, bare });
        for ([_]bool{ false, true }) |one_stream| {
            var per: [classes][reps]f64 = undefined;
            var calls: [classes]usize = @splat(0);
            var totals: [reps]f64 = undefined;
            for (0..reps) |rep| {
                try setup(e, saved, prompt.len, cont[0]);
                try e.upload(cont[1]);
                try e.ops().upload(e.b.ids, std.mem.sliceAsBytes(cont[1..][0..r]));
                for (0..32) |_| try e.ops().copy(gate.ptr + gate_bytes / 2, gate.ptr, gate_bytes / 2);
                var m: nemotron.Marks = .{ .ev = ev[1..], .class = class };
                try ev[0].record(e.stream);
                var f = e.forward(null);
                f.marks = &m;
                if (one_stream) f.side = null;
                try f.window(r);
                try ev[m.n].synchronize();
                var sums: [classes]f64 = @splat(0);
                for (0..m.n) |i| {
                    sums[@intFromEnum(class[i])] += try cuda.Event.elapsedMs(ev[i], ev[i + 1]);
                    if (rep == 0) calls[@intFromEnum(class[i])] += 1;
                }
                for (0..classes) |c| per[c][rep] = sums[c];
                totals[rep] = try cuda.Event.elapsedMs(ev[0], ev[m.n]);
            }
            const total = median(&totals);
            var marks: usize = 0;
            for (calls) |n| marks += n;
            std.debug.print("  eager with {d} marks, shared expert {s}: {d:.3} ms\n", .{ marks, if (one_stream) "on one stream" else "forked", total });
            for (0..classes) |c| {
                if (calls[c] == 0) continue;
                const ms = median(&per[c]);
                std.debug.print("    {s:13} {d:4} calls {d:8.3} ms {d:5.1}%\n", .{ @tagName(@as(nemotron.Class, @enumFromInt(c))), calls[c], ms, 100 * ms / total });
            }
        }
    }
    try e.reset();
}
