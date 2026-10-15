//! The wide prompt chain's marks on synthetic rows: each state and conv window equal to a segment ending there.
const std = @import("std");
const mtl = @import("metal");
const delta = @import("core_delta");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const Params = extern struct {
    rows: u32,
    nk: u32,
    nv: u32,
    dk: u32,
    dv: u32,
    taps: u32,
    slots: u32,
    weight_flags: u32,
    eps: f32,
    zba_stride: u32 = 0,
};
const Segment = extern struct { first: u32, rows: u32, state_slot: u32, next_slot: u32 };

const nk = 1;
const nv = 2;
const dk = 128;
const dv = 128;
const taps = 4;
const keep = taps - 1;
const channels = 2 * nk * dk + nv * dv;
const conv_width = keep * channels;
const state_words = nv * dv * dk;
const slots = 6;
const max_rows = 64;
const untouched: u8 = 0xff;

/// Two segments of one call: their rows, and the state slot each reads.
const shape = [_]struct { first: u32, rows: u32, slot: u32 }{
    .{ .first = 0, .rows = 24, .slot = 0 },
    .{ .first = 24, .rows = 17, .slot = 1 },
};
const call_rows = shape[1].first + shape[1].rows;

const Case = struct { name: []const u8, marks: [2][]const u32 };
const cases = [_]Case{
    .{ .name = "no marks", .marks = .{ &.{}, &.{} } },
    .{ .name = "first row", .marks = .{ &.{1}, &.{1} } },
    .{ .name = "last row", .marks = .{ &.{24}, &.{17} } },
    .{ .name = "interior row", .marks = .{ &.{7}, &.{12} } },
    .{ .name = "four marks", .marks = .{ &.{ 1, 9, 16, 24 }, &.{ 2, 5, 11, 17 } } },
    .{ .name = "one segment marked", .marks = .{ &.{ 3, 20 }, &.{} } },
};

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    k: delta.Kernels,

    fn buffer(g: Gpu, len: usize) !mtl.Buffer {
        const b = try g.device.buffer(len, opts);
        @memset(b.contents()[0..len], untouched);
        return b;
    }

    fn run(g: Gpu, encode: anytype) !void {
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        encode.go(e);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.debug.print("GPU failed: {s}\n", .{message});
            return error.GpuFailed;
        }
    }
};

/// The rows' inputs, both slots' committed states and conv windows, and every output buffer.
const Data = struct {
    query: mtl.Buffer,
    key: mtl.Buffer,
    value: mtl.Buffer,
    decay: mtl.Buffer,
    mixing: mtl.Buffer,
    committed: mtl.Buffer,
    history: mtl.Buffer,
    projection: mtl.Buffer,
    windows: mtl.Buffer,
    segments: mtl.Buffer,
    keeps: mtl.Buffer,
    kept_rows: mtl.Buffer,
    marks: mtl.Buffer,
    output: mtl.Buffer,
    next_state: mtl.Buffer,
    next_history: mtl.Buffer,
    mark_state: mtl.Buffer,
    mark_conv: mtl.Buffer,

    fn init(g: Gpu) !Data {
        var d: Data = undefined;
        inline for (.{
            .{ "query", max_rows * nk * dk * 2 },
            .{ "key", max_rows * nk * dk * 2 },
            .{ "value", max_rows * nv * dv * 2 },
            .{ "decay", max_rows * nv * 4 },
            .{ "mixing", max_rows * nv * 2 },
            .{ "committed", slots * state_words * 4 },
            .{ "history", slots * conv_width * 2 },
            .{ "projection", max_rows * channels * 2 },
            .{ "windows", max_rows * taps * 4 },
            .{ "segments", 2 * @sizeOf(Segment) },
            .{ "keeps", 2 * @sizeOf(Segment) },
            .{ "kept_rows", max_rows * 4 },
            .{ "marks", 2 * delta.MARKS * 4 },
            .{ "output", max_rows * nv * dv * 2 },
            .{ "next_state", slots * state_words * 4 },
            .{ "next_history", slots * conv_width * 2 },
            .{ "mark_state", 2 * delta.MARKS * state_words * 4 },
            .{ "mark_conv", 2 * delta.MARKS * conv_width * 2 },
        }) |f| @field(d, f[0]) = try g.buffer(f[1]);
        var prng = std.Random.DefaultPrng.init(548);
        const r = prng.random();
        for (d.query.slice(u16, max_rows * nk * dk)) |*x| x.* = toBf((r.float(f32) * 2 - 1) * 0.09);
        for (d.key.slice(u16, max_rows * nk * dk)) |*x| x.* = toBf((r.float(f32) * 2 - 1) * 0.09);
        for (d.value.slice(u16, max_rows * nv * dv)) |*x| x.* = toBf(r.float(f32) * 2 - 1);
        for (d.decay.slice(f32, max_rows * nv)) |*x| x.* = 0.5 + r.float(f32) * 0.5;
        for (d.mixing.slice(u16, max_rows * nv)) |*x| x.* = toBf(r.float(f32));
        for (d.committed.slice(f32, slots * state_words)) |*x| x.* = r.float(f32) - 0.5;
        for (d.history.slice(u16, slots * conv_width)) |*x| x.* = toBf(r.float(f32) * 2 - 1);
        for (d.projection.slice(u16, max_rows * channels)) |*x| x.* = toBf(r.float(f32) * 2 - 1);
        const windows = d.windows.slice(u32, max_rows * taps);
        for (shape) |s| for (0..s.rows) |row| for (0..taps) |tap| {
            const at = @as(i64, @intCast(row + tap)) - keep;
            windows[(s.first + row) * taps + tap] = @intCast(if (at < 0) at + keep else keep + s.first + at);
        };
        return d;
    }

    fn params(rows: u32) Params {
        return .{
            .rows = rows,
            .nk = nk,
            .nv = nv,
            .dk = dk,
            .dv = dv,
            .taps = taps,
            .slots = slots,
            .weight_flags = 0,
            .eps = 1e-6,
        };
    }

    /// One wide chain over `segs`, with `marks` (one list a segment) or the unmarked kernel.
    fn chain(d: *const Data, g: Gpu, segs: []const Segment, marks: ?[]const [delta.MARKS]u32) !void {
        @memcpy(d.segments.slice(Segment, segs.len), segs);
        if (marks) |m| @memcpy(d.marks.slice([delta.MARKS]u32, m.len), m);
        const Enc = struct {
            d: *const Data,
            n: usize,
            k: delta.Kernels,
            marked: bool,
            fn go(x: @This(), e: mtl.ComputeEncoder) void {
                e.setPipeline(if (x.marked) x.k.chain_wide_marks else x.k.chain_wide);
                for ([_]mtl.Buffer{
                    x.d.query,
                    x.d.key,
                    x.d.value,
                    x.d.decay,
                    x.d.mixing,
                    x.d.committed,
                    x.d.segments,
                }, 0..) |b, i| e.setBuffer(b, 0, i);
                e.setValue(params(call_rows), 7);
                e.setBuffer(x.d.output, 0, 8);
                e.setBuffer(x.d.next_state, 0, 9);
                if (x.marked) {
                    e.setBuffer(x.d.marks, 0, 10);
                    e.setBuffer(x.d.mark_state, 0, 11);
                }
                e.dispatchThreads(.{ .width = 32, .height = dv / 4, .depth = nv * x.n }, .{ .width = 32, .height = 4 });
                if (!x.marked) return;
                e.barrier();
                e.setPipeline(x.k.mark_conv);
                for ([_]mtl.Buffer{
                    x.d.history,
                    x.d.projection,
                    x.d.segments,
                    x.d.windows,
                }, 0..) |b, i| e.setBuffer(b, 0, i);
                e.setValue(params(call_rows), 4);
                e.setBuffer(x.d.marks, 0, 5);
                e.setBuffer(x.d.mark_conv, 0, 6);
                e.dispatchThreads(.{ .width = conv_width, .height = delta.MARKS, .depth = x.n }, .{ .width = 32 });
            }
        };
        try g.run(Enc{ .d = d, .n = segs.len, .k = g.k, .marked = marks != null });
    }

    /// conv_commit's window after the first `rows` rows of segment `s`, into next_history slot `to`.
    fn convCommit(d: *const Data, g: Gpu, s: Segment, rows: u32, to: u32) !void {
        d.keeps.slice(Segment, 1)[0] = .{ .first = 0, .rows = rows, .state_slot = s.state_slot, .next_slot = to };
        for (d.kept_rows.slice(u32, rows), 0..) |*x, i| x.* = s.first + @as(u32, @intCast(i));
        const Enc = struct {
            d: *const Data,
            k: delta.Kernels,
            fn go(x: @This(), e: mtl.ComputeEncoder) void {
                e.setPipeline(x.k.pipelines.conv_commit);
                for ([_]mtl.Buffer{
                    x.d.history,
                    x.d.projection,
                    x.d.kept_rows,
                    x.d.keeps,
                    x.d.windows,
                }, 0..) |b, i| e.setBuffer(b, 0, i);
                e.setValue(params(call_rows), 5);
                e.setBuffer(x.d.next_history, 0, 6);
                e.dispatchThreads(.{ .width = conv_width, .height = 1 }, .{ .width = 32 });
            }
        };
        try g.run(Enc{ .d = d, .k = g.k });
    }
};

fn bytes(b: mtl.Buffer, off: usize, len: usize) []const u8 {
    return b.contents()[off..][0..len];
}

fn segmentsOf(next: [2]u32) [2]Segment {
    return .{
        .{ .first = shape[0].first, .rows = shape[0].rows, .state_slot = shape[0].slot, .next_slot = next[0] },
        .{ .first = shape[1].first, .rows = shape[1].rows, .state_slot = shape[1].slot, .next_slot = next[1] },
    };
}

/// One case: the marked call against the unmarked one, then each mark against a segment ending there.
fn check(g: Gpu, d: *const Data, c: Case) !usize {
    const state_bytes = state_words * 4;
    const conv_bytes = conv_width * 2;
    const segs = segmentsOf(.{ 2, 3 });
    try d.chain(g, &segs, null);
    const plain_out = try std.heap.page_allocator.dupe(u8, bytes(d.output, 0, call_rows * nv * dv * 2));
    defer std.heap.page_allocator.free(plain_out);
    const plain_state = try std.heap.page_allocator.dupe(u8, bytes(d.next_state, 2 * state_bytes, 2 * state_bytes));
    defer std.heap.page_allocator.free(plain_state);
    var lists: [2][delta.MARKS]u32 = undefined;
    for (&lists, c.marks, shape) |*l, m, s| l.* = try delta.markList(s.rows, m);
    @memset(d.mark_state.contents()[0 .. 2 * delta.MARKS * state_bytes], untouched);
    @memset(d.mark_conv.contents()[0 .. 2 * delta.MARKS * conv_bytes], untouched);
    @memset(d.output.contents()[0 .. call_rows * nv * dv * 2], untouched);
    try d.chain(g, &segs, &lists);
    var failed: usize = 0;
    if (!std.mem.eql(u8, plain_out, bytes(d.output, 0, plain_out.len))) {
        std.debug.print("  {s}: marked outputs differ from the unmarked call\n", .{c.name});
        failed += 1;
    }
    if (!std.mem.eql(u8, plain_state, bytes(d.next_state, 2 * state_bytes, plain_state.len))) {
        std.debug.print("  {s}: marked end state differs from the unmarked call\n", .{c.name});
        failed += 1;
    }
    const mark_state = try std.heap.page_allocator.dupe(u8, bytes(d.mark_state, 0, 2 * delta.MARKS * state_bytes));
    defer std.heap.page_allocator.free(mark_state);
    const mark_conv = try std.heap.page_allocator.dupe(u8, bytes(d.mark_conv, 0, 2 * delta.MARKS * conv_bytes));
    defer std.heap.page_allocator.free(mark_conv);
    for (0..2) |si| for (0..delta.MARKS) |m| {
        const at = lists[si][m];
        const got_state = mark_state[(si * delta.MARKS + m) * state_bytes ..][0..state_bytes];
        const got_conv = mark_conv[(si * delta.MARKS + m) * conv_bytes ..][0..conv_bytes];
        if (at == 0) {
            for (got_state) |x| if (x != untouched) {
                std.debug.print("  {s}: segment {d} unused mark {d} state written\n", .{ c.name, si, m });
                failed += 1;
                break;
            };
            for (got_conv) |x| if (x != untouched) {
                std.debug.print("  {s}: segment {d} unused mark {d} conv window written\n", .{ c.name, si, m });
                failed += 1;
                break;
            };
            continue;
        }
        const s = segs[si];
        try d.chain(g, &.{.{ .first = s.first, .rows = at, .state_slot = s.state_slot, .next_slot = 4 }}, null);
        try d.convCommit(g, s, at, 4);
        const state_ok = std.mem.eql(u8, got_state, bytes(d.next_state, 4 * state_bytes, state_bytes));
        const conv_ok = std.mem.eql(u8, got_conv, bytes(d.next_history, 4 * conv_bytes, conv_bytes));
        if (!state_ok or !conv_ok) {
            std.debug.print("  {s}: segment {d} mark at row {d}: state {s}, conv window {s}\n", .{
                c.name,
                si,
                at,
                if (state_ok) "equal" else "DIFFERS",
                if (conv_ok) "equal" else "DIFFERS",
            });
            failed += 1;
        }
    };
    return failed;
}

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const k = try delta.Kernels.init(device);
    defer k.deinit();
    const g = Gpu{ .device = device, .queue = queue, .k = k };
    const d = try Data.init(g);
    std.debug.print("device: {s}\n", .{device.name()});
    var failed: usize = 0;
    for (cases) |c| {
        const f = try check(g, &d, c);
        var n: usize = 0;
        for (c.marks) |m| n += m.len;
        std.debug.print("{s}: {d} marks, {s}\n", .{
            c.name,
            n,
            if (f == 0) "every state and conv window equal" else "FAILED",
        });
        failed += f;
    }
    if (failed > 0) {
        std.debug.print("{d} mismatches\n", .{failed});
        std.process.exit(1);
    }
    std.debug.print("all marks match\n", .{});
}
