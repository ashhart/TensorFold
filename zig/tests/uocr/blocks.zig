//! tf-uocr-blocks: Unlimited-OCR-2's image encoder step by step, each step alone against float32.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const uo = tf.uocr;
const gate = uo.gate;
const Encoder = uo.vision.Encoder;

const usage =
    \\tf-uocr-blocks MODEL_DIR BLOCKS
    \\BLOCKS is `tools/zig/uocr_reference.py blocks` for one view: each step's float32 input rounded to bf16, its
    \\float32 output and torch bf16's output on that same input, for every SAM block (sam_*), the neck and its
    \\convolutions (neck_*), CLIP's embedding and pre-norm (front_*), every CLIP layer (clip_*) and the projector
    \\(project_*). Each step runs alone here on its input and must be as close to float32 as twice torch bf16's (or
    \\bf16 rounding's) cosine error and worst row; every step then runs again and must repeat byte for byte.
;

fn halves(f: *const tf.safetensors.File, name: []const u8) ![]const u16 {
    const t = f.get(name) orelse return error.MissingTensor;
    if (t.dtype != .bf16) return error.BadBlocksTensor;
    return @alignCast(std.mem.bytesAsSlice(u16, t.bytes));
}

fn floats(a: std.mem.Allocator, f: *const tf.safetensors.File, name: []const u8) ![]f32 {
    const t = f.get(name) orelse return error.MissingTensor;
    const out = try a.alloc(f32, t.numel());
    switch (t.dtype) {
        .f32 => @memcpy(std.mem.sliceAsBytes(out), t.bytes),
        .bf16 => for (out, std.mem.bytesAsSlice(u16, t.bytes)) |*o, h| {
            o.* = uo.resize.widen(h);
        },
        else => return error.BadBlocksTensor,
    }
    return out;
}

/// One kind of step: its tensors' prefix, the rows it reads, the width of its output rows.
const Kind = struct { name: []const u8, width: usize, input: enum { x, m3, cx, both } };

const kinds = [_]Kind{
    .{ .name = "sam", .width = 768, .input = .x },
    .{ .name = "neck", .width = 1024, .input = .x },
    .{ .name = "front", .width = 1024, .input = .m3 },
    .{ .name = "clip", .width = 1024, .input = .cx },
    .{ .name = "project", .width = 1280, .input = .both },
};

const Tally = struct { pass: bool = true, same: bool = true };

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    var f = try tf.safetensors.File.open(a, io, args[2]);
    defer f.close(io);
    const pix = f.get("pixels") orelse return error.MissingTensor;
    const size = pix.dim(3);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    if (!device.tensorUnits()) {
        std.debug.print("{s} has no tensor units; this family runs on the M5's alone\n", .{device.name()});
        return 1;
    }
    const queue = try device.queue();
    defer queue.deinit();
    var w = try uo.weights.load(std.heap.smp_allocator, io, device, args[1]);
    defer w.deinit();
    var k = try uo.kernels.Kernels.load(std.heap.smp_allocator, device);
    defer k.deinit();
    var enc = try Encoder.init(std.heap.smp_allocator, device, queue, &k, &w);
    defer enc.deinit();
    std.debug.print("{s}: a view of {d}\n", .{ device.name(), size });

    var tally: Tally = .{};
    for (kinds) |kind| {
        const exact = try floats(a, &f, try std.fmt.allocPrint(a, "{s}_f32", .{kind.name}));
        const theirs = try floats(a, &f, try std.fmt.allocPrint(a, "{s}_bf16", .{kind.name}));
        const ft = f.get(try std.fmt.allocPrint(a, "{s}_f32", .{kind.name})).?;
        // the towers stack their blocks [blocks, rows, width]; the steps between are one [rows, width]
        const count: usize = if (ft.rank == 3) ft.dim(0) else 1;
        const n = exact.len / count;
        const in_name = try std.fmt.allocPrint(a, "{s}_in", .{kind.name});
        const ins: ?[]const u16 = if (kind.input == .both) null else try halves(&f, in_name);
        const outs = try a.alloc(u16, n);
        const again = try a.alloc(u16, n);
        const ours = try a.alloc(f32, n);
        var passed: usize = 0;
        var most: f64 = 0;
        for (0..count) |i| {
            var rows: Encoder.Rows = .{};
            switch (kind.input) {
                .x => rows.x = ins.?[i * (ins.?.len / count) ..][0 .. ins.?.len / count],
                .m3 => rows.m3 = ins.?,
                .cx => rows.cx = ins.?[i * n ..][0..n],
                .both => {
                    rows.cx = try halves(&f, "project_cx");
                    rows.m3 = try halves(&f, "project_m3");
                },
            }
            const st: Encoder.Step = if (std.mem.eql(u8, kind.name, "sam"))
                .{ .sam = i }
            else if (std.mem.eql(u8, kind.name, "neck"))
                .neck
            else if (std.mem.eql(u8, kind.name, "front"))
                .front
            else if (std.mem.eql(u8, kind.name, "clip"))
                .{ .clip = i }
            else
                .project;
            try enc.step(size, st, rows, outs);
            try enc.step(size, st, rows, again);
            tally.same = tally.same and std.mem.eql(u16, outs, again);
            for (ours, outs) |*o, h| o.* = uo.resize.widen(h);
            const ex = exact[i * n ..][0..n];
            const ref = gate.match(theirs[i * n ..][0..n], ex, kind.width);
            const v = gate.verdict(gate.match(ours, ex, kind.width), ref, try gate.rounding(a, ex, kind.width));
            most = @max(most, @max(v.cosine_err / v.cosine_limit, v.row / v.row_limit));
            std.debug.print("  {s} {d}: 1-cosine {e:.2} of {e:.2}, worst row {d:.5} (row {d}) of {d:.5}, torch bf16 {d:.5}: {s}\n", .{ kind.name, i, v.cosine_err, v.cosine_limit, v.row, v.row_at, v.row_limit, ref.worst_row, if (v.pass) "pass" else "FAIL" });
            passed += @intFromBool(v.pass);
            tally.pass = tally.pass and v.pass;
        }
        std.debug.print("{s}: {d} of {d} pass, at most {d:.2} of a limit\n", .{ kind.name, passed, count, most });
    }
    const ok = tally.pass and tally.same;
    std.debug.print("repeat byte for byte: {}\n{s}\n", .{ tally.same, if (ok) "PASS" else "FAIL" });
    return if (ok) 0 else 1;
}
