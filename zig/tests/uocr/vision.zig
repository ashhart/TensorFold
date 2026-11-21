//! tf-uocr-vision: Unlimited-OCR-2's image encoder on a reference dump's own pixels, each stage against float32.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const uo = tf.uocr;
const cfg = uo.config;
const gate = uo.gate;

const usage =
    \\tf-uocr-vision MODEL_DIR DUMP_BF16 DUMP_F32 [PAGE]
    \\MODEL_DIR is the baidu/Unlimited-OCR-2 folder. The dumps (`tools/zig/uocr_reference.py dump`) hold, for every
    \\encoder call i, its pixels pixels{i} [views, 3, size, size] and the reference's SAM (sam{i}), CLIP (clip{i}) and
    \\projector (projector{i}) outputs, from torch bf16 and from float32 on the same bf16 pixels. Each view is encoded
    \\here from the bf16 dump's pixels; SAM's output must be as close to float32 as twice torch bf16's (or bf16
    \\rounding's) cosine error and worst row. CLIP's and the projector's are reported against the same limits but not
    \\gated: CLIP's input is SAM's output, and a few of its rows magnify bf16 rounding there by more than twice, on
    \\torch's devices as on ours; tf-uocr-blocks gates each CLIP layer and the projector alone. The views are then
    \\encoded again and must repeat byte for byte. With PAGE (the image the dump was made from), the views are made
    \\here from it instead, and must equal the dump's pixels bit for bit first.
;

fn floats(a: std.mem.Allocator, f: *const tf.safetensors.File, name: []const u8) ![]f32 {
    const t = f.get(name) orelse return error.MissingTensor;
    const out = try a.alloc(f32, t.numel());
    switch (t.dtype) {
        .f32 => @memcpy(std.mem.sliceAsBytes(out), t.bytes),
        .bf16 => for (out, std.mem.bytesAsSlice(u16, t.bytes)) |*o, h| {
            o.* = uo.resize.widen(h);
        },
        else => return error.BadDumpTensor,
    }
    return out;
}

fn widen(a: std.mem.Allocator, h: []const u16) ![]f32 {
    const out = try a.alloc(f32, h.len);
    for (out, h) |*o, x| o.* = uo.resize.widen(x);
    return out;
}

/// SAM's [channels, side, side] as rows [side^2, channels].
fn rowsOf(a: std.mem.Allocator, chw: []const f32, channels: usize, positions: usize) ![]f32 {
    const out = try a.alloc(f32, chw.len);
    for (0..channels) |c| for (0..positions) |p| {
        out[p * channels + c] = chw[c * positions + p];
    };
    return out;
}

fn report(label: []const u8, view: usize, theirs: gate.Match, v: gate.Verdict, gated: bool) void {
    const verdict = if (v.pass) "within" else if (gated) "FAIL" else "over (reported)";
    std.debug.print("  {s} view {d}: 1-cosine {e:.2} of {e:.2}, worst row {d:.5} (row {d}) of {d:.5}, torch bf16 {d:.5}: {s}\n", .{ label, view, v.cosine_err, v.cosine_limit, v.row, v.row_at, v.row_limit, theirs.worst_row, verdict });
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4 and args.len != 5) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    var bf = try tf.safetensors.File.open(a, io, args[2]);
    defer bf.close(io);
    var ex = try tf.safetensors.File.open(a, io, args[3]);
    defer ex.close(io);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    if (!device.tensorUnits()) {
        std.debug.print("{s} has no tensor units; this family runs on the M5's alone\n", .{device.name()});
        return 1;
    }
    const queue = try device.queue();
    defer queue.deinit();
    const t0 = mtl.clock.seconds();
    var w = try uo.weights.load(std.heap.smp_allocator, io, device, args[1]);
    defer w.deinit();
    var k = try uo.kernels.Kernels.load(std.heap.smp_allocator, device);
    defer k.deinit();
    var enc = try uo.vision.Encoder.init(std.heap.smp_allocator, device, queue, &k, &w);
    defer enc.deinit();
    std.debug.print("{s}: loaded in {d:.1} s\n", .{ device.name(), mtl.clock.seconds() - t0 });

    var mine: ?uo.preprocess.Views = null;
    if (args.len == 5) {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[4], a, .limited(64 << 20));
        var page = try uo.preprocess.decode(io, a, bytes);
        const gundam = bf.get("pixels1") != null;
        mine = try uo.preprocess.views(a, page.rgb, page.width, page.height, gundam);
        page.deinit(a);
        std.debug.print("page {s}: {d} x {d}, {s}, tiles {d} x {d}\n", .{ args[4], page.width, page.height, if (gundam) "gundam" else "base", mine.?.across, mine.?.down });
    }

    var all_pass = true;
    var repeat_same = true;
    var call: usize = 0;
    while (true) : (call += 1) {
        var name_buf: [32]u8 = undefined;
        const pix_name = try std.fmt.bufPrint(&name_buf, "pixels{d}", .{call});
        const pix = bf.get(pix_name) orelse break;
        if (pix.dtype != .bf16) return error.BadDumpTensor;
        const views = pix.dim(0);
        const size = pix.dim(2);
        const side = cfg.side(size);
        const n = side * side;
        const sam_bf = try floats(a, &bf, try std.fmt.allocPrint(a, "sam{d}", .{call}));
        const sam_ex = try floats(a, &ex, try std.fmt.allocPrint(a, "sam{d}", .{call}));
        const clip_bf = try floats(a, &bf, try std.fmt.allocPrint(a, "clip{d}", .{call}));
        const clip_ex = try floats(a, &ex, try std.fmt.allocPrint(a, "clip{d}", .{call}));
        const proj_bf = try floats(a, &bf, try std.fmt.allocPrint(a, "projector{d}", .{call}));
        const proj_ex = try floats(a, &ex, try std.fmt.allocPrint(a, "projector{d}", .{call}));
        std.debug.print("call {d}: {d} view(s) of {d}\n", .{ call, views, size });
        const out = try a.alloc(u16, n * cfg.hidden);
        const again = try a.alloc(u16, n * cfg.hidden);
        const sam = try a.alloc(u16, n * cfg.net3_width);
        const clip = try a.alloc(u16, (n + 1) * cfg.clip_width);
        for (0..views) |vi| {
            const dumped: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, pix.bytes)[vi * 3 * size * size ..][0 .. 3 * size * size]);
            const px = if (mine) |m| (if (size == cfg.base_size) m.global else m.tile(vi)) else dumped;
            if (mine != null) {
                var differ: usize = 0;
                for (px, dumped) |x, y| differ += @intFromBool(x != y);
                std.debug.print(" view {d}: pixels {d} of {d} differ from the dump's\n", .{ vi, differ, px.len });
                all_pass = all_pass and differ == 0;
            }
            const t = mtl.clock.seconds();
            try enc.encode(size, @alignCast(px), out, .{ .sam = sam, .clip = clip });
            const dt = mtl.clock.seconds() - t;
            try enc.encode(size, @alignCast(px), again, .{});
            repeat_same = repeat_same and std.mem.eql(u16, out, again);
            const sam_off = vi * n * cfg.net3_width;
            const s_bf = try rowsOf(a, sam_bf[sam_off..][0 .. n * cfg.net3_width], cfg.net3_width, n);
            const s_ex = try rowsOf(a, sam_ex[sam_off..][0 .. n * cfg.net3_width], cfg.net3_width, n);
            const stages = .{
                .{ "sam", try widen(a, sam), s_bf, s_ex, cfg.net3_width, true },
                .{ "clip", try widen(a, clip), clip_bf[vi * (n + 1) * cfg.clip_width ..][0 .. (n + 1) * cfg.clip_width], clip_ex[vi * (n + 1) * cfg.clip_width ..][0 .. (n + 1) * cfg.clip_width], cfg.clip_width, false },
                .{ "projector", try widen(a, out), proj_bf[vi * n * cfg.hidden ..][0 .. n * cfg.hidden], proj_ex[vi * n * cfg.hidden ..][0 .. n * cfg.hidden], cfg.hidden, false },
            };
            std.debug.print(" view {d}: encoded in {d:.3} s\n", .{ vi, dt });
            inline for (stages) |s| {
                const theirs = gate.match(s[2], s[3], s[4]);
                const v = gate.verdict(gate.match(s[1], s[3], s[4]), theirs, try gate.rounding(a, s[3], s[4]));
                report(s[0], vi, theirs, v, s[5]);
                if (s[5]) all_pass = all_pass and v.pass;
            }
        }
    }
    if (call == 0) {
        std.debug.print("the dump has no pixels0\n", .{});
        return 1;
    }
    std.debug.print("repeat byte for byte: {}\n{s}\n", .{ repeat_same, if (all_pass and repeat_same) "PASS" else "FAIL" });
    return if (all_pass and repeat_same) 0 else 1;
}
