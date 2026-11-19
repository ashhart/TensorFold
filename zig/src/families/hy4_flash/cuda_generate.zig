//! Greedy generation with the CUDA Hy4-flash text model: prefill token ids, decode N, write timings and logits.
const std = @import("std");
const cuda = @import("cuda");
const model = @import("cuda_model.zig");

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 5) {
        std.debug.print("usage: tf-hy4-cuda-generate MODEL_DIR PROMPT_IDS.u32 NEW_TOKENS OUT_DIR\n", .{});
        return error.BadArgs;
    }
    const prompt_bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(1 << 20));
    const prompt = try a.alloc(u32, prompt_bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(prompt), prompt_bytes[0 .. prompt.len * 4]);
    const new_tokens = try std.fmt.parseInt(usize, args[3], 10);
    const out = try std.Io.Dir.cwd().openDir(io, args[4], .{});
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    if (try ctx.capability() != 121) return error.NeedGb10;
    var s = try cuda.Stream.init(&driver, true);
    defer s.deinit();
    var kernels = try model.Kernels.load(a, &driver);
    const t0 = std.Io.Clock.awake.now(io);
    const capacity = prompt.len + new_tokens;
    const hy4 = try model.Model.open(a, io, &driver, s, &kernels, args[1], capacity, @max(prompt.len, 1));
    const load_ns = t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    std.debug.print("loaded {d} layers, {d:.2} GiB device weights in {d:.1} s\n", .{ hy4.layers.len, @as(f64, @floatFromInt(hy4.weight_bytes)) / (1 << 30), @as(f64, @floatFromInt(load_ns)) / 1e9 });
    const logits = try a.alloc(f32, hy4.text.vocab_size);
    var copy: std.ArrayList(u32) = .empty;
    const p0 = std.Io.Clock.awake.now(io);
    var u32buf: [1]u32 = undefined;
    try hy4.pass(prompt, logits);
    const prefill_ns = p0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    try out.writeFile(io, .{ .sub_path = "prefill-logits.f32", .data = std.mem.sliceAsBytes(logits) });
    const d0 = std.Io.Clock.awake.now(io);
    for (0..new_tokens) |step| {
        var best: usize = 0;
        for (logits, 0..) |v, i| if (v > logits[best]) {
            best = i;
        };
        try copy.append(a, @intCast(best));
        if (step + 1 == new_tokens) break;
        u32buf[0] = @intCast(best);
        try hy4.pass(&u32buf, logits);
    }
    const decode_ns = d0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    try out.writeFile(io, .{ .sub_path = "generated.u32", .data = std.mem.sliceAsBytes(copy.items) });
    std.debug.print("prompt {d} tokens: prefill {d:.1} ms ({d:.1} tok/s); decode {d} tokens {d:.1} ms/token\ngenerated={any}\n", .{
        prompt.len,
        @as(f64, @floatFromInt(prefill_ns)) / 1e6,
        @as(f64, @floatFromInt(prompt.len)) * 1e9 / @as(f64, @floatFromInt(prefill_ns)),
        new_tokens,
        @as(f64, @floatFromInt(decode_ns)) / 1e6 / @as(f64, @floatFromInt(@max(new_tokens, 2) - 1)),
        copy.items,
    });
}
