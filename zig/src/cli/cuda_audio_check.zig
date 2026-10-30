//! audio-check: a WAV clip through Transcribe's audio encoder, every stage to safetensors under NeMo's names.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemo_audio = @import("nemotron").nemo_audio;

pub fn run(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, model: []const u8, rest: []const []const u8) !u8 {
    if (rest.len != 2) {
        std.debug.print("audio-check: give MODEL WAV OUT.safetensors\n", .{});
        return 2;
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, rest[0], gpa, .limited(1 << 31));
    defer gpa.free(bytes);
    const wav = try nemo_audio.mel.decodeWav(gpa, bytes);
    defer gpa.free(wav);
    const e = try nemo_audio.Encoder.init(gpa, io, ctx.d, model);
    defer e.deinit();
    var s = try cuda.Stream.init(ctx.d, true);
    defer s.deinit();
    const n = nemo_audio.rows(wav.len);
    var out = try cuda.DeviceBuffer.alloc(ctx.d, @max(1, n) * e.c.text_hidden * 2);
    defer out.free();
    var dump: nemo_audio.Dump = .{ .gpa = gpa, .d = ctx.d };
    defer dump.deinit();
    // one pass without the dump for the time, then the dumped one
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const got = try e.encode(s, wav, out.ptr, null);
    try s.synchronize();
    const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
    _ = try e.encode(s, wav, out.ptr, &dump);
    var tensors: std.ArrayList(core.shard_edit.New) = .empty;
    defer tensors.deinit(gpa);
    for (dump.items.items) |*it| try tensors.append(gpa, .{ .name = it.name, .dtype = .f32, .shape = &it.shape, .bytes = std.mem.sliceAsBytes(it.data) });
    std.Io.Dir.cwd().deleteFile(io, rest[1]) catch |err| if (err != error.FileNotFound) return err;
    try core.shard_edit.create(gpa, rest[1], &.{}, tensors.items);
    std.debug.print("RESULT audio-check: {d} samples ({d:.2} s) -> {d} rows in {d:.1} ms; {d} stages to {s}\n", .{ wav.len, @as(f64, @floatFromInt(wav.len)) / 16000.0, got, ms, dump.items.items.len, rest[1] });
    return 0;
}
