//! Diagnostic: generate()'s one-row codebook-9 head on identical inputs, through the production tower's plan.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");
const max_rows = 128;
const width = 2048;
const vocab = 4096;
const projection_count = 6; // tower.zig `projections.len`
const head_kind = 5; // tower.zig `projections[5]` = { 4096, hidden }, the codebook head shape
const head_index = 8; // codebook 9: audio_decoder_mtp.heads.8, the oracle's `p.heads[8]`
const head_rows = 1; // generate() and the oracle project one row through the head
const row_counts = [_]usize{ 1, 7, max_rows };
const origins = [_][]const u8{ "native", "reference" };

fn readExact(io: std.Io, dir: std.Io.Dir, name: []const u8, bytes: []u8) !void {
    const file = try dir.openFile(io, name, .{});
    defer file.close(io);
    if (try file.length(io) != bytes.len or try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.BadFixtureLength;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: tf-drama-tower-head-replay CHECKPOINT INPUT_DIRECTORY OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    var input = try std.Io.Dir.cwd().openDir(init.io, args[2], .{});
    defer input.close(init.io);
    var out = try std.Io.Dir.cwd().openDir(init.io, args[3], .{});
    defer out.close(init.io);
    var total_rows: usize = 0;
    for (row_counts) |n| total_rows += n * origins.len;
    const rows = try init.gpa.alloc(u16, total_rows * width);
    defer init.gpa.free(rows);
    var inputs: [row_counts.len][origins.len][]u16 = undefined;
    var name_buffer: [128]u8 = undefined;
    // Every input is read before any CUDA call, so a missing or short fixture never touches the device.
    var offset: usize = 0;
    for (row_counts, 0..) |n, i| for (origins, 0..) |origin, j| {
        inputs[i][j] = rows[offset .. offset + n * width];
        offset += n * width;
        const name = try std.fmt.bufPrint(&name_buffer, "input-{d}-{s}.bf16", .{ n, origin });
        try readExact(init.io, input, name, std.mem.sliceAsBytes(inputs[i][j]));
    };
    var logits: [vocab]u16 = undefined;
    var repeated: [vocab]u16 = undefined;
    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    // Same options as drama_tower_oracle.zig, so the plan list, weights and buffers are the ones it ran.
    var tower = try drama.tower.Model.init(init.gpa, init.io, &ctx, args[1], "", .{ .context = max_rows, .max_rows = max_rows });
    defer tower.deinit();
    const p = tower.impl;
    if (p.plans.items.len != max_rows * projection_count or p.head_logits.len != vocab * 2 or p.output_hidden.len != max_rows * width * 2)
        return error.UnexpectedTowerLayout;
    const plan = &p.plans.items[(head_rows - 1) * projection_count + head_kind];
    for (row_counts, 0..) |n, i| for (origins, 0..) |origin, j| {
        // Full N rows at offset 0 of the production buffer: the head reads row N - 1 at the oracle's address.
        try p.output_hidden.upload(0, std.mem.sliceAsBytes(inputs[i][j]));
        for ([_]*[vocab]u16{ &logits, &repeated }) |destination| {
            // 0xffff is a BF16 NaN: a head that writes nothing cannot pass the CPU-side self-control.
            try p.head_logits.fill8(0xff, stream.handle);
            try plan.run(p.output_hidden.ptr + (n - 1) * width * 2, p.heads[head_index].ptr, p.head_logits.ptr, p.workspace.ptr, p.workspace.len, stream.handle);
            try stream.synchronize();
            try p.head_logits.download(0, std.mem.sliceAsBytes(destination));
        }
        if (!std.mem.eql(u16, &logits, &repeated)) return error.HeadReplayNotReproducible;
        const name = try std.fmt.bufPrint(&name_buffer, "native-op-{d}-{s}.bf16", .{ n, origin });
        try out.writeFile(init.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(&logits) });
        std.debug.print("head_rows={d} input={s} last_row={d} repeat_bit_exact=true\n", .{ n, origin, n - 1 });
    };
    const used = cuda.usage(false);
    std.debug.print("cuda_bytes={d} pinned_bytes={d}\n", .{ used.device, used.host });
    return 0;
}
