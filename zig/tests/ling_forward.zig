//! Native bf16 Ling full-forward receipts with whole, chunked and snapshotted execution.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("ling_checkpoint");
const Rows = @import("ling_rows").Rows;
const Forward = @import("ling_forward").Forward;
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3 or args.len > 4) return error.ModelAndOutputRequired;
    const count = if (args.len == 4) try std.fmt.parseInt(u32, args[3], 10) else 2;
    if (count < 2 or count > 32) return error.TokenCount;
    const ids = try init.arena.allocator().alloc(u32, count);
    for (ids, 0..) |*id, t| id.* = @intCast((37 * t + 11) % 157182);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    var model = try ck.Checkpoint.open(init.gpa, init.io, device, args[1]);
    defer model.deinit();
    var rows = try Rows.init(init.gpa, device, count);
    defer rows.deinit();
    const init_mark = rows.owned.items.len;
    var forward = try Forward.init(&rows, &model, 64);
    const init_buffers = rows.owned.items.len - init_mark;
    defer forward.deinit();
    const logits = try forward.run(ids);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[2], .data = std.mem.sliceAsBytes(logits) });
    const picks = try std.fmt.allocPrint(init.arena.allocator(), "{s}.picks", .{args[2]});
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = picks, .data = forward.picks.contents()[0 .. 23 * count * 8 * 4] });
    std.debug.print("Ling full forward: 24 layers, {d} tokens, logits={d}; written with route picks\n", .{ count, logits.len });
    var split = try Forward.init(&rows, &model, 64);
    defer split.deinit();
    const split_logits = try init.arena.allocator().alloc(f32, logits.len);
    @memcpy(split_logits[0..157184], try split.run(ids[0..1]));
    const saved = try split.snapshot(init.arena.allocator());
    var resumed = try Forward.init(&rows, &model, 64);
    defer resumed.deinit();
    try std.testing.expectError(error.BadLingState, resumed.restore(saved[0 .. saved.len - 1]));
    try resumed.restore(saved);
    const resumed_logits = try resumed.run(ids[1..]);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(logits[157184..]), std.mem.sliceAsBytes(resumed_logits));
    for (1..count) |t| @memcpy(split_logits[t * 157184 ..][0..157184], try split.run(ids[t .. t + 1]));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(split_logits));
    try sameState(&forward, &split, count);
    try sameState(&forward, &resumed, count);
    const mark = rows.owned.items.len;
    rows.fail_buffer_at = mark + init_buffers - 1;
    try std.testing.expectError(error.OutOfMemory, Forward.init(&rows, &model, 64));
    try std.testing.expectEqual(mark, rows.owned.items.len);
    rows.fail_buffer_at = mark + 2;
    try std.testing.expectError(error.OutOfMemory, @import("ling_mla").Mla.init(&rows, 16, 64));
    try std.testing.expectEqual(mark, rows.owned.items.len);
    rows.fail_buffer_at = null;
    std.debug.print("Ling init failures: Forward late-buffer and MLA early-buffer errors rolled back owned buffers\n", .{});

    std.debug.print("Ling full forward whole/chunk/resume: logits/KDAstate/conv/MLAcache exact; truncated snapshot rejected\n", .{});
}

fn sameState(whole: *Forward, other: *Forward, count: u32) !void {
    try std.testing.expectEqual(whole.position, other.position);
    for (&whole.attention, &other.attention) |*attention, *part| switch (attention.*) {
        .kda => |*k| {
            try std.testing.expectEqualSlices(u8, k.state.contents()[0 .. 16 * 128 * 128 * 4], part.kda.state.contents()[0 .. 16 * 128 * 128 * 4]);
            try std.testing.expectEqualSlices(u8, k.conv.contents()[0 .. 9 * 2048 * 2], part.kda.conv.contents()[0 .. 9 * 2048 * 2]);
        },
        .mla => |*m| try std.testing.expectEqualSlices(u8, m.cache.contents()[0 .. count * 576 * 2], part.mla.cache.contents()[0 .. count * 576 * 2]),
    };
}
