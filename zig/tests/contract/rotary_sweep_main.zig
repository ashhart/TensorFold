//! Write only a complete exhaustive receipt; no partial sweep can replace a previously accepted table.
const std = @import("std");
const mtl = @import("metal");
const f = @import("tensorfold").flashnext_vision_contract;
const accuracy = f.rotary_accuracy;
const deployment = @import("rotary_deployment.zig");
const gpu = @import("rotary_sweep_gpu.zig");
const normalization = @import("rotary_sweep_norm.zig");

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3 or !std.mem.eql(u8, args[1], "--out")) {
        std.debug.print("usage: tf-rotary-sweep --out RECEIPT.json\n", .{});
        return error.Usage;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const identity = try deployment.current(a, init.io, device);
    std.debug.print("ROTARY SWEEP compiler {s}/{s}, OS {s} ({s}), GPU generation {d}; no partial receipt\n", .{ identity.compiler_service, identity.compiler_framework, identity.os_version, identity.os_build, identity.device_generation });
    var rows: [12]accuracy.Row = undefined;
    var made: usize = 0;
    var total: u64 = 0;
    for (std.enums.values(accuracy.Role)) |role| {
        const normalized = try normalization.run(a, device, queue, role);
        for (std.enums.values(accuracy.Mode)) |mode| {
            const row = &rows[made];
            row.* = .{ .role = role, .mode = mode, .normalization = normalized };
            try gpu.run(a, device, queue, row);
            total += row.cases;
            made += 1;
            std.debug.print("SWEPT {t}/{t}: {d} cases, {d} normalization fixtures; cos {e} abs/{d:.3} ulp, sin {e} abs/{d:.3} ulp\n", .{ role, mode, row.cases, normalized.len, row.cosine.absolute, row.cosine.ulps, row.sine.absolute, row.sine.ulps });
        }
    }
    if (made != rows.len or total != 302008320) return error.IncompleteRotarySweep;
    const receipt: accuracy.Receipt = .{ .complete = true, .identity = identity, .rows = &rows };
    try receipt.validate(identity);
    const bytes = try std.json.Stringify.valueAlloc(a, receipt, .{ .whitespace = .indent_2 });
    const temporary = try std.fmt.allocPrint(a, "{s}.part-{d}", .{ args[2], @import("sys").c.getpid() });
    const dir = std.Io.Dir.cwd();
    try dir.writeFile(init.io, .{ .sub_path = temporary, .data = bytes, .flags = .{ .exclusive = true } });
    errdefer dir.deleteFile(init.io, temporary) catch {};
    try dir.rename(temporary, dir, args[2], init.io);
    std.debug.print("COMPLETE {d} cases: wrote {s}; rerun the source-matched probes before qualification\n", .{ total, args[2] });
}
