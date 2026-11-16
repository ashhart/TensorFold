//! Complete routed-half G1-G5 checks from source-built images, without loading unrelated model operators.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const checks = @import("check.zig");
const fixture = @import("contract_exl3_routed_half.zig");
const limits = @import("contract_limits.zig");
const g = core.contract.gates;
var served_image: []align(16) u8 = &.{};

fn resolve(name: []const u8) error{UnknownContractImage}![]const u8 {
    if (std.mem.eql(u8, name, "exl3_routed_half")) return served_image;
    return error.UnknownContractImage;
}

fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]align(16) u8 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}.fatbin", .{ dir, name });
    defer a.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 28));
    defer a.free(bytes);
    const aligned = try a.alignedAlloc(u8, .@"16", bytes.len);
    @memcpy(aligned, bytes);
    return aligned;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const offline = args.len == 3 and std.mem.eql(u8, args[1], "--offline");
    if ((!offline and args.len != 2) or (args.len == 2 and std.mem.eql(u8, args[1], "--offline"))) {
        std.debug.print("usage: tf-exl3-routed-half-contract [--offline] FATBIN_DIR\n", .{});
        return 2;
    }
    const dir = args[if (offline) @as(usize, 2) else 1];
    var records: std.ArrayList(core.contract.registry.CudaRecord) = .empty;
    defer records.deinit(init.gpa);
    for (core.contract.registry.all_cuda) |record| if (std.mem.eql(u8, record.launch.image, "exl3_routed_half")) {
        try records.append(init.gpa, record);
    };
    if (records.items.len != 1) return error.RotationRecordMissingOrDuplicate;
    const host: checks.Gpu = .{ .d = undefined, .ctx = undefined, .gpa = init.gpa, .io = init.io };
    try limits.fatbinsFor(host, dir, records.items);
    if (offline) return 0;
    served_image = try read(init.gpa, init.io, dir, "exl3_routed_half");
    defer init.gpa.free(served_image);
    const wide = try read(init.gpa, init.io, dir, "exl3_experts");
    defer init.gpa.free(wide);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: checks.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    var stream = try cuda.Stream.init(&driver, false);
    defer stream.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(init.gpa);
    const set = try fixture.Set.initWithImages(gpu, stream, init.gpa, &list, .{ .served = served_image, .wide = wide });
    defer set.deinit();
    var tally: g.Tally = .{};
    try g.runAll(init.gpa, list.items, 0x786c3368, &tally);
    const problems = try limits.liveWith(gpu, &.{"exl3 routed half"}, resolve);
    std.debug.print("SUMMARY {d} kernels: {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed; {d} limit problems\n", .{ tally.kernels, tally.passed, tally.failed, tally.known, tally.controls, tally.caught, tally.missed, problems });
    return if (tally.problems() + problems == 0) 0 else 1;
}
