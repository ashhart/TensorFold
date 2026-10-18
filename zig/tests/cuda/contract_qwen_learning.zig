//! The dense learner's component gates and explicit pending compiler receipts.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const registry = core.contract.registry;
fn image(name: []const u8) []const u8 {
    inline for (.{ "qwen27_train", "attn_train", "gdn_train", "train", "qwen_vision" }) |n|
        if (std.mem.eql(u8, name, n)) return @field(cuda.kernels, n);
    unreachable;
}
pub fn numerical(gpu: check.Gpu) !void {
    try @import("q27_back_tests.zig").run(gpu, .all);
    try @import("q27_gemm_tests.zig").run(gpu);
    try @import("gdn_back_tests.zig").runShape(gpu, 5120, 64);
}
pub fn live(gpu: check.Gpu) !usize {
    var missing = try registry.checkCuda(gpu.gpa, &registry.qwen_learning_pending, false);
    const sm = try gpu.ctx.capability();
    const at = for (core.contract.limits.sms, 0..) |target, i| {
        if (target.sm == sm) break i;
    } else return error.UnsupportedDevice;
    for (registry.qwen_learning_pending) |record| {
        var module = try cuda.Module.load(gpu.d, image(record.launch.image));
        defer module.unload();
        const symbol = try gpu.gpa.dupeSentinel(u8, record.launch.symbol, 0);
        defer gpu.gpa.free(symbol);
        const f = try module.function(symbol);
        const regs: u32 = @intCast(try f.attribute(.num_regs));
        const shared: u32 = @intCast(try f.attribute(.shared_size_bytes));
        const bound: u32 = @intCast(try f.attribute(.max_threads_per_block));
        const usage = record.usage[at];
        if (usage.regs != regs or usage.static != shared or record.launch.threads > bound) missing += 1;
    }
    return missing;
}
pub fn offline(gpu: check.Gpu, dir: []const u8) !usize {
    var missing = try registry.checkCuda(gpu.gpa, &registry.qwen_learning_pending, false);
    for (registry.qwen_learning_pending) |record| {
        const path = try std.fmt.allocPrint(gpu.gpa, "{s}/{s}.fatbin", .{ dir, record.launch.image });
        defer gpu.gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 29));
        defer gpu.gpa.free(bytes);
        var report: std.ArrayList(u8) = .empty;
        defer report.deinit(gpu.gpa);
        const images = try core.contract.cubin.images(gpu.gpa, bytes);
        defer gpu.gpa.free(images);
        for (core.contract.limits.sms, 0..) |target, i| {
            const img = for (images) |entry| {
                if (entry.sm == target.sm) break entry;
            } else {
                missing += 1;
                continue;
            };
            const kernels = try core.contract.cubin.kernels(gpu.gpa, img.elf, target.sm);
            defer gpu.gpa.free(kernels);
            const k = for (kernels) |entry| {
                if (std.mem.eql(u8, entry.name, record.launch.symbol)) break entry;
            } else return error.SymbolNotInImage;
            missing += try core.contract.limits.cuda(record.launch, k.usage, target, &report, gpu.gpa);
            if (!std.meta.eql(k.usage, record.usage[i])) missing += 1;
        }
    }
    return missing;
}
