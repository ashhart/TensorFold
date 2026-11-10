//! Inspect the embedded ELF records on the host without opening the CUDA driver.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");

pub fn main(init: std.process.Init) !void {
    const cubin = core.contract.cubin;
    const images = try cubin.images(init.gpa, cuda.kernels.prefill_attention);
    defer cubin.freeImages(init.gpa, images);
    var seen: [2]bool = @splat(false);
    for (images) |image| {
        const index: usize = switch (image.sm) {
            120 => 0,
            121 => 1,
            else => continue,
        };
        const release = try cubin.release(init.gpa, image.elf) orelse return error.MissingNvccRelease;
        if (release.major != 13 or release.minor != 5) return error.WrongNvccRelease;
        const kernels = try cubin.kernels(init.gpa, image.elf, image.sm);
        defer init.gpa.free(kernels);
        for (kernels) |kernel| {
            if (!std.mem.eql(u8, kernel.name, cuda.kv_expand.symbol)) continue;
            if (seen[index]) return error.DuplicateExpandSymbol;
            seen[index] = true;
            if (kernel.usage.regs != 24 or kernel.usage.static != 0 or kernel.usage.bound != 256)
                return error.WrongExpandResources;
            std.debug.print("PASS host ELF sm={d} symbol={s} registers={d} static={d} bound={d} nvcc={d}.{d}\n", .{
                image.sm, kernel.name, kernel.usage.regs, kernel.usage.static, kernel.usage.bound, release.major, release.minor,
            });
        }
    }
    if (!seen[0] or !seen[1]) return error.MissingExpandProfile;
}
