//! Optional 128-pair NVFP4 FACT prompt entries and their module setup.
const prompt_layout = @import("experts_prompt_layout.zig");
const Module = @import("module.zig").Module;
const Nvfp4Experts = @import("experts.zig").Nvfp4Experts;

pub const symbols = [2][:0]const u8{ "tf_nvfp4_fact128_swiglu", "tf_nvfp4_fact128_down_f32" };
pub const wn = [2]usize{ 4, 4 }; // WM=2, RT=4: 128 rows and 256 threads for both entries

pub fn load(mod: *Module, e: *Nvfp4Experts) !void {
    for (symbols, &e.fact128_fns, [2]usize{ 2, 1 }, wn) |name, *f, parts, n| {
        f.* = try mod.function(name);
        try f.allowDynamicShared(prompt_layout.promptSmemWM(parts, 2, n));
    }
}
