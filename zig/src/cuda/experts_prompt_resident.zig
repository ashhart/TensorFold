//! NVFP4 original and opt-in R3 prompt entry loading, with decode warp selection unchanged.

const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const kernels = @import("kernels.zig");
const prompt_layout = @import("experts_prompt_layout.zig");
const experts = @import("experts.zig");
const Nvfp4Experts = experts.Nvfp4Experts;

pub fn load(d: *const Driver, sms: usize, warps: usize, resident: bool) !Nvfp4Experts {
    if (warps != 1 and warps != 4) return error.BadExpertWarps;
    if (!kernels.available) return error.BuiltWithoutKernels;
    var mod = try Module.load(d, kernels.nvfp4_experts);
    errdefer mod.unload();
    var e: Nvfp4Experts = .{ .mod = mod, .fns = undefined, .resident = undefined, .prompt_fns = undefined, .pair_fns = undefined, .fact_fns = undefined, .warps = warps, .prompt_r3 = resident };
    for ([_][:0]const u8{ Nvfp4Experts.symbols.gate_up, Nvfp4Experts.symbols.down_f32, Nvfp4Experts.symbols.down_bf16, Nvfp4Experts.symbols.up_relu2 } ++ @import("experts_waves.zig").symbols, 0..) |name, i| {
        var narrow: [256]u8 = undefined;
        const selected = if (warps == 1) blk: {
            const marker = std.mem.indexOf(u8, name, "ELi4E") orelse return error.MissingSymbol;
            @memcpy(narrow[0..name.len], name);
            narrow[marker + 3] = '1';
            narrow[name.len] = 0;
            break :blk narrow[0..name.len :0];
        } else name;
        e.fns[i] = try mod.function(selected);
        e.resident[i] = @max(1, try e.fns[i].occupancy(@intCast(warps * 32), 0)) * sms;
    }
    for (Nvfp4Experts.symbols.prompt, &e.prompt_fns, [3]usize{ 1, 2, 1 }, Nvfp4Experts.prompt_wn) |name, *f, parts, wn| {
        f.* = try mod.function(name);
        try f.allowDynamicShared(prompt_layout.promptSmem(parts, wn));
    }
    for (Nvfp4Experts.symbols.resident_prompt, &e.resident_prompt_fns, [2]usize{ 2, 1 }, [2]usize{ 4, 8 }) |name, *f, parts, wn| {
        f.* = try mod.function(name);
        try f.allowDynamicShared(prompt_layout.promptResidentSmem(parts, wn));
    }
    for (Nvfp4Experts.symbols.pair_prompt, &e.pair_fns, [2]usize{ 2, 1 }, Nvfp4Experts.pair_prompt_wn) |name, *f, parts, wn| {
        f.* = try mod.function(name);
        try f.allowDynamicShared(prompt_layout.pairPromptSmem(parts, wn));
    }
    for (Nvfp4Experts.symbols.fact, &e.fact_fns, [2]usize{ 2, 1 }, Nvfp4Experts.fact_wn) |name, *f, parts, wn| {
        f.* = try mod.function(name);
        try f.allowDynamicShared(prompt_layout.promptSmem(parts, wn));
    }
    return e;
}
