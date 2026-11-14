//! TF_DSV41_PROMPT_UNITS: DeepSeek-V4.1 prompt rows on M5 tensor units (on by default; their bits are not decode's).
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const environment = @import("sys").environment;

/// Tensor-unit prompt rows: on by default on the M5 (where their quality was checked), 1 on later GPUs, never with 0.
pub fn on(device: mtl.Device) !bool {
    return device.tensorUnits() and try environment.read(bool, "TF_DSV41_PROMPT_UNITS", device.generation() == 17);
}

/// `text` with nax.h inlined, compiled into its f32 and bf16 kernels `names`.
pub fn load(device: mtl.Device, text: []const u8, names: [2][:0]const u8) ![2]mtl.Pipeline {
    const a = std.heap.page_allocator;
    const body = try std.mem.replaceOwned(u8, a, text, "#include \"../nax.h\"", sources.nax);
    defer a.free(body);
    const lib = try mtl.Library.fromSource(device, body, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const f = try mtl.Pipeline.init(device, lib, names[0], false);
    errdefer f.deinit();
    return .{ f, try mtl.Pipeline.init(device, lib, names[1], false) };
}

/// The serving kernels: null without tensor units, with TF_DSV41_PROMPT_UNITS=0, or when they do not build.
pub fn serving(device: mtl.Device, text: []const u8, names: [2][:0]const u8) !?[2]mtl.Pipeline {
    if (!try on(device)) return null;
    return load(device, text, names) catch |err| {
        std.log.warn("dsv41 metal: tensor-unit prompt kernels {s} did not build ({t}); the decode-equal kernels run", .{ names[0], err });
        return null;
    };
}
