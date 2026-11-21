//! LFM2's loader only builds its optional prompt-attention libraries on tensor-unit devices.
const mtl = @import("metal");
const ks = @import("kernel_sources");
const Set = @import("set.zig").Set;
const source = @import("../src/families/lfm2_moe/source.zig");

pub fn add(s: *Set, device: mtl.Device) !void {
    try s.raw("lfm2", "base", ks.lfm2);
    if (source.optionalAvailable(device)) {
        try s.raw("lfm2", "attention", try source.prompt(device, s.a, false));
        try s.raw("lfm2", "masked-prompt", try source.prompt(device, s.a, true));
    }
}
