//! GLM-5.3-Flash's load plan: the checkpoint's identity, the bytes its weights and caches take, the chunk height, the limit.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const st = @import("state.zig");
const prompt_mod = @import("prompt.zig");
const ep_mod = @import("ep.zig");

/// The checkpoint's identity for a peer: its config and weight index, hashed.
pub fn modelHash(gpa: std.mem.Allocator, dir: []const u8, config: []const u8) !u64 {
    const path = try std.fmt.allocPrintSentinel(gpa, "{s}/model.safetensors.index.json", .{dir}, 0);
    defer gpa.free(path);
    const f = try mtl.MappedFile.open(path);
    defer f.deinit();
    var h = std.hash.Wyhash.init(0x474c4d);
    h.update(config);
    h.update(f.bytes[0..f.size]);
    return h.final();
}

/// The weights' plan from the headers: names, dtypes, shapes and bytes, nothing read.
pub fn planBytes(gpa: std.mem.Allocator, device: mtl.Device, dir: []const u8, c: *const cfg.Config) !usize {
    const plan = try wts.load(gpa, device, dir, c, 16, true);
    defer gpa.destroy(plan);
    defer plan.deinit();
    return plan.bytes;
}

/// The fewest bytes of caches and buffers a load of `cap` tokens takes (the shortest prompt chunks), counted, not allocated.
pub fn leastArena(gpa: std.mem.Allocator, c: *const cfg.Config, cap: u32, chunked: bool) !usize {
    var dry: st.Arena = .{ .device = undefined, .gpa = gpa, .dry = true };
    const both = try st.init(&dry, c, cap);
    const chunk = if (chunked) prompt_mod.chunkBytes(gpa, c, &both.scratch, cap, prompt_mod.heights[prompt_mod.heights.len - 1]) else 0;
    return dry.bytes + @as(usize, cap) * 4 + 256 + chunk;
}

/// The tallest prompt chunk whose buffers fit under `limit` beside `used` bytes (expert parallel: one exchange's rows).
pub fn chunkHeight(gpa: std.mem.Allocator, c: *const cfg.Config, sc: *const st.Scratch, cap: u32, used: usize, limit: usize, ep: bool) u32 {
    for (prompt_mod.heights) |h| {
        if (ep and h > ep_mod.PROMPT_ROWS) continue;
        if (used + prompt_mod.chunkBytes(gpa, c, sc, cap, h) <= limit) return h;
    }
    return prompt_mod.heights[prompt_mod.heights.len - 1];
}

/// The one-Mac limit: 70% of RAM in GiB read as GB (179.2 at 256 GiB), or GLM_LOAD_LIMIT_GB on a Mac cleared for more, at most 70% of its bytes.
pub fn loadLimit() usize {
    var mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &mem, &len, null, 0) != 0 or mem == 0) return 0;
    const ram: f64 = @floatFromInt(mem);
    const asked: f64 = if (std.c.getenv("GLM_LOAD_LIMIT_GB")) |v| std.fmt.parseFloat(f64, std.mem.span(v)) catch 0 else 0;
    if (asked > 0) return @intFromFloat(@min(asked * 1e9, ram * 0.7));
    return @intFromFloat(ram / (1 << 30) * 0.7 * 1e9);
}
