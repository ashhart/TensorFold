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

/// The one-Mac load limit, read once a load: the load check, stream admission and the cache budget all read `bytes`.
pub const Limit = struct {
    bytes: usize,
    default: usize, // 70% of RAM in GiB read as GB (179.2 at 256 GiB)
    asked: bool = false, // GLM_LOAD_LIMIT_GB set it
    capped: bool = false, // GLM_LOAD_LIMIT_GB asked for more than the GPU's recommended working set

    /// Where the limit came from, for the refusal and the MTP head's note.
    pub fn source(l: Limit) []const u8 {
        return if (l.asked) "GLM_LOAD_LIMIT_GB" else "70% of RAM";
    }
};

/// 70% of `ram`, or GLM_LOAD_LIMIT_GB's GB (`text`) up to the GPU's working set; text that isn't a number is refused.
pub fn limitFor(ram: u64, working_set: u64, text: ?[]const u8) error{BadLoadLimit}!Limit {
    const default: usize = @intFromFloat(@as(f64, @floatFromInt(ram)) / (1 << 30) * 0.7 * 1e9);
    const t = std.mem.trim(u8, text orelse "", " \t");
    if (t.len == 0) return .{ .bytes = default, .default = default };
    const gb = std.fmt.parseFloat(f64, t) catch return error.BadLoadLimit;
    if (!std.math.isFinite(gb) or gb <= 0) return error.BadLoadLimit;
    const most: f64 = @floatFromInt(@min(ram, working_set));
    return .{ .bytes = @intFromFloat(@min(gb * 1e9, most)), .default = default, .asked = true, .capped = gb * 1e9 > most };
}

/// This Mac's limit (RAM, GPU working set, GLM_LOAD_LIMIT_GB); past the 70% default the log names the panic risk.
pub fn loadLimit(device: mtl.Device) error{BadLoadLimit}!Limit {
    var mem: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &mem, &len, null, 0) != 0) mem = 0; // RAM unknown: a zero limit refuses every load
    const text: ?[]const u8 = if (std.c.getenv("GLM_LOAD_LIMIT_GB")) |v| std.mem.span(v) else null;
    const l = limitFor(mem, device.maxWorkingSet(), text) catch |err| {
        std.log.err("glm: GLM_LOAD_LIMIT_GB={s} is not a load limit: give a number of GB above 0, or unset it for 70% of RAM", .{text.?});
        return err;
    };
    if (l.capped) std.log.info("glm: GLM_LOAD_LIMIT_GB={s} passes this GPU's recommended working set; the limit is {d:.1} GB", .{ text.?, toGb(l.bytes) });
    if (l.bytes > l.default) std.log.warn("glm: GLM_LOAD_LIMIT_GB lets this Mac load {d:.1} GB, past the default {d:.1} GB (70% of RAM): a load that wires most of the RAM can stall macOS until its watchdog restarts the Mac, so run nothing else large beside it", .{ toGb(l.bytes), toGb(l.default) });
    return l;
}

fn toGb(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1e9;
}

test "the load limit: 70% of RAM, or GLM_LOAD_LIMIT_GB up to the GPU's working set; bad text refused" {
    const ram: u64 = 256 << 30;
    const ws: u64 = 239_000_000_000;
    const default = try limitFor(ram, ws, null);
    try std.testing.expectEqual(@as(usize, 179_200_000_000), default.bytes);
    try std.testing.expect(!default.asked and std.mem.eql(u8, "70% of RAM", default.source()));
    try std.testing.expectEqual(default, try limitFor(ram, ws, " "));
    const lower = try limitFor(ram, ws, "40");
    try std.testing.expect(lower.asked and lower.bytes == 40_000_000_000 and lower.bytes <= lower.default);
    try std.testing.expectEqual(@as(usize, 183_500_000_000), (try limitFor(ram, ws, "183.5")).bytes);
    const past = try limitFor(ram, ws, " 210 ");
    try std.testing.expect(past.bytes == 210_000_000_000 and past.bytes > past.default and !past.capped);
    try std.testing.expect(past.bytes > @as(usize, @intFromFloat(@as(f64, @floatFromInt(ram)) * 0.7))); // past 70% of the RAM's bytes too
    const most = try limitFor(ram, ws, "1000");
    try std.testing.expect(most.bytes == ws and most.capped and std.mem.eql(u8, "GLM_LOAD_LIMIT_GB", most.source()));
    for ([_][]const u8{ "abc", "0", "-5", "nan", "inf", "1e400", "12GB" }) |bad| try std.testing.expectError(error.BadLoadLimit, limitFor(ram, ws, bad));
    try std.testing.expectEqual(@as(usize, 0), (try limitFor(0, ws, "100")).bytes); // RAM unknown: nothing loads
}
