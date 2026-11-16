//! Generation stop metadata, kept separate from the architecture's single eos_token_id.
const std = @import("std");
const config = @import("config.zig");

pub const Generation = struct {
    bos: u32,
    pad: u32,
    sample: bool,
    eos: [8]u32,
    eos_n: usize,

    pub fn stops(g: *const Generation, id: u32) bool {
        return std.mem.indexOfScalar(u32, g.eos[0..g.eos_n], id) != null;
    }
};

pub fn parse(a: std.mem.Allocator, bytes: []const u8, vocab: u32, why: *config.Why) !Generation {
    why.len = 0;
    var doc = std.json.parseFromSlice(std.json.Value, a, bytes, .{}) catch |err| {
        if (err == error.OutOfMemory) return err;
        return config.bad(why, "generation_config.json");
    };
    defer doc.deinit();
    if (doc.value != .object) return config.bad(why, "generation_config.json");
    const o = doc.value.object;
    const bos = try config.uint(o, "bos_token_id", why);
    if (bos >= vocab) return config.bad(why, "bos_token_id");
    const pad = try config.uint(o, "pad_token_id", why);
    if (pad >= vocab) return config.bad(why, "pad_token_id");
    const sample = try config.field(o, "do_sample", why);
    if (sample != .bool) return config.bad(why, "do_sample");
    var out: Generation = .{ .bos = bos, .pad = pad, .sample = sample.bool, .eos = @splat(0), .eos_n = 0 };
    const eos = try config.field(o, "eos_token_id", why);
    const values: []const std.json.Value = if (eos == .array) eos.array.items else &.{eos};
    if (values.len == 0 or values.len > out.eos.len) return config.bad(why, "eos_token_id");
    for (values) |v| {
        if (v != .integer or v.integer < 0 or v.integer >= vocab) return config.bad(why, "eos_token_id");
        const id: u32 = @intCast(v.integer);
        if (std.mem.indexOfScalar(u32, out.eos[0..out.eos_n], id) != null) return config.bad(why, "eos_token_id");
        out.eos[out.eos_n] = id;
        out.eos_n += 1;
    }
    return out;
}
