//! Portable dense pocket saves preserve the checkpoint's tensor namespace and config scope.
const std = @import("std");
const living = @import("living");
const core = @import("core");

pub fn save(a: std.mem.Allocator, io: std.Io, dir: []const u8, lay: living.Layout, layers: []const usize, neurons: usize, gate: []f32, up: []f32, down: []f32) !usize {
    // The native loader accepts both public namespaces. Keep the folder's names and config key scope as stored.
    const scoped = try namespace(a, io, dir);
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    defer a.free(text);
    var doc = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer doc.deinit();
    if (doc.value != .object) return error.BadConfig;
    const wrapped = doc.value.object.contains("text_config");
    inline for (.{ false, true }) |scope| inline for (.{ false, true }) |wrap| {
        if (scoped == scope and wrapped == wrap) return living.shards.Io(core).writePending(a, io, dir, storedFamily(scope, wrap), lay, layers, neurons, &.{ gate, up }, down);
    };
    unreachable;
}

// A save owns the pending transaction; inspect headers only, without reopening it as a serving checkpoint.
fn namespace(a: std.mem.Allocator, io: std.Io, dir: []const u8) !bool {
    const paths = try core.checkpoint_host.shardFiles(a, io, dir);
    defer core.checkpoint_host.freeShardFiles(a, paths);
    var scoped = false;
    var plain = false;
    for (paths) |path| {
        var f = try core.safetensors.File.open(a, io, path);
        defer f.close(io);
        scoped = scoped or f.get("language_model.model.layers.0.mlp.gate_proj.weight") != null;
        plain = plain or f.get("model.layers.0.mlp.gate_proj.weight") != null;
    }
    if (scoped == plain) return error.BadPocketNamespace;
    return scoped;
}

fn storedFamily(comptime scoped: bool, comptime wrapped: bool) living.family.Family {
    var f = living.families.qwen3_5;
    if (!scoped) f.names.layers = "model.layers.";
    if (!wrapped) f.keys.scope = null;
    return f;
}
