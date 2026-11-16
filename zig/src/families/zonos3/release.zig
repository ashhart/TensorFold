//! Where a ZONOS3 Mini release (a snapshot of its Hugging Face repo) keeps its parts, read from its own registries.
const std = @import("std");
const cfg = @import("config.zig");

pub const lm_dir = "lm/zonos3_mini";
/// The reference servers' default decoder: the pruned multi-resolution iSTFT head.
pub const default_decoder = "keep2_plain";
/// A decoder's registry: codec_registry.json, or codec_registry_<id>.json.
const registries = [_][]const u8{ "registries/codec_registry.json", "registries/codec_registry_{s}.json" };

pub const Release = struct {
    arena: std.heap.ArenaAllocator,
    root: []const u8,
    lm: []const u8, // params.json, model.pth and the tokenizer
    tokenizer: []const u8,
    generator: []const u8,
    head: []const u8, // the chosen decoder's checkpoint
    decoder: []const u8,
    silence: [cfg.codebooks]i32, // the codec's frame of digital silence
    headroom_db: f32,

    pub fn deinit(r: *Release) void {
        r.arena.deinit();
    }
};

/// A registry's path inside the release: relative, never out of the folder.
fn inside(path: []const u8) ![]const u8 {
    if (path.len == 0 or path[0] == '/' or std.mem.indexOf(u8, path, "..") != null or std.mem.indexOf(u8, path, "://") != null) return error.BadRegistry;
    return path;
}

fn string(o: std.json.ObjectMap, name: []const u8) ![]const u8 {
    const v = o.get(name) orelse return error.BadRegistry;
    return if (v == .string) v.string else error.BadRegistry;
}

fn readJson(a: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.ObjectMap {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    return if (doc == .object) doc.object else error.BadRegistry;
}

/// `decoder`: a decoder id (the folder of its head under autoencoder/heads), or null for the default.
pub fn open(gpa: std.mem.Allocator, io: std.Io, root: []const u8, decoder: ?[]const u8) !Release {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const id = decoder orelse default_decoder;
    for (id) |ch| if (!std.ascii.isAlphanumeric(ch) and ch != '_' and ch != '-') return error.NoSuchDecoder;
    const lm = try std.fs.path.join(a, &.{ root, lm_dir });
    const model = (try readJson(a, io, try std.fs.path.join(a, &.{ lm, "params.json" }))).get("model") orelse return error.BadConfig;
    const codec = try string(model.object, "audio_codec");
    const spec = model.object.get("text_tokenizer_spec") orelse return error.BadConfig;
    const tokenizer = try std.fs.path.join(a, &.{ lm, try inside(try string(spec.object, "path")) });

    var r: Release = .{ .arena = undefined, .root = try a.dupe(u8, root), .lm = lm, .tokenizer = tokenizer, .generator = "", .head = "", .decoder = try a.dupe(u8, id), .silence = undefined, .headroom_db = 0 };
    var found = false;
    for (registries) |pattern| {
        const name = if (std.mem.indexOf(u8, pattern, "{s}") != null) try std.fmt.allocPrint(a, "registries/codec_registry_{s}.json", .{id}) else pattern;
        const doc = readJson(a, io, try std.fs.path.join(a, &.{ root, name })) catch |e| switch (e) {
            error.FileNotFound => continue,
            else => return e,
        };
        var it = doc.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* != .object) continue;
            const entry = e.value_ptr.object;
            if (!std.mem.eql(u8, string(entry, "codec") catch "", codec)) continue;
            const d = (entry.get("decoder") orelse continue).object;
            const campaign = try inside(try string(d, "campaign"));
            if (!std.mem.eql(u8, std.fs.path.basename(campaign), id)) continue;
            r.generator = try std.fs.path.join(a, &.{ root, try inside(try string(entry, "checkpoint")) });
            r.head = try std.fs.path.join(a, &.{ root, campaign, try inside(try string(d, "checkpoint")) });
            if (d.get("headroom_db")) |h| r.headroom_db = switch (h) {
                .float => |f| @floatCast(f),
                .integer => |n| @floatFromInt(n),
                else => return error.BadRegistry,
            };
            found = true;
        }
    }
    if (!found) return error.NoSuchDecoder;
    const silence = try readJson(a, io, try std.fmt.allocPrint(a, "{s}/registries/silence_frame_{s}.json", .{ root, try inside(codec) }));
    const frame = silence.get("frame") orelse return error.BadRegistry;
    if (frame != .array or frame.array.items.len != cfg.codebooks) return error.BadRegistry;
    for (&r.silence, frame.array.items) |*code, v| {
        if (v != .integer or v.integer < 0 or v.integer >= cfg.codebook_size) return error.BadRegistry;
        code.* = @intCast(v.integer);
    }
    r.arena = arena;
    return r;
}

test "inside: a path in the release, never out of the folder" {
    const t = std.testing;
    try t.expectEqualStrings("autoencoder/generator.pt", try inside("autoencoder/generator.pt"));
    try t.expectError(error.BadRegistry, inside("autoencoder/../../x"));
    try t.expectError(error.BadRegistry, inside("/abs/path"));
    try t.expectError(error.BadRegistry, inside("hf://owner/repo/x"));
    try t.expectError(error.BadRegistry, inside(""));
}
