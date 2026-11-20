//! The DeepSeek paired GLP file, strictly: version 3, the add operation, rank one, at the hc fold site and its writer.

const std = @import("std");
const gguf = @import("glp_container.zig");
const glp = @import("glp.zig");
const Why = glp.Why;
const Error = glp.Error;

fn refuse(why: *Why, comptime fmt: []const u8, args: anytype) Error {
    why.set(fmt, args);
    return error.BadGlpVector;
}

/// The text a key holds, null when the key is absent; a key of another type is refused, not skipped.
fn text(file: gguf.File, key: []const u8, why: *Why) Error!?[]const u8 {
    const v = file.find(key) orelse return null;
    return switch (v) {
        .string => |s| s,
        else => return refuse(why, "{s} is not text", .{key}),
    };
}

/// The unsigned integer a key holds, null when the key is absent; text, a float or a negative number is refused.
fn count(file: gguf.File, key: []const u8, why: *Why) Error!?u64 {
    if (file.find(key) == null) return null;
    return file.num(key) orelse return refuse(why, "{s} is not an unsigned integer", .{key});
}

/// The layer id a tensor name carries: `direction.N` with N written canonically, or a refusal naming why.
fn directionId(name: []const u8, why: *Why) Error!u32 {
    const prefix = "direction.";
    if (!std.mem.startsWith(u8, name, prefix)) {
        return refuse(why, "the file holds the tensor \"{f}\", which is not a direction", .{std.zig.fmtString(name)});
    }
    const tail = name[prefix.len..];
    if (std.mem.indexOfScalar(u8, tail, '.') != null) {
        return refuse(why, "the file holds \"{f}\": a rank above one; this reader implements rank 1", .{std.zig.fmtString(name)});
    }
    for (tail) |ch| if (ch < '0' or ch > '9') return refuse(why, "\"{f}\" does not name a layer", .{std.zig.fmtString(name)});
    if (tail.len == 0 or (tail.len > 1 and tail[0] == '0')) {
        return refuse(why, "\"{f}\" does not name a layer as direction.N", .{std.zig.fmtString(name)});
    }
    return std.fmt.parseInt(u32, tail, 10) catch return refuse(why, "\"{f}\" names a layer past 32 bits", .{std.zig.fmtString(name)});
}

/// Keys that carry another reading of the file, or an interpretation this reader lacks, and a key given twice.
fn checkKeys(a: std.mem.Allocator, file: gguf.File, why: *Why) Error!void {
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(a);
    try seen.ensureTotalCapacity(a, @intCast(file.entries.len));
    for (file.entries) |e| {
        if (std.mem.startsWith(u8, e.key, "dspark.")) {
            return refuse(why, "the file carries dspark.* keys: a pre-GLP internal build, to be re-exported as GLP", .{});
        }
        if (std.mem.eql(u8, e.key, "glp.dir_scales") or std.mem.eql(u8, e.key, "glp.layer_scales")) {
            return refuse(why, "the file carries {s}: per-direction and per-layer scales are not implemented", .{e.key});
        }
        if (seen.getOrPutAssumeCapacity(e.key).found_existing) {
            return refuse(why, "the file holds the key \"{f}\" twice", .{std.zig.fmtString(e.key)});
        }
    }
}

/// The hook the file declares, once its architecture, operation and rank are ones this reader applies.
fn declared(file: gguf.File, why: *Why) Error![]const u8 {
    const arch = try text(file, "general.architecture", why) orelse return refuse(why, "the file names no general.architecture", .{});
    if (!std.mem.eql(u8, arch, "controlvector")) return refuse(why, "the file's architecture is not controlvector", .{});

    // Never infer the operation: a missing mode could silently turn an addition into a projection.
    const mode = try text(file, "glp.mode", why) orelse return refuse(why, "the file names no glp.mode; this reader applies add vectors only", .{});
    if (std.mem.eql(u8, mode, "project")) return refuse(why, "the file's mode is project; this reader applies add vectors only", .{});
    if (!std.mem.eql(u8, mode, "add")) return refuse(why, "the file's mode is not one this reader applies; it applies add vectors only", .{});

    const rank = try count(file, "glp.rank", why) orelse 1;
    if (rank != 1) return refuse(why, "glp.rank {d}; this reader implements rank 1", .{rank});
    if (file.find("glp.orthonormal")) |v| if (std.meta.activeTag(v) != .boolv) return refuse(why, "glp.orthonormal is not a boolean", .{});

    const hook = try text(file, "glp.hook_point", why) orelse return refuse(why, "the file names no glp.hook_point", .{});
    if (!std.mem.eql(u8, hook, glp.paired_site)) {
        return refuse(why, "the hook point \"{f}\" is not applied; version 3 applies at {s} only", .{ std.zig.fmtString(hook), glp.paired_site });
    }
    return glp.paired_site;
}

/// File strength defaults to 1; otherwise it is zero or a normal FP32 from 0, exactly as --glp-alpha is.
fn alphaDefault(file: gguf.File, why: *Why) Error!f32 {
    const v = file.find("glp.alpha_default") orelse return 1.0;
    const wide: f64 = switch (v) {
        .f32v => |x| x,
        .f64v => |x| x,
        else => return refuse(why, "glp.alpha_default is not a floating-point strength", .{}),
    };
    return glp.strength(wide) orelse
        return refuse(why, "glp.alpha_default is not a finite strength from 0 that FP32 holds: zero, or a normal float", .{});
}

const Found = struct { layers: []glp.Layer, width: usize };

/// The layers the file's tensors carry, ascending: every tensor a finite one-dimensional FP32 direction of one width.
fn directions(a: std.mem.Allocator, file: gguf.File, why: *Why) Error!Found {
    var layers: std.ArrayList(glp.Layer) = .empty;
    var width: usize = 0;
    for (file.tensors) |tensor| {
        const id = try directionId(tensor.name, why);
        if (id == 0) return refuse(why, "the file holds direction.0; layer 0 is not steerable", .{});
        if (tensor.kind != gguf.f32_type) return refuse(why, "{s} is not fp32", .{tensor.name});
        if (tensor.dims.len != 1) return refuse(why, "{s} is not one dimensional", .{tensor.name});
        if (width == 0) width = @intCast(tensor.dims[0]);
        if (width != tensor.dims[0]) return refuse(why, "{s} is {d} wide; the other directions are {d}", .{ tensor.name, tensor.dims[0], width });
        const raw = file.bytesOf(tensor) orelse return refuse(why, "{s} points past the file", .{tensor.name});
        const dir = try a.alloc(f32, width);
        for (dir, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
        for (dir) |v| if (!std.math.isFinite(v)) return refuse(why, "layer {d}'s direction holds a value that is not finite", .{id});
        try layers.append(a, .{ .id = id, .dir = dir });
    }
    if (layers.items.len == 0) return refuse(why, "the file holds no direction tensors", .{});
    std.mem.sort(glp.Layer, layers.items, {}, struct {
        fn lessThan(_: void, x: glp.Layer, y: glp.Layer) bool {
            return x.id < y.id;
        }
    }.lessThan);
    for (layers.items[1..], 1..) |l, i| {
        if (l.id == layers.items[i - 1].id) return refuse(why, "the file holds two directions for layer {d}", .{l.id});
    }
    return .{ .layers = layers.items, .width = width };
}

/// glp.layer_ids_zero_based, when the file carries it, names exactly the layers the tensors do, in any order.
fn checkLayerIds(a: std.mem.Allocator, file: gguf.File, layers: []const glp.Layer, why: *Why) Error!void {
    const list = try text(file, "glp.layer_ids_zero_based", why) orelse return;
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |part| {
        const id = std.fmt.parseInt(u32, std.mem.trim(u8, part, " "), 10) catch return refuse(why, "the file's layer id list is not a number list", .{});
        try ids.append(a, id);
    }
    if (ids.items.len > layers.len) return refuse(why, "the file names more layers than it holds", .{});
    if (ids.items.len < layers.len) return refuse(why, "the file names {d} layers and holds {d}", .{ ids.items.len, layers.len });
    std.mem.sort(u32, ids.items, {}, std.sort.asc(u32));
    for (ids.items, layers) |id, l| {
        if (id != l.id) return refuse(why, "the file's layer id {d} does not match its tensor's {d}", .{ id, l.id });
    }
}

/// The digest over the tensors' bytes, in the file's order; null when a tensor points past the file.
fn digest(file: gguf.File) ?[32]u8 {
    var h: std.crypto.hash.sha2.Sha256 = .init(.{});
    for (file.tensors) |tensor| h.update(file.bytesOf(tensor) orelse return null);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

/// The base checkpoint as the file names it: the repo id and the commit pin, either empty when unnamed.
const Base = struct { repo: []const u8 = "", revision: []const u8 = "" };

/// The base model GGUF's own keys name: one repo, by URL or by organization and name, and its version.
fn canonicalBase(a: std.mem.Allocator, file: gguf.File, why: *Why) Error!Base {
    const prefix = "general.base_model.0.";
    const url = try text(file, prefix ++ "repo_url", why);
    const org = try text(file, prefix ++ "organization", why);
    const name = try text(file, prefix ++ "name", why);
    const version = try text(file, prefix ++ "version", why);
    if (try count(file, "general.base_model.count", why)) |n| {
        if (n > 1) return refuse(why, "the file names {d} base models; a vector is for one checkpoint", .{n});
        if (n == 0 and (url != null or org != null or name != null or version != null)) {
            return refuse(why, "general.base_model.count is 0 but the file names a base model", .{});
        }
    }
    var repo: []const u8 = "";
    if (url) |u| repo = glp.repoId(u);
    if (org != null and name != null and org.?.len > 0 and name.?.len > 0) {
        const joined = try std.fmt.allocPrint(a, "{s}/{s}", .{ org.?, name.? });
        if (repo.len == 0) {
            repo = joined;
        } else if (std.mem.startsWith(u8, url.?, glp.hub_prefix) and !std.ascii.eqlIgnoreCase(repo, joined)) {
            return refuse(why, "general.base_model.0 names {f} by its URL and {f} by its organization and name", .{ std.zig.fmtString(repo), std.zig.fmtString(joined) });
        }
    } else if (repo.len == 0) {
        repo = name orelse "";
    }
    return .{ .repo = repo, .revision = version orelse "" };
}

/// The base checkpoint: GGUF's keys, and glp.base_model and glp.base_revision where named too; they must agree.
fn baseOf(a: std.mem.Allocator, file: gguf.File, why: *Why) Error!Base {
    var base = try canonicalBase(a, file, why);
    if (try text(file, "glp.base_model", why)) |own| if (own.len > 0) {
        const repo = glp.repoId(own);
        if (base.repo.len == 0) {
            base.repo = repo;
        } else if (!std.ascii.eqlIgnoreCase(base.repo, repo)) {
            return refuse(why, "the file names the base model {f} as glp.base_model and {f} as general.base_model.0", .{ std.zig.fmtString(repo), std.zig.fmtString(base.repo) });
        }
    };
    if (try text(file, "glp.base_revision", why)) |own| if (own.len > 0) {
        if (base.revision.len == 0) {
            base.revision = own;
        } else if (!std.ascii.eqlIgnoreCase(base.revision, own)) {
            return refuse(why, "the file pins the base revision {f} as glp.base_revision and {f} as general.base_model.0.version", .{ std.zig.fmtString(own), std.zig.fmtString(base.revision) });
        }
    };
    return base;
}

/// Read the version 3 file the container `file` holds and accept it, or refuse it; `bytes` are the file's own.
pub fn read(a: std.mem.Allocator, bytes: []const u8, file: gguf.File, why: *Why) Error!glp.Vector {
    try checkKeys(a, file, why);
    const hook = try declared(file, why);
    const alpha = try alphaDefault(file, why);
    const found = try directions(a, file, why);
    try checkLayerIds(a, file, found.layers, why);

    const sha = digest(file) orelse return refuse(why, "the file's tensors point past its end", .{});
    const claimed = try text(file, "glp.content_sha256", why);
    if (claimed) |hex_claimed| {
        const hex = std.fmt.bytesToHex(sha, .lower);
        if (!std.ascii.eqlIgnoreCase(hex[0..], hex_claimed)) return refuse(why, "the file's content digest does not match its vectors", .{});
    }

    const derived = try text(file, "glp.derived_at", why) orelse hook;
    var file_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &file_sha256, .{});
    const base = try baseOf(a, file, why);
    return .{
        .spec_version = glp.paired_version,
        .alpha = alpha,
        .hook_point = hook,
        .derived_at = derived,
        .base_model = base.repo,
        .base_revision = base.revision,
        .method = try text(file, "glp.method", why) orelse "",
        .contrast = try text(file, "glp.contrast", why) orelse "",
        .width = found.width,
        .layers = found.layers,
        .sha256 = sha,
        .file_sha256 = file_sha256,
        .digest_claimed = claimed != null,
        .transferred = !std.mem.eql(u8, derived, hook),
        .publication = .{
            .name = try text(file, "general.name", why),
            .description = try text(file, "general.description", why),
            .author = try text(file, "general.author", why),
            .license = try text(file, "general.license", why),
            .model_hint = try text(file, "controlvector.model_hint", why),
            .measured_effect = try text(file, "glp.measured_effect", why),
            .created = try text(file, "glp.created", why),
        },
    };
}
