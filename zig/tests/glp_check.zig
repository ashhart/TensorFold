//! The GLP control vector container as the outside world sees it: the reader's view of a file, and a rewrite of it.
const std = @import("std");
const glp = @import("glp");

const help =
    \\tf-glp-check read FILE OUT.json OUT.bin  the reader's view: the metadata, the layers and their numbers
    \\tf-glp-check rewrite FILE OUT.gguf [--repo ID] [--revision REV]
    \\                                    the file again through this writer
    \\
    \\--repo and --revision re-stamp the checkpoint the file names, for weights published under another
    \\name: the directions stay the same bytes, so only do it for a checkpoint with the same weights.
    \\
    \\The binary sidecar holds the directions in ascending layer order, each one's fp32 payload as it
    \\stands in the file, so another reader's numbers can be compared byte for byte.
;

/// What the command line asks for, settled before any file is opened.
const Plan = union(enum) {
    help,
    read: struct { file: []const u8, json: []const u8, blob: []const u8 },
    rewrite: struct { file: []const u8, out: []const u8, repo: ?[]const u8, revision: ?[]const u8 },
};

/// The plan the arguments name; otherwise `why` says what they got wrong.
fn parse(args: []const [:0]const u8, why: *[]const u8) error{Usage}!Plan {
    if (args.len < 2 or std.mem.eql(u8, args[1], "--help")) return .help;
    const operands = args[2..];
    if (std.mem.eql(u8, args[1], "read")) {
        if (operands.len != 3) {
            why.* = "read takes FILE OUT.json OUT.bin";
            return error.Usage;
        }
        return .{ .read = .{ .file = operands[0], .json = operands[1], .blob = operands[2] } };
    }
    if (std.mem.eql(u8, args[1], "rewrite")) {
        if (operands.len < 2) {
            why.* = "rewrite takes FILE OUT.gguf [--repo ID] [--revision REV]";
            return error.Usage;
        }
        var plan: Plan = .{ .rewrite = .{ .file = operands[0], .out = operands[1], .repo = null, .revision = null } };
        var i: usize = 2;
        while (i < operands.len) : (i += 2) {
            const flag = operands[i];
            const is_repo = std.mem.eql(u8, flag, "--repo");
            if (!is_repo and !std.mem.eql(u8, flag, "--revision")) {
                why.* = "rewrite takes only --repo and --revision after its files";
                return error.Usage;
            }
            if (i + 1 >= operands.len) {
                why.* = "--repo and --revision each take a value";
                return error.Usage;
            }
            if (is_repo) plan.rewrite.repo = operands[i + 1] else plan.rewrite.revision = operands[i + 1];
        }
        return plan;
    }
    why.* = "the command is read or rewrite";
    return error.Usage;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var problem: []const u8 = "";
    const plan = parse(args, &problem) catch {
        try std.Io.File.stderr().writeStreamingAll(io, try std.fmt.allocPrint(a, "tf-glp-check: {s}\n{s}\n", .{ problem, help }));
        std.process.exit(2);
    };
    const file = switch (plan) {
        .help => return std.Io.File.stdout().writeStreamingAll(io, help ++ "\n"),
        .read => |p| p.file,
        .rewrite => |p| p.file,
    };
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, file, a, .limited(64 << 20));
    var why: glp.Why = .{};
    const vector = glp.read(a, bytes, &why) catch |err| {
        if (err == error.OutOfMemory) return err;
        const line = try std.fmt.allocPrint(a, "refused: {s}\n", .{why.text()});
        try std.Io.File.stdout().writeStreamingAll(io, line);
        return error.BadVector;
    };
    switch (plan) {
        .help => unreachable,
        .rewrite => |p| {
            // The checkpoint the file names may be re-stamped: the same weights under another name.
            const out = try glp.write(a, .{
                .spec_version = vector.spec_version,
                .alpha = vector.alpha,
                .hook_point = vector.hook_point,
                .derived_at = vector.derived_at,
                .base_model = p.repo orelse vector.base_model,
                .base_revision = p.revision orelse vector.base_revision,
                .method = vector.method,
                .contrast = vector.contrast,
                .publication = vector.publication,
                .layers = vector.layers,
            });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p.out, .data = out });
        },
        .read => |p| {
            // The directions as bytes, in the order the file holds them.
            var blob: std.ArrayList(u8) = .empty;
            for (vector.layers) |l| {
                for (l.dir) |v| try blob.appendSlice(a, &std.mem.toBytes(@as(u32, @bitCast(v))));
            }
            var ids: std.ArrayList(u32) = .empty;
            for (vector.layers) |l| try ids.append(a, l.id);
            const receipt = try std.json.Stringify.valueAlloc(a, .{
                .spec_version = vector.spec_version,
                .alpha = vector.alpha,
                .hook_point = vector.hook_point,
                .derived_at = vector.derived_at,
                .base_model = vector.base_model,
                .base_revision = vector.base_revision,
                .method = vector.method,
                .contrast = vector.contrast,
                .width = vector.width,
                .sha256 = std.fmt.bytesToHex(vector.sha256, .lower),
                .file_sha256 = std.fmt.bytesToHex(vector.file_sha256, .lower),
                .transferred = vector.transferred,
                .name = vector.publication.name,
                .description = vector.publication.description,
                .author = vector.publication.author,
                .license = vector.publication.license,
                .model_hint = vector.publication.model_hint,
                .measured_effect = vector.publication.measured_effect,
                .created = vector.publication.created,
                .layers = ids.items,
                .bytes = blob.items.len,
            }, .{});
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p.json, .data = receipt });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p.blob, .data = blob.items });
        },
    }
}
