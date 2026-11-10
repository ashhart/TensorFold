//! tf-h3-image-prompt's edit inputs: references' vision rows (tf-h3-image-vision), anchors (tf-h3-image-encoder).
const std = @import("std");
const mtl = @import("metal");
const hp = @import("h3_prompt");
const fixture = @import("checkpoint_fixture");

const rules = hp.rules;

/// Each `--reference FILE` and `--anchors PREFIX` on the command line, in order.
pub const Options = struct { references: []const []const u8 = &.{}, anchors: []const []const u8 = &.{} };

/// `args` without its edit options, and those options; error.BadArguments when one lacks its value.
pub fn take(a: std.mem.Allocator, args: []const []const u8) !struct { []const []const u8, Options } {
    var rest: std.ArrayList([]const u8) = .empty;
    var refs: std.ArrayList([]const u8) = .empty;
    var prefixes: std.ArrayList([]const u8) = .empty;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const list = if (std.mem.eql(u8, args[i], "--reference")) &refs else if (std.mem.eql(u8, args[i], "--anchors")) &prefixes else {
            try rest.append(a, args[i]);
            continue;
        };
        i += 1;
        if (i == args.len) return error.BadArguments;
        try list.append(a, args[i]);
    }
    return .{ rest.items, .{ .references = refs.items, .anchors = prefixes.items } };
}

/// The references' vision rows in upload order: each one's token count and patch grid, and every row back to back.
pub const References = struct {
    counts: []usize,
    grids: [][2]u32,
    rows: []u16,
    deep: [3][]u16,

    /// Each tf-h3-image-vision file: grid I32 [3] (1, h, w patches), rows BF16 [tokens, hidden], deepstack [3, ...].
    pub fn read(gpa: std.mem.Allocator, a: std.mem.Allocator, device: mtl.Device, paths: []const []const u8, hidden: usize) !References {
        var r: References = .{ .counts = try a.alloc(usize, paths.len), .grids = try a.alloc([2]u32, paths.len), .rows = &.{}, .deep = .{ &.{}, &.{}, &.{} } };
        var total: usize = 0;
        var files = try a.alloc(hp.tensors.Tensors, paths.len);
        var opened: usize = 0;
        defer for (files[0..opened]) |*f| f.deinit();
        for (paths, 0..) |path, k| {
            files[k] = try hp.tensors.Tensors.open(gpa, device, try a.dupeSentinel(u8, path, 0));
            opened += 1;
            const g = try files[k].copy(i32, a, try files[k].at("grid", .i32, 3), 3);
            if (g[0] != 1 or g[1] <= 0 or g[2] <= 0) return error.H3BadVisionGrid;
            r.grids[k] = .{ @intCast(g[1]), @intCast(g[2]) };
            r.counts[k] = try rules.visionTokens(r.grids[k][0], r.grids[k][1]);
            total += r.counts[k];
        }
        r.rows = try a.alloc(u16, total * hidden);
        for (&r.deep) |*d| d.* = try a.alloc(u16, total * hidden);
        var at: usize = 0;
        for (files, r.counts) |*f, n| {
            const m = n * hidden;
            @memcpy(r.rows[at..][0..m], try f.copy(u16, a, try f.at("rows", .bf16, m), m));
            const deep = try f.copy(u16, a, try f.at("deepstack", .bf16, 3 * m), 3 * m);
            for (&r.deep, 0..) |*d, j| @memcpy(d.*[at..][0..m], deep[j * m ..][0..m]);
            at += m;
        }
        return r;
    }

    /// The encoder's vision input for presentation `p`: its image pads, the rows and DeepStack's; null with none.
    pub fn vision(r: References, a: std.mem.Allocator, p: rules.Presentation, marks: rules.Marks) !?hp.Vision {
        if (r.counts.len == 0) return null;
        return .{ .pads = try rules.pads(a, p, marks, r.counts), .rows = r.rows, .deepstack = .{ r.deep[0], r.deep[1], r.deep[2] } };
    }
};

/// Each tf-h3-image-encoder prefix's grid and anchor rows as a case's `references` I32 [n, 2] and `anchors` F32.
pub fn anchors(io: std.Io, a: std.mem.Allocator, prefixes: []const []const u8, entries: *std.ArrayList(fixture.Entry)) !void {
    if (prefixes.len == 0) return;
    const grids = try a.alloc(i32, 2 * prefixes.len);
    var rows: std.ArrayList(f32) = .empty;
    for (prefixes, 0..) |prefix, k| {
        const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}.grid.json", .{prefix}), a, .limited(1 << 16));
        const g = try std.json.parseFromSliceLeaky(struct { rows: usize, columns: usize }, a, text, .{ .ignore_unknown_fields = true });
        const want = try hp.picture.gridRows(.{ .rows = g.rows, .columns = g.columns }) * hp.picture.row_width;
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}.anchors.f32", .{prefix}), a, .limited(1 << 30));
        if (bytes.len != want * 4) {
            std.debug.print("{s}.anchors.f32 holds {d} bytes; its grid {d} x {d} needs {d}\n", .{ prefix, bytes.len, g.rows, g.columns, want * 4 });
            return error.H3BadAnchors;
        }
        const at = rows.items.len;
        try rows.resize(a, at + want);
        @memcpy(std.mem.sliceAsBytes(rows.items[at..]), bytes);
        grids[2 * k ..][0..2].* = .{ @intCast(g.rows), @intCast(g.columns) };
    }
    try entries.appendSlice(a, &.{
        .{ .name = "references", .dtype = "I32", .shape = try a.dupe(usize, &.{ prefixes.len, 2 }), .bytes = std.mem.sliceAsBytes(grids) },
        .{ .name = "anchors", .dtype = "F32", .shape = try a.dupe(usize, &.{ rows.items.len / hp.picture.row_width, hp.picture.row_width }), .bytes = std.mem.sliceAsBytes(rows.items) },
    });
}
