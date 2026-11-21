//! dots.mocr's OCR reader: our layout prompt, and the model's JSON block list to normalized Markdown blocks.
const std = @import("std");
const api = @import("engine_api");
const image = @import("image.zig");
const ocr = api.ocr;
const Allocator = std.mem.Allocator;
const log = std.log.scoped(.dots_ocr);

/// Our layout prompt (prompt.txt, also the oracle's `--prompt`); ocr_test.zig checks it against each oracle run.
pub const prompt = @embedFile("prompt.txt");

/// The reader the family offers (Offer.ocr).
pub const reader: ocr.Ocr = .{ .prompt = prompt, .parse = parse };

const categories = std.StaticStringMap(ocr.Kind).initComptime(.{
    .{ "Caption", .caption },
    .{ "Footnote", .footnote },
    .{ "Formula", .formula },
    .{ "List-item", .list },
    .{ "Page-footer", .page_footer },
    .{ "Page-header", .page_header },
    .{ "Picture", .picture },
    .{ "Section-header", .section_header },
    .{ "Table", .table },
    .{ "Text", .text },
    .{ "Title", .title },
});

/// The prepared image the tower reads, with the frame its boxes are in (the resized size, one patch a grid cell).
pub fn prepared(p: image.Prepared, patch: u32) api.PreparedImage {
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash, .frame = .{ .width = p.gw * patch, .height = p.gh * patch } };
}

/// The model's JSON array of blocks; ``a`` is a request arena (the parsed JSON tree stays in it).
pub fn parse(a: Allocator, text: []const u8, original: api.Size, frame: api.Size) anyerror![]ocr.OcrBlock {
    if (frame.width == 0 or frame.height == 0) return error.InvalidFrame;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, std.mem.trim(u8, text, &std.ascii.whitespace), .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.Unreadable,
    };
    if (doc != .array) return error.Unreadable;
    var blocks: std.ArrayList(ocr.OcrBlock) = try .initCapacity(a, doc.array.items.len);
    var unknown = false;
    for (doc.array.items) |item| {
        if (item != .object) return error.Unreadable;
        const name = item.object.get("category") orelse return error.Unreadable;
        if (name != .string) return error.Unreadable;
        const kind = categories.get(name.string) orelse kind: {
            if (!unknown) log.warn("unknown dots.mocr category \"{s}\" read as text", .{name.string});
            unknown = true;
            break :kind .text;
        };
        const quad = try box(item.object.get("bbox") orelse return error.Unreadable, original, frame);
        if (kind == .picture) {
            try blocks.append(a, .{ .content = "", .quad = quad, .kind = .picture });
            continue;
        }
        const field = item.object.get("text") orelse return error.Unreadable;
        if (field != .string) return error.Unreadable;
        const body = std.mem.trim(u8, field.string, &std.ascii.whitespace);
        const content = try normalize(a, kind, body);
        if (content.len == 0) continue; // the oracle shows blocks with empty text
        try blocks.append(a, .{ .content = content, .quad = quad, .kind = kind });
    }
    return blocks.toOwnedSlice(a);
}

/// A [x1, y1, x2, y2] box in frame pixels as a quad in original pixels; reversed corners are swapped.
fn box(v: std.json.Value, original: api.Size, frame: api.Size) ocr.Error!ocr.Quad {
    if (v != .array or v.array.items.len != 4) return error.Unreadable;
    var n: [4]f64 = undefined;
    for (v.array.items, &n) |x, *dst| dst.* = switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        .number_string => |s| std.fmt.parseFloat(f64, s) catch return error.Unreadable,
        else => return error.Unreadable,
    };
    for (n) |x| if (!std.math.isFinite(x)) return error.Unreadable;
    const sx = @as(f64, @floatFromInt(original.width)) / @as(f64, @floatFromInt(frame.width));
    const sy = @as(f64, @floatFromInt(original.height)) / @as(f64, @floatFromInt(frame.height));
    const x1: f32 = @floatCast(@min(n[0], n[2]) * sx);
    const x2: f32 = @floatCast(@max(n[0], n[2]) * sx);
    const y1: f32 = @floatCast(@min(n[1], n[3]) * sy);
    const y2: f32 = @floatCast(@max(n[1], n[3]) * sy);
    if (!std.math.isFinite(x2) or !std.math.isFinite(y2) or !std.math.isFinite(x1) or !std.math.isFinite(y1)) return error.Unreadable;
    return .{ .{ x1, y1 }, .{ x2, y1 }, .{ x2, y2 }, .{ x1, y2 } };
}

/// A block's trimmed text as normalized Markdown (docs/api.md OCR: Reader Markdown); empty: nothing to keep.
fn normalize(a: Allocator, kind: ocr.Kind, body: []const u8) anyerror![]const u8 {
    if (body.len == 0) return "";
    return switch (kind) {
        .title => heading(a, "# ", body),
        .section_header => heading(a, "## ", body),
        .formula => formula(a, body),
        .table => table(a, body),
        .list => if (listMarker(body)) body else std.mem.concat(a, u8, &.{ "- ", body }),
        else => body,
    };
}

fn heading(a: Allocator, marker: []const u8, body: []const u8) ![]const u8 {
    return if (body[0] == '#') body else std.mem.concat(a, u8, &.{ marker, body });
}

/// Display math with one outer `$$…$$`, `\[…\]` or `$…$` removed; LaTeX otherwise kept verbatim.
fn formula(a: Allocator, body: []const u8) ![]const u8 {
    const latex = std.mem.trim(u8, unwrap(body), &std.ascii.whitespace);
    if (latex.len == 0) return "";
    return std.mem.concat(a, u8, &.{ "$$\n", latex, "\n$$" });
}

fn unwrap(body: []const u8) []const u8 {
    for ([_][2][]const u8{ .{ "$$", "$$" }, .{ "\\[", "\\]" }, .{ "$", "$" } }) |pair| {
        if (body.len >= pair[0].len + pair[1].len and std.mem.startsWith(u8, body, pair[0]) and std.mem.endsWith(u8, body, pair[1]))
            return body[pair[0].len .. body.len - pair[1].len];
    }
    return body;
}

/// One HTML table as a GFM pipe table, or as clean HTML when it has spans; a table that does not parse is unreadable.
fn table(a: Allocator, body: []const u8) ![]const u8 {
    const t = ocr.html.parse(a, body) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.Malformed => return error.Unreadable,
    };
    return if (t.spanned) ocr.html.normalized(a, t) else ocr.html.pipe(a, t);
}

/// Whether a list item already starts with a Markdown marker: `-`, `*`, `+`, or digits and `.` or `)`, then a space.
fn listMarker(body: []const u8) bool {
    if (body.len >= 2 and std.mem.indexOfScalar(u8, "-*+", body[0]) != null and body[1] == ' ') return true;
    var i: usize = 0;
    while (i < body.len and std.ascii.isDigit(body[i])) i += 1;
    return i > 0 and i + 1 < body.len and (body[i] == '.' or body[i] == ')') and body[i + 1] == ' ';
}
