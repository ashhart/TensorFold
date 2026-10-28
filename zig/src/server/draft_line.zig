//! The serving line names available proposals without promising a request will use one.
const std = @import("std");
const api = @import("engine_api");

pub fn shown(no_drafts: bool, info: api.Info) Shown {
    return .{ .off = no_drafts or info.plain_only, .sources = info.draft_sources };
}

const Shown = struct {
    off: bool,
    sources: api.DraftSources,

    pub fn format(s: Shown, w: *std.Io.Writer) std.Io.Writer.Error!void {
        if (s.off) return w.writeAll("off");
        const head = s.sources.mtp or s.sources.drafter != null;
        if (s.sources.mtp) try w.writeAll("MTP");
        if (s.sources.drafter) |d| {
            if (s.sources.mtp) try w.writeAll(" + ");
            try oneLine(w, d.kind);
            if (d.name.len > 0) {
                try w.writeAll(" (");
                try oneLine(w, d.name);
                try w.writeByte(')');
            }
        }
        if (s.sources.copies) try w.writeAll(if (head) " + copies" else "copies only");
        if (!head and !s.sources.copies) try w.writeAll("none");
    }
};

fn oneLine(w: *std.Io.Writer, text: []const u8) std.Io.Writer.Error!void {
    for (text) |c| try w.writeByte(if (std.ascii.isControl(c)) ' ' else c);
}

test "draft sources: serving line text reflects engine info and off overrides" {
    const cases = [_]struct { sources: api.DraftSources, no_drafts: bool = false, plain_only: bool = false, text: []const u8 }{
        .{ .sources = .{ .copies = true }, .text = "copies only" },
        .{ .sources = .{ .copies = true, .mtp = true }, .text = "MTP + copies" },
        .{ .sources = .{ .mtp = true }, .text = "MTP" },
        .{ .sources = .{ .copies = true, .drafter = .{ .kind = "DFlash2", .name = "z-lab/Qwen3.8-27B-DFlash2" } }, .text = "DFlash2 (z-lab/Qwen3.8-27B-DFlash2) + copies" },
        .{ .sources = .{ .drafter = .{ .kind = "DSpark" } }, .text = "DSpark" },
        .{ .sources = .{ .drafter = .{ .kind = "EAGLE", .name = "example/draft" } }, .text = "EAGLE (example/draft)" },
        .{ .sources = .{}, .text = "none" },
        .{ .sources = .{ .copies = true, .mtp = true }, .no_drafts = true, .text = "off" },
        .{ .sources = .{ .copies = true, .mtp = true }, .plain_only = true, .text = "off" },
        .{ .sources = .{ .drafter = .{ .kind = "DFlash2", .name = "example/one\ntwo\r\x1b" } }, .text = "DFlash2 (example/one two  )" },
    };
    for (cases) |c| {
        var buf: [256]u8 = undefined;
        const text = try std.fmt.bufPrint(&buf, "{f}", .{shown(c.no_drafts, .{ .plain_only = c.plain_only, .draft_sources = c.sources })});
        try std.testing.expectEqualStrings(c.text, text);
    }
}

test "draft sources: resolved directories retain the original drafter name in engine options" {
    const cli = @import("cli.zig");
    const a = std.testing.allocator;
    const o = try cli.request(a, "cache/target", "qwen3_5", .{ .drafter = "cache/drafter", .drafter_name = "example/DFlash2" });
    try std.testing.expectEqualStrings("cache/drafter", o.drafter.?);
    try std.testing.expectEqualStrings("example/DFlash2", o.drafter_name.?);
    const direct = try cli.request(a, "cache/target", "qwen3_5", .{ .drafter = "local-drafter" });
    try std.testing.expectEqualStrings("local-drafter", direct.drafter_name.?);
}
