//! A family's image header checked before its state payload is loaded or a target state allocated.
const file = @import("rows_file.zig");
const chain = @import("rows_chain.zig");
const Header = @import("rows_store.zig").Image.Header;

pub fn check(dir: []const u8, name: chain.Name, at: u32, h: Header) !void {
    var path: [1200]u8 = undefined;
    const p = try file.path(&path, dir, &chain.hex(name), ".st", false);
    var opened = try file.open(p, .state, name);
    defer opened.close();
    if (opened.trailer.at != at) return error.RowsRead;
    var bytes: [file.alignment]u8 align(file.alignment) = undefined;
    try h.check(try file.prefix(&opened, h.bytes, &bytes));
}
