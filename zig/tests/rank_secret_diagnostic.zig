//! The actual TP parser's diagnostics must name the setting without disclosing its key-file path.
const std = @import("std");
const Config = @import("tp").Config;
const Missing = struct {
    fn get(name: []const u8) ?[]const u8 {
        return if (std.mem.eql(u8, name, "TF_RANK_SECRET_FILE")) "diagnostic-private-key-missing" else null;
    }
};
const Control = struct {
    fn get(name: []const u8) ?[]const u8 {
        return if (std.mem.eql(u8, name, "TF_RANK_SECRET_FILE")) "diagnostic-private\nkey" else null;
    }
};
pub fn main() !void {
    if (Config.parse(&Missing.get)) |_| return error.ExpectedRefusal else |e| if (e != error.BadSetting) return e;
    if (Config.parse(&Control.get)) |_| return error.ExpectedRefusal else |e| if (e != error.BadSetting) return e;
}
