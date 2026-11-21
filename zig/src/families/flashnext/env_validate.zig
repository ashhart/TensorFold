//! Validate optional Flash Next settings before opening resources, including settings read only by later paths.
const sys = @import("sys");
const environment = sys.environment;

/// Every TF_FLASHNEXT_* row of the shared table, parsed exactly as each later reader parses it.
pub fn startup() !void {
    try environment.checkScope(.flashnext, environment.Process{});
}
