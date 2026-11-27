//! Validate optional Flash Next settings before opening resources, including settings read only by later paths.
const sys = @import("sys");
const environment = sys.environment;

/// Every TF_FLASHNEXT_* row of the shared table, parsed exactly as each later reader parses it.
pub fn startup() !void {
    const process = environment.Process{};
    environment.checkScope(.flashnext, process) catch |err| {
        // The loader's longstanding public error survives the shared scope check.
        _ = environment.read(usize, "TF_FLASHNEXT_LOAD_THREADS", 8) catch return error.BadLoadThreads;
        return err;
    };
}
