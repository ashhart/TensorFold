//! Flash Next owns this opt-in: shared families and captured entries retain their launch policy.
const cuda = @import("cuda");
const env = @import("sys").environment;

pub fn enabled(on: bool, capability: u32) bool {
    return on and capability >= 90;
}

pub fn read(d: *const cuda.Driver) !bool {
    const on = try env.read(bool, "TF_FLASHNEXT_PDL", false);
    return on and enabled(on, try d.currentCapability());
}
