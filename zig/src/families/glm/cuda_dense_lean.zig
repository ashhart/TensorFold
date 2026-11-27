//! GLM's opt-in dense prompt geometry; the shared parser rejects every value except blank, 0 and 1.
const environment = @import("sys").environment;
pub const name = "TF_GLM_DENSE_LEAN_TILE";
pub fn wanted() !bool {
    return environment.read(bool, name, false);
}
