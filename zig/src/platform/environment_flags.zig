//! Which scopes a startup checks: each backend or feature switch the server or a family entry point turns on.
const Scope = @import("environment.zig").Scope;
const Rule = @import("environment.zig").Rule;

pub const Flags = struct {
    cuda: bool = false,
    metal: bool = false,
    xpu: bool = false,
    cpu: bool = false,
    slide: bool = false,
    hooks: bool = false,
    dsv41: bool = false,
    gemma4: bool = false,
    drama3: bool = false,
    glm: bool = false,
    flashnext: bool = false,
    nemotron: bool = false,
    minimax_m3: bool = false,
    chat: bool = false,
    hub: bool = false,
    device: ?u32 = null,
    segments: ?u32 = null,
    carveout: bool = false,
    /// --parallel named: TENSORFOLD_PARALLEL then stays out of the way, as a flag-named device does.
    parallel_named: bool = false,

    pub fn includes(f: Flags, scope: Scope) bool {
        return switch (scope) {
            .common => true,
            .cuda => f.cuda,
            .metal => f.metal,
            .xpu => f.xpu,
            .cpu => f.cpu,
            .slide => f.slide,
            .hooks => f.hooks,
            .dsv41 => f.dsv41,
            .gemma4 => f.gemma4,
            .drama3 => f.drama3,
            .glm => f.glm,
            .flashnext => f.flashnext,
            .nemotron => f.nemotron,
            .minimax_m3 => f.minimax_m3,
            .chat => f.chat,
            .hub => f.hub,
        };
    }

    pub fn includesRule(f: Flags, r: Rule) bool {
        return f.includes(r.scope) or if (r.also) |s| f.includes(s) else false;
    }
};
