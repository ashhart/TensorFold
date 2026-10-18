# Unwired experiments

These twenty prototypes have their own tests but no production callers. They are preserved here for future work, excluded from app entrypoints, and still typechecked and included in the isolated test runner. Their presence does not mean TensorFold ships these features.

Integration requires a real host or renderer entrypoint, explicit permissions and lifecycle ownership, and tests for the user flow. Move the implementation and its tests back into the relevant production area together when that work is complete. The maintained ACP SDK server and standalone acp-server-main.ts remain production source.
