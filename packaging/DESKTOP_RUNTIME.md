# Automatic desktop runtime publication

Stable engine releases must become downloadable from the existing app without an app rebuild. The release workflow prepares a Mac runtime, publishes it as an immutable GitHub asset and sends a signed catalog to the website. The website keeps this catalog in its existing Durable Object binding. Other website deployments cannot reset the catalog to an older static file.

The current approved stable platform is Apple silicon macOS. Windows and Linux preview qualification remains separate. Existing signed entries for other platforms are preserved.

## One-time activation

Deploy the website's reviewed runtime-store changes first. Configure the GitHub `runtime-release` environment with `TF_RUNTIME_SIGNING_KEY` containing the existing established Ed25519 release identity. Restrict this environment to the `main` branch, with no tag policy. The production signer refuses another key. Keep this secret limited to the signing job. The Mac preparation job receives no release signing key and executes the official engine only after checking its authenticated checksum.

Publish the reviewed workflow and tools on the default branch. That change automatically reconciles the latest stable engine, including a missed release. A manual dispatch with `v1.2.4` also backfills that exact release. Future published stable `v1.x.y` releases trigger it directly. The weekly run renews expiring catalog signatures without replacing existing runtime assets. It does not build the GUI, run models or publish Windows/Linux previews.

The release trigger has no signing secret and dispatches the publisher on `main`. Release events reconcile the highest stable engine version. A delayed event therefore cannot leave a newer release behind the queue. Exact tags remain available for explicit backfills.

Releases created with GitHub's repository token do not trigger another workflow. An automated release job using that token dispatches `runtime-feed.yml` on `main` after publishing, using actions-write permission. A release is complete only after the desktop publication job passes its live checks.

## Trust and failure behavior

The preparer verifies the official checksum's `tensorfold-native-checksum-v1` signature, complete archive hash, bounded safe extraction, exact version, native architecture and required licenses. It creates a deterministic flat archive and checks the app's schema-1 capabilities, Metal backend and grammar library using an isolated environment. This CPU probe does not substitute for the engine release's model qualification.

The Linux signing job reauthenticates the original archive and independently reconstructs and verifies every prepared file. It never executes the Mac binary. It reuses an existing immutable GitHub runtime only when its complete bytes match. It signs a catalog only after the canonical asset is downloadable and verified. Previous package records are retained unchanged and ordered by engine version.

The website accepts only a fresh catalog signed by the app's established root. Its transactional store refuses rollback, same-timestamp equivocation, package removal and changes to existing package IDs. Repeated delivery is idempotent. No separate website deployment token is needed for a signed catalog submission. Invalid requests cannot change the stored catalog.

Publication succeeds only after the signed live endpoint lists the exact package and its complete downloaded archive and inventory pass again. Preparation and publication each get three bounded attempts. Exhausted attempts leave the job failed; a rerun reuses verified existing assets. Engine-tag publication alone is not proof that this stage succeeded.

Run local offline fixtures with:

```sh
node --test tools/runtime_release/*.test.mjs
python3 -m unittest discover -s tools/runtime_release -p 'test_*.py'
```

The workflow and website endpoint require deployment before this automation is active. Preparing these files locally does not change a release, feed, key, running job or installed app.
