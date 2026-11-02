import assert from 'node:assert/strict';
import { desktopReleaseMetadata } from './desktop-release-metadata.mjs';

const stable = desktopReleaseMetadata('1.0.0');
assert.equal(stable.tag, 'app-v1.0.0');
assert.equal(stable.downloadUrl, 'https://github.com/ashhart/TensorFold/releases/download/app-v1.0.0/TensorFold-1.0.0-arm64.dmg');
assert.equal(stable.zipUrl, stable.downloadUrl.replace(/dmg$/, 'zip'));
assert.equal(desktopReleaseMetadata('1.0.0-beta.10').tag, 'v1.0.0-beta.10');
for (const value of ['1.0.0/other', 'v1.0.0', '1.0.0-rc.1', '01.0.0', '1.0.0+local', '1.0.0-beta.0']) {
  assert.throws(() => desktopReleaseMetadata(value), /canonical desktop release version/);
}
console.log('Desktop stable tags remain separate from engine tags and retain canonical immutable beta URLs.');
