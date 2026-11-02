export function desktopReleaseMetadata(version) {
  const beta = /^1\.0\.0-beta\.[1-9]\d*$/.test(version);
  const stable = /^[1-9]\d*\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)$/.test(version);
  if (!beta && !stable) throw new Error('Expected a canonical desktop release version.');
  const tag = `${beta ? 'v' : 'app-v'}${version}`;
  const assetRoot = `https://github.com/ashhart/TensorFold/releases/download/${tag}`;
  return {
    tag,
    downloadUrl: `${assetRoot}/TensorFold-${version}-arm64.dmg`,
    zipUrl: `${assetRoot}/TensorFold-${version}-arm64.zip`,
    releaseNotesUrl: `https://github.com/ashhart/TensorFold/releases/tag/${tag}`,
  };
}
