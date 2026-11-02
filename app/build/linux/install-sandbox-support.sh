#!/bin/bash
set -euo pipefail
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
unset BASH_ENV ENV CDPATH
umask 077
base=/opt/tensorfold-appimage
stage=
policy_tmp=
new_version=0
new_policy=0
complete=0
fail() { printf '%s\n' "$1" >&2; exit 1; }
valid_hash() { [[ "$1" =~ ^[a-f0-9]{64}$ ]] || fail 'Invalid AppImage digest.'; }
policy_text() {
  printf 'abi <abi/4.0>,\ninclude <tunables/global>\nprofile tensorfold_appimage_%s %s/%s/app/tensorfold-studio flags=(unconfined) {\n  userns,\n}\n' "$1" "$base" "$1"
}
if [[ ${1:-} == --policy && $# == 2 ]]; then valid_hash "$2"; policy_text "$2"; exit 0; fi
[[ ${1:-} == --install && $# == 3 || ${1:-} == --remove && $# == 2 ]] || fail 'Choose --install IMAGE DIGEST or --remove DIGEST.'
[[ $(id -u) == 0 ]] || fail 'Administrator approval is required for AppImage sandbox integration.'
operation=$1
if [[ $operation == --install ]]; then source_image=$2; digest=$3; else digest=$2; fi
valid_hash "$digest"
version=$base/$digest
policy=/etc/apparmor.d/tensorfold-appimage-$digest
safe_directory() {
  local directory=$1 mode
  [[ -d $directory && ! -L $directory && $(stat -c %u -- "$directory") == 0 ]] || fail 'A sandbox integration directory is not trusted.'
  mode=$(stat -c %a -- "$directory")
  (( (8#$mode & 022) == 0 )) || fail 'A sandbox integration directory is writable by other users.'
}
safe_file() {
  local file=$1 mode
  [[ -f $file && ! -L $file && $(stat -c %u -- "$file") == 0 ]] || fail 'An existing sandbox integration file is not trusted.'
  mode=$(stat -c %a -- "$file")
  (( (8#$mode & 022) == 0 )) || fail 'An existing sandbox integration file is writable by other users.'
}
safe_directory /opt
safe_directory /etc/apparmor.d
if [[ -e $base || -L $base ]]; then safe_directory "$base"; else install -d -m0755 -- "$base"; fi
[[ ! -L $base/.lock ]] || fail 'The integration lock is invalid.'
if [[ -e $base/.lock ]]; then safe_file "$base/.lock"; fi
exec 9> "$base/.lock"
flock -w 30 9 || fail 'Another sandbox integration operation is still running.'
if [[ -e $version || -L $version ]]; then
  safe_directory "$version"
  safe_file "$version/TensorFold.AppImage"
  safe_directory "$version/app"
  safe_file "$version/app/tensorfold-studio"
  [[ -z $(find "$version/app" \( ! -user root -o \( ! -type l -a -perm /022 \) \) -print -quit) ]] || fail 'The installed AppImage payload is writable or has an unexpected owner.'
  [[ $(sha256sum -- "$version/TensorFold.AppImage" | cut -d' ' -f1) == "$digest" ]] || fail 'The installed AppImage digest changed. Ask an administrator to review it.'
fi
if [[ -e $policy || -L $policy ]]; then
  safe_file "$policy"
  cmp -s "$policy" <(policy_text "$digest") || fail 'The existing AppArmor policy differs. Ask an administrator to review it.'
fi
if [[ $operation == --remove ]]; then
  if [[ -f $policy ]]; then
    if [[ -r /sys/kernel/security/apparmor/profiles ]] && grep -qF "tensorfold_appimage_$digest (" /sys/kernel/security/apparmor/profiles; then
      apparmor_parser -R "$policy" || fail 'The AppArmor policy could not be unloaded. No integration files were removed.'
    fi
    rm -- "$policy"
  fi
  if [[ -d $version ]]; then rm -rf -- "$version"; fi
  printf 'Removed integration for %s. User data was not changed.\n' "$digest"
  exit 0
fi
caller=${PKEXEC_UID:-${SUDO_UID:-}}
[[ $caller =~ ^[1-9][0-9]*$ ]] || fail 'Start setup from a normal user account using sudo or the desktop approval dialog.'
user=$(getent passwd "$caller" | cut -d: -f1)
[[ -n $user && $(id -u -- "$user") == "$caller" ]] || fail 'The requesting user could not be identified.'
[[ $source_image == /* && -f $source_image && ! -L $source_image ]] || fail 'Choose a regular AppImage file using its full path.'
[[ -r /sys/module/apparmor/parameters/enabled ]] && grep -q '^Y' /sys/module/apparmor/parameters/enabled || fail 'This setup requires an active AppArmor service.'
command -v apparmor_parser >/dev/null || fail 'Install the AppArmor utilities before running setup.'
command -v unsquashfs >/dev/null || fail 'Install squashfs-tools before running sandbox setup.'
if [[ -r /proc/sys/kernel/unprivileged_userns_clone ]]; then
  [[ $(cat /proc/sys/kernel/unprivileged_userns_clone) == 1 ]] || fail 'This kernel disables user namespaces. Ask an administrator to review its sandbox support.'
fi
cleanup() {
  if [[ $complete != 1 ]]; then
    if [[ $new_policy == 1 ]]; then apparmor_parser -R "$policy" >/dev/null 2>&1 || true; rm -f -- "$policy"; fi
    if [[ $new_version == 1 ]]; then rm -rf -- "$version"; fi
  fi
  [[ -z $stage ]] || rm -rf -- "$stage"
  [[ -z $policy_tmp ]] || rm -f -- "$policy_tmp"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
if [[ ! -d $version ]]; then
  stage=$(mktemp -d "$base/.install-XXXXXXXX")
  timeout 120 runuser -u "$user" -- head -c 2147483649 -- "$source_image" > "$stage/TensorFold.AppImage" || fail 'The selected AppImage could not be read as its requesting user.'
  (( $(stat -c %s -- "$stage/TensorFold.AppImage") <= 2147483648 )) || fail 'The AppImage exceeds the supported setup size.'
  [[ $(sha256sum -- "$stage/TensorFold.AppImage" | cut -d' ' -f1) == "$digest" ]] || fail 'The AppImage changed during setup. Run it again to review the new file.'
  magic=$(od -An -tx1 -N11 -- "$stage/TensorFold.AppImage" | tr -d ' \n')
  [[ $magic == 7f454c46020101??414902 ]] || fail 'The selected file is not a supported AppImage.'
  machine=$(od -An -tx1 -j18 -N2 -- "$stage/TensorFold.AppImage" | tr -d ' \n')
  case $(uname -m):$machine in x86_64:3e00|aarch64:b700) ;; *) fail 'The AppImage does not match this computer architecture.' ;; esac
  offset=
  while IFS= read -r candidate; do
    [[ $candidate =~ ^[0-9]+$ ]] || continue
    if unsquashfs -s -o "$candidate" "$stage/TensorFold.AppImage" >/dev/null 2>&1; then
      [[ -z $offset ]] || fail 'The AppImage payload is ambiguous.'
      offset=$candidate
    fi
  done < <(head -c 2097152 "$stage/TensorFold.AppImage" | LC_ALL=C grep -abo hsqs | cut -d: -f1)
  [[ -n $offset ]] || fail 'The AppImage payload could not be located.'
  unsquashfs -no-progress -no-xattrs -processors 2 -o "$offset" -d "$stage/app" "$stage/TensorFold.AppImage" >/dev/null || fail 'The AppImage payload could not be extracted.'
  [[ -z $(find "$stage/app" ! \( -type f -o -type d -o -type l \) -print -quit) ]] || fail 'The AppImage contains unsupported special files.'
  while IFS= read -r -d '' link; do
    target=$(readlink -- "$link")
    [[ $target != /* ]] || fail 'The AppImage contains an external symbolic link.'
    resolved=$(realpath -m -- "$link")
    [[ $resolved == "$stage/app/"* ]] || fail 'An AppImage symbolic link escapes its application directory.'
  done < <(find "$stage/app" -type l -print0)
  [[ -f $stage/app/tensorfold-studio && ! -L $stage/app/tensorfold-studio ]] || fail 'The AppImage application executable is invalid.'
  chown -hR root:root "$stage/app"
  find "$stage/app" -type d -exec chmod 0555 {} +
  find "$stage/app" -type f -perm /111 -exec chmod 0555 {} +
  find "$stage/app" -type f ! -perm /111 -exec chmod 0444 {} +
  chmod 0555 "$stage/TensorFold.AppImage"
  chmod 0755 "$stage"
fi
if [[ ! -f $policy ]]; then
  policy_tmp=$(mktemp /etc/apparmor.d/.tensorfold-appimage-XXXXXXXX)
  policy_text "$digest" > "$policy_tmp"
  chmod 0644 "$policy_tmp"
  apparmor_parser -Q -T "$policy_tmp" || fail 'The AppArmor policy could not be validated.'
fi
if [[ -n $stage ]]; then mv -T -- "$stage" "$version"; stage=; new_version=1; fi
if [[ -n $policy_tmp ]]; then mv -T -- "$policy_tmp" "$policy"; policy_tmp=; new_policy=1; fi
apparmor_parser -r "$policy" || fail 'The AppArmor policy could not be activated. New integration files have been rolled back.'
complete=1
printf '%s\n' "$version/app/tensorfold-studio"
