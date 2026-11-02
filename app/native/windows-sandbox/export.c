#include "sandbox.h"
#include <winioctl.h>

typedef struct {
  const wchar_t *root;
  BOOL input_links;
  DWORD entries;
  ULONGLONG bytes;
} ExportState;

static BOOL safe_name(const wchar_t *name) {
  size_t length = wcslen(name);
  if (!length || name[length - 1] == L'.' || name[length - 1] == L' ') return FALSE;
  wchar_t base[16]; size_t prefix = wcscspn(name, L".");
  if (prefix < 16) {
    wcsncpy(base, name, prefix); base[prefix] = L'\0';
    if (!_wcsicmp(base, L"CON") || !_wcsicmp(base, L"PRN") || !_wcsicmp(base, L"AUX") || !_wcsicmp(base, L"NUL")
      || !_wcsicmp(base, L"CONIN$") || !_wcsicmp(base, L"CONOUT$")) return FALSE;
    if (prefix == 4 && (!_wcsnicmp(base, L"COM", 3) || !_wcsnicmp(base, L"LPT", 3))
      && ((base[3] >= L'1' && base[3] <= L'9') || base[3] == 0xB9 || base[3] == 0xB2 || base[3] == 0xB3)) return FALSE;
  }
  if (wcspbrk(name, L"\\/:\r\n")) return FALSE;
  for (size_t i = 0; i < length; i++) if (name[i] < 32) return FALSE;
  return TRUE;
}

static HANDLE pinned(const wchar_t *path, BOOL directory) {
  return CreateFileW(path, directory ? FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES : GENERIC_READ,
    FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, NULL);
}

static BOOL internal_link(ExportState *state, HANDLE link, const wchar_t *source, wchar_t *target) {
  BYTE data[MAXIMUM_REPARSE_DATA_BUFFER_SIZE]; DWORD received;
  if (!DeviceIoControl(link, FSCTL_GET_REPARSE_POINT, NULL, 0, data, sizeof(data), &received, NULL) || received < 16) return FALSE;
  DWORD tag = *(DWORD *)data;
  WORD offset = *(WORD *)(data + 8), length = *(WORD *)(data + 10);
  DWORD base = tag == IO_REPARSE_TAG_SYMLINK ? 20 : 16;
  if ((tag != IO_REPARSE_TAG_SYMLINK && tag != IO_REPARSE_TAG_MOUNT_POINT) || length % 2
    || base + offset + length > received || length >= 32760 * sizeof(wchar_t)) return FALSE;
  wchar_t raw[32768], combined[32768];
  memcpy(raw, data + base + offset, length); raw[length / sizeof(wchar_t)] = L'\0';
  BOOL relative = tag == IO_REPARSE_TAG_SYMLINK && received >= 20 && (*(DWORD *)(data + 16) & 1);
  if (relative) {
    wcsncpy(combined, source, 32768);
    wchar_t *separator = wcsrchr(combined, L'\\'); if (!separator) return FALSE;
    separator[1] = L'\0';
    if (wcslen(combined) + wcslen(raw) >= 32768) return FALSE;
    wcscat(combined, raw);
  } else {
    const wchar_t *value = wcsncmp(raw, L"\\??\\", 4) == 0 ? raw + 4 : raw;
    if (wcslen(value) >= 32768) return FALSE;
    wcscpy(combined, value);
  }
  DWORD size = GetFullPathNameW(combined, 32768, target, NULL);
  return size && size < 32768 && sandbox_inside(state->root, target) && sandbox_validate_input(target);
}

static BOOL export_entry(ExportState *state, const wchar_t *source, const wchar_t *destination, DWORD depth) {
  if (++state->entries > 200000 || depth > 80 || !sandbox_inside(state->root, source)) return FALSE;
  DWORD attributes = GetFileAttributesW(source);
  if (attributes == INVALID_FILE_ATTRIBUTES) return FALSE;
  BOOL directory = (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  HANDLE input = pinned(source, directory);
  if (input == INVALID_HANDLE_VALUE) return FALSE;
  FILE_ATTRIBUTE_TAG_INFO tag;
  BY_HANDLE_FILE_INFORMATION identity;
  BOOL ok = FALSE;
  if (!GetFileInformationByHandleEx(input, FileAttributeTagInfo, &tag, sizeof(tag))
    || !GetFileInformationByHandle(input, &identity)) goto done;
  if (tag.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) {
    wchar_t resolved[32768];
    if (state->input_links && internal_link(state, input, source, resolved)) ok = export_entry(state, resolved, destination, depth + 1);
    goto done;
  }
  directory = (tag.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  if (directory) {
    if (!CreateDirectoryW(destination, NULL) && GetLastError() != ERROR_ALREADY_EXISTS) goto done;
    wchar_t pattern[32768], child[32768], target[32768];
    if (swprintf(pattern, 32768, L"%ls\\*", source) < 0) goto done;
    WIN32_FIND_DATAW entry;
    HANDLE find = FindFirstFileW(pattern, &entry);
    if (find == INVALID_HANDLE_VALUE) { ok = GetLastError() == ERROR_FILE_NOT_FOUND; goto done; }
    ok = TRUE;
    do {
      if (!wcscmp(entry.cFileName, L".") || !wcscmp(entry.cFileName, L"..")) continue;
      if (!safe_name(entry.cFileName) || swprintf(child, 32768, L"%ls\\%ls", source, entry.cFileName) < 0
        || swprintf(target, 32768, L"%ls\\%ls", destination, entry.cFileName) < 0
        || !export_entry(state, child, target, depth + 1)) { ok = FALSE; break; }
    } while (FindNextFileW(find, &entry));
    if (ok && GetLastError() != ERROR_NO_MORE_FILES) ok = FALSE;
    FindClose(find);
    if (ok && (tag.FileAttributes & FILE_ATTRIBUTE_READONLY)) ok = SetFileAttributesW(destination, FILE_ATTRIBUTE_READONLY);
  } else {
    if (identity.nNumberOfLinks != 1 && !state->input_links) goto done;
    ULONGLONG size = ((ULONGLONG)identity.nFileSizeHigh << 32) | identity.nFileSizeLow;
    if (size > 2147483648ULL || state->bytes + size > 2147483648ULL) goto done;
    state->bytes += size;
    HANDLE output = CreateFileW(destination, GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
    if (output == INVALID_HANDLE_VALUE) goto done;
    BYTE buffer[65536]; DWORD received, written;
    ULONGLONG total = 0;
    ok = TRUE;
    while (TRUE) {
      if (!ReadFile(input, buffer, sizeof(buffer), &received, NULL)) { ok = FALSE; break; }
      if (!received) break;
      total += received;
      if (total > size) { ok = FALSE; break; }
      DWORD offset = 0;
      while (offset < received) {
        if (!WriteFile(output, buffer + offset, received - offset, &written, NULL) || !written) { ok = FALSE; break; }
        offset += written;
      }
      if (!ok) break;
    }
    if (total != size || !FlushFileBuffers(output)) ok = FALSE;
    CloseHandle(output);
    if (ok && (tag.FileAttributes & FILE_ATTRIBUTE_READONLY)) ok = SetFileAttributesW(destination, FILE_ATTRIBUTE_READONLY);
  }
done:
  CloseHandle(input);
  return ok;
}

BOOL sandbox_export(const wchar_t *source, const wchar_t *destination, BOOL input_links) {
  if (!sandbox_validate_input(source) || !sandbox_validate_directory(source) || !sandbox_validate_input(destination)
    || !sandbox_validate_directory(destination) || sandbox_inside(source, destination)) return FALSE;
  HANDLE ancestors[128]; DWORD count = 0;
  wchar_t prefix[32768];
  BOOL ok = FALSE;
  size_t length = wcslen(source);
  for (size_t i = 3; i < length; i++) {
    if (source[i] != L'\\') continue;
    if (count >= 128) goto done;
    wcsncpy(prefix, source, i); prefix[i] = L'\0';
    ancestors[count] = pinned(prefix, TRUE);
    if (ancestors[count] == INVALID_HANDLE_VALUE) goto done;
    FILE_ATTRIBUTE_TAG_INFO info;
    if (!GetFileInformationByHandleEx(ancestors[count], FileAttributeTagInfo, &info, sizeof(info))
      || (info.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) { CloseHandle(ancestors[count]); goto done; }
    count++;
  }
  ExportState state = {source, input_links, 0, 0};
  ok = export_entry(&state, source, destination, 0);
done:
  while (count) CloseHandle(ancestors[--count]);
  if (!ok) fprintf(stderr, "Windows isolation snapshot refused a busy, linked, oversized or unsafe tree (%lu).\n", GetLastError());
  return ok;
}
