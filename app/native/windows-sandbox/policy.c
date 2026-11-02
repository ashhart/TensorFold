#include "sandbox.h"

BOOL sandbox_validate_input(const wchar_t *path) {
  size_t length = wcslen(path);
  if (length < 3 || length >= 32768 || path[1] != L':' || path[2] != L'\\' || wcsstr(path + 2, L":")) return FALSE;
  wchar_t prefix[32768];
  wcsncpy(prefix, path, 3); prefix[3] = L'\0';
  UINT drive = GetDriveTypeW(prefix);
  if (drive != DRIVE_FIXED && drive != DRIVE_REMOVABLE) return FALSE;
  for (size_t i = 3; i <= length; i++) {
    if (i != length && path[i] != L'\\') continue;
    wcsncpy(prefix, path, i); prefix[i] = L'\0';
    DWORD attributes = GetFileAttributesW(prefix);
    if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_REPARSE_POINT)) return FALSE;
  }
  return TRUE;
}

BOOL sandbox_inside(const wchar_t *root, const wchar_t *path) {
  size_t length = wcslen(root);
  return _wcsnicmp(root, path, length) == 0 && (path[length] == L'\0' || path[length] == L'\\');
}

BOOL sandbox_validate_directory(const wchar_t *path) {
  if (wcslen(path) < 4 || path[1] != L':' || path[2] != L'\\' || wcsstr(path, L"\\..") || wcsstr(path, L"\\.\\")) return FALSE;
  HANDLE file = CreateFileW(path, FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
    NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (file == INVALID_HANDLE_VALUE) return FALSE;
  FILE_ATTRIBUTE_TAG_INFO info;
  wchar_t canonical[32768];
  DWORD length = GetFinalPathNameByHandleW(file, canonical, 32768, FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
  BOOL valid = GetFileInformationByHandleEx(file, FileAttributeTagInfo, &info, sizeof(info))
    && (info.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) && !(info.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)
    && length > 4 && length < 32768 && _wcsicmp(canonical + 4, path) == 0;
  CloseHandle(file);
  return valid;
}

BOOL sandbox_prepare_directory(const wchar_t *path) {
  if (!sandbox_validate_directory(path)) return FALSE;
  wchar_t pattern[32768];
  if (swprintf(pattern, 32768, L"%ls\\*", path) < 0) return FALSE;
  WIN32_FIND_DATAW data;
  HANDLE find = FindFirstFileW(pattern, &data);
  if (find != INVALID_HANDLE_VALUE) {
    BOOL empty = TRUE;
    do { if (wcscmp(data.cFileName, L".") && wcscmp(data.cFileName, L"..")) { empty = FALSE; break; } } while (FindNextFileW(find, &data));
    FindClose(find);
    if (!empty) return FALSE;
  } else if (GetLastError() != ERROR_FILE_NOT_FOUND) return FALSE;
  HANDLE token = NULL, directory = INVALID_HANDLE_VALUE;
  BYTE user_bytes[1024]; DWORD length;
  LPWSTR sid = NULL;
  PSECURITY_DESCRIPTOR descriptor = NULL, previous = NULL;
  PSID owner = NULL; PACL acl = NULL;
  BOOL present, defaulted, ok = FALSE;
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)
    || !GetTokenInformation(token, TokenUser, user_bytes, sizeof(user_bytes), &length)) goto done;
  directory = CreateFileW(path, READ_CONTROL | WRITE_DAC, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
    NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (directory == INVALID_HANDLE_VALUE || GetSecurityInfo(directory, SE_FILE_OBJECT, OWNER_SECURITY_INFORMATION,
    &owner, NULL, NULL, NULL, &previous) != ERROR_SUCCESS || !EqualSid(owner, ((TOKEN_USER *)user_bytes)->User.Sid)) goto done;
  wchar_t sddl[512];
  if (!ConvertSidToStringSidW(owner, &sid) || swprintf(sddl, 512, L"D:P(A;OICI;FA;;;%ls)(A;OICI;FA;;;SY)", sid) < 0
    || !ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, SDDL_REVISION_1, &descriptor, NULL)
    || !GetSecurityDescriptorDacl(descriptor, &present, &acl, &defaulted) || !present || !acl) goto done;
  ok = SetSecurityInfo(directory, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
    NULL, NULL, acl, NULL) == ERROR_SUCCESS;
done:
  if (descriptor) LocalFree(descriptor);
  if (previous) LocalFree(previous);
  if (sid) LocalFree(sid);
  if (token) CloseHandle(token);
  if (directory != INVALID_HANDLE_VALUE) CloseHandle(directory);
  return ok;
}

static BOOL git_control(const wchar_t *path) {
  for (const wchar_t *part = path; *part; part++) {
    if (part != path && part[-1] != L'\\') continue;
    if (_wcsnicmp(part, L".git", 4) == 0 && (part[4] == L'\0' || part[4] == L'\\')) return TRUE;
  }
  return FALSE;
}

static BOOL grant(SandboxPlan *plan, const wchar_t *path, BOOL directory, BOOL writable) {
  HANDLE file = CreateFileW(path, MAXIMUM_ALLOWED, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
    NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, NULL);
  if (file == INVALID_HANDLE_VALUE) return FALSE;
  PACL original = NULL, updated = NULL;
  PSECURITY_DESCRIPTOR descriptor = NULL, label = NULL;
  BOOL ok = FALSE;
  if (GetSecurityInfo(file, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, NULL, NULL, &original, NULL, &descriptor) != ERROR_SUCCESS || !original) goto done;
  EXPLICIT_ACCESSW access = {0};
  access.grfAccessPermissions = FILE_GENERIC_READ | FILE_GENERIC_EXECUTE;
  if (writable) access.grfAccessPermissions |= FILE_GENERIC_WRITE | DELETE;
  access.grfAccessMode = GRANT_ACCESS;
  access.grfInheritance = directory ? SUB_CONTAINERS_AND_OBJECTS_INHERIT : NO_INHERITANCE;
  access.Trustee.TrusteeForm = TRUSTEE_IS_SID;
  access.Trustee.TrusteeType = TRUSTEE_IS_UNKNOWN;
  access.Trustee.ptstrName = (LPWSTR)plan->sid;
  if (SetEntriesInAclW(1, &access, original, &updated) != ERROR_SUCCESS) goto done;
  if (SetSecurityInfo(file, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, NULL, NULL, updated, NULL) != ERROR_SUCCESS) goto done;
  if (writable) {
    PACL sacl = NULL;
    BOOL present, defaulted;
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(L"S:(ML;OICI;NW;;;LW)", SDDL_REVISION_1, &label, NULL)
      || !GetSecurityDescriptorSacl(label, &present, &sacl, &defaulted) || !present
      || SetSecurityInfo(file, SE_FILE_OBJECT, LABEL_SECURITY_INFORMATION, NULL, NULL, NULL, sacl) != ERROR_SUCCESS) goto done;
  }
  ok = TRUE;
done:
  if (label) LocalFree(label);
  if (updated) LocalFree(updated);
  if (descriptor) LocalFree(descriptor);
  CloseHandle(file);
  return ok;
}

static BOOL walk(SandboxPlan *plan, const wchar_t *path, DWORD depth, DWORD *count) {
  if (depth > 80 || ++*count > 300000 || !sandbox_inside(plan->root, path)) return FALSE;
  DWORD attributes = GetFileAttributesW(path);
  if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_REPARSE_POINT)) return FALSE;
  BOOL directory = (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
  BOOL writable = sandbox_inside(plan->scratch, path) || (plan->writable_workspace && sandbox_inside(plan->workspace, path) && !git_control(path));
  if (!grant(plan, path, directory, writable)) return FALSE;
  if (!directory) return TRUE;
  wchar_t pattern[32768], next[32768];
  if (swprintf(pattern, 32768, L"%ls\\*", path) < 0) return FALSE;
  WIN32_FIND_DATAW data;
  HANDLE find = FindFirstFileW(pattern, &data);
  if (find == INVALID_HANDLE_VALUE) return GetLastError() == ERROR_FILE_NOT_FOUND;
  BOOL ok = TRUE;
  do {
    if (wcscmp(data.cFileName, L".") == 0 || wcscmp(data.cFileName, L"..") == 0) continue;
    if (swprintf(next, 32768, L"%ls\\%ls", path, data.cFileName) < 0 || !walk(plan, next, depth + 1, count)) { ok = FALSE; break; }
  } while (FindNextFileW(find, &data));
  if (ok && GetLastError() != ERROR_NO_MORE_FILES) ok = FALSE;
  FindClose(find);
  return ok;
}

BOOL sandbox_grant_tree(SandboxPlan *plan) {
  DWORD count = 0;
  if (!sandbox_validate_directory(plan->root) || !sandbox_validate_directory(plan->workspace) || !sandbox_validate_directory(plan->scratch)
    || !sandbox_inside(plan->root, plan->workspace) || !sandbox_inside(plan->root, plan->scratch)
    || _wcsicmp(plan->root, plan->workspace) == 0 || _wcsicmp(plan->workspace, plan->scratch) == 0) return FALSE;
  return walk(plan, plan->root, 0, &count);
}
