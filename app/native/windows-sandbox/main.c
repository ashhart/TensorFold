#include "sandbox.h"

wchar_t *sandbox_command_line(int count, wchar_t **args) {
  size_t capacity = 1;
  for (int i = 0; i < count; i++) capacity += wcslen(args[i]) * 2 + 4;
  if (capacity > 32767) return NULL;
  wchar_t *line = calloc(capacity, sizeof(wchar_t));
  if (!line) return NULL;
  const wchar_t *base = wcsrchr(args[0], L'\\');
  if (count == 5 && _wcsicmp(base ? base + 1 : args[0], L"cmd.exe") == 0
    && _wcsicmp(args[1], L"/d") == 0 && _wcsicmp(args[2], L"/s") == 0 && _wcsicmp(args[3], L"/c") == 0) {
    swprintf(line, capacity, L"\"%ls\" /d /s /c \"%ls\"", args[0], args[4]);
    return line;
  }
  wchar_t *out = line;
  for (int i = 0; i < count; i++) {
    if (i) *out++ = L' ';
    *out++ = L'"';
    const wchar_t *input = args[i];
    while (*input) {
      size_t slashes = 0;
      while (*input == L'\\') { slashes++; input++; }
      if (*input == L'"' || !*input) slashes *= 2;
      while (slashes--) *out++ = L'\\';
      if (*input == L'"') *out++ = L'\\';
      if (*input) *out++ = *input++;
    }
    *out++ = L'"';
  }
  *out = L'\0';
  return line;
}

DWORD sandbox_child_probe(void) {
  HANDLE token;
  DWORD appcontainer = 0, lpac = 0, length = 0;
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) return 126;
  BOOL valid = GetTokenInformation(token, TokenIsAppContainer, &appcontainer, sizeof(appcontainer), &length)
    && GetTokenInformation(token, TokenIsLessPrivilegedAppContainer, &lpac, sizeof(lpac), &length) && appcontainer && lpac;
  CloseHandle(token);
  if (!valid) return 126;
  puts("{\"abi\":1,\"appContainer\":true,\"lessPrivileged\":true}");
  return 0;
}

DWORD sandbox_probe(void) {
  wchar_t temporary[32768], root[32768], workspace[32768], scratch[32768], source[32768], binary[32768];
  if (!GetTempPathW(32768, temporary) || !GetTempFileNameW(temporary, L"tfs", 0, root)) return 126;
  DeleteFileW(root);
  if (!CreateDirectoryW(root, NULL) || !sandbox_prepare_directory(root)) return 126;
  swprintf(workspace, 32768, L"%ls\\workspace", root);
  swprintf(scratch, 32768, L"%ls\\scratch", root);
  swprintf(binary, 32768, L"%ls\\probe.exe", root);
  DWORD result = 126;
  SandboxPlan plan = {0};
  if (!CreateDirectoryW(workspace, NULL) || !CreateDirectoryW(scratch, NULL)
    || !GetModuleFileNameW(NULL, source, 32768) || !CopyFileW(source, binary, TRUE)) goto done;
  wchar_t *args[] = {binary, L"--child-probe"};
  plan.root = root; plan.workspace = workspace; plan.scratch = scratch;
  plan.executable = binary; plan.command_line = sandbox_command_line(2, args); plan.parent_pid = GetCurrentProcessId(); plan.timeout_ms = 10000;
  if (plan.command_line) result = sandbox_execute(&plan);
done:
  free(plan.command_line);
  swprintf(source, 32768, L"%ls\\ready", root); DeleteFileW(source);
  DeleteFileW(binary); RemoveDirectoryW(workspace); RemoveDirectoryW(scratch); RemoveDirectoryW(root);
  return result;
}

int wmain(int argc, wchar_t **argv) {
  if (argc == 4 && (wcscmp(argv[1], L"--export-input") == 0 || wcscmp(argv[1], L"--export-staged") == 0)) {
    return sandbox_export(argv[2], argv[3], wcscmp(argv[1], L"--export-input") == 0) ? 0 : 126;
  }
  if (argc == 3 && wcscmp(argv[1], L"--validate-input") == 0) return sandbox_validate_input(argv[2]) ? 0 : 126;
  if (argc == 3 && wcscmp(argv[1], L"--prepare-directory") == 0) return sandbox_prepare_directory(argv[2]) ? 0 : 126;
  if (argc == 2 && wcscmp(argv[1], L"--child-probe") == 0) return (int)sandbox_child_probe();
  if (argc == 2 && wcscmp(argv[1], L"--probe") == 0) return (int)sandbox_probe();
  if (argc < 12 || wcscmp(argv[1], L"--run") != 0 || wcscmp(argv[10], L"--") != 0) return 126;
  SandboxPlan plan = {0};
  plan.root = argv[2]; plan.workspace = argv[3]; plan.scratch = argv[4];
  if ((wcscmp(argv[5], L"0") && wcscmp(argv[5], L"1")) || (wcscmp(argv[6], L"0") && wcscmp(argv[6], L"1"))) return 126;
  plan.network = wcscmp(argv[5], L"1") == 0; plan.writable_workspace = wcscmp(argv[6], L"1") == 0;
  wchar_t *end;
  plan.timeout_ms = wcstoul(argv[7], &end, 10); if (*end) return 126;
  plan.parent_pid = wcstoul(argv[8], &end, 10); if (*end || !plan.parent_pid) return 126;
  if (wcscmp(argv[9], L"abi1") != 0) return 126;
  plan.executable = argv[11]; plan.command_line = sandbox_command_line(argc - 11, argv + 11);
  if (!plan.command_line) return 126;
  DWORD result = sandbox_execute(&plan);
  free(plan.command_line);
  return (int)result;
}
