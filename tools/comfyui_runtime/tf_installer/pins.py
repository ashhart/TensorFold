SOURCE_REVISION = "1ddac962008367ec2c50e10202c910ecf430b5eb"
SOURCE_PATCH_SHA256 = "e6156ed11263a2d32586245b1702a6bc47ac0049e65bf1724afad69242eb617f"
SOURCE_SHA256 = "013d687c5b72718220ac8e137252492d72be85d27ff9cec37525d714c52dc2ba"
SOURCE_URL = f"https://codeload.github.com/ashhart/TensorFold/tar.gz/{SOURCE_REVISION}"
SOURCE_ROOT = f"TensorFold-{SOURCE_REVISION}"
ZIG_VERSION = "0.17.0"
RELEASE_API = "https://api.github.com/repos/ashhart/TensorFold/releases/latest"
RELEASE_BASE = "https://github.com/ashhart/TensorFold/releases/download"
MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
MAX_EXTRACTED_BYTES = 3 * 1024 * 1024 * 1024
MAX_SOURCE_BYTES = 256 * 1024 * 1024
MAX_SOURCE_EXTRACTED_BYTES = 1024 * 1024 * 1024


class InstallError(Exception):
    pass
