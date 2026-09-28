cmake_minimum_required(VERSION 3.24)
project(_hostsync LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -c "import mlx, pathlib; print(pathlib.Path(mlx.__path__[0]) / 'share' / 'cmake' / 'MLX')"
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE MLX_CMAKE)
find_package(MLX CONFIG REQUIRED PATHS "${MLX_CMAKE}" NO_DEFAULT_PATH)
nanobind_add_module(_hostsync NB_STATIC STABLE_ABI LTO NOMINSIZE NB_DOMAIN mlx hostsync.cpp)
target_link_libraries(_hostsync PRIVATE mlx)
