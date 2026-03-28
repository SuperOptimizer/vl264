cmake_minimum_required(VERSION 3.25)
project(vl264 VERSION 0.1.0 LANGUAGES C)

if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    message(WARNING "vl264 is tested with Clang and GCC. Other compilers may work but are untested.")
endif()

set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(VL264_BUILD_TESTS  "Build vl264 test suite"            ON)
option(VL264_BUILD_FUZZ   "Build AFL++ fuzz targets"           OFF)
option(VL264_ASAN         "Enable AddressSanitizer"            OFF)
option(VL264_MSAN         "Enable MemorySanitizer"             OFF)
option(VL264_UBSAN        "Enable UndefinedBehaviorSanitizer"  OFF)
option(VL264_TSAN         "Enable ThreadSanitizer"             OFF)
option(VL264_COVERAGE     "Enable code coverage profiling"     OFF)
option(VL264_PROFILE      "Enable performance profiling"       OFF)
option(VL264_FTIME_TRACE  "Enable -ftime-trace compile stats"  OFF)

# ── Sanitizer flags (layerable) ─────────────────────────────────────────────
set(VL264_SAN_FLAGS "")
set(VL264_SAN_LINK "")

if(VL264_ASAN)
    list(APPEND VL264_SAN_FLAGS -fsanitize=address -fno-omit-frame-pointer)
    list(APPEND VL264_SAN_LINK  -fsanitize=address)
endif()
if(VL264_MSAN)
    list(APPEND VL264_SAN_FLAGS -fsanitize=memory -fno-omit-frame-pointer -fsanitize-memory-track-origins)
    list(APPEND VL264_SAN_LINK  -fsanitize=memory)
endif()
if(VL264_UBSAN)
    list(APPEND VL264_SAN_FLAGS -fsanitize=undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
    list(APPEND VL264_SAN_LINK  -fsanitize=undefined)
endif()
if(VL264_TSAN)
    list(APPEND VL264_SAN_FLAGS -fsanitize=thread -fno-omit-frame-pointer)
    list(APPEND VL264_SAN_LINK  -fsanitize=thread)
endif()

# ── Coverage flags ───────────────────────────────────────────────────────────
set(VL264_COV_FLAGS "")
if(VL264_COVERAGE)
    set(VL264_COV_FLAGS -fprofile-instr-generate -fcoverage-mapping)
endif()

# ── Profile flags ────────────────────────────────────────────────────────────
set(VL264_PROF_FLAGS "")
if(VL264_PROFILE)
    set(VL264_PROF_FLAGS -fno-omit-frame-pointer -g -fxray-instrument)
endif()

# ── Compile-time trace ───────────────────────────────────────────────────────
set(VL264_FTIME_FLAGS "")
if(VL264_FTIME_TRACE)
    set(VL264_FTIME_FLAGS -ftime-trace)
endif()

# ── Library ──────────────────────────────────────────────────────────────────
add_library(vl264 STATIC vl264.c)
target_include_directories(vl264 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(vl264 PRIVATE
    -march=native -mtune=native
    -fvectorize -fslp-vectorize
    -ffunction-sections -fdata-sections
    -fvisibility=hidden
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    -Wno-unused-function
    ${VL264_SAN_FLAGS} ${VL264_COV_FLAGS} ${VL264_PROF_FLAGS} ${VL264_FTIME_FLAGS}
)
target_link_options(vl264 PRIVATE ${VL264_SAN_LINK})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(vl264 PRIVATE -O0 -g3)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(vl264 PRIVATE -O3 -flto -DNDEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(vl264 PRIVATE -O2 -g)
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    target_compile_options(vl264 PRIVATE -Oz -flto -DNDEBUG)
endif()

target_link_libraries(vl264 PRIVATE m)

# ── Tests ────────────────────────────────────────────────────────────────────
if(VL264_BUILD_TESTS)
    enable_testing()
    add_executable(vl264_test vl264-test.c)
    target_link_libraries(vl264_test PRIVATE vl264 m)
    target_compile_options(vl264_test PRIVATE
        -march=native -Wall -Wextra -O2 -g
        -Wno-unused-function
        ${VL264_SAN_FLAGS} ${VL264_COV_FLAGS} ${VL264_PROF_FLAGS}
    )
    target_link_options(vl264_test PRIVATE ${VL264_SAN_LINK} ${VL264_COV_FLAGS})

    add_test(NAME bitstream COMMAND vl264_test --bitstream)
    add_test(NAME dct       COMMAND vl264_test --dct)
    add_test(NAME quant     COMMAND vl264_test --quant)
    add_test(NAME intra     COMMAND vl264_test --intra)
    add_test(NAME inter     COMMAND vl264_test --inter)
    add_test(NAME cavlc     COMMAND vl264_test --cavlc)
    add_test(NAME nal       COMMAND vl264_test --nal)
    add_test(NAME axis      COMMAND vl264_test --axis)
    add_test(NAME roundtrip COMMAND vl264_test --roundtrip)
    add_test(NAME bench     COMMAND vl264_test --bench)
    add_test(NAME all       COMMAND vl264_test --all)
endif()

# ── AFL++ fuzz targets ───────────────────────────────────────────────────────
if(VL264_BUILD_FUZZ)
    # Build with: cmake -B build-fuzz -DVLA264_BUILD_FUZZ=ON -DCMAKE_C_COMPILER=afl-clang-fast
    add_executable(vl264_fuzz_decode vl264-fuzz-decode.c)
    target_link_libraries(vl264_fuzz_decode PRIVATE vl264 m)
    target_compile_options(vl264_fuzz_decode PRIVATE -O2 -g -fsanitize=address,undefined)
    target_link_options(vl264_fuzz_decode PRIVATE -fsanitize=address,undefined)

    add_executable(vl264_fuzz_roundtrip vl264-fuzz-roundtrip.c)
    target_link_libraries(vl264_fuzz_roundtrip PRIVATE vl264 m)
    target_compile_options(vl264_fuzz_roundtrip PRIVATE -O2 -g -fsanitize=address,undefined)
    target_link_options(vl264_fuzz_roundtrip PRIVATE -fsanitize=address,undefined)
endif()

# ── Coverage report target ───────────────────────────────────────────────────
if(VL264_COVERAGE)
    add_custom_target(coverage
        COMMAND ${CMAKE_COMMAND} -E env LLVM_PROFILE_FILE=vl264_test.profraw
                $<TARGET_FILE:vl264_test> --all
        COMMAND llvm-profdata merge -sparse vl264_test.profraw -o vl264_test.profdata
        COMMAND llvm-cov report $<TARGET_FILE:vl264_test> -instr-profile=vl264_test.profdata
        COMMAND llvm-cov show $<TARGET_FILE:vl264_test> -instr-profile=vl264_test.profdata
                -format=html -output-dir=coverage-report
        COMMENT "Generating coverage report in coverage-report/"
        DEPENDS vl264_test
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()

# ── Riced build: LTO + PGO + aggressive opts ────────────────────────────────
# Usage: cmake --build build --target riced
# This produces a maximally optimized vl264_test binary in build-riced/
add_custom_target(riced
    # Step 1: PGO instrumented build
    COMMAND ${CMAKE_COMMAND} -B ${CMAKE_BINARY_DIR}-pgo-gen
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_C_FLAGS=-fprofile-generate
            -DCMAKE_EXE_LINKER_FLAGS=-fprofile-generate\ -fuse-ld=lld
            -DVL264_BUILD_TESTS=ON
            ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}-pgo-gen --clean-first
    # Step 2: PGO training on real workloads
    COMMAND ${CMAKE_BINARY_DIR}-pgo-gen/vl264_test --roundtrip --sweep --props --edge
    # Step 3: Merge profiles
    COMMAND llvm-profdata merge -sparse
            ${CMAKE_BINARY_DIR}-pgo-gen/default_*.profraw
            -o ${CMAKE_BINARY_DIR}-pgo.profdata
    # Step 4: Final build with PGO + LTO + aggressive flags
    COMMAND ${CMAKE_COMMAND} -B ${CMAKE_BINARY_DIR}-riced
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_C_FLAGS=-fprofile-use=${CMAKE_BINARY_DIR}-pgo.profdata\ -flto
            -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld\ -Wl,--icf=all\ -Wl,--gc-sections\ -flto
            -DVL264_BUILD_TESTS=ON
            ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}-riced --clean-first
    COMMENT "Building riced binary: PGO + LTO + ICF + GC"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS pgo-gen
)
