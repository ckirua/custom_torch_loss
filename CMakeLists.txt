# Standalone LibTorch demo: test_losses (Python extension uses setup.py, not this file.)
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(custom_loss_functions)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TORCH_USE_PIP_PYTHON
    "Set CMAKE_PREFIX_PATH from pip torch (torch.utils.cmake_prefix_path). Disable if you pass -DCMAKE_PREFIX_PATH to LibTorch."
    ON)

set(TORCH_PYTHON "" CACHE FILEPATH
    "Python that can `import torch` (empty: use .venv or python3 on PATH)")

if(TORCH_USE_PIP_PYTHON)
    if(TORCH_PYTHON)
        set(_torch_py "${TORCH_PYTHON}")
    elseif(WIN32 AND EXISTS "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
        set(_torch_py "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
        set(_torch_py "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    else()
        find_program(_torch_py NAMES python3.14 python3 python)
    endif()

    if(_torch_py)
        execute_process(
            COMMAND "${_torch_py}" -c "import torch; print(torch.utils.cmake_prefix_path)"
            OUTPUT_VARIABLE _torch_prefix
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _torch_rc
            ERROR_VARIABLE _torch_err
        )
        if(_torch_rc EQUAL 0 AND _torch_prefix)
            list(FIND CMAKE_PREFIX_PATH "${_torch_prefix}" _pfx_idx)
            if(_pfx_idx LESS 0)
                list(PREPEND CMAKE_PREFIX_PATH "${_torch_prefix}")
            endif()
            message(STATUS "LibTorch prefix (from ${_torch_py}): ${_torch_prefix}")
        else()
            message(FATAL_ERROR
                "Could not read torch.utils.cmake_prefix_path with \"${_torch_py}\".\n"
                "Install torch in that environment, or run:\n"
                "  cmake .. -DTORCH_USE_PIP_PYTHON=OFF -DCMAKE_PREFIX_PATH=<path-to-libtorch>\n"
                "Details: ${_torch_err}")
        endif()
    else()
        message(FATAL_ERROR
            "No Python found to locate pip LibTorch. Create .venv with torch, or set -DTORCH_PYTHON=..., "
            "or -DTORCH_USE_PIP_PYTHON=OFF -DCMAKE_PREFIX_PATH=...")
    endif()
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(test_losses
    examples/test_losses.cpp
    cpp/custom_loss_functions.cpp
)
target_include_directories(test_losses PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cpp")
target_link_libraries(test_losses "${TORCH_LIBRARIES}")

set_target_properties(test_losses PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET test_losses
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:test_losses>)
endif ()

message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
