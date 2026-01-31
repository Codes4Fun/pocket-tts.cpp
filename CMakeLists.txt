cmake_minimum_required(VERSION 3.14)
project("pocket_tts" C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(POCKET_TTS_STANDALONE ON)

    # configure project version
    # TODO
else()
    set(POCKET_TTS_STANDALONE OFF)
endif()

include_directories(include)

function(setup_rpath target)
    if (APPLE)
        # On macOS, set RPATH to look in the same directory as the executable
        set_target_properties(${target} PROPERTIES
            INSTALL_RPATH "@executable_path"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    else()
        # On Linux, set RPATH to look in the same directory as the executable
        set_target_properties(${target} PROPERTIES
            INSTALL_RPATH "$ORIGIN"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endfunction()

find_package(Threads REQUIRED)
find_package(GGML REQUIRED)
find_package(SentencePiece REQUIRED)

option(POCKET_TTS_BUILD_DEMOS    "pocket-tts: build demos"          ${POCKET_TTS_STANDALONE})

add_library(pocket_tts SHARED src/pocket_tts.cpp src/json.cpp src/safetensor.cpp)
target_link_libraries(pocket_tts PRIVATE ${GGML_LIBRARIES} ${SentencePiece_LIBRARIES})
target_include_directories(pocket_tts PRIVATE ${GGML_INCLUDE_DIRS} ${SentencePiece_INCLUDE_DIRS})
setup_rpath(pocket_tts)

if (POCKET_TTS_BUILD_DEMOS)
    add_subdirectory(demos)
endif ()
