#pragma once

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef POCKET_TTS_BUILD
#        define PTTS_API __declspec(dllexport) extern
#    else
#        define PTTS_API __declspec(dllimport) extern
#    endif
#else
#    define PTTS_API __attribute__ ((visibility ("default"))) extern
#endif

PTTS_API void ptts_run_test(
    std::string voice,
    std::string text_to_generate,
    std::string output_filepath
);

