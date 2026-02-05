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


PTTS_API void ptts_set_seed( unsigned int seed );
PTTS_API unsigned int ptts_get_seed();

struct ptts_context_t;

PTTS_API ptts_context_t * ptts_init(
    ggml_backend * backend,
    ggml_backend * backend_cpu,
    const char * model_path
);
PTTS_API int ptts_get_sample_rate( ptts_context_t * ptts_ctx );
PTTS_API int ptts_get_frame_size( ptts_context_t * ptts_ctx );


struct ptts_stream_t;

PTTS_API ptts_stream_t * ptts_stream_from_safetensors(
    ptts_context_t * ptts_ctx,
    const char * voice,
    float temp = 0.7f
);
PTTS_API void ptts_stream_reset( ptts_stream_t * stream );
PTTS_API void ptts_stream_flush( ptts_stream_t * stream );
PTTS_API void ptts_stream_send( ptts_stream_t * stream, const char * chunk );
PTTS_API bool ptts_stream_receive( ptts_stream_t * stream, float * samples );
