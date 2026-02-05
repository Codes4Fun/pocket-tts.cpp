
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <deque>

#include <sentencepiece_processor.h>

#define POCKET_TTS_BUILD
#include <pocket_tts/pocket_tts.h>
#include <pocket_tts/ptrs.h>
#include <pocket_tts/safetensor.h>

#define CAPTURE(...)
#define CAPTURE_GROUP(...)
#define ONCE(code) {static bool once=false; if (!once) {{code;}; once=true;}}
#define ON_NTH(nth, code) {static int count=0; if (count++ == (nth)) {code;}}

#include "context.h"
#include "loader.h"
#include "torch.h"
#include "config.h"
#include "pocket_tts/modules/conv.h"
#include "pocket_tts/modules/seanet.h"
#include "pocket_tts/modules/mimi_transformer.h"
#include "pocket_tts/models/defaults.h"
#include "pocket_tts/models/mimi.h"
#include "pocket_tts/conditioners/text.h"
#include "pocket_tts/modules/transformer.h"
#include "pocket_tts/modules/mlp.h"
#include "pocket_tts/models/flow_lm.h"

#include "wav.h"

//////////////////
// MARK: tts
//////////////////

std::tuple<ggml_tensor*, bool> _run_flow_lm(
    ScratchContext & ctx,
    flow_lm_t * flow_lm,
    flow_lm_states_t * model_states,
    std::vector<int> * text_tokens,
    ggml_tensor * backbone_input_latents,
    ggml_tensor * audio_conditioning,
    float temp
) {
    ggml_tensor * text_embeddings = NULL;
    if ( text_tokens && text_tokens->size() ) {
        auto tokens = ctx.input( GGML_NE(text_tokens->size()), *text_tokens );
        text_embeddings = conditioner_forward( ctx, flow_lm->conditioner, tokens );
    }
    if ( audio_conditioning ) {
        if ( text_embeddings ) {
            text_embeddings = ggml_concat( ctx, text_embeddings, audio_conditioning, 1 );
        } else {
            text_embeddings = audio_conditioning;
        }
    }

    return flow_lm_sample_next_latent(
        ctx,
        flow_lm,
        backbone_input_latents,
        text_embeddings,
        model_states,
        temp
    );
}

std::tuple<ggml_tensor*, bool> _run_flow_lm_and_increment_step(
    ScratchContext & ctx,
    flow_lm_t * flow_lm,
    flow_lm_states_t * model_states,
    std::vector<int> * text_tokens,
    ggml_tensor * backbone_input_latents,
    ggml_tensor * audio_conditioning,
    float temp
) {
    auto output = _run_flow_lm(
        ctx, flow_lm,
        model_states,
        text_tokens,
        backbone_input_latents,
        audio_conditioning,
        temp
    );
    int increment_by = 0;
    if ( text_tokens )
        increment_by += (int)text_tokens->size();
    if ( backbone_input_latents )
        increment_by += (int)backbone_input_latents->ne[1];
    if ( audio_conditioning )
        increment_by += (int)audio_conditioning->ne[1];
    increment_states( model_states, increment_by );
    return output;
}

void get_state_for_audio_prompt(
    ScratchContext & ctx,
    flow_lm_t * flow_lm,
    std::string audio_conditioning,
    flow_lm_states_t * model_states,
    float temp
) {
    //auto prompt = load_predefined_voice( audio_conditioning );

    auto loader = WeightLoader::from_safetensor( audio_conditioning.c_str(), &ctx, ctx.backend );
    if ( ! loader ) {
        fprintf(stderr, "error: failed to open voice %s\n", audio_conditioning.c_str() );
        exit(1);
    }
    ggml_tensor * prompt;
    auto n = loader->fetch( &prompt, "voice.audio_prompt" );
    assert( n );
    loader->load();

    init_states( ctx, flow_lm, model_states );

    _run_flow_lm_and_increment_step( ctx, flow_lm, model_states, NULL, NULL, prompt, temp );

    // TODO: look into _slice_kv_cache
}

void _generate_audio_stream_short_text(
    ScratchContext & ctx,
    flow_lm_t * flow_lm,
    flow_lm_states_t * start_states,
    flow_lm_states_t * model_states,
    mimi_model_t * mimi,
    mimi_states_t * mimi_states,
    float temp,
    std::string text_to_generate,
    int frames_after_eos,
    std::vector<short> & pcm
) {
    copy_states( ctx, start_states, model_states );

    // TODO: do we need to expand kv_cache?

    init( &ctx, mimi_states, mimi );

    float gen_len_sec = count_words(text_to_generate) + 2.0f;
    int max_gen_len = int(gen_len_sec * 12.5f);

    std::vector<int> prepared;
    conditioner_prepare( flow_lm->conditioner, text_to_generate, prepared );

    _run_flow_lm_and_increment_step( ctx, flow_lm, model_states, &prepared, NULL, NULL, temp );

    auto backbone_input = flow_lm->bos_emb;
    int eos_step = -1;

    std::vector<float> fy;

    for ( int generation_step = 0; generation_step < max_gen_len; ++generation_step ) {
        auto [next_latent, is_eos] = _run_flow_lm_and_increment_step( ctx, flow_lm,
            model_states, NULL, backbone_input, NULL, temp );
        if ( is_eos && eos_step == -1 )
            eos_step = generation_step;
        if ( eos_step != -1 && generation_step >= eos_step + frames_after_eos )
            break;

        // Add generated latent to queue for immediate decoding
        auto latent = next_latent;

        auto mimi_decoding_input = ggml_add( ctx,
            ggml_mul( ctx, flow_lm->emb_std, latent ), flow_lm->emb_mean );

        auto transposed = ggml_transpose( ctx, mimi_decoding_input );
        auto quantized = mimi_quantizer( ctx, mimi, transposed );

        auto y = mimi_decode_from_latent( ctx, mimi_states, mimi, quantized );

        y = ggml_scale( ctx, y, 32767.f );
        fy.resize( ggml_nelements(y) );
        ctx.build_forward_expand( y, fy.data() );
        ctx.compute();

        for ( auto value : fy ) {
            pcm.push_back( (short)(value * 32767.f) );
        }

        backbone_input = next_latent;
    }
}

void generate_audio(
    ScratchContext & ctx,
    flow_lm_t * flow_lm,
    flow_lm_states_t * start_states,
    flow_lm_states_t * scratch_states,
    mimi_model_t * mimi,
    mimi_states_t * mimi_states,
    float temp,
    std::string text_to_generate,
    std::vector<short> & pcm
) {
    auto chunks = split_into_best_sentences( flow_lm->conditioner->tokenizer, text_to_generate );
    for ( auto & chunk : chunks ) {
        auto [text_to_generate, frames_after_eos_guess] = prepare_text_prompt( chunk );
        frames_after_eos_guess += 2;
        _generate_audio_stream_short_text( ctx,
            flow_lm, start_states, scratch_states,
            mimi, mimi_states,
            temp,
            text_to_generate,
            frames_after_eos_guess,
            pcm
        );
    }
}

void generate_audio_to_wav(
    ScratchContext & ctx,
    flow_lm_t * flow_lm,
    flow_lm_states_t * start_states,
    flow_lm_states_t * scratch_states,
    mimi_model_t * mimi,
    mimi_states_t * mimi_states,
    float temp,
    std::string text_to_generate,
    std::string wav_path
) {
    std::vector<short> pcm;
    generate_audio( ctx,
        flow_lm, start_states, scratch_states,
        mimi, mimi_states,
        temp, text_to_generate,
        pcm
    );
    printf("%d\n", (int)pcm.size());
    save_wav( wav_path.c_str(), pcm, 24000 );
}

//////////////////
// MARK: api
//////////////////

std::map<std::string, std::string> default_voices = {
    {"alba",    "embeddings/alba.safetensors"},
    {"azelma",  "embeddings/azelma.safetensors"},
    {"cosette", "embeddings/cosette.safetensors"},
    {"eponine", "embeddings/eponine.safetensors"},
    {"fantine", "embeddings/fantine.safetensors"},
    {"javert",  "embeddings/javert.safetensors"},
    {"jean",    "embeddings/jean.safetensors"},
    {"marius",  "embeddings/marius.safetensors"}
};

void ptts_set_seed( unsigned int seed ) {
    random_seed = seed;
    gen.seed( seed );
}

unsigned int ptts_get_seed() {
    return random_seed;
}

struct ptts_context_t {
    ggml_backend * backend;
    ggml_backend * backend_cpu;
    ScratchContext * scratch;
    ScratchContext * scratch_cpu;
    std::string model_path;
    mimi_model_t * mimi;
    flow_lm_t * flow_lm;
    WeightLoader * weights;
    std::vector<int> end_of_sentence_tokens;
};

ptts_context_t * ptts_init(
    ggml_backend * backend,
    ggml_backend * backend_cpu,
    const char * model_path
) {
    auto scratch_cpu = new ScratchContext( 256, backend_cpu );
    auto scratch = new ScratchContext( 256, backend );

    ////////////
    // allocate
    ////////////
    auto mimi = ptts_mimi_alloc_default();

    auto flow_lm = from_config( config.flow_lm, model_path, config.mimi.quantizer.dimension );

    ////////
    // load
    ////////

    std::string filename = model_path;
    filename += "tts_b6369a24.safetensors";
    auto weights = WeightLoader::from_safetensor( filename.c_str(), scratch_cpu, backend );
    if ( ! weights ) {
        fprintf(stderr, "error: weights not found %s\n", filename.c_str() );
        exit(1);
    }

    // flow_lm
    get_weights( weights, "pts.flow_lm.", flow_lm );

    // mimi
    get_weights( weights, "pts.", mimi );

    weights->load();

    std::vector<int> end_of_sentence_tokens;
    flow_lm->conditioner->tokenizer.Encode( ".!...?", &end_of_sentence_tokens );
    if ( end_of_sentence_tokens.size() )
        end_of_sentence_tokens.erase( end_of_sentence_tokens.begin() );

    return new ptts_context_t {
        backend, backend_cpu,
        scratch, scratch_cpu,
        model_path,
        mimi,
        flow_lm,
        weights,
        end_of_sentence_tokens
    };
}

int ptts_get_sample_rate( ptts_context_t * ptts_ctx ) {
    return 24000;
}

int ptts_get_frame_size( ptts_context_t * ptts_ctx ) {
    return 1920;
}


struct ptts_stream_t {
    ptts_context_t * ptts_ctx;
    std::string voice;
    StateContext * state_ctx;

    mimi_states_t * mimi_states;
    flow_lm_states_t * model_states;
    flow_lm_states_t * cond_model_states;
    float temp;

    str_processor_t sproc;
    int frames_after_eos;
    int max_gen_len;
    int generation_step;
    int eos_step;
    ggml_tensor * backbone_input;
};

ptts_stream_t * ptts_stream_from_safetensors(
    ptts_context_t * ptts_ctx,
    const char * voice_c_str,
    float temp
) {
    std::string voice = voice_c_str;
    auto it = default_voices.find( voice );
    if ( it != default_voices.end() )
        voice = ptts_ctx->model_path + it->second;

    //////////
    // alloc states
    //////////
    auto state_ctx = new StateContext( ptts_ctx->backend );

    auto mimi_states = create_mimi_states( state_ctx, ptts_ctx->mimi );
    auto model_states = new_states( state_ctx, ptts_ctx->flow_lm, 1000 );
    auto cond_model_states = new_states( state_ctx, ptts_ctx->flow_lm, 1000 );

    state_ctx->alloc();
    state_ctx->init();

    //////////////
    // init state
    //////////////
    init( ptts_ctx->scratch, mimi_states, ptts_ctx->mimi );

    get_state_for_audio_prompt( *ptts_ctx->scratch, ptts_ctx->flow_lm, voice,
        cond_model_states, temp );

    auto stream = new ptts_stream_t {
        ptts_ctx,
        voice,
        state_ctx,
        mimi_states,
        model_states,
        cond_model_states,
        temp
    };

    ptts_stream_reset( stream );

    return stream;
}

void ptts_stream_reset( ptts_stream_t * stream ) {
    stream->max_gen_len = 0;
    stream->generation_step = 0;
    str_processor_reset( stream->sproc );
}

void ptts_stream_flush( ptts_stream_t * stream ) {
    str_processor_flush( stream->sproc );
}

void ptts_stream_send( ptts_stream_t * stream, const char * chunk ) {
    // empty chunk signals a flush
    if ( chunk[0] == '\0' ) {
        ptts_stream_flush( stream );
        return;
    }

    str_processor_ingest( stream->sproc, chunk );
}

void _stream_sentence_init(
    ptts_stream_t * stream,
    std::string text_to_generate,
    int frames_after_eos
) {
    ScratchContext & ctx = *stream->ptts_ctx->scratch;

    copy_states( ctx, stream->cond_model_states, stream->model_states );

    // TODO: do we need to expand kv_cache?

    init( &ctx, stream->mimi_states, stream->ptts_ctx->mimi );

    float gen_len_sec = count_words(text_to_generate) + 2.0f;
    int max_gen_len = int(gen_len_sec * 12.5f);

    // process the prompt
    std::vector<int> prepared;
    conditioner_prepare( stream->ptts_ctx->flow_lm->conditioner,
        text_to_generate, prepared );
    _run_flow_lm_and_increment_step( ctx, stream->ptts_ctx->flow_lm,
        stream->model_states, &prepared, NULL, NULL, stream->temp );

    stream->frames_after_eos = frames_after_eos;
    stream->max_gen_len = max_gen_len;
    stream->backbone_input = stream->ptts_ctx->flow_lm->bos_emb;
    stream->generation_step = 0;
    stream->eos_step = -1;
}

bool _stream_sentence_step(
    ptts_stream_t * stream,
    float * samples
) {
    if ( stream->generation_step >= stream->max_gen_len ) {
        fprintf(stderr, "warning: called with high gen step\n");
        return false;
    }

    ScratchContext & ctx = *stream->ptts_ctx->scratch;

    auto [next_latent, is_eos] = _run_flow_lm_and_increment_step( ctx,
        stream->ptts_ctx->flow_lm,
        stream->model_states,
        NULL, stream->backbone_input, NULL, stream->temp );

    if ( is_eos && stream->eos_step == -1 )
        stream->eos_step = stream->generation_step;
    if ( stream->eos_step != -1 && stream->generation_step >= stream->eos_step + stream->frames_after_eos ) {
        stream->generation_step = stream->max_gen_len;
        return false;
    }

    // Add generated latent to queue for immediate decoding
    auto latent = next_latent;

    auto mimi_decoding_input = ggml_add( ctx,
        ggml_mul( ctx, stream->ptts_ctx->flow_lm->emb_std, latent ),
        stream->ptts_ctx->flow_lm->emb_mean );

    //auto emb = mimi_decode_latent( ctx, mimi->quantizer, codes );
    auto transposed = ggml_cont( ctx, ggml_transpose( ctx, mimi_decoding_input ) );
    auto quantized = mimi_quantizer( ctx, stream->ptts_ctx->mimi, transposed );

    auto y = mimi_decode_from_latent( ctx, stream->mimi_states,
        stream->ptts_ctx->mimi, quantized );

    assert( ggml_nelements(y) == 1920 );
    ctx.build_forward_expand( y, samples );
    ctx.compute();

    stream->backbone_input = next_latent;

    stream->generation_step++;

    return true;
}

bool ptts_stream_receive( ptts_stream_t * stream, float * samples ) {
    if ( stream->generation_step < stream->max_gen_len ) {
        if ( _stream_sentence_step( stream, samples ) )
            return true;
    }

    if ( stream->sproc.sentences.size() ) {
        auto text_to_generate = stream->sproc.sentences.front();
        stream->sproc.sentences.pop_front();
        //printf("%s\n", text_to_generate.c_str());
        int number_of_words = count_words( text_to_generate );
        int frames_after_eos_guess = number_of_words <= 4 ? 3 : 1;
        frames_after_eos_guess += 2;

        _stream_sentence_init(
            stream,
            text_to_generate,
            frames_after_eos_guess
        );

        if ( _stream_sentence_step( stream, samples ) )
            return true;
    }

    return false;
}
