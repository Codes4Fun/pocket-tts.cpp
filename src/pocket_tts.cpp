
#include <stdio.h>
#include <assert.h>
#include <math.h>

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
#include "pocket_tts/modules/conv.h"
#include "pocket_tts/modules/seanet.h"
#include "pocket_tts/modules/mimi_transformer.h"
#include "pocket_tts/models/defaults.h"

#include "wav.h"

static bool ggml_backends_loaded = false;

std::string model_root;

void ensure_path( std::string & path ) {
    auto path_size = path.size();
    if ( path_size > 1 && path[path_size - 1] != '/' ) {
        path += "/";
    }
}

/**********************\
 * torch
\**********************/

void torch_chunk_2( GraphContext & ctx,
    ggml_tensor * x,
    ggml_tensor ** left,
    ggml_tensor ** right
) {
    assert( ( x->ne[0] % 2 ) == 0 );
    *left = ggml_view_4d( ctx, x,
        x->ne[0] / 2, x->ne[1], x->ne[2], x->ne[3],
        x->nb[1], x->nb[2], x->nb[3],
        0
    );
    *right = ggml_view_4d( ctx, x,
        x->ne[0] / 2, x->ne[1], x->ne[2], x->ne[3],
        x->nb[1], x->nb[2], x->nb[3],
        x->nb[1] / 2
    );
}

void torch_chunk_3( GraphContext & ctx,
    ggml_tensor * x,
    ggml_tensor ** left,
    ggml_tensor ** mid,
    ggml_tensor ** right
) {
    assert( ( x->ne[0] % 3 ) == 0 );
    *left = ggml_view_4d( ctx, x,
        x->ne[0] / 3, x->ne[1], x->ne[2], x->ne[3],
        x->nb[1], x->nb[2], x->nb[3],
        0
    );
    *mid = ggml_view_4d( ctx, x,
        x->ne[0] / 3, x->ne[1], x->ne[2], x->ne[3],
        x->nb[1], x->nb[2], x->nb[3],
        x->nb[1] / 3
    );
    *right = ggml_view_4d( ctx, x,
        x->ne[0] / 3, x->ne[1], x->ne[2], x->ne[3],
        x->nb[1], x->nb[2], x->nb[3],
        x->nb[1] * 2 / 3
    );
}

/********************\
 * config
\********************/

struct config_flow_lm_flow_t {
    int64_t depth; // 6
    int64_t dim; // 512
};

struct config_flow_lm_transformer_t {
    int64_t d_model; // 1024
    int64_t hidden_scale; // 4
    int64_t max_period; // 10000
    int64_t num_heads; // 16
    int64_t num_layers; // 6
};

struct config_flow_lm_lookup_table_t {
    int64_t dim; // 1024
    int64_t n_bins; // 4000
    std::string tokenizer; // "sentencepiece"
    std::string tokenizer_path; // "hf://kyutai/pocket-tts-without-voice-cloning/tokenizer.model@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"
};

struct config_flow_lm_t {
    std::string dtype; // "float32"
    config_flow_lm_flow_t flow;
    config_flow_lm_transformer_t transformer;
    config_flow_lm_lookup_table_t lookup_table;
};

struct config_mimi_quantizer_t {
    int dimension; // 32
    int output_dimension; // 512
};

struct config_mimi_seanet_t {};
struct config_mimi_transformer_t {};

struct config_mimi_t {
    std::string dtype; // "float32"
    int sample_rate; // 24000
    int channels; // 1
    float frame_rate; // 12.5
    config_mimi_seanet_t seanet;
    config_mimi_transformer_t transformer;
    config_mimi_quantizer_t quantizer;
};

struct config_t {
    config_flow_lm_t flow_lm;
    config_mimi_t mimi;
};

config_t config = {
    /*.flow_lm=*/ {
        /*.dtype=*/ "float32",
        /*.flow=*/ {
            /*.depth=*/ 6,
            /*.dim*/ 512
        },
        /*.transformer=*/ {
            /*.d_model=*/ 1024,
            /*.hidden_scale=*/ 4,
            /*.max_period=*/ 10000,
            /*.num_heads=*/ 16,
            /*.num_layers=*/ 6
        },
        /*.lookup_table=*/ {
            /*.dim=*/ 1024,
            /*.n_bins=*/ 4000,
            /*.tokenizer=*/ "sentencepiece",
            /*.tokenizer_path=*/ "kyutai/pocket-tts-without-voice-cloning/tokenizer.model",
            //"hf://kyutai/pocket-tts-without-voice-cloning/tokenizer.model@d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3",
        }
    },
    /*.mimi=*/ {
        /*.dtype=*/ "float32",
        /*.sample_rate=*/ 24000,
        /*.channels=*/ 1,
        /*.frame_rate=*/ 12.5f,
        /*.seanet=*/ {},
        /*.transformer=*/ {},
        /*.quantizer=*/ {
            /*.dimensions=*/ 32,
            /*.output_dimension=*/ 512
        }
    }
};

/**********************\
 * mimi
\**********************/

struct mimi_model_t {
    ggml_tensor * quantizer;
    moshi_streaming_conv_transpose_1d_t * upsample;
    moshi_streaming_transformer_t * decoder_transformer;
    ptts_seanet_decoder_t * decoder;
};

mimi_model_t * ptts_mimi_alloc_default() {
    auto upsample = new moshi_streaming_conv_transpose_1d_t{
        /*.in_channels=*/ 512,
        /*.out_channels=*/ 512,
        /*.kernel_size=*/ 32,
        /*.stride=*/ 16,
        /*.groups=*/ 512,
    };
    auto decoder_transformer = ptts_mimi_decoder_transformer_alloc_default();
    auto decoder = ptts_mimi_decoder_alloc_default();

    auto mimi = new mimi_model_t {
        /*.quantizer=*/ NULL,
        /*.upsample=*/ upsample,
        /*.decoder_transformer=*/ decoder_transformer,
        /*.decoder=*/ decoder,
    };
    return mimi;
}

void get_weights( WeightLoader * loader, std::string prefix, mimi_model_t * mimi ) {
    auto n = loader->fetch(
        &mimi->quantizer,
        prefix + "mimi.quantizer.output_proj.weight",
        (void*)ggml_conv_1d );
    assert( n );
    get_weights( loader, prefix + "mimi.upsample.convtr.",
        mimi->upsample );
    get_weights( loader, prefix + "mimi.decoder_transformer.transformer.",
        mimi->decoder_transformer );
    get_weights( loader, prefix + "mimi.decoder.", mimi->decoder );
}

struct mimi_states_t {
    ggml_tensor * upsample;
    moshi_streaming_transformer_state_t * decoder_transformer;
    ptts_seanet_decoder_states_t * decoder;
};

mimi_states_t * create_mimi_states(
    StateContext * state_ctx,
    mimi_model_t * mimi
) {
    auto states = new mimi_states_t;

    GGML_NE upsample_ne(1, 512, 1);
    moshi_streaming_conv_transpose_1d_state( state_ctx,
            mimi->upsample, upsample_ne, states->upsample );

    states->decoder_transformer = moshi_streaming_transformer_state(
        state_ctx, mimi->decoder_transformer, NULL );

    GGML_NE x_ne(16, 512, 1);
    states->decoder = create_ptts_seanet_decoder_states(
        state_ctx,
        mimi->decoder,
        x_ne
    );
    return states;
};

void init( ScratchContext * scratch, mimi_states_t * states, mimi_model_t * mimi ) {
    init( scratch, states->upsample, mimi->upsample );
    init( scratch, states->decoder_transformer, mimi->decoder_transformer , NULL );
    init( scratch, states->decoder, mimi->decoder );
}

ggml_tensor * mimi_quantizer(
    GraphContext & ctx,
    mimi_model_t * mimi,
    ggml_tensor * x
) {
    return ggml_conv_1d( ctx, mimi->quantizer, x, /*stride*/1, 0, 1 );
}

ggml_tensor * mimi_decode_from_latent(
    ScratchContext & ctx,
    mimi_states_t * states,
    mimi_model_t * mimi,
    ggml_tensor * latent
) {
    auto emb = moshi_streaming_conv_transpose_1d( ctx,
        states->upsample,
        mimi->upsample,
        latent );

    emb = moshi_projected_transformer( ctx,
        states->decoder_transformer,
        mimi->decoder_transformer,
        emb );

    auto out = ptts_seanet_decoder( ctx, states->decoder, mimi->decoder, emb );

    return out;
}

/**********************\
 * text
\**********************/

// pocket_tts.conditioners.text.LUTConditioner

struct lut_conditioner_t {
    sentencepiece::SentencePieceProcessor tokenizer;
    ggml_tensor * embed;
};

lut_conditioner_t * new_lut_conditioner( std::string filename ) {
    auto conditioner = new lut_conditioner_t;
    conditioner->tokenizer.Load( filename.c_str() );
    return conditioner;
}

void get_weights( WeightLoader * loader, std::string path, lut_conditioner_t * conditioner ) {
    int n = loader->fetch( &conditioner->embed, path + "embed.weight", (void*)ggml_get_rows );
    assert( n );
}

void conditioner_prepare(
    lut_conditioner_t * conditioner,
    std::string x,
    std::vector<int> & tokens
) {
    conditioner->tokenizer.Encode( x, &tokens );
}

ggml_tensor * conditioner_forward(
    GraphContext & ctx,
    lut_conditioner_t * conditioner,
    ggml_tensor * inputs
) {
    // _get_condition
    auto embeds = ggml_get_rows( ctx, conditioner->embed, inputs );
    return embeds;
}

bool iswhitespace( char c ) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

std::string strip( std::string text ) {
    int start = 0;
    while ( start < text.size() && iswhitespace( text[start] ) ) {
        start++;
    }
    if ( start == text.size() )
        return "";
    int end = text.size() - 1;
    while ( end >= 0 && iswhitespace( text[end] ) ) {
        end--;
    }
    return text.substr( start, end - start + 1 );
}

std::string merge_lines( std::string text ) {
    // TODO
    // replace '\n' '\r' with space if no spaces around it.
    return text;
}

int skip_whitespaces( std::string & text, int offset ) {
    while ( offset < text.size() && iswhitespace( text[offset] ) )
        offset++;
    return offset;
}

int find_whitespaces( std::string & text, int offset ) {
    while ( offset < text.size() && ! iswhitespace( text[offset] ) )
        offset++;
    return offset;
}

int word_count( std::string & text ) {
    int offset = 0;
    int words = 0;
    offset = skip_whitespaces( text, offset );
    while ( offset < text.size() ) {
        words++;
        offset = find_whitespaces( text, offset );
        if ( offset == text.size() )
            return words;
        offset = skip_whitespaces( text, offset );
    }

    return words;
}

bool islower( char c ) {
    return (c >= 97 && c <= 122);
}

bool isupper( char c ) {
    return ! islower( c );
}

bool toupper( char c ) {
    if ( ! islower(c) )
        return c;
    return c - 32;
}

std::tuple<std::string, int> prepare_text_prompt( std::string text ) {
    text = strip( text );
    if ( ! text.size() )
        throw new std::runtime_error( "Text prompt cannot be empty" );
    text = merge_lines( text );
    int number_of_words = word_count( text );
    int frames_after_eos_guess = number_of_words <= 4 ? 3 : 1;

    // Make sure it starts with an uppercase letter
    if ( ! isupper( text[0] ) )
        text[0] = toupper( text[0] );

    // Let's make sure it ends with some kind of punctuation
    // If it ends with a letter or digit, we add a period.
    if ( isalnum(text[text.size() - 1]) )
        text += '.';

    // The model does not perform well when there are very few tokens, so
    // we can add empty spaces at the beginning to increase the token count.
    if ( number_of_words < 5 )
        text = "        " + text;

    return std::make_tuple( text, frames_after_eos_guess );
}

std::vector<std::string> split_into_best_sentences(
    sentencepiece::SentencePieceProcessor & tokenizer,
    std::string text_to_generate
) {
    auto [scratch_transformer_out, _] = prepare_text_prompt( text_to_generate );

    std::vector<int> list_of_tokens;
    tokenizer.Encode( text_to_generate, &list_of_tokens );

    std::vector<int> end_of_sentence_tokens;
    tokenizer.Encode( ".!...?", &end_of_sentence_tokens );
    /* returns end_of_sentence_tokens
        260 ▁
        263 .
        682 !
        799 ...
        292 ?
    */

    std::vector<std::vector<int>> sentence_tokens = {{}};
    auto begin = ++end_of_sentence_tokens.begin(); // skip first token
    auto end = end_of_sentence_tokens.end();
    for ( auto & token : list_of_tokens ) {
        sentence_tokens.back().push_back( token );
        if ( std::find( begin, end, token ) != end ) {
            sentence_tokens.push_back({});
        }
    }
    if ( ! sentence_tokens.back().size() )
        sentence_tokens.pop_back();

    const int max_nb_tokens_in_a_chunk = 50;
    std::vector<std::string> chunks = {""};
    int current_nb_of_tokens_in_chunk = 0;
    for ( auto & tokens : sentence_tokens ) {
        if ( current_nb_of_tokens_in_chunk != 0 ) {
            if ( current_nb_of_tokens_in_chunk + tokens.size() > max_nb_tokens_in_a_chunk ) {
                // new chunk
                current_nb_of_tokens_in_chunk = 0;
                chunks.push_back("");
            } else {
                // appending to current chunk
                chunks.back() += " ";
            }
        }
        std::string sentence;
        tokenizer.Decode( tokens, &sentence );
        chunks.back() += sentence;
        current_nb_of_tokens_in_chunk += tokens.size();
    }

    return chunks;
}

/****************\
 * transformer
\****************/

struct streaming_multihead_attention_t {
    int embed_dim;
    int num_heads;
    torch_nn_linear_t * in_proj;
    torch_nn_linear_t * out_proj;
};

struct streaming_multihead_attention_state_t {
    ggml_tensor * keys;
    ggml_tensor * values;
    int current_end;
};

void get_weights( WeightLoader * loader, std::string path, streaming_multihead_attention_t * attn ) {
    get_weights( loader, path + "in_proj.", attn->in_proj );
    get_weights( loader, path + "out_proj.", attn->out_proj );
}

streaming_multihead_attention_state_t * new_states(
    StateContext * state_ctx,
    streaming_multihead_attention_t * attn,
    int sequence_length
) {
    auto states = new streaming_multihead_attention_state_t;
    int dim_per_head = attn->embed_dim / attn->num_heads;
    GGML_NE ne( dim_per_head, attn->num_heads, sequence_length );
    state_ctx->new_tensor( ne, GGML_TYPE_F32, &states->keys );
    state_ctx->new_tensor( ne, GGML_TYPE_F32, &states->values );
    states->current_end = 0;
    return states;
}

void init_states( streaming_multihead_attention_state_t * states ) {
    states->current_end = 0;
}

void copy_states(
    ScratchContext & ctx,
    streaming_multihead_attention_state_t * src,
    streaming_multihead_attention_state_t * dst
) {
    dst->current_end = src->current_end;
    auto cpy_keys = ggml_cpy( ctx, src->keys, dst->keys );
    auto cpy_values = ggml_cpy( ctx, src->values, dst->values );
    ctx.build_forward_expand( cpy_keys );
    ctx.build_forward_expand( cpy_values );
}

void increment_states( streaming_multihead_attention_state_t * states, int increment ) {
    states->current_end += increment;
}

ggml_tensor * streaming_multihead_attention_forward( ScratchContext & ctx,
    streaming_multihead_attention_t * attn,
    streaming_multihead_attention_state_t * attn_states,
    ggml_tensor * query
) {
    ctx.debug("query", query);
    int H = attn->num_heads;
    auto projected = torch_nn_linear( ctx, attn->in_proj, query );
    ctx.debug("projected", projected);
    auto q = ggml_view_3d( ctx, projected,
        projected->ne[0] / 3,
        projected->ne[1],
        projected->ne[2],
        projected->nb[1],
        projected->nb[2],
        0 );
    q = ggml_cont( ctx, q );
    // b t (h d) -> b t h d
    q = ggml_reshape_4d( ctx, q,
        q->ne[0] / H,
        H,
        q->ne[1],
        q->ne[2] );

    auto k = ggml_view_3d( ctx, projected,
        projected->ne[0] / 3,
        projected->ne[1],
        projected->ne[2],
        projected->nb[1],
        projected->nb[2],
        projected->nb[1] / 3 );
    // b t (h d) -> b t h d
    k = ggml_cont( ctx, k );
    k = ggml_reshape_4d( ctx, k,
        k->ne[0] / H,
        H,
        k->ne[1],
        k->ne[2] );

    auto v = ggml_view_3d( ctx, projected,
        projected->ne[0] / 3,
        projected->ne[1],
        projected->ne[2],
        projected->nb[1],
        projected->nb[2],
        projected->nb[1] * 2 / 3 );
    // b t (h d) -> b t h d
    v = ggml_cont( ctx, v );
    v = ggml_reshape_4d( ctx, v,
        v->ne[0] / H,
        H,
        v->ne[1],
        v->ne[2] );

    // q, k = self._apply_rope(q, k, state)
    int streaming_offset = attn_states->current_end;
    timestep_embedding_t tsemb = { NULL, NULL };
    int D = q->ne[0];
    int T = q->ne[2];
    auto toffset = ctx.constant( (float)streaming_offset );
    int rope_max_period = 10000;
    moshi_get_timestep_embedding_new( ctx, (int)T, (int)D, toffset, rope_max_period, tsemb );
    std::tie(q, k) = moshi_apply_rope_new( ctx, q, k, &tsemb );
    ctx.debug("rope q", q);
    ctx.debug("rope k", k);

    // k, v = self._complete_kv(k, v, state)
    int offset = attn_states->current_end * attn_states->keys->nb[2];
    auto cache_0 = ggml_view_3d( ctx, attn_states->keys,
        attn_states->keys->ne[0], attn_states->keys->ne[1], k->ne[2],
        attn_states->keys->nb[1], attn_states->keys->nb[2],
        offset
    );
    cache_0 = ggml_cpy( ctx, k, cache_0 );
    ctx.build_forward_expand( cache_0 );
    k = ggml_view_3d( ctx, cache_0,
        attn_states->keys->ne[0],
        attn_states->keys->ne[1],
        attn_states->current_end + k->ne[2],
        attn_states->keys->nb[1],
        attn_states->keys->nb[2],
        -offset
    );

    auto cache_1 = ggml_view_3d( ctx, attn_states->values,
        attn_states->values->ne[0], attn_states->values->ne[1], v->ne[2],
        attn_states->values->nb[1], attn_states->values->nb[2],
        offset
    );
    cache_1 = ggml_cpy( ctx, v, cache_1 );
    ctx.build_forward_expand( cache_1 );
    v = ggml_view_3d( ctx, cache_1,
        attn_states->values->ne[0],
        attn_states->values->ne[1],
        attn_states->current_end + v->ne[2],
        attn_states->values->nb[1],
        attn_states->values->nb[2],
        -offset
    );
    ctx.debug("_complete_kv k", k);
    ctx.debug("_complete_kv v", v);

    ggml_tensor * attn_mask = NULL;
    int query_shape_1 = query->ne[1];
    if ( query_shape_1 != 1 ) {
        int shift = attn_states->current_end;
        GGML_NE mask_shape( query_shape_1 + attn_states->current_end, query_shape_1 );
        std::vector<float> mask( mask_shape.nelements() );
        for ( int y = 0; y < mask_shape.ne[1]; y++ ) {
            for ( int x = 0; x < mask_shape.ne[0]; x++ ) {
                mask[mask_shape.ne[0] * y + x] = y + shift >= x? 0 : -INFINITY;
            }
        }
        attn_mask = ctx.input( mask_shape, mask );
    }
#ifdef DUMP_ATTN_MASK
    printf("attn_mask %d %d %d\n", (int)attn_mask->ne[0], (int)attn_mask->ne[1], (int)attn_mask->ne[2] );
    auto am = 0;
    for ( int y = 0; y < attn_mask->ne[1]; y++ ) {
        if (pixel_offset + 512 <= pixels.size()) {
            int nel = attn_mask->ne[0];
            for ( int i = 0; i < nel && i < 512; i++ ) {
                auto value = ((float*)transformer_attn_mask->b.data())[am++];
                assert( value == 0 || value == -INFINITY );
                pixels[pixel_offset + i] = value == 0? 255 : 0;
            }
            for ( int i = nel; i < 512; i++ ) {
                pixels[pixel_offset + i] = 64;
            }
            pixel_offset += 512;
        }
    }
#endif

    q = ggml_permute( ctx, q, 0, 2 ,1, 3 );
    k = ggml_permute( ctx, k, 0, 2 ,1, 3 );
    v = ggml_permute( ctx, v, 0, 2 ,1, 3 );
    auto x = torch_nn_functional_scaled_dot_product_attention( ctx, q, k, v, NULL, attn_mask );
    ctx.debug("scaled_dot_product", x);
    x = ggml_cont( ctx, ggml_permute( ctx, x, 0, 2 ,1, 3 ) );
    x = ggml_reshape_3d( ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3] );
    x = torch_nn_linear( ctx, attn->out_proj, x );
    ctx.debug("out_proj", x);
    return x;
}

struct streaming_transformer_layer_t {
    streaming_multihead_attention_t * self_attn;
    torch_nn_layer_norm_t * norm1;
    torch_nn_layer_norm_t * norm2;
    torch_nn_linear_t * linear1;
    torch_nn_linear_t * linear2;
    moshi_layer_scale_t * layer_scale_1;
    moshi_layer_scale_t * layer_scale_2;
};

struct streaming_transformer_layer_state_t {
    streaming_multihead_attention_state_t * self_attn;
};

void get_weights( WeightLoader * loader, std::string path, streaming_transformer_layer_t * layer ) {
    get_weights( loader, path + "self_attn.", layer->self_attn );
    get_weights( loader, path + "norm1.", layer->norm1 );
    get_weights( loader, path + "norm2.", layer->norm2 );
    get_weights( loader, path + "linear1.", layer->linear1 );
    get_weights( loader, path + "linear2.", layer->linear2 );
    if ( layer->layer_scale_1 )
        get_weights( loader, path + "layer_scale_1.", layer->layer_scale_1 );
    if ( layer->layer_scale_2 )
        get_weights( loader, path + "layer_scale_2.", layer->layer_scale_2 );
}

streaming_transformer_layer_state_t * new_states( 
    StateContext * state_ctx,
    streaming_transformer_layer_t * layer,
    int sequence_length
) {
    auto states = new streaming_transformer_layer_state_t;
    states->self_attn = new_states( state_ctx, layer->self_attn, sequence_length );
    return states;
}

void init_states( streaming_transformer_layer_state_t * states ) {
    init_states( states->self_attn );
}

void copy_states(
    ScratchContext & ctx,
    streaming_transformer_layer_state_t * src,
    streaming_transformer_layer_state_t * dst
) {
    copy_states( ctx, src->self_attn, dst->self_attn );
}

void increment_states( streaming_transformer_layer_state_t * states, int increment ) {
    increment_states( states->self_attn, increment );
}

ggml_tensor * streaming_transformer_layer_forward( ScratchContext & ctx,
    streaming_transformer_layer_t * layer,
    streaming_transformer_layer_state_t * layer_states,
    ggml_tensor * x
) {
    // _sa_block
    auto x_orig = x;
    x = torch_nn_layer_norm( ctx, layer->norm1, x );
    auto update = streaming_multihead_attention_forward( ctx,
        layer->self_attn, layer_states->self_attn, x );
    if ( layer->layer_scale_1 ) {
        update = moshi_layer_scale( ctx, layer->layer_scale_1, update );
    }
    x = ggml_add( ctx, x_orig, update );
    // _ff_block
    x_orig = x;
    x = torch_nn_layer_norm( ctx, layer->norm2, x );
    update = torch_nn_linear( ctx, layer->linear1, x );
    update = ggml_gelu( ctx, update );
    update = torch_nn_linear( ctx, layer->linear2, update );
    if ( layer->layer_scale_2 ) {
        update = moshi_layer_scale( ctx, layer->layer_scale_2, update );
    }
    x = ggml_add( ctx, x_orig, update );
    return x;
}



struct streaming_transformer_t {
    std::vector<streaming_transformer_layer_t*> layers;
};

struct streaming_transformer_states_t {
    std::vector<streaming_transformer_layer_state_t*> layers;
    int shift;
};

streaming_transformer_t * from_config( config_flow_lm_transformer_t & config ) {
    auto transformer = new streaming_transformer_t;

    transformer->layers.resize( config.num_layers );
    for ( int i = 0; i < transformer->layers.size(); i++ ) {
        auto layer = new streaming_transformer_layer_t {
            /*.self_attn=*/ new streaming_multihead_attention_t {
                /*.embed_dim=*/ 1024,
                /*.num_heads=*/ 16,
                /*.in_proj=*/ new torch_nn_linear_t,
                /*.out_proj=*/ new torch_nn_linear_t,
            },
            /*.norm1=*/ new torch_nn_layer_norm_t{1e-5},
            /*.norm2=*/ new torch_nn_layer_norm_t{1e-5},
            /*.linear1=*/ new torch_nn_linear_t,
            /*.linear2=*/ new torch_nn_linear_t,
            /*.layer_scale_1=*/ NULL,
            /*.layer_scale_2=*/ NULL,
        };
        transformer->layers[i] = layer;
    }

    return transformer;
}

void get_weights( WeightLoader * loader, std::string path, streaming_transformer_t * transformer ) {
    for ( int i = 0; i < transformer->layers.size(); i++ ) {
        auto layer = transformer->layers[i];
        get_weights( loader, path + "layers." + std::to_string(i) + ".", layer );
    }
}

streaming_transformer_states_t * new_states(
    StateContext * state_ctx,
    streaming_transformer_t * transformer,
    int sequence_length
) {
    auto states = new streaming_transformer_states_t;
    states->shift = 0;
    states->layers.resize( transformer->layers.size() );
    for ( int i = 0; i < states->layers.size(); i++ ) {
        states->layers[i] = new_states( state_ctx, transformer->layers[i], sequence_length );
    }
    return states;
}

void init_states( streaming_transformer_states_t * states ) {
    states->shift = 0;
    for ( int i = 0; i < states->layers.size(); i++ ) {
        init_states( states->layers[i] );
    }
}

void copy_states(
    ScratchContext & ctx,
    streaming_transformer_states_t * src,
    streaming_transformer_states_t * dst
) {
    assert( src->layers.size() == dst->layers.size() );
    for ( int i = 0; i < src->layers.size(); i++ ) {
        copy_states( ctx, src->layers[i], dst->layers[i] );
    }
    dst->shift = src->shift;
}

void increment_states( streaming_transformer_states_t * states, int increment ) {
    for ( int i = 0; i < states->layers.size(); i++ ) {
        increment_states( states->layers[i], increment );
    }
    states->shift += increment;
}

ggml_tensor * streaming_transformer_forward( ScratchContext & ctx,
    streaming_transformer_t * transformer,
    streaming_transformer_states_t * states,
    ggml_tensor * x
) {
    for ( int i = 0; i < transformer->layers.size(); i++ ) {
        auto layer = transformer->layers[i];
        auto layer_state = states->layers[i];
        x = streaming_transformer_layer_forward( ctx, layer, layer_state, x );
    }
    return x;
}

/****************\
 * mlp
\****************/

ggml_tensor * mlp_modulate( GraphContext & ctx, ggml_tensor * x,
    ggml_tensor * shift, ggml_tensor * scale
) {
    return ggml_add(ctx,
        ggml_mul( ctx, x, ggml_add( ctx, scale, ctx.constant(1.f) ) ),
        shift );
}

// pocket_tts.modules.mlp.RMSNorm

struct mlp_rms_norm_t {
    float eps;
    ggml_tensor * alpha;
};

ggml_tensor * mlp_rms_norm(
        GraphContext & ctx,
        mlp_rms_norm_t * norm,
        ggml_tensor * x ) {
    if ( x->type != GGML_TYPE_F32 )
        x = ggml_cast( ctx, x, GGML_TYPE_F32 );
    assert( x->ne[0] > 1 );

    auto mean = ggml_mean( ctx, x );
    auto varsqr = ggml_sub( ctx, x, mean );
    auto var = ggml_sum( ctx, ggml_mul( ctx, varsqr, varsqr ) );
    var = ggml_scale( ctx, var, 1.f / (x->ne[0] - 1) );

    var = ggml_add( ctx, var, ctx.constant(norm->eps) );
    auto y = ggml_div( ctx, x, ggml_sqrt( ctx, var ) );

    //auto y = ggml_rms_norm( ctx, x, norm->eps );
    //auto y = ggml_norm( ctx, x, norm->eps );
    return ggml_mul( ctx, norm->alpha, y );
}

void get_weights( WeightLoader * loader, std::string path, mlp_rms_norm_t * norm ) {
    auto n = loader->fetch( &norm->alpha, path + "alpha", (void*)ggml_rms_norm );
    assert( n );
}

// pocket_tts.modules.mlp.LayerNorm

struct mlp_layer_norm_t {
    float eps;
    ggml_tensor * weight;
    ggml_tensor * bias;
};

ggml_tensor * mlp_layer_norm(
        ggml_context * ctx,
        mlp_layer_norm_t * norm,
        ggml_tensor * x ) {
    if ( x->type != GGML_TYPE_F32 )
        x = ggml_cast( ctx, x, GGML_TYPE_F32 );
    x = ggml_norm( ctx, x, norm->eps );
    if ( norm->weight )
        x = ggml_mul( ctx, x, norm->weight );
    if ( norm->bias )
        x = ggml_add( ctx, x, norm->bias );
    return x;
}

void get_weights( WeightLoader * loader, std::string path, mlp_layer_norm_t * norm ) {
    loader->fetch( &norm->weight, path + "weight", (void*)ggml_mul );
    loader->fetch( &norm->bias, path + "bias", (void*)ggml_add );
}

// pocket_tts.modules.mlp.TimestepEmbedder

struct mlp_timestep_embedder_t {
    torch_nn_linear_t * mlp_0;
    torch_nn_linear_t * mlp_2;
    mlp_rms_norm_t * mlp_3;
    ggml_tensor * freqs;
};

void get_weights( WeightLoader * loader, std::string path,
    mlp_timestep_embedder_t * time_emb
) {
    get_weights( loader, path + "mlp.0.", time_emb->mlp_0 );
    get_weights( loader, path + "mlp.2.", time_emb->mlp_2 );
    get_weights( loader, path + "mlp.3.", time_emb->mlp_3 );
    loader->fetch( &time_emb->freqs, path + "freqs", (void*)ggml_mul );
}

// TODO: remove debug code
//std::vector<float> _embedding;

ggml_tensor * mlp_timestep_embedder_forward( ScratchContext & ctx,
    mlp_timestep_embedder_t * time_embed, ggml_tensor * t
) {
    auto args = ggml_mul( ctx, time_embed->freqs, t );
    auto embedding = ggml_concat( ctx, ggml_cos( ctx, args ), ggml_sin( ctx, args ), 0 );
    auto t_emb = torch_nn_linear( ctx, time_embed->mlp_0, embedding );
    t_emb = ggml_silu( ctx, t_emb );
    t_emb = torch_nn_linear( ctx, time_embed->mlp_2, t_emb );
    t_emb = mlp_rms_norm( ctx, time_embed->mlp_3, t_emb );
    //if ( ! _embedding.size() ) {
    //    _embedding.resize( ggml_nelements(t_emb) );
    //    ctx.build_forward_expand( t_emb, _embedding.data() );
    //}
    return t_emb;
}

// pocket_tts.modules.mlp.ResBlock

struct mlp_res_block_t {
    mlp_layer_norm_t * in_ln; // eps=1e-6
    torch_nn_linear_t * mlp_0;
    torch_nn_linear_t * mlp_2;
    torch_nn_linear_t * adaLN_modulation_1;
};

void get_weights( WeightLoader * loader, std::string path, mlp_res_block_t * res_block ) {
    get_weights( loader, path + "in_ln.", res_block->in_ln );
    get_weights( loader, path + "mlp.0.", res_block->mlp_0 );
    get_weights( loader, path + "mlp.2.", res_block->mlp_2 );
    get_weights( loader, path + "adaLN_modulation.1.", res_block->adaLN_modulation_1 );
}

ggml_tensor * mlp_res_block_forward( GraphContext & ctx,
    mlp_res_block_t * res_block, ggml_tensor * x, ggml_tensor * y
) {
    y = ggml_silu( ctx, y );
    y = torch_nn_linear( ctx, res_block->adaLN_modulation_1, y );
    ggml_tensor * shift_mlp;
    ggml_tensor * scale_mlp;
    ggml_tensor * gate_mlp;
    torch_chunk_3( ctx, y, &shift_mlp, &scale_mlp, &gate_mlp );
    auto h = mlp_modulate( ctx,
        mlp_layer_norm( ctx, res_block->in_ln, x ),
        shift_mlp, scale_mlp );
    h = torch_nn_linear( ctx, res_block->mlp_0, h );
    h = ggml_silu( ctx, h );
    h = torch_nn_linear( ctx, res_block->mlp_2, h );
    return ggml_add( ctx, x, ggml_mul( ctx, gate_mlp, h) );
}

// pocket_tts.modules.mlp.FinalLayer

struct final_layer_t {
    mlp_layer_norm_t * norm_final;
    torch_nn_linear_t * linear;
    torch_nn_linear_t * adaLN_modulation_1;
};

void get_weights( WeightLoader * loader, std::string path, final_layer_t * final_layer ) {
    get_weights( loader, path + "norm_final.", final_layer->norm_final );
    get_weights( loader, path + "linear.", final_layer->linear );
    get_weights( loader, path + "adaLN_modulation.1.", final_layer->adaLN_modulation_1 );
}

ggml_tensor * final_layer_forward(
    ScratchContext & ctx,
    final_layer_t * final_layer,
    ggml_tensor * x,
    ggml_tensor * c
) {
    c = ggml_silu( ctx, c );
    c = torch_nn_linear( ctx, final_layer->adaLN_modulation_1, c );
    ggml_tensor * shift, * scale;
    torch_chunk_2( ctx, c, &shift, &scale );
    x = mlp_modulate( ctx, mlp_layer_norm( ctx, final_layer->norm_final, x ),
        shift, scale );
    x = torch_nn_linear( ctx, final_layer->linear, x );
    return x;
}

// pocket_tts.modules.mlp.SimpleMLPAdaLN

struct simple_mlp_adaln_t {
    std::vector<mlp_res_block_t*> res_blocks;
    final_layer_t * final_layer;
    torch_nn_linear_t * input_proj;
    torch_nn_linear_t * cond_embed;
    std::vector<mlp_timestep_embedder_t*> time_embed;
};

simple_mlp_adaln_t * from_config( config_flow_lm_flow_t & config ) {
    auto flow_net = new simple_mlp_adaln_t;

    flow_net->res_blocks.resize( config.depth );
    for ( int i = 0; i < flow_net->res_blocks.size(); i++ ) {
        flow_net->res_blocks[i] = new mlp_res_block_t {
            /*.in_ln=*/ new mlp_layer_norm_t{1e-6},
            /*.mlp_0=*/ new torch_nn_linear_t,
            /*.mlp_2=*/ new torch_nn_linear_t,
            /*.adaLN_modulation_1=*/ new torch_nn_linear_t
        };
    }

    flow_net->final_layer = new final_layer_t{
        /*.norm_final=*/ new mlp_layer_norm_t{1e-6},
        /*.linear=*/ new torch_nn_linear_t,
        /*.adaLN_modulation_1=*/  new torch_nn_linear_t,
    };

    flow_net->input_proj = new torch_nn_linear_t;
    flow_net->cond_embed = new torch_nn_linear_t;

    flow_net->time_embed.resize(2);
    flow_net->time_embed[0] = new mlp_timestep_embedder_t{
        /*.mlp_0=*/ new torch_nn_linear_t,
        /*.mlp_2=*/ new torch_nn_linear_t,
        /*.mlp_3=*/ new mlp_rms_norm_t{1e-5},
        /*.freqs=*/NULL
    };
    flow_net->time_embed[1] = new mlp_timestep_embedder_t{
        /*.mlp_0=*/ new torch_nn_linear_t,
        /*.mlp_2=*/ new torch_nn_linear_t,
        /*.mlp_3=*/ new mlp_rms_norm_t{1e-5},
        /*.freqs=*/NULL
    };

    return flow_net;
}

void get_weights( WeightLoader * loader, std::string path, simple_mlp_adaln_t * flow_net ) {
    for ( int i = 0; i < flow_net->res_blocks.size(); i++ ) {
        get_weights( loader, path + "res_blocks." + std::to_string(i) + ".", flow_net->res_blocks[i] );
    }
    get_weights( loader, path + "final_layer.", flow_net->final_layer );
    get_weights( loader, path + "input_proj.", flow_net->input_proj );
    get_weights( loader, path + "cond_embed.", flow_net->cond_embed );
    for ( int i = 0; i < flow_net->time_embed.size(); i++ ) {
        get_weights( loader, path + "time_embed." + std::to_string(i) + ".", flow_net->time_embed[i] );
    }
}

ggml_tensor * simple_mlp_adaln_forward( ScratchContext & ctx,
    simple_mlp_adaln_t * flow_net,
    ggml_tensor * c,
    ggml_tensor * s,
    ggml_tensor * t,
    ggml_tensor * x
) {
    x = torch_nn_linear( ctx, flow_net->input_proj, x );
    auto t_combined = ggml_scale( ctx, ggml_add( ctx,
        mlp_timestep_embedder_forward( ctx, flow_net->time_embed[1], t ),
        mlp_timestep_embedder_forward( ctx, flow_net->time_embed[0], s ) ),
        1.f / 2.f );
    c = torch_nn_linear( ctx, flow_net->cond_embed, c );
    auto y = ggml_add( ctx, t_combined, c );
    for ( auto & res_block : flow_net->res_blocks ) {
        x = mlp_res_block_forward( ctx, res_block, x, y );
    }
    return final_layer_forward( ctx, flow_net->final_layer, x, y );
}

// pocket_tts.models.flow_lm.FlowLMModel

struct flow_lm_t {
    int ldim;
    lut_conditioner_t * conditioner;
    simple_mlp_adaln_t * flow_net;
    ggml_tensor * emb_std;
    ggml_tensor * emb_mean;
    ggml_tensor * bos_emb;
    torch_nn_linear_t * input_linear;
    streaming_transformer_t * transformer;
    torch_nn_layer_norm_t * out_norm;
    torch_nn_linear_t * out_eos;
};

struct flow_lm_states_t {
    streaming_transformer_states_t * transformer;
    ggml_tensor * output;
};

flow_lm_t * from_config( config_flow_lm_t & config, int latent_dim = 64 ) {
    auto flow_lm = new flow_lm_t;
    flow_lm->ldim = latent_dim;
    flow_lm->conditioner = new_lut_conditioner( model_root + config.lookup_table.tokenizer_path );
    flow_lm->flow_net = from_config( config.flow );
    flow_lm->emb_std = NULL;
    flow_lm->emb_mean = NULL;
    flow_lm->bos_emb = NULL;
    flow_lm->input_linear = new torch_nn_linear_t;
    flow_lm->transformer = from_config( config.transformer );
    flow_lm->out_norm = new torch_nn_layer_norm_t{1e-5};
    flow_lm->out_eos = new torch_nn_linear_t;
    return flow_lm;
}

void get_weights( WeightLoader * loader, std::string path, flow_lm_t * flow_lm ) {
    get_weights( loader, path + "conditioner.", flow_lm->conditioner );
    get_weights( loader, path + "flow_net.", flow_lm->flow_net );
    int n;
    n = loader->fetch( &flow_lm->emb_std, "pts.flow_lm.emb_std", (void*)ggml_mul );
    assert( n );
    n = loader->fetch( &flow_lm->emb_mean, "pts.flow_lm.emb_mean", (void*)ggml_add );
    assert( n );
    n = loader->fetch( &flow_lm->bos_emb, "pts.flow_lm.bos_emb" );
    assert( n );
    get_weights( loader, path + "input_linear.", flow_lm->input_linear );
    get_weights( loader, path + "transformer.", flow_lm->transformer );
    get_weights( loader, path + "out_norm.", flow_lm->out_norm );
    get_weights( loader, path + "out_eos.", flow_lm->out_eos );
}

flow_lm_states_t * new_states(
    StateContext * state_ctx,
    flow_lm_t * flow_lm,
    int sequence_length
) {
    auto states = new flow_lm_states_t;
    states->transformer = new_states( state_ctx, flow_lm->transformer, sequence_length );
    state_ctx->new_tensor( GGML_NE(32), GGML_TYPE_F32, &states->output );
    return states;
}

void init_states( ScratchContext & ctx, flow_lm_t * flow_lm, flow_lm_states_t * states ) {
    ctx.build_forward_expand( ggml_cpy( ctx, flow_lm->bos_emb, states->output ) );
    init_states( states->transformer );
}

void copy_states(
    ScratchContext & ctx,
    flow_lm_states_t * src,
    flow_lm_states_t * dst
) {
    copy_states( ctx, src->transformer, dst->transformer );
    ctx.build_forward_expand( ggml_cpy( ctx, src->output, dst->output) );
    ctx.compute(); // maybe should be done by the root caller?
}

void increment_states( flow_lm_states_t * states, int increment ) {
    increment_states( states->transformer, increment );
}

//////////////////
// MARK: tts
//////////////////

std::tuple<ggml_tensor*, bool> flow_lm_sample_next_latent(
    ScratchContext & ctx,
    flow_lm_t * flow_lm,
    ggml_tensor * sequence,
    ggml_tensor * text_embeddings,
    flow_lm_states_t * model_states
) {
    // lsd_decode_steps = 1
    float temp = 0.7f;
    // noise_clamp = None
    const float eos_threshold = -4;

    // sequence should already be initialized to bos_emb?
    ggml_tensor * input_ = NULL;
    if ( sequence ) {
        input_ = torch_nn_linear( ctx, flow_lm->input_linear, sequence );
    }
    if ( text_embeddings ) {
        if ( input_ )
            input_ = ggml_concat( ctx, text_embeddings, input_, 1 );
        else
            input_ = text_embeddings;
    }

    auto transformer_out = streaming_transformer_forward( ctx,
        flow_lm->transformer,
        model_states->transformer,
        input_
    );

    if ( flow_lm->out_norm )
        transformer_out = torch_nn_layer_norm( ctx, flow_lm->out_norm, transformer_out );
    // self.backbone also trims transformer_out to sequence shape but
    // the whole thing gets trimmed anyway by the caller self.forward
    // after self.backbone
    transformer_out = ggml_view_1d( ctx,
        transformer_out,
        transformer_out->ne[0],
        transformer_out->nb[1] * (transformer_out->ne[1] - 1)
    );

    auto out_eos = torch_nn_linear( ctx, flow_lm->out_eos, transformer_out );
    out_eos = ggml_sub( ctx, out_eos, ctx.constant( eos_threshold ) );
    out_eos = ggml_sum( ctx, out_eos );
    float eos;
    ctx.build_forward_expand( out_eos, &eos );

    GGML_NE noise_shape( flow_lm->ldim, transformer_out->ne[1], transformer_out->ne[2] );
    float std = sqrtf( temp );
    auto noise = ctx.normal_( noise_shape, /*mean=*/0.0f, std );

    // lsd_decode num_steps = 1
    auto c = transformer_out;
    auto s = ctx.constant(0.f);
    auto t = ctx.constant(1.f);
    auto x = noise;
    auto flow_dir = simple_mlp_adaln_forward( ctx, flow_lm->flow_net, c, s, t, x );
    auto latent = ggml_add( ctx, x, flow_dir );
    ctx.build_forward_expand( latent, model_states->output );

    ctx.compute();
    bool is_eos = eos > 0.f;
    return std::make_tuple( model_states->output, is_eos );
}

std::tuple<ggml_tensor*, bool> _run_flow_lm(
    ScratchContext & ctx,
    flow_lm_t * flow_lm,
    flow_lm_states_t * model_states,
    std::vector<int> * text_tokens,
    ggml_tensor * backbone_input_latents,
    ggml_tensor * audio_conditioning
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
        model_states
    );
}

std::tuple<ggml_tensor*, bool> _run_flow_lm_and_increment_step(
    ScratchContext & ctx,
    flow_lm_t * flow_lm,
    flow_lm_states_t * model_states,
    std::vector<int> * text_tokens,
    ggml_tensor * backbone_input_latents,
    ggml_tensor * audio_conditioning
) {
    auto output = _run_flow_lm(
        ctx, flow_lm,
        model_states,
        text_tokens,
        backbone_input_latents,
        audio_conditioning
    );
    int increment_by = 0;
    if ( text_tokens )
        increment_by += text_tokens->size();
    if ( backbone_input_latents )
        increment_by += backbone_input_latents->ne[1];
    if ( audio_conditioning )
        increment_by += audio_conditioning->ne[1];
    increment_states( model_states, increment_by );
    return output;
}

void get_state_for_audio_prompt(
    ScratchContext & ctx,
    flow_lm_t * flow_lm,
    std::string audio_conditioning,
    flow_lm_states_t * model_states
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

    _run_flow_lm_and_increment_step( ctx, flow_lm, model_states, NULL, NULL, prompt );

    // TODO: look into _slice_kv_cache
}


void _generate_audio_stream_short_text(
    ScratchContext & ctx,
    flow_lm_t * flow_lm,
    flow_lm_states_t * start_states,
    flow_lm_states_t * model_states,
    mimi_model_t * mimi,
    mimi_states_t * mimi_states,
    std::string text_to_generate,
    int frames_after_eos,
    std::vector<short> & pcm
) {
    copy_states( ctx, start_states, model_states );

    // TODO: do we need to expand kv_cache?

    init( &ctx, mimi_states, mimi );

    float gen_len_sec = word_count(text_to_generate) + 2.0f;
    int mag_gen_len = int(gen_len_sec * 12.5f);

    std::vector<int> prepared;
    conditioner_prepare( flow_lm->conditioner, text_to_generate, prepared );

    _run_flow_lm_and_increment_step( ctx, flow_lm, model_states, &prepared, NULL, NULL );

    auto backbone_input = flow_lm->bos_emb;
    int eos_step = -1;

    std::vector<float> fy;

    for ( int generation_step = 0; generation_step < mag_gen_len; ++generation_step ) {
        auto [next_latent, is_eos] = _run_flow_lm_and_increment_step( ctx, flow_lm,
            model_states, NULL, backbone_input, NULL );
        if ( is_eos && eos_step == -1 )
            eos_step = generation_step;
        if ( eos_step != -1 && generation_step >= eos_step + frames_after_eos )
            break;

        // Add generated latent to queue for immediate decoding
        auto latent = next_latent;

        auto mimi_decoding_input = ggml_add( ctx,
            ggml_mul( ctx, flow_lm->emb_std, latent ), flow_lm->emb_mean );
        //auto mimi_decoding_input = event_0_x;

        //auto emb = mimi_decode_latent( ctx, mimi->quantizer, codes );
        auto transposed = ggml_transpose( ctx, mimi_decoding_input );
        auto quantized = mimi_quantizer( ctx, mimi, transposed );

        auto y = mimi_decode_from_latent( ctx, mimi_states, mimi, quantized );

        y = ggml_scale( ctx, y, 32767.f );
        fy.resize( ggml_nelements(y) );
        ctx.build_forward_expand( y, fy.data() );
        ctx.compute();
        //printf("%d %d %d %d\n", (int)y->ne[0], (int)y->ne[1], (int)y->ne[2], (int)y->ne[3]);
        for ( auto value : fy ) {
            pcm.push_back( (short)value );
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
    std::string text_to_generate,
    std::string wav_path
) {
    std::vector<short> pcm;
    generate_audio( ctx,
        flow_lm, start_states, scratch_states,
        mimi, mimi_states,
        text_to_generate,
        pcm
    );
    printf("%d\n", (int)pcm.size());
    save_wav( wav_path.c_str(), pcm, 24000 );
}

//////////////////
// MARK: test
//////////////////

std::map<std::string, std::string> default_voices = {
    {"alba",    "kyutai/pocket-tts-without-voice-cloning/embeddings/alba.safetensors"},
    {"azelma",  "kyutai/pocket-tts-without-voice-cloning/embeddings/azelma.safetensors"},
    {"cosette", "kyutai/pocket-tts-without-voice-cloning/embeddings/cosette.safetensors"},
    {"eponine", "kyutai/pocket-tts-without-voice-cloning/embeddings/eponine.safetensors"},
    {"fantine", "kyutai/pocket-tts-without-voice-cloning/embeddings/fantine.safetensors"},
    {"javert",  "kyutai/pocket-tts-without-voice-cloning/embeddings/javert.safetensors"},
    {"jean",    "kyutai/pocket-tts-without-voice-cloning/embeddings/jean.safetensors"},
    {"marius",  "kyutai/pocket-tts-without-voice-cloning/embeddings/marius.safetensors"}
};

void ptts_run_test(
    std::string voice,
    std::string text_to_generate,
    std::string output_filepath
) {
    printf("hello world\n");

    const char * model_cache = getenv("MODEL_CACHE");
    model_root = model_cache? model_cache : "";
    ensure_path( model_root );

    auto it = default_voices.find( voice );
    if ( it != default_voices.end() )
        voice = model_root + it->second;

    if ( ! ggml_backends_loaded ) {
        ggml_backend_load_all();
        ggml_backends_loaded = true;
    }
    //ggml_backend * backend;
    //if ( device ) {
    //    backend = ggml_backend_init_by_name( device, NULL );
    //} else {
    //    backend = ggml_backend_init_best();
    //}
    auto backend = ggml_backend_init_best();
    if ( ! backend ) {
        fprintf( stderr, "error: failed to initialize backend.\n" );
        exit(1);
    }
    auto backend_cpu = ggml_backend_init_by_type( GGML_BACKEND_DEVICE_TYPE_CPU, NULL );
    if ( ! backend_cpu ) {
        fprintf( stderr, "error: failed to initialize cpu device.\n" );
        exit(1);
    }
    auto scratch_cpu = new ScratchContext( 256, backend_cpu );
    auto scratch = new ScratchContext( 256, backend );
    ScratchContext & ctx = *scratch;

    ////////////
    // allocate
    ////////////
    auto mimi = ptts_mimi_alloc_default();

    auto flow_lm = from_config( config.flow_lm, config.mimi.quantizer.dimension );

    ////////
    // load
    ////////

    std::string filename = model_root + "kyutai/pocket-tts-without-voice-cloning/tts_b6369a24.safetensors";
    auto weights = WeightLoader::from_safetensor( filename.c_str(), scratch_cpu, backend );
    if ( ! weights ) {
        fprintf(stderr, "error: weights not found\n" );
        exit(1);
    }

    // flow_lm
    get_weights( weights, "pts.flow_lm.", flow_lm );

    // mimi
    get_weights( weights, "pts.", mimi );

    weights->load();

    //////////
    // alloc states
    //////////
    auto state_ctx = new StateContext( backend );

    auto mimi_states = create_mimi_states( state_ctx, mimi );
    auto model_states = new_states( state_ctx, flow_lm, 1000 );
    auto cond_model_states = new_states( state_ctx, flow_lm, 1000 );

    state_ctx->alloc();
    state_ctx->init();

    //////////////
    // init state
    //////////////
    init( scratch, mimi_states, mimi );

    ////////////
    // generate
    ////////////

    //std::string text_to_generate = "She sells sea shells by the sea shore. The quick brown fox jumped over the sleeping dog. How much wood could a wood chuck chuck if a wood chuck could chuck wood. Nevermind, how on earth a wood chuck could chuck wood or why they would chuck wood, but rather if a wood chuck could chuck wood and would chuck wood, how much would a wood chuck chuck?";
    //std::string output_filepath = "ptts-c.wav";

    get_state_for_audio_prompt( ctx, flow_lm,
        voice,
        cond_model_states );

    generate_audio_to_wav( ctx,
        flow_lm, cond_model_states, model_states,
        mimi, mimi_states,
        text_to_generate,
        output_filepath
    );
}
