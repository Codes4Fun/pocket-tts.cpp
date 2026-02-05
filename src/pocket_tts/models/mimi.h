#pragma once

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
