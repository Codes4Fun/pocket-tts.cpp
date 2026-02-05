#pragma once

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

flow_lm_t * from_config( config_flow_lm_t & config, std::string model_root, int latent_dim = 64 ) {
    auto flow_lm = new flow_lm_t;
    flow_lm->ldim = latent_dim;
    flow_lm->conditioner = new_lut_conditioner( model_root + config.lookup_table.tokenizer_path );
    flow_lm->flow_net = from_config( config.flow );
    flow_lm->emb_std = NULL;
    flow_lm->emb_mean = NULL;
    flow_lm->bos_emb = NULL;
    flow_lm->input_linear = new torch_nn_linear_t;
    flow_lm->transformer = from_config( config.transformer );
    flow_lm->out_norm = new torch_nn_layer_norm_t{1e-5f};
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

std::tuple<ggml_tensor*, bool> flow_lm_sample_next_latent(
    ScratchContext & ctx,
    flow_lm_t * flow_lm,
    ggml_tensor * sequence,
    ggml_tensor * text_embeddings,
    flow_lm_states_t * model_states,
    float temp
) {
    // lsd_decode_steps = 1
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
