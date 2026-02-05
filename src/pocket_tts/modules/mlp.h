#pragma once

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
            /*.in_ln=*/ new mlp_layer_norm_t{1e-6f},
            /*.mlp_0=*/ new torch_nn_linear_t,
            /*.mlp_2=*/ new torch_nn_linear_t,
            /*.adaLN_modulation_1=*/ new torch_nn_linear_t
        };
    }

    flow_net->final_layer = new final_layer_t{
        /*.norm_final=*/ new mlp_layer_norm_t{1e-6f},
        /*.linear=*/ new torch_nn_linear_t,
        /*.adaLN_modulation_1=*/  new torch_nn_linear_t,
    };

    flow_net->input_proj = new torch_nn_linear_t;
    flow_net->cond_embed = new torch_nn_linear_t;

    flow_net->time_embed.resize(2);
    flow_net->time_embed[0] = new mlp_timestep_embedder_t{
        /*.mlp_0=*/ new torch_nn_linear_t,
        /*.mlp_2=*/ new torch_nn_linear_t,
        /*.mlp_3=*/ new mlp_rms_norm_t{1e-5f},
        /*.freqs=*/NULL
    };
    flow_net->time_embed[1] = new mlp_timestep_embedder_t{
        /*.mlp_0=*/ new torch_nn_linear_t,
        /*.mlp_2=*/ new torch_nn_linear_t,
        /*.mlp_3=*/ new mlp_rms_norm_t{1e-5f},
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
