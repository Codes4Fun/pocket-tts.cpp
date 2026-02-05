#pragma once

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
    int D = (int)q->ne[0];
    int T = (int)q->ne[2];
    auto toffset = ctx.constant( (float)streaming_offset );
    int rope_max_period = 10000;
    moshi_get_timestep_embedding_new( ctx, (int)T, (int)D, toffset, rope_max_period, tsemb );
    std::tie(q, k) = moshi_apply_rope_new( ctx, q, k, &tsemb );
    ctx.debug("rope q", q);
    ctx.debug("rope k", k);

    // k, v = self._complete_kv(k, v, state)
    int offset = attn_states->current_end * (int)attn_states->keys->nb[2];
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
    int query_shape_1 = (int)query->ne[1];
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
            /*.norm1=*/ new torch_nn_layer_norm_t{1e-5f},
            /*.norm2=*/ new torch_nn_layer_norm_t{1e-5f},
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
