#pragma once

moshi_streaming_transformer_t * ptts_mimi_decoder_transformer_alloc_default() {
    auto mimi_decoder__transformer = new moshi_streaming_transformer_t;
    mimi_decoder__transformer->context = 250;
    mimi_decoder__transformer->weights_per_step = 0;
    mimi_decoder__transformer->capacity = 250;
    mimi_decoder__transformer->layers.resize( 2 );
    mimi_decoder__transformer->rope_max_period = 10000;
    mimi_decoder__transformer->dim_per_head = 512 / 8;
    for ( int64_t i = 0; i < 2; i++ ) {
        auto layer = new moshi_streaming_transformer_layer_t{
            /*.norm1_rms=*/ NULL,
            /*.norm1=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*self_attn=*/ new moshi_smha_t{
                /*.embed_dim=*/ 512,
                /*.num_heads=*/ 8,
                /*.cross_attention=*/ false,
                /*.cache_cross_attention=*/ true,
                /*.causal=*/ true,
                /*.rope_max_period=*/ 10000,
                /*.context=*/ 250,
                /*.weights_per_step=*/ 0,
                /*.weights_per_step_schedule=*/ {},
                /*.in_projs=*/ { new torch_nn_linear_t },
                /*.out_projs=*/ { new torch_nn_linear_t }
            },
            /*.layer_scale_1=*/ new moshi_layer_scale_t,
            /*.norm_cross=*/ NULL,
            /*.cross_attention=*/ NULL,
            /*.norm2_rms=*/ NULL,
            /*.norm2=*/ new torch_nn_layer_norm_t{ /*.eps=*/ 0.000000 },
            /*.weights_per_step_schedule=*/ {},
            /*.gating=*/ {},
            /*.linear1=*/ new torch_nn_linear_t,
            /*.linear2=*/ new torch_nn_linear_t,
            /*.layer_scale_2=*/ new moshi_layer_scale_t
        };
        mimi_decoder__transformer->layers[i] = layer;
    }
    return mimi_decoder__transformer;
}

ptts_seanet_decoder_t * ptts_mimi_decoder_alloc_default( ) {
    auto mimi_decoder = new ptts_seanet_decoder_t{
        /*.model_0=*/ new moshi_streaming_conv_1d_t{
            /*.in_channels=*/ 512,
            /*.out_channels=*/ 512,
            /*.kernel_size=*/ 7,
            /*.stride=*/ 1,
        },
        /*.model_2=*/ new moshi_streaming_conv_transpose_1d_t{
            /*.in_channels=*/ 512,
            /*.out_channels=*/ 256,
            /*.kernel_size=*/ 12,
            /*.stride=*/ 6,
            /*.groups=*/ 1,
        },
        /*.model_3=*/ new moshi_seanet_resnet_block_t{
            /*.block_1=*/ new moshi_streaming_conv_1d_t{
                /*.in_channels=*/ 256,
                /*.out_channels=*/ 128,
                /*.kernel_size=*/ 3,
                /*.stride=*/ 1,
            },
            /*.block_3=*/ new moshi_stateless_conv_1d_t{
                /*.in_channels=*/ 128,
                /*.out_channels=*/ 256,
                /*.kernel_size=*/ 1,
            }
        },
        /*.model_5=*/ new moshi_streaming_conv_transpose_1d_t{
            /*.in_channels=*/ 256,
            /*.out_channels=*/ 128,
            /*.kernel_size=*/ 10,
            /*.stride=*/ 5,
            /*.groups=*/ 1,

        },
        /*.model_6=*/ new moshi_seanet_resnet_block_t{
            /*.block_1=*/ new moshi_streaming_conv_1d_t{
                /*.in_channels=*/ 128,
                /*.out_channels=*/ 64,
                /*.kernel_size=*/ 3,
                /*.stride=*/ 1,
            },
            /*.block_3=*/ new moshi_stateless_conv_1d_t{
                /*.in_channels=*/ 64,
                /*.out_channels=*/ 128,
                /*.kernel_size=*/ 1,
            }
        },
        /*.model_8=*/ new moshi_streaming_conv_transpose_1d_t{
            /*.in_channels=*/ 128,
            /*.out_channels=*/ 64,
            /*.kernel_size=*/ 8,
            /*.stride=*/ 4,
            /*.groups=*/ 1,
        },
        /*.model_9=*/ new moshi_seanet_resnet_block_t{
            /*.block_1=*/ new moshi_streaming_conv_1d_t{
                /*.in_channels=*/ 64,
                /*.out_channels=*/ 32,
                /*.kernel_size=*/ 3,
                /*.stride=*/ 1,
            },
            /*.block_3=*/ new moshi_stateless_conv_1d_t{
                /*.in_channels=*/ 32,
                /*.out_channels=*/ 64,
                /*.kernel_size=*/ 1,
            }
        },
        /*.model_11=*/ new moshi_streaming_conv_1d_t{
            /*.in_channels=*/ 64,
            /*.out_channels=*/ 1,
            /*.kernel_size=*/ 3,
            /*.stride=*/ 1,
        },
    };

    return mimi_decoder;
}

