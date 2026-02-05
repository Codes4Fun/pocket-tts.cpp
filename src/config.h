#pragma once

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
            /*.tokenizer_path=*/ "tokenizer.model",
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
