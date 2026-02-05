
#include <assert.h>

#include <string>
#include <vector>

#include <pocket_tts/ptrs.h>

#include <pocket_tts/pocket_tts.h>

#include "ffmpeg_helpers.h"
#include "sdl_helper.h"
#include "util.h"

static void print_usage(const char * program) {
    fprintf( stderr, R"(usage: %s [option(s)] \"Hey, how is it going?\"

streams to sdl audio if output file not specified.

option(s):
  -h,       --help             show this help message

  -l,       --list-devices     list hardware and exit.
  -d NAME,  --device NAME      use named hardware.
            --threads N        number of CPU threads.

  -r PATH,  --model-root PATH  path to where all kyutai models are stored and
                               replaces MODEL_CACHE environment variable. the
                               models at root are in subdirectories of
                               'organization/model'
  -m PATH,  --model PATH       path to where model is, can be relative to the
                               MODEL_CACHE environment variable, or program
                               directory, or working directory. by default is
                               'kyutai/pocket-tts-without-voice-cloning'
  -v VOICE, --voice VOICE      path to voice model/prefix, or one of:
                                 alba
                                 azelma
                                 cosette
                                 eponine
                                 fantine
                                 javert
                                 jean
                                 marius

  -o FNAME, --output FNAME     output to file, can be wav, mp3, ogg, etc.
  -i FNAME, --input FNAME      input text file.

  -s N,     --seed N           seed value.
  -t N,     --temperature N    consistency vs creativity, default 0.6
            --bench            sets defaults for benching.

)", program );
    exit(1);
}

SDL_mutex * stdin_mutex;
SDL_cond * stdin_ready;
std::string stdin_text;

int stdin_thread_func( void * arg ) {
    char buffer[1024];
    while (true) {
        char * read = fgets( buffer, sizeof(buffer) - 1, stdin );
        if (! read ) {
            printf("fgets returned NULL\n");
            break;
        }
        SDL_LockMutex( stdin_mutex );
        stdin_text += buffer;
        SDL_CondSignal( stdin_ready );
        SDL_UnlockMutex( stdin_mutex );
    }
    return 0;
}

bool get_text( std::string & text, bool block ) {
    bool ready = false;
    SDL_LockMutex( stdin_mutex );
    if ( block ) {
        while ( ! stdin_text.size() ) {
            SDL_CondWait( stdin_ready, stdin_mutex );
        }
    }
    if ( stdin_text.size() ) {
        text = stdin_text;
        stdin_text = "";
        ready = true;
    }
    SDL_UnlockMutex( stdin_mutex );
    return ready;
}


#include <signal.h>
void signal_handler(int dummy) {
    printf("exit\n");
    exit(1);
}

////////////////
// MARK: Main
////////////////

int main( int argc, char ** argv ) {
    signal(SIGINT, signal_handler);

    const char * device = NULL;
    int n_threads = 0;

    const char * model_cache = getenv("MODEL_CACHE");
    std::string model_root = model_cache? model_cache : "";
    std::string model_path = "kyutai/pocket-tts-without-voice-cloning/";
    bool tts_path_set = false;
    std::string voice = "alba";

    const char * input_filepath = NULL;
    const char * output_filepath = NULL;

    bool seed_set = false;
    bool temperature_set = false;
    float temperature = 0.7f;
    bool bench = false;

    const char * text = NULL;

    //////////////////////
    // MARK: Parse Args
    //////////////////////

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
        }
        if (arg == "-l" || arg == "--list-devices") {
            list_devices();
        }
        if (arg == "-d" || arg == "--device") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires name of device\n", argv[i] );
                exit(1);
            }
            device = argv[++i];
            continue;
        }
        if (arg == "--threads") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            n_threads = std::stoi( argv[++i] );
            continue;
        }
        if (arg == "-r" || arg == "--model-root") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires path to models\n", argv[i] );
                exit(1);
            }
            model_root = argv[++i];
            continue;
        }
        if (arg == "-m" || arg == "--model") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires filepath to model\n", argv[i] );
                exit(1);
            }
            model_path = argv[++i];
            tts_path_set = true;
            continue;
        }
        if (arg == "-v" || arg == "--voice") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires filepath to voice\n", argv[i] );
                exit(1);
            }
            voice = argv[++i];
            continue;
        }
        if (arg == "-o" || arg == "--output") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires filepath to output file\n", argv[i] );
                exit(1);
            }
            output_filepath = argv[++i];
            continue;
        }
        if (arg == "-i" || arg == "--input") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires filepath to input file\n", argv[i] );
                exit(1);
            }
            input_filepath = argv[++i];
            continue;
        }
        if (arg == "-s" || arg == "--seed") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            seed_set = true;
            ptts_set_seed( std::stoi( argv[++i] ) );
            continue;
        }
        if (arg == "-t" || arg == "--temperature") {
            if (i + 1 >= argc) {
                fprintf( stderr, "error: \"%s\" requires value\n", argv[i] );
                exit(1);
            }
            temperature_set = true;
            temperature = (float) std::stod(argv[++i]);
            continue;
        }
        if (arg == "--bench") {
            bench = true;
            continue;
        }
        if (arg[0] == '-') {
            fprintf( stderr, "error: unrecognized option \"%s\"\n", argv[i] );
            exit(1);
        }
        if (!text) {
            text = argv[i];
        } else {
            fprintf( stderr, "error: unexpected extra argument \"%s\"\n", argv[i] );
            exit(1);
        }
    }

    bool use_sdl = ! output_filepath;
    if ( bench ) {
        if ( ! text && !input_filepath )
            text = "The quick brown fox jumped over the sleeping dog.";
        if ( ! seed_set ) ptts_set_seed( 0 );
        if ( ! temperature_set ) temperature = 0;
        use_sdl = false;
    }

    /////////////////////////
    // MARK: Validate Args
    /////////////////////////

    const char * ext = NULL;
    if ( output_filepath ) {
        ext = get_ext( output_filepath );
        if ( ! ext ) {
            fprintf( stderr, "unable to determine output file type without ext.\n" );
            print_usage(argv[0]);
        }
    }

    // initialize device

    ggml_backend_load_all();
    ggml_backend * backend;
    if ( device ) {
        backend = ggml_backend_init_by_name( device, NULL );
    } else {
        backend = ggml_backend_init_best();
    }
    if ( ! backend ) {
        fprintf( stderr, "error: failed to initialize backend.\n" );
        exit(1);
    }
    auto dev = ggml_backend_get_device( backend );
    if ( n_threads > 0 ) {
        auto reg = ggml_backend_dev_backend_reg( dev );
        auto set_n_threads = (ggml_backend_set_n_threads_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if ( set_n_threads ) {
            set_n_threads( backend, n_threads );
        }
    }

    ggml_backend * backend_cpu;
    if ( ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU ) {
        backend_cpu = backend;
    } else {
        backend_cpu = ggml_backend_init_by_type( GGML_BACKEND_DEVICE_TYPE_CPU, NULL );
        if ( ! backend_cpu ) {
            fprintf( stderr, "error: failed to initialize cpu device.\n" );
            exit(1);
        }
        if ( n_threads > 0 ) {
            auto dev_cpu = ggml_backend_get_device( backend_cpu );
            auto reg_cpu = ggml_backend_dev_backend_reg( dev_cpu );
            auto set_n_threads_cpu = (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(reg_cpu, "ggml_backend_set_n_threads");
            if ( set_n_threads_cpu ) {
                set_n_threads_cpu( backend, n_threads );
            }
        }
    }

    std::string program_path = get_program_path(argv[0]);
    ensure_path( program_path );
    ensure_path( model_root );
    ensure_path( model_path );

    bool found_file, found_dir;
    check_arg_path( model_path, found_file, found_dir );
    if ( ! found_dir ) {
        if ( found_file ) {
            fprintf( stderr, "error: expected directory but found file: %s\n",
                    model_path.c_str() );
            exit(1);
        }

        if ( is_abs_or_rel( model_path ) ) {
            fprintf( stderr, "error: could not find directory: %s\n",
                model_path.c_str() );
            exit(1);
        }

        std::vector<std::string> paths;
        paths.push_back( model_root + model_path );
        paths.push_back( program_path + model_path );
        for ( auto & path : paths ) {
            check_arg_path( path, found_file, found_dir );
            if ( found_dir ) {
                model_path = path;
                break;
            }
        }
        if ( ! found_dir ) {
            fprintf( stderr, "error: could not find a default model directory\n" );
            exit(1);
        }
    }
    printf( "found model path: %s\n", model_path.c_str() );

    if ( input_filepath ) {
        if ( ! file_exists( input_filepath ) ) {
            fprintf( stderr, "error: failed to find input file: \"%s\"\n", input_filepath );
            exit(1);
        }
    }

    ///////////////////////////////////////////////
    // MARK: Initialize
    ///////////////////////////////////////////////

    stdin_mutex = SDL_CreateMutex();
    stdin_ready = SDL_CreateCond();

    // read in text file
    // TODO: stream it in chunks instead of reading the whole file in
    own_ptr<char> input_file_text;
    if ( input_filepath ) {
        auto f = fopen( input_filepath, "rb" );
        if ( ! f ) {
            fprintf( stderr, "error: unable to open \"%s\"\n", input_filepath );
            exit( 1 );
        }
        auto e = fseek( f, 0, SEEK_END );
        assert( e == 0 );
        auto size = ftell( f );
        assert( size > 0 );
        e = fseek( f, 0, SEEK_SET );
        assert( e == 0 );
        input_file_text = new char[size];
        assert( input_file_text );
        auto n = fread( input_file_text, size, 1, f );
        assert( n == 1 );
        fclose( f );
        text = input_file_text;
    }

    auto ctx = ptts_init( backend, backend_cpu, model_path.c_str() );
    const int sample_rate = ptts_get_sample_rate( ctx );
    const int frame_size = ptts_get_frame_size( ctx );
    auto stream = ptts_stream_from_safetensors( ctx, voice.c_str(), temperature );

    AVChannelLayout mono;
    av_channel_layout_default( &mono, 1 );
    unref_ptr<FILE> mimi_file;
    own_ptr<Encoder> encoder;
    if ( output_filepath ) {
        encoder = new Encoder();
        encoder->init_from( output_filepath, 24000, AV_SAMPLE_FMT_FLT, mono );
    }
    if ( use_sdl ) {
        if ( SDL_Init(SDL_INIT_AUDIO | SDL_INIT_TIMER) < 0 ) {
            fprintf( stderr, "error: Could not initialize SDL: %s\n", SDL_GetError() );
            exit( 1 );
        }
    }

    unref_ptr<AVFrame> av_frame;
    own_ptr<Resampler> resampler;
    AudioState state;
    if ( encoder ) {
        av_frame = av_frame_alloc();
        av_frame->nb_samples     = frame_size;
        av_frame->ch_layout      = mono;
        av_frame->format         = AV_SAMPLE_FMT_FLT;
        av_frame->sample_rate    = sample_rate;
        check_error( av_frame_get_buffer( av_frame, 0 ),
            "Error making frame buffer" );

        resampler = new Resampler;
        resampler->set_input( sample_rate, AV_SAMPLE_FMT_FLT, mono, frame_size );
        resampler->set_output( encoder->codec_ctx );
        resampler->init();
    }
    if ( use_sdl ) {
        int format = AUDIO_F32;
        int nb_samples = frame_size;
        int nb_bytes = nb_samples * 4;

        SDL_AudioSpec want, have;
        SDL_zero(want);
        want.freq = sample_rate;
        want.format = format;
        want.channels = 1;
        want.samples = nb_samples; // Buffer size
        want.callback = sdl_audio_callback;
        want.userdata = &state;

        state.device_id = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
        if (state.device_id == 0) {
            fprintf(stderr, "Failed to open SDL audio device: %s\n", SDL_GetError());
            SDL_Quit();
            return 1;
        }

        // do we need a resampler?
        if (have.freq != sample_rate) {
            fprintf(stderr, "error: sample_rate %d\n", have.freq);
            return 1;
        }
        if (have.format != format) {
            fprintf(stderr, "error: format %d\n", have.format);
            return 1;
        }
        if (have.channels != 1) {
            fprintf(stderr, "error: channels %d\n", have.channels);
            return 1;
        }
        if (have.samples != nb_samples) {
            fprintf(stderr, "error: samples %d\n", have.samples);
            return 1;
        }

        sdl_init_frames( state, 3, nb_bytes );
    }

    if ( use_sdl )
        SDL_PauseAudioDevice(state.device_id, 0);

    /////////////////////
    // MARK: Main Loop
    /////////////////////

    printf("seed: %d\n", ptts_get_seed() );

    int64_t gen_start = ggml_time_ms();
    int64_t lm_start = gen_start;
    int64_t lm_delta_time = 0;
    int64_t lm_frames = 0;

    int count = (int)strlen( text );

    bool active = true;
    while ( active ) {
        active = false;

        if ( count ) {
            // artificially feed pocket_tts 15 chars at a time to simulate streaming
            int chunk_size = count > 15 ? 15 : count;
            std::string chunk( text, chunk_size );
            text += chunk_size;
            count -= chunk_size;

            auto start = ggml_time_ms();
            ptts_stream_send( stream, chunk.c_str() );
            if ( count == 0 ) {
                ptts_stream_flush( stream );
            }
            lm_delta_time += ggml_time_ms() - start;
            active = true;
        }

        if ( encoder ) {
            auto start = ggml_time_ms();
            if ( ptts_stream_receive( stream, (float*)av_frame->data[0] ) ) {
                lm_delta_time += ggml_time_ms() - start;
                lm_frames++;
                auto resampler_frame = resampler->frame( av_frame );
                while ( resampler_frame ) {
                    encoder->frame( resampler_frame );
                    resampler_frame = resampler->frame();
                }
                active = true;
            }
        } else if ( use_sdl ) {
            sdl_frame_t * sdl_frame = sdl_get_frame( state );
            auto start = ggml_time_ms();
            if ( ptts_stream_receive( stream, (float*)sdl_frame->data ) ) {
                lm_delta_time += ggml_time_ms() - start;
                lm_frames++;
                sdl_send_frame( state, sdl_frame );
                active = true;
            } else {
                sdl_free_frame( state, sdl_frame );
            }
        } else {
            static std::vector<float> samples( frame_size );
            auto start = ggml_time_ms();
            if ( ptts_stream_receive( stream, samples.data() ) ) {
                lm_delta_time += ggml_time_ms() - start;
                lm_frames++;
                active = true;
            }
        }
    }

    auto gen_end = ggml_time_ms();
    printf("done generating. %f\n", (gen_end - gen_start) / 1000.f);
    printf("frame count: %4d frames\n", (int)lm_frames);
    printf("frame rate:  %f frames/s\n", lm_frames * 1000.f / lm_delta_time );

    ////////////////
    // MARK: Exit
    ////////////////

    if ( encoder )
        encoder->flush();

    if ( use_sdl ) {
        SDL_Delay(1);
        SDL_CloseAudioDevice(state.device_id);
        SDL_Quit();
    }

    return 0;
}
