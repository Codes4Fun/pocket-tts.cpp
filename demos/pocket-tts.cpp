
#include <string>

#include <pocket_tts/pocket_tts.h>

static void print_usage(const char * program) {
    fprintf( stderr, R"(usage: %s <voice> \"Hey, how is it going?\" <file.wav>

where:
<voice> is a file path to a pre-encoded safetensor or one of the following:
    alba
    azelma
    cosette
    eponine
    fantine
    javert
    jean
    marius
<file.wav> is the file path to save a wav file to.

model should be in the subdirectory in the current directory:
  kyutai/pocket-tts-without-voice-cloning
or a subdirectory of the environment variable MODEL_CACHE

)", program );
    exit(1);
}

int main( int argc, char ** argv ) {
    if ( argc < 4 ) {
        print_usage( argv[0] );
        exit(-1);
    }

    ptts_run_test(argv[1], argv[2], argv[3]);

    return 0;
}
