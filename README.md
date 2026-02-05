
# pocket-tts.cpp

A port of Kyutai's Pocket TTS to C++ and ggml.
 * https://github.com/kyutai-labs/pocket-tts

Similar to the moshi.cpp project:
 * https://github.com/Codes4Fun/moshi.cpp

## Status

It's in a pre-alpha development state, but enough to generate a wav file from voice and text arguments.

TODO:
* bug fix, the audio is slightly off
* optimizations, GGUF conversion, renaming stuff
* lots and lots of testing

## Data / Weights

They can be downloaded here:
* https://huggingface.co/kyutai/pocket-tts-without-voice-cloning

The demo will look for them in the subdirectory "kyutai/pocket-tts-without-voice-cloning", in the current working directory or if you set the environment variable `MODEL_CACHE`.

## Build

The same instructions as moshi.cpp:
 * https://github.com/Codes4Fun/moshi.cpp?tab=readme-ov-file#build-dependencies

At the end you will end up with `bin/pocket-tts` which allows you to pick a voice, for example `-v cosette`, and the text you want to be spoken, and it will output to your speakers or you can specify an output file `-v output.mp3` or a wav file, ogg, etc. It has several other options `pocket-tts -h`, you can also run a benchmark `pocket-tts --bench`.
