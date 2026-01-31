
# pocket-tts.cpp

A port of Kyutai's Pocket TTS to C++ and ggml.
 * https://github.com/kyutai-labs/pocket-tts

Similar to the moshi.cpp project:
 * https://github.com/Codes4Fun/moshi.cpp

## Status

It's in a pre-alpha development state, but enough to generate a wav file from voice and text arguments.

TODO:
* refactor - move alot of code into files.
* streaming api - similar to moshi.cpp but chunks of text in, chunks of pcm out.
* add ffmpeg / sdl to demo.
* benchmarking
* GGUF conversion
* lots and lots of testing

## Build

The same instructions as moshi.cpp:
 * https://github.com/Codes4Fun/moshi.cpp?tab=readme-ov-file#build-dependencies

at the end you will end up with `bin/pocket-tts` that takes 3 arguments, a voice (like "cosette"), text you want to be spoken, and the path location of the output wav file.

## Data / Weights

They can be downloaded here:
* https://huggingface.co/kyutai/pocket-tts-without-voice-cloning

The demo will look for them in the subdirectory "kyutai/pocket-tts-without-voice-cloning", in the current working directory or if you set the environment variable `MODEL_CACHE`.
