
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

Early testing shows the CPU running faster than the GPU on a 9 year old computer, running at about 19 frames per second which is fast enough for realtime.

I'm skeptical on the latency, they only show giving the model a full sentence before it starts generating, where as with moshi tts it only needs two tokens to start generating. So for realtime applications your llm needs to run at a faster rate to get faster responses. But pocket-tts performance is good enough for some applications, such as modding games to have voices, maybe integrating into browsers to read web pages to you, or listening to messages and emails.

## Data / Weights

For convenience I've provided an aria2c script that will download all the files needed from hugging face, which can be downloaded via the command:

```
aria2c --disable-ipv6 -i kyutai_pocket-tts-without-voice-cloning.txt
```

if you do not have aria2c, it can be [downloaded for windows](https://github.com/aria2/aria2/releases/tag/release-1.37.0) or installed via a package manager `sudo apt install aria2`.

You can also download the individual model files from the web:
* https://huggingface.co/kyutai/pocket-tts-without-voice-cloning

The demo will look for them in the subdirectory "kyutai/pocket-tts-without-voice-cloning", in the current working directory or if you set the environment variable `MODEL_CACHE`.

## Build

The same instructions as moshi.cpp:
 * https://github.com/Codes4Fun/moshi.cpp?tab=readme-ov-file#build-dependencies

At the end you will end up with `bin/pocket-tts` which allows you to pick a voice, for example `-v cosette`, and the text you want to be spoken, and it will output to your speakers or you can specify an output file `-v output.mp3` or a wav file, ogg, etc. It has several other options `pocket-tts -h`, you can also run a benchmark `pocket-tts --bench`.
