#pragma once

// pocket_tts.conditioners.text.LUTConditioner

struct lut_conditioner_t {
    sentencepiece::SentencePieceProcessor tokenizer;
    ggml_tensor * embed;
};

lut_conditioner_t * new_lut_conditioner( std::string filename ) {
    auto conditioner = new lut_conditioner_t;
    conditioner->tokenizer.Load( filename.c_str() );
    return conditioner;
}

void get_weights( WeightLoader * loader, std::string path, lut_conditioner_t * conditioner ) {
    int n = loader->fetch( &conditioner->embed, path + "embed.weight", (void*)ggml_get_rows );
    assert( n );
}

void conditioner_prepare(
    lut_conditioner_t * conditioner,
    std::string x,
    std::vector<int> & tokens
) {
    conditioner->tokenizer.Encode( x, &tokens );
}

ggml_tensor * conditioner_forward(
    GraphContext & ctx,
    lut_conditioner_t * conditioner,
    ggml_tensor * inputs
) {
    // _get_condition
    auto embeds = ggml_get_rows( ctx, conditioner->embed, inputs );
    return embeds;
}

std::string strip( std::string text ) {
    int start = 0;
    while ( start < text.size() && isspace( text[start] ) ) {
        start++;
    }
    if ( start == text.size() )
        return "";
    int end = (int)text.size() - 1;
    while ( end >= 0 && isspace( text[end] ) ) {
        end--;
    }
    return text.substr( start, end - start + 1 );
}

std::string merge_whitespaces( std::string text ) {
    std::string out;
    bool was_space = true;
    for ( int i = 0; i < text.size(); i++ ) {
        char c = text[i];
        bool is_space = isspace( c );
        if ( ! is_space ) {
            out += c;
        } else if ( ! was_space ) {
            out += ' ';
        }
        was_space = is_space;
    }
    return out;
}

int skip_whitespaces( std::string & text, int offset ) {
    while ( offset < text.size() && isspace( text[offset] ) )
        offset++;
    return offset;
}

int find_whitespaces( std::string & text, int offset ) {
    while ( offset < text.size() && ! isspace( text[offset] ) )
        offset++;
    return offset;
}

int count_words( std::string & text ) {
    int offset = 0;
    int words = 0;
    offset = skip_whitespaces( text, offset );
    while ( offset < text.size() ) {
        words++;
        offset = find_whitespaces( text, offset );
        if ( offset == text.size() )
            return words;
        offset = skip_whitespaces( text, offset );
    }

    return words;
}

char ensure_upper( char c ) {
    if ( ! islower( c ) )
        return c;
    return toupper( c );
}

std::tuple<std::string, int> prepare_text_prompt( std::string text ) {
    text = strip( text );
    if ( ! text.size() )
        throw new std::runtime_error( "Text prompt cannot be empty" );
    text = merge_whitespaces( text );
    int number_of_words = count_words( text );
    int frames_after_eos_guess = number_of_words <= 4 ? 3 : 1;

    // Make sure it starts with an uppercase letter
    text[0] = ensure_upper( text[0] );

    // Let's make sure it ends with some kind of punctuation
    // If it ends with a letter or digit, we add a period.
    if ( isalnum(text[text.size() - 1]) )
        text += '.';

    // The model does not perform well when there are very few tokens, so
    // we can add empty spaces at the beginning to increase the token count.
    if ( number_of_words < 5 )
        text = "        " + text;

    return std::make_tuple( text, frames_after_eos_guess );
}

std::vector<std::string> split_into_best_sentences(
    sentencepiece::SentencePieceProcessor & tokenizer,
    std::string text_to_generate
) {
    auto [scratch_transformer_out, _] = prepare_text_prompt( text_to_generate );

    std::vector<int> list_of_tokens;
    tokenizer.Encode( text_to_generate, &list_of_tokens );

    std::vector<int> end_of_sentence_tokens;
    tokenizer.Encode( ".!...?", &end_of_sentence_tokens );
    /* returns end_of_sentence_tokens
        260 ▁
        263 .
        682 !
        799 ...
        292 ?
    */

    std::vector<std::vector<int>> sentence_tokens = {{}};
    auto begin = ++end_of_sentence_tokens.begin(); // skip first token
    auto end = end_of_sentence_tokens.end();
    for ( auto & token : list_of_tokens ) {
        sentence_tokens.back().push_back( token );
        if ( std::find( begin, end, token ) != end ) {
            sentence_tokens.push_back({});
        }
    }
    if ( ! sentence_tokens.back().size() )
        sentence_tokens.pop_back();

    const int max_nb_tokens_in_a_chunk = 50;
    std::vector<std::string> chunks = {""};
    int current_nb_of_tokens_in_chunk = 0;
    for ( auto & tokens : sentence_tokens ) {
        if ( current_nb_of_tokens_in_chunk != 0 ) {
            if ( current_nb_of_tokens_in_chunk + tokens.size() > max_nb_tokens_in_a_chunk ) {
                // new chunk
                current_nb_of_tokens_in_chunk = 0;
                chunks.push_back("");
            } else {
                // appending to current chunk
                chunks.back() += " ";
            }
        }
        std::string sentence;
        tokenizer.Decode( tokens, &sentence );
        chunks.back() += sentence;
        current_nb_of_tokens_in_chunk += (int)tokens.size();
    }

    return chunks;
}


bool is_eos_char( char c ) {
    switch (c) {
    case '.':
    case '!':
    case '?':
        return true;
    }
    return false;
}

struct str_processor_t {
    std::string tail;
    std::deque<std::string> sentences;
    bool was_whitespace;
    bool was_eos;
    bool leading_char;
};

void str_processor_reset( str_processor_t & processor ) {
    processor.tail = "";
    processor.sentences.clear();
    processor.was_whitespace = true; // skip leading white spaces
    processor.was_eos = false;
    processor.leading_char = true;
}

void str_processor_ingest( str_processor_t & processor, std::string chunk ) {
    if ( ! chunk.size() )
        return;
    bool was_whitespace = processor.was_whitespace;
    bool was_eos = processor.was_eos;
    bool leading_char = processor.leading_char;
    for ( int i = 0; i < chunk.size(); i++ ) {
        char c = chunk[i];
        bool is_eos = is_eos_char( c );
        if ( ! is_eos && was_eos ) {
            processor.sentences.push_back( processor.tail );
            processor.tail = "";
            was_whitespace = true; // skip leading white spaces
            leading_char = true;
        }
        bool is_whitespace = isspace( c );
        if ( is_whitespace && ! was_whitespace ) {
            processor.tail += ' '; // merge/replace with single space
        } else if ( ! is_whitespace ) {
            if ( leading_char ) {
                c = ensure_upper( c );
                leading_char = false;
            }
            processor.tail += c;
        }
        was_whitespace = is_whitespace;
        was_eos = is_eos;
    }
    processor.was_whitespace = was_whitespace;
    processor.was_eos = was_eos;
    processor.leading_char = leading_char;
}

void str_processor_flush( str_processor_t & processor ) {
    int tail_size = (int)processor.tail.size();
    if ( tail_size ) {
        if ( isalnum( processor.tail[tail_size - 1] ) )
            processor.tail += '.';
        processor.sentences.push_back( processor.tail );
        processor.tail = "";
    }
    processor.was_whitespace = true; // skip leading white spaces
    processor.was_eos = false;
    processor.leading_char = true;
}
