/// \file language_table.hpp
/// \brief language_table class
/// \author FastFlowLM Team
/// \date 2025-10-17
/// \version 0.9.24
/// \note This is a header file for the language_table class
#pragma once
#include <string>
#include <unordered_map>

namespace langmap {

static const std::unordered_map<std::string, std::string> TOKEN_TO_LANGUAGE = {
    {"<|en|>", "English"},
    {"<|zh|>", "Chinese"},
    {"<|de|>", "German"},
    {"<|es|>", "Spanish"},
    {"<|ru|>", "Russian"},
    {"<|ko|>", "Korean"},
    {"<|fr|>", "French"},
    {"<|ja|>", "Japanese"},
    {"<|pt|>", "Portuguese"},
    {"<|tr|>", "Turkish"},
    {"<|pl|>", "Polish"},
    {"<|ca|>", "Catalan"},
    {"<|nl|>", "Dutch"},
    {"<|ar|>", "Arabic"},
    {"<|sv|>", "Swedish"},
    {"<|it|>", "Italian"},
    {"<|id|>", "Indonesian"},
    {"<|hi|>", "Hindi"},
    {"<|fi|>", "Finnish"},
    {"<|vi|>", "Vietnamese"},
    {"<|he|>", "Hebrew"},
    {"<|uk|>", "Ukrainian"},
    {"<|el|>", "Greek"},
    {"<|ms|>", "Malay"},
    {"<|cs|>", "Czech"},
    {"<|ro|>", "Romanian"},
    {"<|da|>", "Danish"},
    {"<|hu|>", "Hungarian"},
    {"<|ta|>", "Tamil"},
    {"<|no|>", "Norwegian"},
    {"<|th|>", "Thai"},
    {"<|ur|>", "Urdu"},
    {"<|hr|>", "Croatian"},
    {"<|bg|>", "Bulgarian"},
    {"<|lt|>", "Lithuanian"},
    {"<|la|>", "Latin"},
    {"<|mi|>", "Maori"},
    {"<|ml|>", "Malayalam"},
    {"<|cy|>", "Welsh"},
    {"<|sk|>", "Slovak"},
    {"<|te|>", "Telugu"},
    {"<|fa|>", "Persian"},
    {"<|lv|>", "Latvian"},
    {"<|bn|>", "Bengali"},
    {"<|sr|>", "Serbian"},
    {"<|az|>", "Azerbaijani"},
    {"<|sl|>", "Slovenian"},
    {"<|kn|>", "Kannada"},
    {"<|et|>", "Estonian"},
    {"<|mk|>", "Macedonian"},
    {"<|br|>", "Breton"},
    {"<|eu|>", "Basque"},
    {"<|is|>", "Icelandic"},
    {"<|hy|>", "Armenian"},
    {"<|ne|>", "Nepali"},
    {"<|mn|>", "Mongolian"},
    {"<|bs|>", "Bosnian"},
    {"<|kk|>", "Kazakh"},
    {"<|sq|>", "Albanian"},
    {"<|sw|>", "Swahili"},
    {"<|gl|>", "Galician"},
    {"<|mr|>", "Marathi"},
    {"<|pa|>", "Punjabi"},
    {"<|si|>", "Sinhala"},
    {"<|km|>", "Khmer"},
    {"<|sn|>", "Shona"},
    {"<|yo|>", "Yoruba"},
    {"<|so|>", "Somali"},
    {"<|af|>", "Afrikaans"},
    {"<|oc|>", "Occitan"},
    {"<|ka|>", "Georgian"},
    {"<|be|>", "Belarusian"},
    {"<|tg|>", "Tajik"},
    {"<|sd|>", "Sindhi"},
    {"<|gu|>", "Gujarati"},
    {"<|am|>", "Amharic"},
    {"<|yi|>", "Yiddish"},
    {"<|lo|>", "Lao"},
    {"<|uz|>", "Uzbek"},
    {"<|fo|>", "Faroese"},
    {"<|ht|>", "Haitian Creole"},
    {"<|ps|>", "Pashto"},
    {"<|tk|>", "Turkmen"},
    {"<|nn|>", "Nynorsk"},
    {"<|mt|>", "Maltese"},
    {"<|sa|>", "Sanskrit"},
    {"<|lb|>", "Luxembourgish"},
    {"<|my|>", "Burmese"},
    {"<|bo|>", "Tibetan"},
    {"<|tl|>", "Tagalog"},
    {"<|mg|>", "Malagasy"},
    {"<|as|>", "Assamese"},
    {"<|tt|>", "Tatar"},
    {"<|haw|>", "Hawaiian"},
    {"<|ln|>", "Lingala"},
    {"<|ha|>", "Hausa"},
    {"<|ba|>", "Bashkir"},
    {"<|jw|>", "Javanese"},
    {"<|su|>", "Sundanese"},
    {"<|yue|>", "Cantonese"}
};

inline std::string to_language_name(const std::string& token) {
    auto it = TOKEN_TO_LANGUAGE.find(token);
    if (it != TOKEN_TO_LANGUAGE.end())
        return it->second;
    return "Unknown";
}

/// Map ISO-639-1 language codes to Whisper token strings
static const std::unordered_map<std::string, std::string> CODE_TO_TOKEN = {
    {"en", "<|en|>"}, {"zh", "<|zh|>"}, {"de", "<|de|>"}, {"es", "<|es|>"},
    {"ru", "<|ru|>"}, {"ko", "<|ko|>"}, {"fr", "<|fr|>"}, {"ja", "<|ja|>"},
    {"pt", "<|pt|>"}, {"tr", "<|tr|>"}, {"pl", "<|pl|>"}, {"ca", "<|ca|>"},
    {"nl", "<|nl|>"}, {"ar", "<|ar|>"}, {"sv", "<|sv|>"}, {"it", "<|it|>"},
    {"id", "<|id|>"}, {"hi", "<|hi|>"}, {"fi", "<|fi|>"}, {"vi", "<|vi|>"},
    {"he", "<|he|>"}, {"uk", "<|uk|>"}, {"el", "<|el|>"}, {"ms", "<|ms|>"},
    {"cs", "<|cs|>"}, {"ro", "<|ro|>"}, {"da", "<|da|>"}, {"hu", "<|hu|>"},
    {"ta", "<|ta|>"}, {"no", "<|no|>"}, {"th", "<|th|>"}, {"ur", "<|ur|>"},
    {"hr", "<|hr|>"}, {"bg", "<|bg|>"}, {"lt", "<|lt|>"}, {"la", "<|la|>"},
    {"mi", "<|mi|>"}, {"ml", "<|ml|>"}, {"cy", "<|cy|>"}, {"sk", "<|sk|>"},
    {"te", "<|te|>"}, {"fa", "<|fa|>"}, {"lv", "<|lv|>"}, {"bn", "<|bn|>"},
    {"sr", "<|sr|>"}, {"az", "<|az|>"}, {"sl", "<|sl|>"}, {"kn", "<|kn|>"},
    {"et", "<|et|>"}, {"mk", "<|mk|>"}, {"br", "<|br|>"}, {"eu", "<|eu|>"},
    {"is", "<|is|>"}, {"hy", "<|hy|>"}, {"ne", "<|ne|>"}, {"mn", "<|mn|>"},
    {"bs", "<|bs|>"}, {"kk", "<|kk|>"}, {"sq", "<|sq|>"}, {"sw", "<|sw|>"},
    {"gl", "<|gl|>"}, {"mr", "<|mr|>"}, {"pa", "<|pa|>"}, {"si", "<|si|>"},
    {"km", "<|km|>"}, {"sn", "<|sn|>"}, {"yo", "<|yo|>"}, {"so", "<|so|>"},
    {"af", "<|af|>"}, {"oc", "<|oc|>"}, {"ka", "<|ka|>"}, {"be", "<|be|>"},
    {"tg", "<|tg|>"}, {"sd", "<|sd|>"}, {"gu", "<|gu|>"}, {"am", "<|am|>"},
    {"yi", "<|yi|>"}, {"lo", "<|lo|>"}, {"uz", "<|uz|>"}, {"fo", "<|fo|>"},
    {"ht", "<|ht|>"}, {"ps", "<|ps|>"}, {"tk", "<|tk|>"}, {"nn", "<|nn|>"},
    {"mt", "<|mt|>"}, {"sa", "<|sa|>"}, {"lb", "<|lb|>"}, {"my", "<|my|>"},
    {"bo", "<|bo|>"}, {"tl", "<|tl|>"}, {"mg", "<|mg|>"}, {"as", "<|as|>"},
    {"tt", "<|tt|>"}, {"haw", "<|haw|>"}, {"ln", "<|ln|>"}, {"ha", "<|ha|>"},
    {"ba", "<|ba|>"}, {"jw", "<|jw|>"}, {"su", "<|su|>"}, {"yue", "<|yue|>"}
};

inline std::string language_code_to_token(const std::string& code) {
    auto it = CODE_TO_TOKEN.find(code);
    if (it != CODE_TO_TOKEN.end())
        return it->second;
    return "";
}

}  // namespace langmap
