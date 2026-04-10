/// \file deepseek.cpp
/// \brief deepseek class
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a source file for the deepseek class


#include "AutoModel/modeling_gemma4e.hpp"
#include "metrices.hpp"
#include "error_measure.hpp" //TODO: FIXME: remove later


/************              Gemma4e family            **************/
Gemma4e::Gemma4e(xrt::device* npu_device_inst) : AutoModel(npu_device_inst, "Gemma4e") {}

void Gemma4e::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    std::cout << "Loading model from: " << model_path << std::endl;
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
    std::cout << "Model loaded: " << this->model_path << std::endl;
    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // lm_config->model_type == qwen3
    this->lm_engine = std::make_unique<gemma4e_npu>(*this->lm_config, this->npu.get(), this->MAX_L);
    std::cout <<"lm_enginer created" << std::endl;
    this->lm_engine->load_weights(*this->q4nx);
    //free the q4nx
    this->q4nx.reset();
    //TODO: FIXME: reenable it
    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    this->enable_tool = (model_info["size"] > 800000000)? true : false;

    sampler_config config;
    config.top_k = 20;
    config.top_p = 0.8;
    config.min_p = 0.0;
    config.temperature = 0.7;
    config.rep_penalty = 1.0;
    config.freq_penalty = 1.0;
    config.pre_penalty = 1.5f;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void Gemma4e::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string Gemma4e::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty() && this->enable_tool)
        inputs.tools = tools;
    return this->chat_tmpl->apply(inputs);
}

bool Gemma4e::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input) {
    // preprocess
    constexpr int image_soft_token_id = 248056;
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    constexpr bool DEBUG_IMAGE_PREPROCESS = false;
    gemma4e_image_payload_t image_payload;
    gemma4e_audio_payload_t audio_payload;
    audio_payload.num_audios = 0;
    image_payload.num_images = 0;

    float max_support_audio_length_seconds = 30.0f;
    if (input.images.size() > 0) {


        // header_print("info", "Processing images...");
        
        // time_utils::time_point preprocess_start = time_utils::now();
        for(const auto& img_str : input.images){
            gemma4e_image_t image = this->load_image(img_str);



            std::vector<bf16> pixel_values;
            std::pair<int, int> patch_element_per_patch;
            uint32_t valid_patch_size = 0;
            uint32_t num_soft_tokens = 0;
            std::vector<int> image_grid_pairs; // [num_of_position_id][x, y]
            preprocess_image(image,
                patch_element_per_patch,
                valid_patch_size, 
                pixel_values,
                image_grid_pairs,
                num_soft_tokens);

            image_payload.image_patch__element_per_patch.push_back(patch_element_per_patch);
            image_payload.valid_patch_size_per_image.push_back(valid_patch_size);
            image_payload.pixel_values.push_back(pixel_values);
            image_payload.image_grid_pairs_per_image.push_back(image_grid_pairs);
            image_payload.num_soft_tokens_per_image.push_back(num_soft_tokens);
            image_payload.num_images++;
        } 
    }
    if(input.audios.size() > 0){
        


        //load a reference tensor
        SafeTensors reference_audio_preprocess_tensor("/home/shouyud/liquid-mega-kernel-npu/python_code/gemma4/audio_preprocess_debug.safetensors");

        std::vector<audio_data_t> audio_data_list;
        for(int i = 0; i < input.audios.size(); i++){
            std::string audio_str = input.audios[i];
            std::cout << "loading audio: " << audio_str << std::endl;
            gemma4e_npu *gemma4e_engine = dynamic_cast<gemma4e_npu*>(this->lm_engine.get());
            audio_data_t audio_data = this->load_audio(audio_str ,gemma4e_engine->Gemma4E_Audio_resample_rate, MonoDownmixMode::MEAN); 
            if(audio_data.channels > 1){
                std::cerr << "only mono audio is supported, but got " << audio_data.original_channels << " channels. Please convert it to mono first." << std::endl;
                exit(-1);
            }


        //     buffer<float> audio_tensor_Ref;
        //     reference_audio_preprocess_tensor.load_weights(audio_tensor_Ref, "raw_speech_"+std::to_string(i));
            
        //    std::cout << "Error analysis for audio preprocessing, comparing with reference tensor from safetensors..." << std::endl;
        //     print_error_metrics<float, float>(
        //         audio_data.samples.data(),
        //         audio_tensor_Ref.data(),
        //         1, 
        //         1, audio_data.num_samples,
        //         1, audio_data.num_samples

        //     );


            // apply clipping 
            //TODO make 30.0 second as a configurable parameter later
            audio_data = this->clip_audio_length(audio_data, max_support_audio_length_seconds); // clip to 30 seconds, which is the max audio length that Gemma4e can handle


 
            audio_data_list.push_back(audio_data);

        }

        // {   
        //     buffer<float>batched_speech_reference;
        //     reference_audio_preprocess_tensor.load_weights(batched_speech_reference, "batched_speech");
        //     std::cout << "Error analysis for batched audio preprocessing tensor, comparing with reference tensor from safetensors..." << std::endl;
        //     // concatenate the audio data into a batched
        //     int length_per_reference_audio = 480000; // for debug

        //     for(int i = 0; i < audio_data_list.size(); i++){

        //         print_error_metrics<float, float>(
        //             audio_data_list[i].samples.data(),
        //             batched_speech_reference.data() + i * length_per_reference_audio,
        //             1, 
        //             1, audio_data_list[i].num_samples,
        //             1, length_per_reference_audio

        //         );
        //         // printout the info of each audio data
        //         header_print("Audio Preprocess Debug", "Audio " + std::to_string(i) + ": original sample rate = " + std::to_string(audio_data_list[i].original_sample_rate
        //         ) + ", resampled sample rate = " + std::to_string(audio_data_list[i].sample_rate) + ", original channels = " + std::to_string(audio_data_list[i].original_channels) + ", resampled channels = " + std::to_string(audio_data_list[i].channels) + ", duration = " + std::to_string(audio_data_list[i].duration_seconds) + " seconds, num_samples = " + std::to_string(audio_data_list[i].num_samples));   

        //     }



        // }


       this->extract_spectrogram(audio_data_list, audio_payload);

        // Calculate the number of soft tokens per audio
        // Mirrors Python's _compute_audio_num_tokens:
        //   1. Mel framing (unfold) count
        //   2. Two Conv2d subsampling layers (kernel=3, stride=2, semicausal pad top=1, bottom=1)
        //   3. Cap at audio_seq_length
        {
            gemma4e_npu *gemma4e_engine = dynamic_cast<gemma4e_npu*>(this->lm_engine.get());
            const unsigned int conv2d_kernel = gemma4e_engine->Gemma4E_Audio_conv2d_kernel_size;
            const unsigned int conv2d_stride = gemma4e_engine->Gemma4E_Audio_conv2d_Stride;
            const unsigned int conv2d_padding = gemma4e_engine->Gemma4e_Audio_conv2d_Padding;
            const unsigned int max_audio_seq_length =max_support_audio_length_seconds * gemma4e_engine->Gemma4E_Audio_resample_rate;    // calculate from 30 second using gemma4e->Gemma4E_Audio_resample_rate



            
            constexpr float frame_length_ms = 20.0f;
            constexpr float hop_length_ms   = 10.0f;

            for(int i = 0; i < audio_payload.num_audios; i++){
                const int num_samples = static_cast<int>(audio_data_list[i].num_samples);
                const int sampling_rate = audio_data_list[i].sample_rate;

                // Step 1: Mel frames (matches feature_extraction_gemma4.py _unfold)
                const int frame_length = static_cast<int>(std::round(sampling_rate * frame_length_ms / 1000.0f));
                const int hop_length   = static_cast<int>(std::round(sampling_rate * hop_length_ms / 1000.0f));
                const int frame_size_for_unfold = frame_length + 1;

                const int pad_left = frame_length / 2;
                const int padded_samples = num_samples + pad_left;
                int num_mel_frames = (padded_samples - frame_size_for_unfold) / hop_length + 1;

                unsigned int num_tokens = 0;
                if (num_mel_frames > 0) {
                    // Step 2: Two SSCP conv layers
                    // Each layer: T_out = (T_in + pad_top + pad_bottom - kernel) // stride + 1
                    int t = num_mel_frames;
                    for (int layer = 0; layer < 2; layer++) {
                        int t_padded = t + 2 * static_cast<int>(conv2d_padding);
                        t = (t_padded - static_cast<int>(conv2d_kernel)) / static_cast<int>(conv2d_stride) + 1;
                    }
                    // Cap at audio_seq_length
                    assert(t< max_audio_seq_length);
                    num_tokens =t;
                }
                std::cout <<"max_audio_seq_length: " << max_audio_seq_length << ", calculated num_tokens for audio " << i << ": " << num_tokens << std::endl;
                audio_payload.num_soft_tokens_per_audio.push_back(num_tokens);
            }
        }

        // print out the actual info for each audio_payload.audio
        for(int i = 0; i < audio_payload.num_audios; i++){
            header_print("Audio Preprocess Debug", "Audio " + std::to_string(i) 
                + ": mel spectrogram frames = " + std::to_string(audio_payload.mel_spectrogram_frames_per_audio[i]) 
                + ", mel spectrogram bins = " + std::to_string(audio_payload.mel_spectrogram_bins_per_audio[i])
                + ", num_soft_tokens = " + std::to_string(audio_payload.num_soft_tokens_per_audio[i]));
        }   
        // {

        //     int length_per_reference_audio_spectrogram = 2999 * 128;
        //     buffer<float> prepared_speech_reference;
        //     reference_audio_preprocess_tensor.load_weights(prepared_speech_reference, "prepared_speech");
        //     std::cout << "Error analysis for prepared spectrogram tensor, comparing with reference tensor from safetensors..." << std::endl;
        //     for(int i = 0; i < audio_payload.num_audios; i++){
        //         print_error_metrics<bf16, float>(
        //             audio_payload.mel_spectrograms[i].data(),
        //             prepared_speech_reference.data() + i * length_per_reference_audio_spectrogram,
        //             1, 
        //             1, audio_payload.mel_spectrogram_frames_per_audio[i] * audio_payload.mel_spectrogram_bins_per_audio[i],
        //             1, audio_payload.mel_spectrogram_frames_per_audio[i] * audio_payload.mel_spectrogram_bins_per_audio[i]
        //         );

        //     }       


        // }    

    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        json qwenvl_message = json::array();
        for (const auto& item : input.messages) {
            if (!item.contains("images")) {
                qwenvl_message.push_back(item);
                continue;
            }

            json newContent = json::array();
            for (const auto& img : item["images"]) {
                newContent.push_back({
                    {"type", "image"},
                    {"image", img}
                });
            }
            //TODO: FIXME: add the audio part later
            newContent.push_back({
                {"type", "text"},
                {"text", item["content"]}
            });

            json newItem = {
                {"role", item["role"]},
                {"content", newContent}
            };

            qwenvl_message.push_back(newItem);
        }
        templated_text = this->apply_chat_template(qwenvl_message, input.tools);
        int total_images = 0;
        for (auto& message : qwenvl_message) {
            auto content = message.value("content", nlohmann::ordered_json::array());
            for (auto& item : content) {
                if (item.contains("type") && item["type"] == "image") {
                    std::string img_str = item.value("image", "");
                    if (!img_str.empty()) {
                        total_images++;
                    }
                    gemma4e_image_t image = this->load_image_base64(img_str);
                    std::vector<bf16> pixel_values;
                    std::pair<int, int> patch_element_per_patch;
                    uint32_t valid_patch_size = 0;
                    uint32_t num_soft_tokens = 0;
                    std::vector<int> image_grid_pairs; // [num_of_position_id][x, y]
                    preprocess_image(image,
                        patch_element_per_patch,
                        valid_patch_size, 
                        pixel_values,
                        image_grid_pairs,
                        num_soft_tokens);

                    image_payload.image_patch__element_per_patch.push_back(patch_element_per_patch);
                    image_payload.valid_patch_size_per_image.push_back(valid_patch_size);
                    image_payload.pixel_values.push_back(pixel_values);
                    image_payload.image_grid_pairs_per_image.push_back(image_grid_pairs);
                    image_payload.num_soft_tokens_per_image.push_back(num_soft_tokens);
                    image_payload.num_images++;
                }
                //TODO: fixme, add the audio stuff
            }
        }
        header_print("FLM", "Total images: " << total_images);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;
        nlohmann::ordered_json content;
        content["role"] = "user";
        content["content"] = nlohmann::ordered_json::array();
        
        // Add image objects to content array
        for (int i = 0; i < input.images.size(); i++) {
            nlohmann::ordered_json image_obj;
            image_obj["type"] = "image";
            image_obj["image"] = input.images[i];
            content["content"].push_back(image_obj);
        }
        //TODO: fixme , add the audio
        
        // Add text object to content array
        nlohmann::ordered_json text_obj;
        text_obj["type"] = "text";
        text_obj["text"] = input.prompt;
        content["content"].push_back(text_obj);
        
        messages.push_back(content);
        templated_text = this->apply_chat_template(messages);
    }
    std::vector<int> tokens_init = this->tokenizer->encode(templated_text);


    //TODO: FIXME: also add the audi token to it

    // update the tokens to include the image tokens
    std::vector<int> tokens;
    int total_image_tokens = 0;
    for (int i = 0; i < input.images.size(); i++) {
        total_image_tokens += image_payload.num_soft_tokens_per_image[i];
    }
    std::cout << "DEBUG: total image tokens is " << total_image_tokens << std::endl;

    tokens.reserve(tokens_init.size() + total_image_tokens);
    int image_counter = 0;
    for (int i = 0; i < tokens_init.size(); i++) {
        if (tokens_init[i] == image_soft_token_id) {
            for (int j = 0; j <  image_payload.num_soft_tokens_per_image[image_counter]; j++) {
                tokens.push_back(image_soft_token_id);
            }
            image_counter++;
        } else {
            //TODO: FIXME: not sure if it can detect token token fr  now
            tokens.push_back(tokens_init[i]);
        }
    }
    std::cout << "DEBUG: current token is size is  " << tokens.size() << "MAX_L is " << this->MAX_L << std::endl;   
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());
    for (const auto& token : tokens) {
        std::cout << token << ", ";
    }
    std::cout << std::endl;
    // hardware

    
    gemma4e_multi_modal_payload_t multi_modal_payload;
    multi_modal_payload.image_payload = image_payload;
    multi_modal_payload.audio_payload = audio_payload;

    if (image_payload.num_images > 0 || audio_payload.num_audios > 0) {
        return this->_shared_insert(meta_info, tokens, &multi_modal_payload);
    }else{
        return this->_shared_insert(meta_info, tokens, nullptr);
    }

}

std::string Gemma4e::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    if (this->enable_think) {
        os << "<think>\n" << std::flush;
    }
    return this->_shared_generate(meta_info, length_limit, os, is_cancelled);
}

std::string Gemma4e::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    if (this->enable_think) {
        os << "<think>\n" << std::flush;
    }
    return this->_shared_generate(meta_info, length_limit, os);
}

// Non-stream
NonStreamResult Gemma4e::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;

    std::string name, arguments;

    std::string start_tag = "<tool_call>";
    std::string end_tag = "</tool_call>";

    size_t start_pos = response_text.find(start_tag);
    size_t end_pos = response_text.find(end_tag);

    if (start_pos == std::string::npos || end_pos == std::string::npos) {
        // pure content
        result.content = response_text;
        return result;
    }

    start_pos += start_tag.length();
    std::string json_str = response_text.substr(start_pos, end_pos - start_pos);

    // Parse "name" 
    std::string key_name = "\"name\": \"";
    size_t name_start = json_str.find(key_name);
    if (name_start != std::string::npos) {
        name_start += key_name.length();
        size_t name_end = json_str.find("\"", name_start);
        if (name_end != std::string::npos) {
            name = json_str.substr(name_start, name_end - name_start);
        }
    }

    // Parse "arguments"
    std::string key_args = "\"arguments\":";
    size_t args_pos = json_str.find(key_args);
    if (args_pos != std::string::npos) {
        size_t brace_start = json_str.find("{", args_pos);
        size_t brace_end = json_str.rfind("}"); // Find the last closing brace

        if (brace_start != std::string::npos && brace_end != std::string::npos && brace_end > brace_start) {
            arguments = json_str.substr(brace_start, brace_end - brace_start);
        }
    }

    result.tool_name = name;
    result.tool_args = arguments;

    return result;
}

// Stream
StreamResult Gemma4e::parse_stream_content(const std::string content) {
    std::string tool_start_tag = "<tool_call>";
    std::string tool_end_tag = "</tool_call>";
    std::string think_start_tag = "<think>";
    std::string think_end_tag = "</think>";

    StreamResult result;
    result.type = StreamEventType::CONTENT;

    if (!is_in_tool_block_ && content.find(think_start_tag) != std::string::npos) {
        current_mode_ = StreamEventType::REASONING;
        result.type = StreamEventType::WAITING;
        return result;
    }

    if (!is_in_tool_block_ && content.find(think_end_tag) != std::string::npos) {
        current_mode_ = StreamEventType::CONTENT;
        result.type = StreamEventType::WAITING;
        return result;
    }

    if (!is_in_tool_block_ && current_mode_ == StreamEventType::REASONING) {
        result.type = StreamEventType::REASONING;
        result.content = content;
        return result;
    }

    if (content.find(tool_start_tag) != std::string::npos) {
        is_in_tool_block_ = true;
        tool_name_.clear();
        result.type = StreamEventType::WAITING;
        return result;
    }

    if (content.find(tool_end_tag) != std::string::npos) {
        is_in_tool_block_ = false;

        try {
            const std::string& block = tool_name_;

            // Parse function name from <function=NAME>
            std::string func_open = "<function=";
            size_t func_start = block.find(func_open);
            if (func_start != std::string::npos) {
                func_start += func_open.length();
                size_t func_end = block.find(">", func_start);
                if (func_end != std::string::npos) {
                    result.tool_name = block.substr(func_start, func_end - func_start);
                }
            }

            // Parse parameters from <parameter=NAME>\nVALUE\n</parameter>
            nlohmann::json args = nlohmann::json::object();
            std::string param_open = "<parameter=";
            std::string param_close = "</parameter>";
            size_t search_pos = 0;
            while (true) {
                size_t p_start = block.find(param_open, search_pos);
                if (p_start == std::string::npos) break;
                p_start += param_open.length();
                size_t p_name_end = block.find(">", p_start);
                if (p_name_end == std::string::npos) break;
                std::string param_name = block.substr(p_start, p_name_end - p_start);

                size_t val_start = p_name_end + 1;
                if (val_start < block.size() && block[val_start] == '\n') val_start++;

                size_t val_end = block.find(param_close, val_start);
                if (val_end == std::string::npos) break;

                std::string param_value = block.substr(val_start, val_end - val_start);
                if (!param_value.empty() && param_value.back() == '\n') param_value.pop_back();

                args[param_name] = param_value;
                search_pos = val_end + param_close.length();
            }

            result.type = StreamEventType::TOOL_DONE;
            result.tool_id = "call_" + std::to_string(std::time(nullptr));
            result.tool_args_str = args.dump();
        }
        catch (...) {
            result.type = StreamEventType::CONTENT;
            result.content = "[Error parsing tool call]";
        }
        return result;
    }

    if (is_in_tool_block_) {
        tool_name_ += content;
        result.type = StreamEventType::WAITING;
        return result;
    }

    result.content = content;
    return result;

}