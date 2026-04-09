/// \file deepseek.cpp
/// \brief deepseek class
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a source file for the deepseek class


#include "AutoModel/modeling_gemma4e.hpp"
#include "metrices.hpp"


/************              Gemma4e family            **************/
Gemma4e::Gemma4e(xrt::device* npu_device_inst) : AutoModel(npu_device_inst, "Gemma4e") {}

void Gemma4e::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    std::cout << "Loading model from: " << model_path << std::endl;
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
    
    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // lm_config->model_type == qwen3
    this->lm_engine = std::make_unique<gemma4e_npu>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);
    //free the q4nx
    this->q4nx.reset();
    //TODO: FIXME: reenable it
    //this->lm_engine->clear_context();
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
    constexpr int image_soft_token_id = 258880;
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    constexpr bool DEBUG_IMAGE_PREPROCESS = false;
    gemma4e_image_payload_t image_payload;
    image_payload.num_images = 0;
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
        
        // Add text object to content array
        nlohmann::ordered_json text_obj;
        text_obj["type"] = "text";
        text_obj["text"] = input.prompt;
        content["content"].push_back(text_obj);
        
        messages.push_back(content);
        templated_text = this->apply_chat_template(messages);
    }
    std::vector<int> tokens_init = this->tokenizer->encode(templated_text);

    // update the tokens to include the image tokens
    std::vector<int> tokens;
    int total_image_tokens = 0;
    for (int i = 0; i < input.images.size(); i++) {
        total_image_tokens += image_payload.num_soft_tokens_per_image[i];
    }
    
    tokens.reserve(tokens_init.size() + total_image_tokens);
    
    int image_counter = 0;
   
    for (int i = 0; i < tokens_init.size(); i++) {
        if (tokens_init[i] == image_soft_token_id) {
            tokens.push_back(255999); // the first image soft token id, which is reserved for the model to identify the image position, the rest of the soft tokens for this image will be continuous following this id
            for (int j = 0; j <  image_payload.num_soft_tokens_per_image[image_counter]; j++) {
                tokens.push_back(image_soft_token_id);
            }
            tokens.push_back(258882); // a separator token between images, not necessary but can help the model to better distinguish different images
            image_counter++;
        } else {
            tokens.push_back(tokens_init[i]);
        }
    }
      
    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // hardware
    if (image_payload.num_images > 0){
        return this->_shared_insert(meta_info, tokens, &image_payload);
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