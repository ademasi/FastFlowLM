/// \file Gemma4e.hpp
/// \brief Gemma4e class
/// \author FastFlowLM Team
/// \date 2025-09-03
/// \version 0.9.24
/// \note This is a source file for the Gemma4e class

#pragma once
#include "AutoModel/automodel.hpp"
#include "metrices.hpp"


#include "typedef.hpp"
#include "image/image_reader.hpp"
#include "image_process_utils/imageproc.hpp"
#include "image_process_utils/imageprocAVX512.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "base64.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

/************              Qwen3VL_4b            **************/
class Gemma4e : public AutoModel {
private:

    bool enable_think = false;
    bool enable_tool = false;
    void setup_tokenizer(std::string model_path);
    
    // Image processing functionality
    ImageReader image_reader_;
    gemma4e_image_t load_image(const std::string& filename);
    gemma4e_image_t load_image_base64(const std::string& base64_string);
    



    int image_softtoken_budget = 280; // set a default value

    int debug_count= 0;

    
    void preprocess_image(
      gemma4e_image_t &image,
      std::pair<int, int> & patch_element_per_patch,
      uint32_t & valid_patch_size, // the unpadded size per image
      std::vector<bf16> &pixel_values,
      std::vector<int> &image_grid_pairs, // [num_of_position_id][x, y]      
      uint32_t &num_soft_tokens
    );


    std::vector<uint8_t>  aspect_ratio_preserving_resize( 
        gemma4e_image_t& image,
        int patch_size,
        int max_patches,
        int pooling_kernel_size
    );




public:
    Gemma4e(xrt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
    //void toggle_enable_think() override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input) override;
    std::string generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled = [] { return false; }) override;
    std::string generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os = std::cout) override;
    std::string apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools = nlohmann::ordered_json::object()) override;
    NonStreamResult parse_nstream_content(const std::string response_text);
    StreamResult parse_stream_content(const std::string content);

    /// \brief Configure a parameter with type-erased value
	/// \param parameter_name the name of the parameter
	/// \param value the value to set (can be any type)
	/// \return true if the parameter was configured successfully, false otherwise
	bool configure_parameter(std::string parameter_name, const std::any& value) override{
        if (parameter_name == "enable_think") {
            try {
                this->enable_think = std::any_cast<bool>(value);
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        else if (parameter_name == "toggle_think") {
            this->enable_think = !this->enable_think;
            return true;
        }
        else if (parameter_name == "system_prompt") {
            try {
                this->user_system_prompt = std::any_cast<std::string>(value);
                this->extra_context["user_system_prompt"] = this->user_system_prompt;
                return true;
            } catch (const std::bad_any_cast&) {
                return false;
            }
        }
        else if (parameter_name == "image_budget") {
            //TODO: use this to sepcific the vision budget #FIXME:
            
            

            this->image_softtoken_budget = std::any_cast<int>(value);
            // sanity checkt
            if(image_softtoken_budget != 70 || image_softtoken_budget != 140 ||
                image_softtoken_budget != 280 || image_softtoken_budget != 560 || image_softtoken_budget != 1120){
                    std::cerr << "Invalid image budget value: " << image_softtoken_budget << ". Supported values are 70, 140, 280, 560, 1120." << std::endl;
                    return false;
                }
            return true;

           
        }
		return false;
	}
};