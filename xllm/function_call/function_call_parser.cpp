/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "function_call/function_call_parser.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <unordered_map>

#include "absl/strings/str_join.h"
#include "core/util/uuid.h"
#include "function_call/deepseekv32_detector.h"
#include "function_call/deepseekv3_detector.h"
#include "function_call/glm45_detector.h"
#include "function_call/glm47_detector.h"
#include "function_call/kimik2_detector.h"
#include "function_call/qwen25_detector.h"
#include "function_call/qwen3_coder_detector.h"

namespace xllm {
namespace function_call {

namespace {

const std::unordered_map<std::string, std::vector<std::string>> auto_paser_map =
    {
        {"qwen25", {"qwen2", "qwen3"}},
        {"qwen3_coder", {"qwen3_coder", "qwen35"}},
        {"kimi_k2", {"kimi_k2", "kimi_k25"}},
        {"deepseekv3", {"deepseek_v3"}},
        {"deepseekv32", {"deepseek_v32"}},
        {"deepseekv4", {"deepseek_v4", "deepseek_v4_mtp"}},
        {"glm5", {"glm_moe_dsa", "glm_moe_dsa_mtp"}},
        // GLM-4.5 and GLM-4.7 are not supported for tool call parser
        // auto-selection
        // {"glm45", {"glm4_moe"}},
        // {"glm47", {"glm4_moe"}},

};

std::string get_auto_paser_map_supported() {
  std::vector<std::string> keys;
  for (const auto& [key, value] : auto_paser_map) {
    for (const auto& v : value) {
      keys.push_back(v);
    }
  }
  return absl::StrJoin(keys, ", ");
}

const std::unordered_map<std::string,
                         std::function<std::unique_ptr<BaseFormatDetector>()>>
    detector_factories = {
        {"qwen25", [] { return std::make_unique<Qwen25Detector>(); }},
        {"qwen3_coder", [] { return std::make_unique<Qwen3CoderDetector>(); }},
        {"kimi_k2", [] { return std::make_unique<KimiK2Detector>(); }},
        {"deepseekv3", [] { return std::make_unique<DeepSeekV3Detector>(); }},
        {"deepseekv32", [] { return std::make_unique<DeepSeekV32Detector>(); }},
        {"deepseekv4", [] { return std::make_unique<DeepSeekV4Detector>(); }},
        {"glm45", [] { return std::make_unique<Glm45Detector>(); }},
        {"glm47", [] { return std::make_unique<Glm47Detector>(); }},
        // glm5 use glm47 detector
        {"glm5", [] { return std::make_unique<Glm47Detector>(); }},
};

std::string get_supported_detector_factories() {
  std::vector<std::string> keys;
  for (const auto& [key, value] : detector_factories) {
    keys.push_back(key);
  }
  return absl::StrJoin(keys, ", ");
}

}  // namespace

std::pair<Status, std::string> FunctionCallParser::resolve_parser(
    const std::string& parser,
    const std::string& model_type,
    bool strict_errors) {
  if (parser.empty()) {
    return {Status(), ""};
  }
  if (parser == "auto") {
    for (const auto& [key, value] : auto_paser_map) {
      if (std::find(value.begin(), value.end(), model_type) != value.end()) {
        return resolve_parser(key, model_type, strict_errors);
      }
    }
    return {Status(StatusCode::INVALID_ARGUMENT,
                   "Unsupported model type for auto tool call parser: " +
                       model_type + ". Supported model types are: " +
                       get_auto_paser_map_supported()),
            ""};
  }
  const std::string normalized = parser == "qwen2" || parser == "qwen3"
                                     ? "qwen25"
                                 : parser == "qwen35" ? "qwen3_coder"
                                                      : parser;
  if (detector_factories.contains(normalized)) {
    if (strict_errors && normalized != "qwen25" &&
        normalized != "qwen3_coder" && normalized != "glm45" &&
        normalized != "glm47" && normalized != "glm5") {
      return {Status(StatusCode::INVALID_ARGUMENT,
                     "Tool call parser does not support strict finalization: " +
                         normalized),
              ""};
    }
    return {Status(), normalized};
  }
  return {Status(StatusCode::INVALID_ARGUMENT,
                 "Unsupported tool call parser: " + parser +
                     ". Supported parsers are: " +
                     get_supported_detector_factories()),
          ""};
}

std::string FunctionCallParser::get_parser_auto(const std::string& parser,
                                                const std::string& model_type) {
  auto [status, resolved] = resolve_parser(parser, model_type);
  CHECK(status.ok()) << status.message();
  if (parser == "auto") {
    LOG(INFO) << "Using tool call parser: " << resolved
              << " for model type: " << model_type;
  }
  return resolved;
}

FunctionCallParser::FunctionCallParser(const std::vector<JsonTool>& tools,
                                       const std::string& tool_call_parser,
                                       bool strict_errors)
    : tools_(tools),
      parser_format_(tool_call_parser),
      strict_errors_(strict_errors) {
  detector_ = create_detector(tool_call_parser);
  CHECK(detector_ != nullptr)
      << "Unsupported tool_call_parser: " << tool_call_parser;
  detector_->set_strict_errors(strict_errors_);
}

bool FunctionCallParser::has_tool_call(const std::string& text) const {
  return detector_->has_tool_call(text);
}

std::tuple<std::string, std::vector<ToolCallItem>>
FunctionCallParser::parse_non_stream(const std::string& full_text) {
  StreamingParseResult parsed_result =
      detector_->detect_and_parse(full_text, tools_);

  if (!parsed_result.calls.empty()) {
    return std::make_tuple(parsed_result.normal_text, parsed_result.calls);
  } else {
    return std::make_tuple(full_text, std::vector<ToolCallItem>());
  }
}

StreamingParseResult FunctionCallParser::parse_streaming_increment(
    const std::string& new_text) {
  if (strict_errors_) {
    stream_text_ += new_text;
  }
  auto result = detector_->parse_streaming_increment(new_text, tools_);
  if (strict_errors_) {
    record_stream_result(result);
  }
  return result;
}

void FunctionCallParser::record_stream_result(
    const StreamingParseResult& result) {
  emitted_text_ += result.normal_text;
  for (const auto& call : result.calls) {
    if (call.tool_index < 0) {
      continue;
    }
    const size_t index = static_cast<size_t>(call.tool_index);
    if (emitted_calls_.size() <= index) {
      emitted_calls_.resize(index + 1);
    }
    auto& emitted = emitted_calls_[index];
    emitted.tool_index = call.tool_index;
    if (call.name.has_value()) {
      emitted.name = call.name;
    }
    emitted.parameters += call.parameters;
  }
}

std::pair<Status, StreamingParseResult> FunctionCallParser::finish_stream(
    bool incomplete) {
  StreamingParseResult tail;
  // Detectors may yield a name before arguments, or retain another complete
  // call. Drain only progress already present in their buffers.
  const size_t drain_limit = stream_text_.size() + 1;
  size_t drained = 0;
  for (; drained < drain_limit; ++drained) {
    auto result = parse_streaming_increment("");
    if (result.normal_text.empty() && result.calls.empty()) {
      break;
    }
    tail.normal_text += result.normal_text;
    tail.calls.insert(tail.calls.end(),
                      std::make_move_iterator(result.calls.begin()),
                      std::make_move_iterator(result.calls.end()));
  }
  if (drained == drain_limit) {
    return {Status(StatusCode::UNKNOWN,
                   "Function parser did not make bounded EOF progress."),
            {}};
  }
  auto remaining = detector_->finish_stream();
  record_stream_result(remaining);
  tail.normal_text += remaining.normal_text;
  tail.calls.insert(tail.calls.end(),
                    std::make_move_iterator(remaining.calls.begin()),
                    std::make_move_iterator(remaining.calls.end()));
  if (!detector_->error_status().ok()) {
    return {detector_->error_status(), {}};
  }
  if (!strict_errors_ || incomplete) {
    return {Status(), std::move(tail)};
  }

  auto verifier = create_detector(parser_format_);
  verifier->set_strict_errors(true);
  const auto expected = verifier->detect_and_parse(stream_text_, tools_);
  if (!verifier->error_status().ok()) {
    return {verifier->error_status(), {}};
  }
  if ((verifier->has_tool_call(stream_text_) && expected.calls.empty()) ||
      expected.calls.size() != emitted_calls_.size()) {
    return {
        Status(StatusCode::UNKNOWN,
               "Model generated an incomplete or unparsable function call."),
        {}};
  }
  if (expected.normal_text != emitted_text_) {
    return {Status(StatusCode::UNKNOWN,
                   "Function parser changed its emitted ordinary text."),
            {}};
  }
  for (size_t index = 0; index < expected.calls.size(); ++index) {
    const auto& expected_call = expected.calls[index];
    const auto& emitted = emitted_calls_[index];
    if (expected_call.name != emitted.name) {
      return {Status(StatusCode::UNKNOWN,
                     "Function parser changed the emitted function identity."),
              {}};
    }
    const auto arguments = nlohmann::json::parse(
        emitted.parameters, nullptr, /*allow_exceptions=*/false);
    const auto expected_arguments = nlohmann::json::parse(
        expected_call.parameters, nullptr, /*allow_exceptions=*/false);
    if (!expected_arguments.is_object()) {
      return {Status(StatusCode::UNKNOWN,
                     "Function parser returned non-object final arguments."),
              {}};
    }
    if (!arguments.is_object() || arguments != expected_arguments) {
      return {Status(StatusCode::UNKNOWN,
                     "Function arguments do not match their streamed output."),
              {}};
    }
  }
  return {Status(), std::move(tail)};
}

std::unique_ptr<BaseFormatDetector> FunctionCallParser::create_detector(
    const std::string& tool_call_parser) {
  if (tool_call_parser.empty()) {
    return nullptr;
  }

  auto it = detector_factories.find(tool_call_parser);
  if (it != detector_factories.end()) {
    return it->second();
  }
  LOG(ERROR) << "Unsupported tool call parser: " << tool_call_parser;

  return nullptr;
}

namespace utils {

std::vector<ToolCallItem> parse_function_calls(
    const std::string& text,
    const std::vector<JsonTool>& tools,
    const std::string& parser_type) {
  try {
    FunctionCallParser parser(tools, parser_type);
    auto [normal_text, calls] = parser.parse_non_stream(text);
    return calls;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error parsing function calls: " << e.what();
    return {};
  }
}

bool has_function_calls(const std::string& text,
                        const std::string& parser_type) {
  try {
    FunctionCallParser parser({}, parser_type);
    return parser.has_tool_call(text);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error checking function calls: " << e.what();
    return false;
  }
}

StreamingParseResult parse_streaming_increment(
    const std::string& new_text,
    const std::vector<JsonTool>& tools,
    const std::string& parser_type) {
  try {
    FunctionCallParser parser(tools, parser_type);
    return parser.parse_streaming_increment(new_text);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Error in streaming parsing: " << e.what();
    return StreamingParseResult();
  }
}

thread_local ShortUUID short_uuid;

std::string generate_tool_call_id() { return "call_" + short_uuid.random(); }

}  // namespace utils

}  // namespace function_call
}  // namespace xllm
