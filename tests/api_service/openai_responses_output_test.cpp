/* Copyright 2026 The xLLM Authors.

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

#include "api_service/openai_responses_output.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/request/finish_reason.h"
#include "function_call/function_call_parser.h"

namespace xllm::api_service {
namespace {

nlohmann::json initial_response() {
  return {{"id", "resp_test"},
          {"object", "response"},
          {"model", "test"},
          {"created_at", 1},
          {"completed_at", nullptr},
          {"tools", nlohmann::json::array()},
          {"store", false}};
}

Usage token_usage() {
  Usage usage;
  usage.num_prompt_tokens = 9;
  usage.num_generated_tokens = 5;
  usage.num_total_tokens = 14;
  usage.num_cached_tokens = 3;
  usage.num_cache_write_tokens = 2;
  usage.num_reasoning_tokens = 1;
  return usage;
}

RequestOutput chunk(std::string text,
                    bool finished = false,
                    std::string finish_reason = "stop") {
  RequestOutput output;
  output.finished = finished;
  output.outputs.emplace_back();
  auto& sequence = output.outputs.back();
  sequence.index = 0;
  sequence.text = std::move(text);
  if (finished) {
    output.usage = token_usage();
    sequence.finish_reason = std::move(finish_reason);
  }
  return output;
}

std::vector<JsonTool> weather_tool() {
  JsonTool tool;
  tool.type = "function";
  tool.function.name = "weather";
  tool.function.parameters = {{"type", "object"},
                              {"properties", {{"city", {{"type", "string"}}}}}};
  return {tool};
}

std::string deltas(const std::vector<nlohmann::json>& events,
                   const std::string& type) {
  std::string text;
  for (const auto& event : events) {
    if (event["type"] != type) {
      continue;
    }
    text += event.at("delta").get<std::string>();
  }
  return text;
}

TEST(OpenAIResponsesOutputTest, TextStreamAndFinalSnapshotAgree) {
  std::vector<nlohmann::json> events;
  ResponsesOutput output(initial_response(),
                         /*stream=*/true,
                         {},
                         /*tool_parser=*/"",
                         /*reasoning_parser=*/"",
                         /*force_reasoning=*/false,
                         [&events](const nlohmann::json& event) {
                           events.emplace_back(event);
                           return true;
                         });
  ASSERT_TRUE(output.append(chunk("hello ")));
  ASSERT_TRUE(output.append(chunk("world", /*finished=*/true)));
  const std::vector<std::string> expected = {"response.created",
                                             "response.in_progress",
                                             "response.output_item.added",
                                             "response.content_part.added",
                                             "response.output_text.delta",
                                             "response.output_text.delta",
                                             "response.output_text.done",
                                             "response.content_part.done",
                                             "response.output_item.done",
                                             "response.completed"};
  ASSERT_EQ(events.size(), expected.size());
  for (size_t index = 0; index < events.size(); ++index) {
    EXPECT_EQ(events[index]["type"], expected[index]);
    EXPECT_EQ(events[index]["sequence_number"], index);
  }
  EXPECT_TRUE(events[2]["item"]["content"].empty());
  EXPECT_EQ(events[3]["part"]["text"], "");
  EXPECT_EQ(events[3]["content_index"], 0);
  EXPECT_EQ(events[4]["item_id"], events[2]["item"]["id"]);
  EXPECT_EQ(deltas(events, "response.output_text.delta"), "hello world");
  EXPECT_EQ(output.snapshot()["output"][0]["content"][0]["text"],
            "hello world");
  EXPECT_EQ(events.back()["response"], output.snapshot());
  EXPECT_TRUE(output.snapshot()["completed_at"].is_number_integer());
  EXPECT_GE(output.snapshot()["completed_at"].get<int64_t>(),
            output.snapshot()["created_at"].get<int64_t>());
  EXPECT_EQ(output.snapshot()["usage"]["input_tokens_details"],
            (nlohmann::json{{"cached_tokens", 3}, {"cache_write_tokens", 2}}));
  EXPECT_EQ(
      output.snapshot()["usage"]["output_tokens_details"]["reasoning_tokens"],
      1);
  EXPECT_FALSE(output.append(chunk("late", /*finished=*/true)));
  EXPECT_FALSE(output.fail(StatusCode::UNKNOWN, "late"));
  EXPECT_EQ(events.size(), expected.size());
}

TEST(OpenAIResponsesOutputTest, Utf8IsBufferedWithoutReplacementOrLoss) {
  std::vector<nlohmann::json> events;
  ResponsesOutput output(initial_response(),
                         /*stream=*/true,
                         {},
                         "",
                         "",
                         false,
                         [&events](const nlohmann::json& event) {
                           events.emplace_back(event);
                           return true;
                         });
  ASSERT_TRUE(output.append(chunk(std::string("A\xE4\xBD", 3))));
  ASSERT_TRUE(output.append(chunk(std::string("\xA0\xF0\x9F", 3))));
  ASSERT_TRUE(
      output.append(chunk(std::string("\x98\x80", 2), /*finished=*/true)));
  EXPECT_EQ(deltas(events, "response.output_text.delta"), "A你😀");
  EXPECT_EQ(output.snapshot()["output"][0]["content"][0]["text"], "A你😀");
  for (const auto& event : events) {
    EXPECT_NO_THROW(event.dump());
  }
}

TEST(OpenAIResponsesOutputTest, InvalidOrUnfinishedUtf8FailsNotReplaces) {
  for (const std::string text : {std::string("\xC0\x80", 2),
                                 std::string("\xED\xA0\x80", 3),
                                 std::string("\xF0\x9F", 2)}) {
    ResponsesOutput output(initial_response(), false, {}, "", "", false, {});
    EXPECT_FALSE(output.append(chunk(text, /*finished=*/true)));
    EXPECT_EQ(output.snapshot()["status"], "failed");
    EXPECT_EQ(output.snapshot()["error"]["code"], "server_error");
    EXPECT_FALSE(output.status().ok());
  }
}

TEST(OpenAIResponsesOutputTest, LengthEndsIncompleteExactlyOnce) {
  std::vector<nlohmann::json> events;
  ResponsesOutput output(initial_response(),
                         true,
                         {},
                         "",
                         "",
                         false,
                         [&events](const nlohmann::json& event) {
                           events.emplace_back(event);
                           return true;
                         });
  ASSERT_TRUE(output.append(chunk("partial", true, "length")));
  EXPECT_EQ(events.back()["type"], "response.incomplete");
  EXPECT_EQ(output.snapshot()["status"], "incomplete");
  EXPECT_TRUE(output.snapshot()["completed_at"].is_null());
  EXPECT_EQ(output.snapshot()["incomplete_details"]["reason"],
            "max_output_tokens");
  EXPECT_EQ(output.snapshot()["output"][0]["status"], "incomplete");
  EXPECT_FALSE(output.append(chunk("")));
  EXPECT_EQ(events.back()["response"], output.snapshot());
}

TEST(OpenAIResponsesOutputTest, GenerationFailureKeepsPartialOutput) {
  std::vector<nlohmann::json> events;
  ResponsesOutput output(initial_response(),
                         true,
                         {},
                         "",
                         "",
                         false,
                         [&events](const nlohmann::json& event) {
                           events.emplace_back(event);
                           return true;
                         });
  ASSERT_TRUE(output.append(chunk("partial")));
  EXPECT_FALSE(output.fail(StatusCode::UNKNOWN, "execution failed"));
  EXPECT_EQ(events.back()["type"], "response.failed");
  EXPECT_EQ(events.back()["response"]["error"]["message"], "execution failed");
  EXPECT_EQ(output.snapshot()["output"][0]["content"][0]["text"], "partial");
  EXPECT_EQ(output.snapshot()["output"][0]["status"], "incomplete");
  const size_t count = events.size();
  EXPECT_FALSE(output.fail(StatusCode::UNKNOWN, "duplicate"));
  EXPECT_EQ(events.size(), count);
}

TEST(OpenAIResponsesOutputTest, CancelAndWriteFailureNeverPretendCompleted) {
  std::vector<nlohmann::json> events;
  ResponsesOutput output(initial_response(),
                         true,
                         {},
                         "",
                         "",
                         false,
                         [&events](const nlohmann::json& event) {
                           events.emplace_back(event);
                           return true;
                         });
  ASSERT_TRUE(output.append(chunk("partial")));
  RequestOutput cancelled;
  cancelled.cancelled = true;
  EXPECT_FALSE(output.append(cancelled));
  EXPECT_EQ(output.status().code(), StatusCode::CANCELLED);
  EXPECT_EQ(output.snapshot()["status"], "cancelled");
  EXPECT_EQ(events.back()["type"], "response.output_text.delta");
  ResponsesOutput disconnected(
      initial_response(), true, {}, "", "", false, [](const nlohmann::json&) {
        return false;
      });
  EXPECT_FALSE(disconnected.append(chunk("text", true)));
  EXPECT_EQ(disconnected.status().code(), StatusCode::CANCELLED);
}

TEST(OpenAIResponsesOutputTest, RawReasoningRemainsSeparateFromTextAndSummary) {
  std::vector<nlohmann::json> events;
  ResponsesOutput output(initial_response(),
                         true,
                         {},
                         "",
                         "qwen3",
                         false,
                         [&events](const nlohmann::json& event) {
                           events.emplace_back(event);
                           return true;
                         });
  ASSERT_TRUE(output.append(chunk("<th")));
  ASSERT_TRUE(output.append(chunk("ink>why</th")));
  ASSERT_TRUE(output.append(chunk("ink>answer", true)));
  ASSERT_EQ(output.snapshot()["output"].size(), 2);
  const auto& reasoning = output.snapshot()["output"][0];
  EXPECT_EQ(reasoning["type"], "reasoning");
  EXPECT_TRUE(reasoning["summary"].empty());
  EXPECT_EQ(reasoning["content"][0]["type"], "reasoning_text");
  EXPECT_EQ(reasoning["content"][0]["text"], "why");
  EXPECT_EQ(deltas(events, "response.reasoning_text.delta"), "why");
  EXPECT_EQ(deltas(events, "response.output_text.delta"), "answer");
  EXPECT_EQ(output.snapshot()["output"][1]["content"][0]["text"], "answer");
}

TEST(OpenAIResponsesOutputTest, ActualTemplateMetadataForcesInitialReasoning) {
  ResponsesOutput output(initial_response(), false, {}, "", "glm47", false, {});
  RequestOutput complete = chunk("why</think>answer", true);
  complete.force_reasoning = true;
  ASSERT_TRUE(output.append(complete));
  ASSERT_EQ(output.snapshot()["output"].size(), 2);
  EXPECT_EQ(output.snapshot()["output"][0]["content"][0]["text"], "why");
  EXPECT_EQ(output.snapshot()["output"][1]["content"][0]["text"], "answer");
}

TEST(OpenAIResponsesOutputTest,
     ActualFalseMetadataOverridesThinkingOnlyDefault) {
  ResponsesOutput output(initial_response(), false, {}, "", "glm5", false, {});
  RequestOutput complete = chunk("answer", true);
  complete.force_reasoning = false;
  ASSERT_TRUE(output.append(complete));
  ASSERT_EQ(output.snapshot()["output"].size(), 1);
  EXPECT_EQ(output.snapshot()["output"][0]["type"], "message");
  EXPECT_EQ(output.snapshot()["output"][0]["content"][0]["text"], "answer");
}

TEST(OpenAIResponsesOutputTest, ToolArgumentsAcrossChunksMatchFinalCall) {
  std::vector<nlohmann::json> events;
  ResponsesOutput output(initial_response(),
                         true,
                         weather_tool(),
                         "qwen25",
                         "",
                         false,
                         [&events](const nlohmann::json& event) {
                           events.emplace_back(event);
                           return true;
                         });
  for (const std::string text : {"<tool_",
                                 "call>\n{\"name\":\"weat",
                                 "her\",\"arguments\":{\"city\":\"Par",
                                 "is\"}}\n</tool_call>"}) {
    ASSERT_TRUE(output.append(chunk(text))) << output.status().message();
  }
  const auto reason = FinishReason(FinishReason::FUNCTION_CALL).to_string();
  ASSERT_TRUE(reason.has_value());
  ASSERT_TRUE(output.append(chunk("", true, reason.value())))
      << output.status().message();
  ASSERT_EQ(output.snapshot()["output"].size(), 1);
  const auto& call = output.snapshot()["output"][0];
  EXPECT_EQ(call["type"], "function_call");
  EXPECT_EQ(call["name"], "weather");
  EXPECT_EQ(nlohmann::json::parse(call["arguments"].get<std::string>()),
            (nlohmann::json{{"city", "Paris"}}));
  EXPECT_EQ(deltas(events, "response.function_call_arguments.delta"),
            call["arguments"].get<std::string>());
  EXPECT_NE(call["id"], call["call_id"]);
  EXPECT_EQ(events.back()["type"], "response.completed");
  const auto found =
      std::find_if(events.begin(), events.end(), [](const auto& event) {
        return event["type"] == "response.function_call_arguments.done";
      });
  ASSERT_NE(found, events.end());
  EXPECT_EQ((*found)["name"], "weather");
  EXPECT_EQ((*found)["arguments"], call["arguments"]);
}

TEST(OpenAIResponsesOutputTest,
     GlmToolAndNormalTextWithinOneChunkArePreserved) {
  ResponsesOutput output(
      initial_response(), false, weather_tool(), "glm47", "", false, {});
  ASSERT_TRUE(
      output.append(chunk("Calling "
                          "it.<tool_call>weather<arg_key>city</"
                          "arg_key><arg_value>Paris</arg_value></tool_call>",
                          true)))
      << output.status().message();
  ASSERT_EQ(output.snapshot()["output"].size(), 2);
  EXPECT_EQ(output.snapshot()["output"][0]["content"][0]["text"],
            "Calling it.");
  const auto& call = output.snapshot()["output"][1];
  EXPECT_EQ(call["type"], "function_call");
  EXPECT_EQ(nlohmann::json::parse(call["arguments"].get<std::string>()),
            (nlohmann::json{{"city", "Paris"}}));
}

TEST(OpenAIResponsesOutputTest, CompletedMalformedOrUndeclaredToolFails) {
  for (const std::string text :
       {"<tool_call>\n{\"name\":\"missing\",\"arguments\":{}}\n</tool_call>",
        "<tool_call>\n{\"name\":\"weather\",\"arguments\":42}\n</tool_call>",
        "<tool_call>\n{broken}\n</tool_call>",
        "<tool_call>\n{\"name\":\"weather\"}\n</tool_call>",
        "<tool_call>\n{\"name\":\"weather\",\"arguments\":"}) {
    ResponsesOutput output(
        initial_response(), false, weather_tool(), "qwen25", "", false, {});
    EXPECT_FALSE(output.append(chunk(text, true))) << text;
    EXPECT_EQ(output.snapshot()["status"], "failed");
    EXPECT_FALSE(output.status().ok());
  }
}

TEST(OpenAIResponsesOutputTest, LiteralPartialMarkerAtEofIsNotDiscarded) {
  ResponsesOutput output(
      initial_response(), false, weather_tool(), "qwen25", "qwen3", false, {});
  ASSERT_TRUE(output.append(chunk("literal<", true)))
      << output.status().message();
  EXPECT_EQ(output.snapshot()["output"][0]["content"][0]["text"], "literal<");
}

TEST(OpenAIResponsesOutputTest, CacheReadAndWriteCountersAreIndependent) {
  ResponsesOutput output(initial_response(), false, {}, "", "", false, {});
  auto complete = chunk("text", true);
  complete.usage->num_cached_tokens = 8;
  complete.usage->num_cache_write_tokens = 7;
  ASSERT_TRUE(output.append(complete));
  EXPECT_EQ(output.snapshot()["usage"]["input_tokens_details"],
            (nlohmann::json{{"cached_tokens", 8}, {"cache_write_tokens", 7}}));
}

TEST(OpenAIResponsesOutputTest, MalformedGlmXmlIsNotConvertedToEmptyArguments) {
  for (const std::string text :
       {"<tool_call>weather<arg_key>city</arg_key></tool_call>",
        "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</tool_call>",
        "<tool_call>weather<arg_key>city</arg_key><arg_key>city</"
        "arg_key><arg_value>Paris</arg_value></tool_call>",
        "<tool_call>weather</tool_call><tool_call>weather<arg"}) {
    ResponsesOutput output(
        initial_response(), false, weather_tool(), "glm47", "", false, {});
    EXPECT_FALSE(output.append(chunk(text, true))) << text;
    EXPECT_EQ(output.snapshot()["status"], "failed");
    EXPECT_FALSE(output.status().ok());
  }
}

TEST(OpenAIResponsesOutputTest, StrictToolFormatsAgreeAtEveryChunkBoundary) {
  const std::vector<std::string> formats = {
      "qwen25", "qwen3_coder", "glm45", "glm47", "glm5"};
  for (const auto& format : formats) {
    const std::string first =
        format == "qwen25" ? "<tool_call>\n{\"name\":\"weather\",\"arguments\":"
                             "{\"city\":\"  Paris  \"}}\n</tool_call>"
        : format == "qwen3_coder"
            ? "<tool_call><function=weather><parameter=city>  Paris  "
              "</parameter></function></tool_call>"
            : "<tool_call>weather\n<arg_key>city</arg_key> \t\n<arg_value>  "
              "Paris  </arg_value> \n</tool_call>";
    const std::string second =
        format == "qwen25" ? "<tool_call>\n{\"name\":\"weather\",\"arguments\":"
                             "{\"city\":\"北京\"}}\n</tool_call>"
        : format == "qwen3_coder"
            ? "<tool_call><function=weather><parameter=city>北京</parameter></"
              "function></tool_call>"
            : "<tool_call>weather<arg_key>city</arg_key><arg_value>\"北京\"</"
              "arg_value></tool_call>";
    const std::string text =
        "before " + first + " between " + second + " after<";
    function_call::FunctionCallParser full(weather_tool(),
                                           format,
                                           /*strict_errors=*/true);
    const auto [normal, calls] = full.parse_non_stream(text);
    ASSERT_TRUE(full.error_status().ok()) << format;
    EXPECT_EQ(normal, "before  between  after<") << format;
    ASSERT_EQ(calls.size(), 2) << format;
    EXPECT_EQ(nlohmann::json::parse(calls[0].parameters),
              (nlohmann::json{{"city", "  Paris  "}}));
    EXPECT_EQ(nlohmann::json::parse(calls[1].parameters),
              (nlohmann::json{{"city", "北京"}}));
    for (const bool stream : {false, true}) {
      for (size_t split = 0; split <= text.size(); ++split) {
        SCOPED_TRACE(format + ":" + std::to_string(split) + ":" +
                     std::to_string(stream));
        std::vector<nlohmann::json> events;
        ResponsesOutput output(initial_response(),
                               stream,
                               weather_tool(),
                               format,
                               "",
                               false,
                               [&events](const nlohmann::json& event) {
                                 events.emplace_back(event);
                                 return true;
                               });
        ASSERT_TRUE(output.append(chunk(text.substr(0, split))))
            << output.status().message();
        ASSERT_TRUE(output.append(chunk(text.substr(split), true)))
            << output.status().message();
        std::vector<nlohmann::json> arguments;
        arguments.reserve(calls.size());
        for (const auto& item : output.snapshot()["output"]) {
          if (item["type"] != "function_call") {
            EXPECT_EQ(item["content"][0]["text"], normal);
            continue;
          }
          EXPECT_EQ(item["name"], "weather");
          arguments.emplace_back(
              nlohmann::json::parse(item["arguments"].get<std::string>()));
        }
        ASSERT_EQ(arguments.size(), calls.size());
        for (size_t index = 0; index < calls.size(); ++index) {
          EXPECT_EQ(arguments[index],
                    nlohmann::json::parse(calls[index].parameters));
        }
        if (stream) {
          EXPECT_EQ(deltas(events, "response.output_text.delta"), normal);
          EXPECT_EQ(events.back()["type"], "response.completed");
          for (size_t index = 0; index < events.size(); ++index) {
            EXPECT_EQ(events[index]["sequence_number"], index);
          }
        }
      }
    }
  }
}

TEST(OpenAIResponsesOutputTest, StrictGlmValueConversionIsChunkIndependent) {
  for (const std::string format : {"glm45", "glm47", "glm5"}) {
    for (const bool declared : {false, true}) {
      const auto tools = declared ? weather_tool() : [&] {
        auto result = weather_tool();
        result[0].function.parameters = {{"type", "object"}};
        return result;
      }();
      for (const std::string value : {"  Paris  ",
                                      "\"  Paris  \"",
                                      "123",
                                      "true",
                                      "{\"key\":1}",
                                      "\"123\""}) {
        const auto parsed = nlohmann::json::parse(value,
                                                  nullptr,
                                                  /*allow_exceptions=*/false);
        const auto expected = declared && !parsed.is_string()
                                  ? nlohmann::json(value)
                              : parsed.is_discarded() ? nlohmann::json(value)
                                                      : parsed;
        const std::string text =
            "<tool_call>weather<arg_key>city</arg_key><arg_value>" + value +
            "</arg_value> \n</tool_call>";
        for (const bool stream : {false, true}) {
          for (size_t split = 0; split <= text.size(); ++split) {
            SCOPED_TRACE(format + ":" + value + ":" + std::to_string(split));
            ResponsesOutput output(initial_response(),
                                   stream,
                                   tools,
                                   format,
                                   "",
                                   false,
                                   [](const nlohmann::json&) { return true; });
            ASSERT_TRUE(output.append(chunk(text.substr(0, split))));
            ASSERT_TRUE(output.append(chunk(text.substr(split), true)))
                << output.status().message();
            ASSERT_EQ(output.snapshot()["output"].size(), 1);
            EXPECT_EQ(nlohmann::json::parse(
                          output.snapshot()["output"][0]["arguments"]
                              .get<std::string>())["city"],
                      expected);
          }
        }
      }
    }
  }
}

TEST(OpenAIResponsesOutputTest,
     StrictToolTerminatorAndMalformedTailFailBothModes) {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"qwen25", "<tool_call>{\"name\":\"weather\",\"arguments\":{}}"},
      {"qwen25",
       "<tool_call>{\"name\":\"weather\",\"arguments\":{}}</"
       "tool_call><tool_call>{"},
      {"qwen3_coder", "<tool_call><function=weather></function>"},
      {"qwen3_coder", "<tool_call><function=weather></tool_call>"},
      {"qwen3_coder",
       "<tool_call><function=weather><parameter=city>Paris</function></"
       "tool_call>"},
      {"qwen3_coder",
       "<tool_call><function=weather></function>JUNK</tool_call>"},
      {"glm45",
       "<tool_call>weather\n</tool_call><tool_call>weather\n<arg_key>city"},
      {"glm47",
       "<tool_call>weather<arg_key>city</arg_key>JUNK<arg_value>Paris</"
       "arg_value></tool_call>"},
      {"glm47",
       "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</"
       "arg_value>JUNK</tool_call>"},
      {"glm5",
       "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</"
       "arg_value>"}};
  for (const auto& [format, text] : cases) {
    for (const bool stream : {false, true}) {
      for (const size_t split : {size_t{0}, text.size() / 2, text.size()}) {
        SCOPED_TRACE(format + ":" + text + ":" + std::to_string(split));
        std::vector<nlohmann::json> events;
        ResponsesOutput output(initial_response(),
                               stream,
                               weather_tool(),
                               format,
                               "",
                               false,
                               [&events](const nlohmann::json& event) {
                                 events.emplace_back(event);
                                 return true;
                               });
        if (output.append(chunk(text.substr(0, split)))) {
          EXPECT_FALSE(output.append(chunk(text.substr(split), true)));
        }
        EXPECT_FALSE(output.status().ok());
        EXPECT_EQ(output.snapshot()["status"], "failed");
        if (stream) {
          ASSERT_FALSE(events.empty());
          EXPECT_EQ(events.back()["type"], "response.failed");
        }
      }
    }
  }
}

TEST(OpenAIResponsesOutputTest, StrictCapabilityDoesNotChangeLegacyResolution) {
  for (const std::string parser :
       {"kimi_k2", "deepseekv3", "deepseekv32", "deepseekv4"}) {
    const auto [legacy, name] =
        function_call::FunctionCallParser::resolve_parser(
            parser, "", /*strict_errors=*/false);
    EXPECT_TRUE(legacy.ok());
    EXPECT_EQ(name, parser);
    const auto [strict, unsupported] =
        function_call::FunctionCallParser::resolve_parser(
            parser, "", /*strict_errors=*/true);
    EXPECT_EQ(strict.code(), StatusCode::INVALID_ARGUMENT);
    EXPECT_TRUE(unsupported.empty());
  }
  EXPECT_FALSE(function_call::FunctionCallParser::resolve_parser(
                   "auto", "deepseek_v32", /*strict_errors=*/true)
                   .first.ok());
  EXPECT_EQ(function_call::FunctionCallParser::resolve_parser(
                "qwen35", "", /*strict_errors=*/true)
                .second,
            "qwen3_coder");
  function_call::FunctionCallParser legacy(weather_tool(), "glm47");
  const auto [text, calls] = legacy.parse_non_stream(
      "<tool_call>weather<arg_key>city</arg_key><arg_value>  Paris  "
      "</arg_value></tool_call>");
  ASSERT_EQ(calls.size(), 1);
  EXPECT_EQ(nlohmann::json::parse(calls[0].parameters)["city"], "Paris");
}

TEST(OpenAIResponsesOutputTest, MissingOrInconsistentUsageIsFailure) {
  for (const bool missing : {false, true}) {
    ResponsesOutput output(initial_response(), false, {}, "", "", false, {});
    RequestOutput complete = chunk("text", true);
    if (missing) {
      complete.usage.reset();
    } else {
      complete.usage->num_total_tokens = 1;
    }
    EXPECT_FALSE(output.append(complete));
    EXPECT_EQ(output.snapshot()["status"], "failed");
  }
}

}  // namespace
}  // namespace xllm::api_service
