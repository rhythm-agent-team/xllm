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
            (nlohmann::json{{"cached_tokens", 3}, {"cache_write_tokens", 0}}));
  const auto& reasoning_tokens =
      output.snapshot()["usage"]["output_tokens_details"]["reasoning_tokens"];
  EXPECT_TRUE(reasoning_tokens.is_number_integer());
  EXPECT_EQ(reasoning_tokens, 0);
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
  const auto& reasoning_tokens =
      output.snapshot()["usage"]["output_tokens_details"]["reasoning_tokens"];
  EXPECT_TRUE(reasoning_tokens.is_number_integer());
  EXPECT_EQ(reasoning_tokens, 0);
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

TEST(OpenAIResponsesOutputTest, InitialFalseOverridesThinkingDefault) {
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

TEST(OpenAIResponsesOutputTest, GlmToolAndTextWithinOneChunk) {
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

TEST(OpenAIResponsesOutputTest, GlmFramesPreserveFunctionArguments) {
  const std::string frame =
      "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
      "</tool_call>";
  const size_t split = frame.find("Paris") + 3;
  for (const bool stream : {false, true}) {
    for (const bool split_frame : {false, true}) {
      std::vector<nlohmann::json> events;
      ResponsesOutput output(initial_response(),
                             stream,
                             weather_tool(),
                             "glm5",
                             "",
                             false,
                             [&events](const nlohmann::json& event) {
                               events.emplace_back(event);
                               return true;
                             });
      if (split_frame) {
        ASSERT_TRUE(output.append(chunk(frame.substr(0, split))));
      }
      ASSERT_TRUE(output.append(chunk(
          split_frame ? frame.substr(split) : frame, true, "function_call")))
          << output.status().message();
      EXPECT_EQ(output.snapshot()["status"], "completed");
      ASSERT_EQ(output.snapshot()["output"].size(), 1);
      const auto& call = output.snapshot()["output"][0];
      EXPECT_EQ(call["type"], "function_call");
      EXPECT_EQ(call["name"], "weather");
      EXPECT_EQ(nlohmann::json::parse(call["arguments"].get<std::string>()),
                (nlohmann::json{{"city", "Paris"}}));
      EXPECT_NE(call["id"], call["call_id"]);
      if (stream) {
        EXPECT_EQ(deltas(events, "response.function_call_arguments.delta"),
                  call["arguments"].get<std::string>());
        ASSERT_GE(events.size(), 3);
        EXPECT_EQ(events[events.size() - 3]["type"],
                  "response.function_call_arguments.done");
        EXPECT_EQ(events[events.size() - 3]["arguments"], call["arguments"]);
        EXPECT_EQ(events.back()["type"], "response.completed");
        EXPECT_EQ(events.back()["response"], output.snapshot());
      }
    }
  }
}

TEST(OpenAIResponsesOutputTest, NonStreamFailureKeepsBufferedToolOutput) {
  for (const bool tool_call : {false, true}) {
    ResponsesOutput output(
        initial_response(), false, weather_tool(), "glm5", "", false, {});
    ASSERT_TRUE(output.append(
        chunk(tool_call ? "<tool_call>weather<arg_key>city</arg_key>"
                          "<arg_value>Paris</arg_value></tool_call>"
                        : "partial")));
    EXPECT_FALSE(output.fail(StatusCode::UNKNOWN, "execution failed"));
    EXPECT_EQ(output.snapshot()["status"], "failed");
    EXPECT_EQ(output.snapshot()["error"]["message"], "execution failed");
    ASSERT_EQ(output.snapshot()["output"].size(), 1);
    const auto& item = output.snapshot()["output"][0];
    EXPECT_EQ(item["status"], "incomplete");
    if (tool_call) {
      EXPECT_EQ(item["name"], "weather");
      EXPECT_EQ(nlohmann::json::parse(item["arguments"].get<std::string>()),
                (nlohmann::json{{"city", "Paris"}}));
    } else {
      EXPECT_EQ(item["content"][0]["text"], "partial");
    }
  }
}

TEST(OpenAIResponsesOutputTest, LiteralPartialMarkerAtEofIsNotDiscarded) {
  ResponsesOutput output(initial_response(), false, {}, "", "qwen3", false, {});
  ASSERT_TRUE(output.append(chunk("literal<", true)))
      << output.status().message();
  EXPECT_EQ(output.snapshot()["output"][0]["content"][0]["text"], "literal<");
}

TEST(OpenAIResponsesOutputTest, CacheWriteUsageIsZeroWithCachedInput) {
  ResponsesOutput output(initial_response(), false, {}, "", "", false, {});
  auto complete = chunk("text", /*finished=*/true);
  complete.usage->num_cached_tokens = 8;
  ASSERT_TRUE(output.append(complete));
  EXPECT_EQ(output.snapshot()["usage"]["input_tokens_details"],
            (nlohmann::json{{"cached_tokens", 8}, {"cache_write_tokens", 0}}));
}

TEST(OpenAIResponsesOutputTest, LegacyToolParserBehaviorIsUnchanged) {
  for (const std::string parser :
       {"kimi_k2", "deepseekv3", "deepseekv32", "deepseekv4"}) {
    EXPECT_EQ(function_call::FunctionCallParser::get_parser_auto(parser, ""),
              parser);
  }
  EXPECT_EQ(function_call::FunctionCallParser::get_parser_auto("qwen35", ""),
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
