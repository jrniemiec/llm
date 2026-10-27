package llm

import (
	"context"
	"encoding/json"
	"os"
	"testing"
)

// Live tests call real APIs and are skipped unless their variable names a model:
//
//	LLM_ANTHROPIC_LIVE=claude-sonnet-5 ANTHROPIC_API_KEY=… go test -run Live -v
//	LLM_OPENAI_LIVE=gpt-5 OPENAI_API_KEY=… go test -run Live -v
//	LLM_OLLAMA_LIVE=qwen3.5:9b go test -run Live -v

var liveTools = []ToolDef{{
	Name:        "get_weather",
	Description: "Get the current weather for a city.",
	InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
}}

// liveModel returns the model named by env, or skips the test.
func liveModel(t *testing.T, env string) string {
	t.Helper()
	model := os.Getenv(env)
	if model == "" {
		t.Skipf("set %s=<model> to run", env)
	}
	return model
}

// liveKey returns the API key in env, or fails the test.
func liveKey(t *testing.T, env string) string {
	t.Helper()
	key := os.Getenv(env)
	if key == "" {
		t.Fatalf("%s is not set", env)
	}
	return key
}

// liveToolRound asks for the weather in two cities and answers every tool
// call until the model gives a final answer. In the first round the last
// result is an error, so the API must accept IsError. It returns every
// assistant turn.
func liveToolRound(t *testing.T, p ToolProvider) []Turn {
	t.Helper()
	ctx := context.Background()
	turns := []Turn{{Role: RoleUser, Text: "What is the weather in Paris and in Rome? Use the tool for each city."}}
	var replies []Turn

	for round := 1; ; round++ {
		reply, err := p.ChatTools(ctx, "Answer briefly.", turns, liveTools)
		if err != nil {
			t.Fatalf("round %d: ChatTools: %v", round, err)
		}
		t.Logf("round %d: stop=%s calls=%d usage=%+v", round, reply.Stop, len(reply.Turn.Calls), reply.Usage)
		replies = append(replies, reply.Turn)
		turns = append(turns, reply.Turn)

		if round == 1 && (reply.Stop != StopToolUse || len(reply.Turn.Calls) == 0) {
			t.Fatalf("round 1: stop=%s calls=%d, want tool calls", reply.Stop, len(reply.Turn.Calls))
		}
		if reply.Stop != StopToolUse {
			if reply.Stop != StopEnd || reply.Turn.Text == "" {
				t.Fatalf("round %d: stop=%s text=%q, want a final answer", round, reply.Stop, reply.Turn.Text)
			}
			t.Logf("answer: %s", reply.Turn.Text)
			return replies
		}
		if round == 4 {
			t.Fatal("no final answer after 4 rounds")
		}

		results := Turn{Role: RoleUser}
		for i, c := range reply.Turn.Calls {
			r := ToolResult{CallID: c.ID, Content: string(c.Input) + ": 15C, cloudy"}
			if round == 1 && i == len(reply.Turn.Calls)-1 && i > 0 {
				r = ToolResult{CallID: c.ID, Content: "weather service unavailable, try again", IsError: true}
			}
			results.Results = append(results.Results, r)
		}
		turns = append(turns, results)
	}
}

func TestAnthropicLive(t *testing.T) {
	model := liveModel(t, "LLM_ANTHROPIC_LIVE")
	key := liveKey(t, "ANTHROPIC_API_KEY")

	t.Run("no thinking", func(t *testing.T) {
		p, err := newAnthropicProvider(key, model, 0, 0, "")
		if err != nil {
			t.Fatalf("newAnthropicProvider: %v", err)
		}
		liveToolRound(t, p)
	})

	// With thinking on, the replies sent back must keep their thinking blocks.
	// The model decides whether to think, so a reply may have none; then the
	// replay isn't exercised and the test only logs it.
	t.Run("adaptive thinking", func(t *testing.T) {
		p, err := newAnthropicProvider(key, model, 8192, 0, ThinkingAdaptive)
		if err != nil {
			t.Fatalf("newAnthropicProvider: %v", err)
		}
		replies := liveToolRound(t, p)
		thought := false
		for _, r := range replies[:len(replies)-1] { // turns that were sent back
			thought = thought || hasThinking(t, r.Raw)
		}
		if !thought {
			t.Log("no thinking block in a replayed turn; thinking replay not exercised")
		}
	})

	// Sending a tool-use reply back without its thinking block must be
	// rejected. This is why Turn.Raw exists. The model may not think, so the
	// first request is tried up to 3 times before the test is skipped.
	t.Run("thinking dropped", func(t *testing.T) {
		p, err := newAnthropicProvider(key, model, 8192, 0, ThinkingAdaptive)
		if err != nil {
			t.Fatalf("newAnthropicProvider: %v", err)
		}
		ctx := context.Background()
		ask := Turn{Role: RoleUser, Text: "Think about which cities to check, then use the tool to get the weather in Paris and in Rome."}

		var first ToolReply
		for try := 1; ; try++ {
			first, err = p.ChatTools(ctx, "", []Turn{ask}, liveTools)
			if err != nil {
				t.Fatalf("ChatTools: %v", err)
			}
			if first.Stop == StopToolUse && hasThinking(t, first.Turn.Raw) {
				break
			}
			if try == 3 {
				t.Skip("no tool-use reply with a thinking block after 3 tries")
			}
		}

		// Rebuilt from Text and Calls only, so the thinking block is gone.
		stripped := Turn{Role: RoleAssistant, Text: first.Turn.Text, Calls: first.Turn.Calls}
		results := Turn{Role: RoleUser}
		for _, c := range first.Turn.Calls {
			results.Results = append(results.Results, ToolResult{CallID: c.ID, Content: "15C, cloudy"})
		}
		_, err = p.ChatTools(ctx, "", []Turn{ask, stripped, results}, liveTools)
		if err == nil {
			t.Fatal("request without the thinking block was accepted; want a 400 error")
		}
		t.Logf("rejected as expected: %v", err)
	})
}

// hasThinking reports whether Anthropic content in raw has a thinking block.
func hasThinking(t *testing.T, raw json.RawMessage) bool {
	t.Helper()
	var blocks []anthropicBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		t.Fatalf("decode Raw: %v", err)
	}
	for _, b := range blocks {
		if b.Type == "thinking" {
			return true
		}
	}
	return false
}

func TestOpenAILive(t *testing.T) {
	model := liveModel(t, "LLM_OPENAI_LIVE")
	p, err := newOpenAIProvider(liveKey(t, "OPENAI_API_KEY"), model)
	if err != nil {
		t.Fatalf("newOpenAIProvider: %v", err)
	}
	liveToolRound(t, p)
}

func TestOllamaLive(t *testing.T) {
	model := liveModel(t, "LLM_OLLAMA_LIVE")
	p, err := newOllamaProvider(ProviderConfig{Model: model})
	if err != nil {
		t.Fatalf("newOllamaProvider: %v", err)
	}
	liveToolRound(t, p)
}
