package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// newTestOpenAI builds an OpenAI provider pointed at a stub server and records
// each request body.
func newTestOpenAI(t *testing.T, respBody string) (*OpenAIProvider, *[]map[string]any) {
	t.Helper()
	var reqs []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		reqs = append(reqs, body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, respBody)
	}))
	t.Cleanup(srv.Close)
	p := &OpenAIProvider{
		client: openai.NewClient(option.WithAPIKey("test-key"), option.WithBaseURL(srv.URL+"/")),
		model:  "gpt-test",
	}
	return p, &reqs
}

const openAIToolCallResp = `{
	"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-test",
	"choices": [{
		"index": 0,
		"finish_reason": "tool_calls",
		"message": {
			"role": "assistant",
			"content": "Reading both.",
			"refusal": null,
			"tool_calls": [
				{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"a.txt\"}"}},
				{"id": "call_2", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"b.txt\"}"}}
			]
		}
	}],
	"usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
}`

func TestOpenAIChatToolsParsesReply(t *testing.T) {
	p, reqs := newTestOpenAI(t, openAIToolCallResp)

	reply, err := p.ChatTools(context.Background(), "sys", []Turn{{Role: RoleUser, Text: "q"}}, testTools)
	if err != nil {
		t.Fatalf("ChatTools: %v", err)
	}

	tools := (*reqs)[0]["tools"].([]any)
	fn := tools[0].(map[string]any)["function"].(map[string]any)
	if fn["name"] != "read_file" || fn["parameters"] == nil {
		t.Errorf("request tool = %v, want read_file with parameters", fn)
	}
	if reply.Stop != StopToolUse {
		t.Errorf("Stop = %q, want %q", reply.Stop, StopToolUse)
	}
	if reply.Turn.Text != "Reading both." {
		t.Errorf("Text = %q", reply.Turn.Text)
	}
	if len(reply.Turn.Calls) != 2 || reply.Turn.Calls[1].ID != "call_2" ||
		string(reply.Turn.Calls[0].Input) != `{"path":"a.txt"}` {
		t.Errorf("Calls = %+v", reply.Turn.Calls)
	}
	if reply.Turn.RawFrom != rawFromOpenAI || len(reply.Turn.Raw) == 0 {
		t.Errorf("Raw = %s (from %q), want the original message", reply.Turn.Raw, reply.Turn.RawFrom)
	}
	if reply.Usage.InputTokens != 50 || reply.Usage.OutputTokens != 20 {
		t.Errorf("Usage = %+v", reply.Usage)
	}
}

// A full round trip: the reply goes back as the assistant message, and each
// result becomes its own "tool" message, followed by the user's text.
func TestOpenAIChatToolsSendsResults(t *testing.T) {
	p, reqs := newTestOpenAI(t, openAIToolCallResp)
	first, err := p.ChatTools(context.Background(), "sys", []Turn{{Role: RoleUser, Text: "q"}}, testTools)
	if err != nil {
		t.Fatalf("ChatTools: %v", err)
	}

	turns := []Turn{
		{Role: RoleUser, Text: "q"},
		first.Turn,
		{Role: RoleUser, Text: "keep going", Results: []ToolResult{
			{CallID: "call_1", Content: "hello"},
			{CallID: "call_2", Content: "no such file", IsError: true},
		}},
	}
	if _, err := p.ChatTools(context.Background(), "sys", turns, testTools); err != nil {
		t.Fatalf("ChatTools: %v", err)
	}

	msgs := (*reqs)[1]["messages"].([]any)
	roles := make([]string, len(msgs))
	for i, m := range msgs {
		roles[i] = m.(map[string]any)["role"].(string)
	}
	want := []string{"system", "user", "assistant", "tool", "tool", "user"}
	if len(roles) != len(want) {
		t.Fatalf("roles = %v, want %v", roles, want)
	}
	for i := range want {
		if roles[i] != want[i] {
			t.Fatalf("roles = %v, want %v", roles, want)
		}
	}

	asst := msgs[2].(map[string]any)
	if calls := asst["tool_calls"].([]any); len(calls) != 2 {
		t.Errorf("assistant tool_calls = %v, want 2", calls)
	}
	failed := msgs[4].(map[string]any)
	if failed["tool_call_id"] != "call_2" || failed["content"] != "Error: no such file" {
		t.Errorf("failed result = %v", failed)
	}
}

// Raw from another provider is ignored; the turn is rebuilt from Text and
// Calls, with empty arguments sent as {}.
func TestOpenAIRebuildsForeignRaw(t *testing.T) {
	p := &OpenAIProvider{model: "gpt-test"}
	turns := []Turn{
		{Role: RoleUser, Text: "q"},
		{Role: RoleAssistant, Text: "ok", Calls: []ToolCall{{ID: "toolu_1", Name: "list"}},
			Raw: json.RawMessage(`[{"type":"thinking"}]`), RawFrom: rawFromAnthropic},
	}
	params, err := p.buildToolParams("", turns, nil)
	if err != nil {
		t.Fatalf("buildToolParams: %v", err)
	}
	asst := params.Messages[1].OfAssistant
	if asst == nil || asst.Content.OfString.Value != "ok" || len(asst.ToolCalls) != 1 {
		t.Fatalf("assistant = %+v", params.Messages[1])
	}
	if args := asst.ToolCalls[0].OfFunction.Function.Arguments; args != "{}" {
		t.Errorf("arguments = %q, want {}", args)
	}
}

func TestOpenAIRejectsBadTurns(t *testing.T) {
	p := &OpenAIProvider{model: "gpt-test"}
	bad := map[string]Turn{
		"results on assistant": {Role: RoleAssistant, Text: "x", Results: []ToolResult{{CallID: "a"}}},
		"calls on user":        {Role: RoleUser, Calls: []ToolCall{{ID: "a"}}},
		"empty assistant":      {Role: RoleAssistant},
		"unknown role":         {Role: "tool"},
	}
	for name, turn := range bad {
		if _, err := p.buildToolParams("", []Turn{turn}, nil); err == nil {
			t.Errorf("%s: expected an error, got nil", name)
		}
	}
}

func TestOpenAIRefusal(t *testing.T) {
	p, _ := newTestOpenAI(t, `{
		"id": "chatcmpl-2", "object": "chat.completion", "created": 1, "model": "gpt-test",
		"choices": [{"index": 0, "finish_reason": "stop",
			"message": {"role": "assistant", "content": null, "refusal": "I can't help with that."}}],
		"usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
	}`)
	reply, err := p.ChatTools(context.Background(), "", []Turn{{Role: RoleUser, Text: "q"}}, nil)
	if err != nil {
		t.Fatalf("ChatTools: %v", err)
	}
	if reply.Stop != StopRefusal || reply.Turn.Text != "I can't help with that." {
		t.Errorf("reply = %+v, want a refusal with its text", reply)
	}
}

func TestOpenAIStop(t *testing.T) {
	tests := map[string]string{
		"stop":           StopEnd,
		"tool_calls":     StopToolUse,
		"length":         StopMaxTokens,
		"content_filter": "content_filter",
	}
	for in, want := range tests {
		if got := openAIStop(in); got != want {
			t.Errorf("openAIStop(%q) = %q, want %q", in, got, want)
		}
	}
}
