package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
)

var testTools = []ToolDef{{
	Name:        "read_file",
	Description: "Read a file.",
	InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}`),
}}

// A reply with thinking, text and two tool calls parses into one assistant
// turn, and the raw content (with the thinking signature) is kept.
func TestChatToolsParsesReply(t *testing.T) {
	var gotReq map[string]any
	p := newTestProvider(t, ThinkingAdaptive, func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&gotReq)
		io.WriteString(w, `{
			"content": [
				{"type": "thinking", "thinking": "plan", "signature": "sig123"},
				{"type": "text", "text": "Reading both."},
				{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.txt"}},
				{"type": "tool_use", "id": "toolu_2", "name": "read_file", "input": {"path": "b.txt"}}
			],
			"stop_reason": "tool_use",
			"usage": {"input_tokens": 50, "output_tokens": 20}
		}`)
	})

	reply, err := p.ChatTools(context.Background(), "sys", []Turn{{Role: RoleUser, Text: "q"}}, testTools)
	if err != nil {
		t.Fatalf("ChatTools: %v", err)
	}

	tools := gotReq["tools"].([]any)
	if len(tools) != 1 || tools[0].(map[string]any)["input_schema"] == nil {
		t.Errorf("request tools = %v, want one tool with input_schema", tools)
	}
	if reply.Stop != StopToolUse {
		t.Errorf("Stop = %q, want %q", reply.Stop, StopToolUse)
	}
	if reply.Turn.Text != "Reading both." {
		t.Errorf("Text = %q", reply.Turn.Text)
	}
	if len(reply.Turn.Calls) != 2 || reply.Turn.Calls[1].ID != "toolu_2" ||
		string(reply.Turn.Calls[0].Input) != `{"path": "a.txt"}` {
		t.Errorf("Calls = %+v", reply.Turn.Calls)
	}
	if reply.Turn.RawFrom != rawFromAnthropic || !strings.Contains(string(reply.Turn.Raw), "sig123") {
		t.Errorf("Raw = %s (from %q), want the original content", reply.Turn.Raw, reply.Turn.RawFrom)
	}
	if reply.Usage.InputTokens != 50 || reply.Usage.OutputTokens != 20 {
		t.Errorf("Usage = %+v", reply.Usage)
	}
}

// The assistant turn goes back verbatim (thinking included), and all results
// of its calls go in one user message, tool_result blocks first.
func TestBuildToolReqReplaysRawAndGroupsResults(t *testing.T) {
	p := newTestProvider(t, ThinkingAdaptive, nil)
	raw := json.RawMessage(`[{"type":"thinking","thinking":"plan","signature":"sig123"},` +
		`{"type":"tool_use","id":"toolu_1","name":"read_file","input":{"path":"a.txt"}}]`)
	turns := []Turn{
		{Role: RoleUser, Text: "q"},
		{Role: RoleAssistant, Calls: []ToolCall{{ID: "toolu_1", Name: "read_file"}}, Raw: raw, RawFrom: rawFromAnthropic},
		{Role: RoleUser, Text: "keep going", Results: []ToolResult{
			{CallID: "toolu_1", Content: "hello"},
			{CallID: "toolu_2", Content: "no such file", IsError: true},
		}},
	}

	body, err := p.buildToolReq("sys", turns, testTools)
	if err != nil {
		t.Fatalf("buildToolReq: %v", err)
	}
	var req struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(req.Messages) != 3 {
		t.Fatalf("got %d messages, want 3", len(req.Messages))
	}
	if string(req.Messages[0].Content) != `"q"` {
		t.Errorf("user content = %s, want a plain string", req.Messages[0].Content)
	}
	if string(req.Messages[1].Content) != string(raw) {
		t.Errorf("assistant content = %s, want raw verbatim", req.Messages[1].Content)
	}

	var blocks []anthropicBlock
	if err := json.Unmarshal(req.Messages[2].Content, &blocks); err != nil {
		t.Fatalf("unmarshal results: %v", err)
	}
	if req.Messages[2].Role != RoleUser || len(blocks) != 3 {
		t.Fatalf("results message = %s, want 2 tool_results + text in one user message", req.Messages[2].Content)
	}
	if blocks[0].Type != "tool_result" || blocks[0].ToolUseID != "toolu_1" || blocks[0].IsError {
		t.Errorf("block 0 = %+v", blocks[0])
	}
	if blocks[1].ToolUseID != "toolu_2" || !blocks[1].IsError {
		t.Errorf("block 1 = %+v, want is_error", blocks[1])
	}
	if blocks[2].Type != "text" {
		t.Errorf("block 2 = %+v, want the text last", blocks[2])
	}
}

// Raw from another provider is ignored; the turn is rebuilt from Text and
// Calls, with an empty input sent as {}.
func TestBuildToolReqRebuildsForeignRaw(t *testing.T) {
	p := newTestProvider(t, "", nil)
	turns := []Turn{
		{Role: RoleUser, Text: "q"},
		{Role: RoleAssistant, Text: "ok", Calls: []ToolCall{{ID: "call_1", Name: "list"}},
			Raw: json.RawMessage(`{"openai":"stuff"}`), RawFrom: "openai"},
	}
	body, err := p.buildToolReq("", turns, nil)
	if err != nil {
		t.Fatalf("buildToolReq: %v", err)
	}
	s := string(body)
	if strings.Contains(s, "openai") {
		t.Errorf("foreign Raw was sent: %s", s)
	}
	want := `[{"type":"text","text":"ok"},{"type":"tool_use","id":"call_1","name":"list","input":{}}]`
	if !strings.Contains(s, want) {
		t.Errorf("body = %s, want assistant content %s", s, want)
	}
}

func TestBuildToolReqRejectsBadTurns(t *testing.T) {
	p := newTestProvider(t, "", nil)
	bad := map[string]Turn{
		"results on assistant": {Role: RoleAssistant, Text: "x", Results: []ToolResult{{CallID: "a"}}},
		"calls on user":        {Role: RoleUser, Calls: []ToolCall{{ID: "a"}}},
		"empty assistant":      {Role: RoleAssistant},
		"unknown role":         {Role: "tool"},
	}
	for name, turn := range bad {
		if _, err := p.buildToolReq("", []Turn{turn}, nil); err == nil {
			t.Errorf("%s: expected an error, got nil", name)
		}
	}
}

func TestAnthropicStop(t *testing.T) {
	tests := map[string]string{
		"end_turn":                      StopEnd,
		"stop_sequence":                 StopEnd,
		"tool_use":                      StopToolUse,
		"max_tokens":                    StopMaxTokens,
		"model_context_window_exceeded": StopContextFull,
		"refusal":                       StopRefusal,
		"pause_turn":                    "pause_turn",
	}
	for in, want := range tests {
		if got := anthropicStop(in); got != want {
			t.Errorf("anthropicStop(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestAnthropicContextFull(t *testing.T) {
	reply := func(status int, body string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			io.Copy(io.Discard, r.Body)
			w.WriteHeader(status)
			io.WriteString(w, body)
		}
	}
	for _, msg := range []string{
		"prompt is too long: 215000 tokens > 200000 maximum",
		"input length and `max_tokens` exceed context limit: 190000 + 16000 > 200000, decrease input length or `max_tokens` and try again",
	} {
		p := newTestProvider(t, "", reply(http.StatusBadRequest,
			`{"type": "error", "error": {"type": "invalid_request_error", "message": "`+msg+`"}}`))
		_, err := p.ChatTools(context.Background(), "", []Turn{{Role: RoleUser, Text: "q"}}, nil)
		if !errors.Is(err, ErrContextFull) || !strings.Contains(err.Error(), msg) {
			t.Errorf("%q: err = %v, want ErrContextFull with Anthropic's message", msg, err)
		}
	}

	p := newTestProvider(t, "", reply(http.StatusBadRequest,
		`{"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens: Field required"}}`))
	if _, err := p.ChatTools(context.Background(), "", []Turn{{Role: RoleUser, Text: "q"}}, nil); err == nil || errors.Is(err, ErrContextFull) {
		t.Errorf("other error: err = %v, want an error that isn't ErrContextFull", err)
	}
}
