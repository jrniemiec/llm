package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newTestOllama builds an Ollama provider pointed at a stub server and records
// each request body.
func newTestOllama(t *testing.T, status int, respBody string) (*OllamaProvider, *[]ollamaToolReq) {
	t.Helper()
	var reqs []ollamaToolReq
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ollamaToolReq
		json.NewDecoder(r.Body).Decode(&req)
		reqs = append(reqs, req)
		w.WriteHeader(status)
		io.WriteString(w, respBody)
	}))
	t.Cleanup(srv.Close)
	p, err := newOllamaProvider(ProviderConfig{Host: srv.URL, Model: "qwen-test"})
	if err != nil {
		t.Fatalf("newOllamaProvider: %v", err)
	}
	return p, &reqs
}

// Shape taken from a real Ollama 0.31 reply.
const ollamaToolCallResp = `{
	"model": "qwen-test",
	"message": {
		"role": "assistant",
		"content": "",
		"thinking": "two cities, two calls",
		"tool_calls": [
			{"id": "call_a", "function": {"index": 0, "name": "read_file", "arguments": {"path": "a.txt"}}},
			{"id": "call_b", "function": {"index": 1, "name": "read_file", "arguments": {"path": "b.txt"}}}
		]
	},
	"done": true,
	"done_reason": "stop",
	"prompt_eval_count": 282,
	"eval_count": 53
}`

func TestOllamaChatToolsParsesReply(t *testing.T) {
	p, reqs := newTestOllama(t, http.StatusOK, ollamaToolCallResp)

	reply, err := p.ChatTools(context.Background(), "sys", []Turn{{Role: RoleUser, Text: "q"}}, testTools)
	if err != nil {
		t.Fatalf("ChatTools: %v", err)
	}

	req := (*reqs)[0]
	if len(req.Tools) != 1 || req.Tools[0].Type != "function" || req.Tools[0].Function.Name != "read_file" ||
		len(req.Tools[0].Function.Parameters) == 0 {
		t.Errorf("request tools = %+v", req.Tools)
	}
	if req.Stream {
		t.Error("request has stream=true, want false")
	}
	// done_reason is "stop", but the reply has calls.
	if reply.Stop != StopToolUse {
		t.Errorf("Stop = %q, want %q", reply.Stop, StopToolUse)
	}
	if len(reply.Turn.Calls) != 2 || reply.Turn.Calls[1].ID != "call_b" ||
		string(reply.Turn.Calls[0].Input) != `{"path": "a.txt"}` {
		t.Errorf("Calls = %+v", reply.Turn.Calls)
	}
	if reply.Turn.RawFrom != rawFromOllama || !strings.Contains(string(reply.Turn.Raw), "two cities") {
		t.Errorf("Raw = %s (from %q), want the message with thinking", reply.Turn.Raw, reply.Turn.RawFrom)
	}
	if reply.Usage.InputTokens != 282 || reply.Usage.OutputTokens != 53 {
		t.Errorf("Usage = %+v", reply.Usage)
	}
}

// The reply goes back with its thinking, and each result becomes its own
// "tool" message carrying the call ID and tool name.
func TestOllamaChatToolsSendsResults(t *testing.T) {
	p, reqs := newTestOllama(t, http.StatusOK, ollamaToolCallResp)
	first, err := p.ChatTools(context.Background(), "sys", []Turn{{Role: RoleUser, Text: "q"}}, testTools)
	if err != nil {
		t.Fatalf("ChatTools: %v", err)
	}

	turns := []Turn{
		{Role: RoleUser, Text: "q"},
		first.Turn,
		{Role: RoleUser, Results: []ToolResult{
			{CallID: "call_a", Content: "hello"},
			{CallID: "call_b", Content: "no such file", IsError: true},
		}},
	}
	if _, err := p.ChatTools(context.Background(), "sys", turns, testTools); err != nil {
		t.Fatalf("ChatTools: %v", err)
	}

	msgs := (*reqs)[1].Messages
	var roles []string
	for _, m := range msgs {
		roles = append(roles, m.Role)
	}
	if got, want := strings.Join(roles, ","), "system,user,assistant,tool,tool"; got != want {
		t.Fatalf("roles = %s, want %s", got, want)
	}
	if msgs[2].Thinking != "two cities, two calls" || len(msgs[2].ToolCalls) != 2 {
		t.Errorf("assistant = %+v, want thinking and 2 calls", msgs[2])
	}
	failed := msgs[4]
	if failed.ToolCallID != "call_b" || failed.ToolName != "read_file" || failed.Content != "Error: no such file" {
		t.Errorf("failed result = %+v", failed)
	}
}

// Older Ollama versions send no call IDs; the provider makes unique ones and
// puts them in Raw too, so the replayed turn matches its results.
func TestOllamaFillsMissingCallIDs(t *testing.T) {
	resp := `{"message": {"role": "assistant", "content": "", "tool_calls": [
		{"function": {"name": "read_file", "arguments": {"path": "a.txt"}}},
		{"function": {"name": "read_file", "arguments": {"path": "b.txt"}}}
	]}, "done_reason": "stop"}`
	p, _ := newTestOllama(t, http.StatusOK, resp)

	reply, err := p.ChatTools(context.Background(), "", []Turn{{Role: RoleUser, Text: "q"}}, testTools)
	if err != nil {
		t.Fatalf("ChatTools: %v", err)
	}
	a, b := reply.Turn.Calls[0].ID, reply.Turn.Calls[1].ID
	if a == "" || b == "" || a == b {
		t.Fatalf("IDs = %q, %q, want two different non-empty IDs", a, b)
	}
	if !strings.Contains(string(reply.Turn.Raw), a) || !strings.Contains(string(reply.Turn.Raw), b) {
		t.Errorf("Raw = %s, want it to contain the generated IDs", reply.Turn.Raw)
	}
}

// Raw from another provider is ignored; the turn is rebuilt from Text and
// Calls, with empty arguments sent as {}.
func TestOllamaRebuildsForeignRaw(t *testing.T) {
	p, _ := newTestOllama(t, http.StatusOK, "")
	turns := []Turn{
		{Role: RoleUser, Text: "q"},
		{Role: RoleAssistant, Text: "ok", Calls: []ToolCall{{ID: "toolu_1", Name: "list"}},
			Raw: json.RawMessage(`[{"type":"thinking"}]`), RawFrom: rawFromAnthropic},
	}
	body, err := p.buildToolReq("", turns, nil)
	if err != nil {
		t.Fatalf("buildToolReq: %v", err)
	}
	var req ollamaToolReq
	json.Unmarshal(body, &req)
	asst := req.Messages[1]
	if asst.Content != "ok" || len(asst.ToolCalls) != 1 || asst.ToolCalls[0].ID != "toolu_1" ||
		string(asst.ToolCalls[0].Function.Arguments) != "{}" {
		t.Errorf("assistant = %+v", asst)
	}
}

func TestOllamaRejectsBadTurns(t *testing.T) {
	p, _ := newTestOllama(t, http.StatusOK, "")
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

// A model without tool support gets a 400 with an error message.
func TestOllamaReportsAPIError(t *testing.T) {
	p, _ := newTestOllama(t, http.StatusBadRequest, `{"error":"gemma3:4b does not support tools"}`)
	_, err := p.ChatTools(context.Background(), "", []Turn{{Role: RoleUser, Text: "q"}}, testTools)
	if err == nil || !strings.Contains(err.Error(), "does not support tools") {
		t.Errorf("err = %v, want the API error message", err)
	}
}

func TestOllamaStop(t *testing.T) {
	tests := []struct {
		reason   string
		hasCalls bool
		want     string
	}{
		{"stop", false, StopEnd},
		{"", false, StopEnd},
		{"stop", true, StopToolUse},
		{"length", true, StopMaxTokens},
		{"unload", false, "unload"},
	}
	for _, tt := range tests {
		if got := ollamaStop(tt.reason, tt.hasCalls); got != tt.want {
			t.Errorf("ollamaStop(%q, %v) = %q, want %q", tt.reason, tt.hasCalls, got, tt.want)
		}
	}
}

// NumCtx and NoTruncate go into every request; unset, the request has none
// of them, as before.
func TestOllamaContextSettings(t *testing.T) {
	p, reqs := newTestOllama(t, http.StatusOK, ollamaToolCallResp)
	if _, err := p.ChatTools(context.Background(), "", []Turn{{Role: RoleUser, Text: "q"}}, nil); err != nil {
		t.Fatal(err)
	}
	if req := (*reqs)[0]; req.Options != nil || req.Truncate != nil || req.Shift != nil {
		t.Errorf("unset: options %v, truncate %v, shift %v; want none", req.Options, req.Truncate, req.Shift)
	}

	p.numCtx, p.noTruncate = 32768, true
	if _, err := p.ChatTools(context.Background(), "", []Turn{{Role: RoleUser, Text: "q"}}, nil); err != nil {
		t.Fatal(err)
	}
	req := (*reqs)[1]
	if req.Options == nil || req.Options.NumCtx != 32768 || req.Truncate == nil || *req.Truncate || req.Shift == nil || *req.Shift {
		t.Errorf("set: options %v, truncate %v, shift %v; want num_ctx 32768, truncate and shift false", req.Options, req.Truncate, req.Shift)
	}
}

// The reply of Ollama 0.31 to a prompt too long with truncate false.
const ollamaContextFullResp = `{"error":"{\"error\":{\"code\":400,\"message\":\"request (8022 tokens) exceeds the available context size (1024 tokens), try increasing it\",\"type\":\"exceed_context_size_error\",\"n_prompt_tokens\":8022,\"n_ctx\":1024}}"}`

func TestOllamaContextFull(t *testing.T) {
	p, _ := newTestOllama(t, http.StatusBadRequest, ollamaContextFullResp)
	_, err := p.ChatTools(context.Background(), "", []Turn{{Role: RoleUser, Text: "q"}}, nil)
	if !errors.Is(err, ErrContextFull) || !strings.Contains(err.Error(), "request (8022 tokens) exceeds") {
		t.Errorf("err = %v, want ErrContextFull with Ollama's message", err)
	}

	p, _ = newTestOllama(t, http.StatusBadRequest, `{"error":"model not found"}`)
	if _, err := p.ChatTools(context.Background(), "", []Turn{{Role: RoleUser, Text: "q"}}, nil); err == nil || errors.Is(err, ErrContextFull) {
		t.Errorf("other error: err = %v, want an error that isn't ErrContextFull", err)
	}
}
