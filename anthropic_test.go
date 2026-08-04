package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newTestProvider builds an Anthropic provider pointed at a stub server.
func newTestProvider(t *testing.T, thinking string, handler http.HandlerFunc) *AnthropicProvider {
	t.Helper()
	p, err := newAnthropicProvider("test-key", "claude-test", 1024, 0, thinking)
	if err != nil {
		t.Fatalf("newAnthropicProvider: %v", err)
	}
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	p.baseURL = srv.URL
	return p
}

// The thinking parameter must be omitted entirely when the mode is "", so that
// models with an untested "disabled" contract keep the original wire format.
func TestBuildReqThinkingParameter(t *testing.T) {
	tests := []struct {
		name     string
		thinking string
		wantKey  bool
		wantType string
	}{
		{"omitted by default", "", false, ""},
		{"disabled", ThinkingDisabled, true, "disabled"},
		{"adaptive", ThinkingAdaptive, true, "adaptive"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := newAnthropicProvider("test-key", "claude-test", 1024, 0, tt.thinking)
			if err != nil {
				t.Fatalf("newAnthropicProvider: %v", err)
			}
			body, err := p.buildReq("sys", []Message{{Role: RoleUser, Content: "hi"}}, false)
			if err != nil {
				t.Fatalf("buildReq: %v", err)
			}

			var got map[string]any
			if err := json.Unmarshal(body, &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			raw, ok := got["thinking"]
			if ok != tt.wantKey {
				t.Fatalf("thinking key present = %v, want %v (body: %s)", ok, tt.wantKey, body)
			}
			if !tt.wantKey {
				return
			}
			gotType := raw.(map[string]any)["type"]
			if gotType != tt.wantType {
				t.Errorf("thinking.type = %v, want %v", gotType, tt.wantType)
			}
		})
	}
}

func TestNewAnthropicProviderRejectsUnknownThinking(t *testing.T) {
	if _, err := newAnthropicProvider("k", "m", 0, 0, "enabled"); err == nil {
		t.Fatal("expected error for unknown thinking mode, got nil")
	}
}

// A thinking-enabled model leads with a thinking block; the answer is a later
// block. Returning Content[0] would yield "".
func TestChatSkipsLeadingThinkingBlock(t *testing.T) {
	p := newTestProvider(t, ThinkingAdaptive, func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"content": [
				{"type": "thinking", "thinking": ""},
				{"type": "text", "text": "the answer"}
			],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`)
	})

	got, usage, err := p.Chat(context.Background(), "sys", []Message{{Role: RoleUser, Content: "q"}})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if got != "the answer" {
		t.Errorf("Chat text = %q, want %q", got, "the answer")
	}
	if usage.InputTokens != 10 || usage.OutputTokens != 5 {
		t.Errorf("usage = %+v, want 10/5", usage)
	}
}

func TestChatConcatenatesTextBlocks(t *testing.T) {
	p := newTestProvider(t, "", func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		io.WriteString(w, `{
			"content": [
				{"type": "text", "text": "part one "},
				{"type": "text", "text": "part two"}
			],
			"usage": {"input_tokens": 1, "output_tokens": 2}
		}`)
	})

	got, _, err := p.Chat(context.Background(), "sys", []Message{{Role: RoleUser, Content: "q"}})
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if got != "part one part two" {
		t.Errorf("Chat text = %q, want %q", got, "part one part two")
	}
}

// A response truncated during thinking has no text block at all. That must be a
// diagnosable error carrying the stop reason, not a silent empty string.
func TestChatNoTextBlockReportsStopReason(t *testing.T) {
	p := newTestProvider(t, ThinkingAdaptive, func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		io.WriteString(w, `{
			"content": [{"type": "thinking", "thinking": ""}],
			"stop_reason": "max_tokens",
			"usage": {"input_tokens": 100, "output_tokens": 1024}
		}`)
	})

	_, usage, err := p.Chat(context.Background(), "sys", []Message{{Role: RoleUser, Content: "q"}})
	if err == nil {
		t.Fatal("expected an error when no text block is present, got nil")
	}
	if !strings.Contains(err.Error(), "max_tokens") {
		t.Errorf("error = %q, want it to mention the stop reason", err)
	}
	// Usage is still reported: a truncated response is billed.
	if usage.OutputTokens != 1024 {
		t.Errorf("usage.OutputTokens = %d, want 1024", usage.OutputTokens)
	}
}
