package llm

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// rawFromOllama tags Turn.Raw produced by this provider.
const rawFromOllama = "ollama"

// --- Tool-aware wire types (/api/chat) ---

type ollamaToolMsg struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	Thinking   string           `json:"thinking,omitempty"`
	ToolCalls  []ollamaToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"` // tool messages
	ToolName   string           `json:"tool_name,omitempty"`    // tool messages
}

type ollamaToolCall struct {
	ID       string `json:"id,omitempty"` // absent in older Ollama versions
	Function struct {
		Index     int             `json:"index"`
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"` // a JSON object, not a string
	} `json:"function"`
}

type ollamaToolDef struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
	} `json:"function"`
}

type ollamaToolReq struct {
	Model    string          `json:"model"`
	Messages []ollamaToolMsg `json:"messages"`
	Tools    []ollamaToolDef `json:"tools,omitempty"`
	Stream   bool            `json:"stream"`
	Think    bool            `json:"think"`
	ollamaLimits
}

type ollamaToolResp struct {
	Message         ollamaToolMsg `json:"message"`
	DoneReason      string        `json:"done_reason"`
	Error           string        `json:"error,omitempty"`
	PromptEvalCount int           `json:"prompt_eval_count"`
	EvalCount       int           `json:"eval_count"`
}

// ChatTools implements ToolProvider. The model must support tools; Ollama
// returns an error for one that does not.
func (p *OllamaProvider) ChatTools(ctx context.Context, systemPrompt string, turns []Turn, tools []ToolDef) (ToolReply, error) {
	body, err := p.buildToolReq(systemPrompt, turns, tools)
	if err != nil {
		return ToolReply{}, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", p.host+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return ToolReply{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.http.Do(req)
	if err != nil {
		return ToolReply{}, err
	}
	defer resp.Body.Close()

	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return ToolReply{}, err
	}
	var out ollamaToolResp
	if err := json.Unmarshal(b, &out); err != nil {
		if resp.StatusCode >= 300 {
			return ToolReply{}, fmt.Errorf("ollama API error %d: %s", resp.StatusCode, b)
		}
		return ToolReply{}, err
	}
	if out.Error != "" {
		return ToolReply{}, ollamaError(out.Error)
	}
	if resp.StatusCode >= 300 {
		return ToolReply{}, fmt.Errorf("ollama API error %d: %s", resp.StatusCode, b)
	}
	return parseOllamaResp(out)
}

func (p *OllamaProvider) buildToolReq(systemPrompt string, turns []Turn, tools []ToolDef) ([]byte, error) {
	msgs := make([]ollamaToolMsg, 0, 1+len(turns))
	if sp := strings.TrimSpace(systemPrompt); sp != "" {
		msgs = append(msgs, ollamaToolMsg{Role: RoleSystem, Content: sp})
	}
	names := map[string]string{} // call ID → tool name, for tool_name on results
	for i, t := range turns {
		var err error
		switch t.Role {
		case RoleSystem:
			msgs = append(msgs, ollamaToolMsg{Role: RoleSystem, Content: t.Text})
		case RoleAssistant:
			var m ollamaToolMsg
			m, err = ollamaAssistant(t)
			for _, c := range m.ToolCalls {
				names[c.ID] = c.Function.Name
			}
			msgs = append(msgs, m)
		case RoleUser, "":
			var ms []ollamaToolMsg
			ms, err = ollamaUser(t, names)
			msgs = append(msgs, ms...)
		default:
			err = fmt.Errorf("unknown role %q", t.Role)
		}
		if err != nil {
			return nil, fmt.Errorf("ollama: turn %d: %w", i, err)
		}
	}

	defs := make([]ollamaToolDef, len(tools))
	for i, t := range tools {
		defs[i].Type = "function"
		defs[i].Function.Name = t.Name
		defs[i].Function.Description = t.Description
		defs[i].Function.Parameters = t.InputSchema
	}

	return json.Marshal(ollamaToolReq{Model: p.model, Messages: msgs, Tools: defs, Think: p.think, ollamaLimits: p.limits()})
}

// ollamaAssistant replays Raw when this provider produced it (keeps the
// thinking text); otherwise it rebuilds the message from Text and Calls.
func ollamaAssistant(t Turn) (ollamaToolMsg, error) {
	if len(t.Results) > 0 {
		return ollamaToolMsg{}, errors.New("assistant turn has tool results")
	}
	if len(t.Raw) > 0 && t.RawFrom == rawFromOllama {
		var m ollamaToolMsg
		if err := json.Unmarshal(t.Raw, &m); err != nil {
			return ollamaToolMsg{}, fmt.Errorf("decode raw: %w", err)
		}
		return m, nil
	}
	if t.Text == "" && len(t.Calls) == 0 {
		return ollamaToolMsg{}, errors.New("empty assistant turn")
	}
	m := ollamaToolMsg{Role: RoleAssistant, Content: t.Text}
	for i, c := range t.Calls {
		var tc ollamaToolCall
		tc.ID = c.ID
		tc.Function.Index = i
		tc.Function.Name = c.Name
		tc.Function.Arguments = c.Input
		if len(tc.Function.Arguments) == 0 {
			tc.Function.Arguments = json.RawMessage(`{}`)
		}
		m.ToolCalls = append(m.ToolCalls, tc)
	}
	return m, nil
}

// ollamaUser returns one "tool" message per result, then the text as a "user"
// message. Ollama has no error flag, so a failed result is prefixed "Error: ".
func ollamaUser(t Turn, names map[string]string) ([]ollamaToolMsg, error) {
	if len(t.Calls) > 0 {
		return nil, errors.New("user turn has tool calls")
	}
	out := make([]ollamaToolMsg, 0, len(t.Results)+1)
	for _, r := range t.Results {
		content := r.Content
		if r.IsError {
			content = "Error: " + content
		}
		out = append(out, ollamaToolMsg{
			Role:       "tool",
			Content:    content,
			ToolCallID: r.CallID,
			ToolName:   names[r.CallID],
		})
	}
	if t.Text != "" || len(t.Results) == 0 {
		out = append(out, ollamaToolMsg{Role: RoleUser, Content: t.Text})
	}
	return out, nil
}

// parseOllamaResp fills in missing call IDs (older Ollama versions send none)
// with random ones, so IDs stay unique across replies, and does it before
// saving Raw, so a replayed turn matches its results.
func parseOllamaResp(out ollamaToolResp) (ToolReply, error) {
	u := Usage{InputTokens: out.PromptEvalCount, OutputTokens: out.EvalCount}
	m := out.Message
	turn := Turn{Role: RoleAssistant, Text: m.Content, RawFrom: rawFromOllama}
	for i := range m.ToolCalls {
		tc := &m.ToolCalls[i]
		if tc.ID == "" {
			tc.ID = "call_" + rand.Text()[:12]
		}
		turn.Calls = append(turn.Calls, ToolCall{ID: tc.ID, Name: tc.Function.Name, Input: tc.Function.Arguments})
	}
	raw, err := json.Marshal(m)
	if err != nil {
		return ToolReply{Usage: u}, fmt.Errorf("ollama: encode raw: %w", err)
	}
	turn.Raw = raw

	return ToolReply{Turn: turn, Stop: ollamaStop(out.DoneReason, len(turn.Calls) > 0), Usage: u}, nil
}

// ollamaStop maps done_reason. Ollama reports "stop" even when the reply has
// tool calls, so their presence decides StopToolUse.
func ollamaStop(reason string, hasCalls bool) string {
	switch {
	case reason == "length":
		return StopMaxTokens
	case hasCalls:
		return StopToolUse
	case reason == "stop" || reason == "":
		return StopEnd
	default:
		return reason
	}
}
