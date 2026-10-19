package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// rawFromAnthropic tags Turn.Raw produced by this provider.
const rawFromAnthropic = "anthropic"

// --- Tool-aware wire types ---

// anthropicToolMsg holds content as raw JSON: a string or a block array.
type anthropicToolMsg struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type anthropicToolReq struct {
	Model     string             `json:"model"`
	MaxTokens int                `json:"max_tokens"`
	System    string             `json:"system,omitempty"`
	Messages  []anthropicToolMsg `json:"messages"`
	Tools     []anthropicToolDef `json:"tools,omitempty"`
	Thinking  *thinkingCfg       `json:"thinking,omitempty"`
}

type anthropicToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// anthropicBlock is one content block, for both requests and responses.
type anthropicBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`        // text
	ID        string          `json:"id,omitempty"`          // tool_use
	Name      string          `json:"name,omitempty"`        // tool_use
	Input     json.RawMessage `json:"input,omitempty"`       // tool_use
	ToolUseID string          `json:"tool_use_id,omitempty"` // tool_result
	Content   string          `json:"content,omitempty"`     // tool_result
	IsError   bool            `json:"is_error,omitempty"`    // tool_result
}

type anthropicToolResp struct {
	Content    json.RawMessage `json:"content"` // kept verbatim for Turn.Raw
	StopReason string          `json:"stop_reason"`
	Usage      anthropicUsage  `json:"usage"`
	Error      *anthropicError `json:"error,omitempty"`
}

// ChatTools implements ToolProvider.
func (p *AnthropicProvider) ChatTools(ctx context.Context, systemPrompt string, turns []Turn, tools []ToolDef) (ToolReply, error) {
	body, err := p.buildToolReq(systemPrompt, turns, tools)
	if err != nil {
		return ToolReply{}, err
	}
	req, err := p.newRequest(ctx, body)
	if err != nil {
		return ToolReply{}, err
	}
	resp, err := p.http.Do(req)
	if err != nil {
		return ToolReply{}, err
	}
	defer resp.Body.Close()
	if err := p.checkHTTPError(resp); err != nil {
		return ToolReply{}, err
	}
	var out anthropicToolResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return ToolReply{}, err
	}
	if out.Error != nil {
		return ToolReply{}, errors.New(out.Error.Message)
	}
	return parseToolResp(out)
}

func (p *AnthropicProvider) buildToolReq(systemPrompt string, turns []Turn, tools []ToolDef) ([]byte, error) {
	msgs := make([]anthropicToolMsg, 0, len(turns))
	for i, t := range turns {
		var (
			content json.RawMessage
			err     error
		)
		switch t.Role {
		case RoleSystem:
			continue
		case RoleAssistant:
			content, err = assistantContent(t)
		case RoleUser, "":
			content, err = userContent(t)
		default:
			err = fmt.Errorf("unknown role %q", t.Role)
		}
		if err != nil {
			return nil, fmt.Errorf("anthropic: turn %d: %w", i, err)
		}
		role := t.Role
		if role == "" {
			role = RoleUser
		}
		msgs = append(msgs, anthropicToolMsg{Role: role, Content: content})
	}

	defs := make([]anthropicToolDef, len(tools))
	for i, t := range tools {
		defs[i] = anthropicToolDef{Name: t.Name, Description: t.Description, InputSchema: t.InputSchema}
	}

	r := anthropicToolReq{
		Model:     p.model,
		MaxTokens: p.maxOutputTokens,
		System:    strings.TrimSpace(systemPrompt),
		Messages:  msgs,
		Tools:     defs,
	}
	if p.thinking != "" {
		r.Thinking = &thinkingCfg{Type: p.thinking}
	}
	return json.Marshal(r)
}

// assistantContent sends Raw back verbatim when this provider produced it, so
// thinking blocks and their signatures survive; otherwise it rebuilds the
// blocks from Text and Calls.
func assistantContent(t Turn) (json.RawMessage, error) {
	if len(t.Results) > 0 {
		return nil, errors.New("assistant turn has tool results")
	}
	if len(t.Raw) > 0 && t.RawFrom == rawFromAnthropic {
		return t.Raw, nil
	}
	blocks := make([]anthropicBlock, 0, 1+len(t.Calls))
	if t.Text != "" {
		blocks = append(blocks, anthropicBlock{Type: "text", Text: t.Text})
	}
	for _, c := range t.Calls {
		input := c.Input
		if len(input) == 0 {
			input = json.RawMessage(`{}`)
		}
		blocks = append(blocks, anthropicBlock{Type: "tool_use", ID: c.ID, Name: c.Name, Input: input})
	}
	if len(blocks) == 0 {
		return nil, errors.New("empty assistant turn")
	}
	return json.Marshal(blocks)
}

// userContent puts all tool results in one message, ahead of any text, as the
// API requires.
func userContent(t Turn) (json.RawMessage, error) {
	if len(t.Calls) > 0 {
		return nil, errors.New("user turn has tool calls")
	}
	if len(t.Results) == 0 {
		return json.Marshal(t.Text)
	}
	blocks := make([]anthropicBlock, 0, len(t.Results)+1)
	for _, r := range t.Results {
		blocks = append(blocks, anthropicBlock{
			Type:      "tool_result",
			ToolUseID: r.CallID,
			Content:   r.Content,
			IsError:   r.IsError,
		})
	}
	if t.Text != "" {
		blocks = append(blocks, anthropicBlock{Type: "text", Text: t.Text})
	}
	return json.Marshal(blocks)
}

func parseToolResp(out anthropicToolResp) (ToolReply, error) {
	u := Usage{InputTokens: out.Usage.InputTokens, OutputTokens: out.Usage.OutputTokens}

	var blocks []anthropicBlock
	if err := json.Unmarshal(out.Content, &blocks); err != nil {
		return ToolReply{Usage: u}, fmt.Errorf("anthropic: decode content: %w", err)
	}
	turn := Turn{Role: RoleAssistant, Raw: out.Content, RawFrom: rawFromAnthropic}
	var sb strings.Builder
	for _, b := range blocks {
		switch b.Type {
		case "text":
			sb.WriteString(b.Text)
		case "tool_use":
			turn.Calls = append(turn.Calls, ToolCall{ID: b.ID, Name: b.Name, Input: b.Input})
		}
		// thinking, redacted_thinking and others live only in Raw.
	}
	turn.Text = sb.String()

	return ToolReply{Turn: turn, Stop: anthropicStop(out.StopReason), Usage: u}, nil
}

func anthropicStop(s string) string {
	switch s {
	case "end_turn", "stop_sequence":
		return StopEnd
	case "tool_use":
		return StopToolUse
	case "max_tokens":
		return StopMaxTokens
	case "refusal":
		return StopRefusal
	default:
		return s
	}
}
