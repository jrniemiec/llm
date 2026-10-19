package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	openai "github.com/openai/openai-go/v3"
)

// rawFromOpenAI tags Turn.Raw produced by this provider.
const rawFromOpenAI = "openai"

// ChatTools implements ToolProvider using Chat Completions.
func (p *OpenAIProvider) ChatTools(ctx context.Context, systemPrompt string, turns []Turn, tools []ToolDef) (ToolReply, error) {
	params, err := p.buildToolParams(systemPrompt, turns, tools)
	if err != nil {
		return ToolReply{}, err
	}
	resp, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return ToolReply{}, err
	}
	u := Usage{
		InputTokens:  int(resp.Usage.PromptTokens),
		OutputTokens: int(resp.Usage.CompletionTokens),
	}
	if len(resp.Choices) == 0 {
		return ToolReply{Usage: u}, errors.New("openai: empty response choices")
	}
	return parseOpenAIChoice(resp.Choices[0], u), nil
}

func (p *OpenAIProvider) buildToolParams(systemPrompt string, turns []Turn, tools []ToolDef) (openai.ChatCompletionNewParams, error) {
	msgs := make([]openai.ChatCompletionMessageParamUnion, 0, 1+len(turns))
	if sp := strings.TrimSpace(systemPrompt); sp != "" {
		msgs = append(msgs, openai.SystemMessage(sp))
	}
	for i, t := range turns {
		var err error
		switch t.Role {
		case RoleSystem:
			msgs = append(msgs, openai.SystemMessage(t.Text))
		case RoleAssistant:
			var m openai.ChatCompletionMessageParamUnion
			m, err = openAIAssistant(t)
			msgs = append(msgs, m)
		case RoleUser, "":
			var ms []openai.ChatCompletionMessageParamUnion
			ms, err = openAIUser(t)
			msgs = append(msgs, ms...)
		default:
			err = fmt.Errorf("unknown role %q", t.Role)
		}
		if err != nil {
			return openai.ChatCompletionNewParams{}, fmt.Errorf("openai: turn %d: %w", i, err)
		}
	}

	defs := make([]openai.ChatCompletionToolUnionParam, len(tools))
	for i, t := range tools {
		var schema openai.FunctionParameters
		if len(t.InputSchema) > 0 {
			if err := json.Unmarshal(t.InputSchema, &schema); err != nil {
				return openai.ChatCompletionNewParams{}, fmt.Errorf("openai: tool %q: schema: %w", t.Name, err)
			}
		}
		defs[i] = openai.ChatCompletionFunctionTool(openai.FunctionDefinitionParam{
			Name:        t.Name,
			Description: openai.String(t.Description),
			Parameters:  schema,
		})
	}

	return openai.ChatCompletionNewParams{
		Model:    p.model,
		Messages: msgs,
		Tools:    defs,
	}, nil
}

// openAIAssistant replays Raw when this provider produced it (keeps fields such
// as the refusal); otherwise it rebuilds the message from Text and Calls.
func openAIAssistant(t Turn) (openai.ChatCompletionMessageParamUnion, error) {
	if len(t.Results) > 0 {
		return openai.ChatCompletionMessageParamUnion{}, errors.New("assistant turn has tool results")
	}
	if len(t.Raw) > 0 && t.RawFrom == rawFromOpenAI {
		var m openai.ChatCompletionMessage
		if err := json.Unmarshal(t.Raw, &m); err != nil {
			return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("decode raw: %w", err)
		}
		return m.ToParam(), nil
	}
	if t.Text == "" && len(t.Calls) == 0 {
		return openai.ChatCompletionMessageParamUnion{}, errors.New("empty assistant turn")
	}

	var asst openai.ChatCompletionAssistantMessageParam
	if t.Text != "" {
		asst.Content.OfString = openai.String(t.Text)
	}
	for _, c := range t.Calls {
		args := string(c.Input)
		if args == "" {
			args = "{}"
		}
		asst.ToolCalls = append(asst.ToolCalls, openai.ChatCompletionMessageToolCallUnionParam{
			OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
				ID: c.ID,
				Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
					Name:      c.Name,
					Arguments: args,
				},
			},
		})
	}
	return openai.ChatCompletionMessageParamUnion{OfAssistant: &asst}, nil
}

// openAIUser returns one "tool" message per result, then the text as a "user"
// message. OpenAI has no error flag, so a failed result is prefixed "Error: ".
func openAIUser(t Turn) ([]openai.ChatCompletionMessageParamUnion, error) {
	if len(t.Calls) > 0 {
		return nil, errors.New("user turn has tool calls")
	}
	out := make([]openai.ChatCompletionMessageParamUnion, 0, len(t.Results)+1)
	for _, r := range t.Results {
		content := r.Content
		if r.IsError {
			content = "Error: " + content
		}
		out = append(out, openai.ToolMessage(content, r.CallID))
	}
	if t.Text != "" || len(t.Results) == 0 {
		out = append(out, openai.UserMessage(t.Text))
	}
	return out, nil
}

func parseOpenAIChoice(c openai.ChatCompletionChoice, u Usage) ToolReply {
	m := c.Message
	turn := Turn{Role: RoleAssistant, Text: m.Content, RawFrom: rawFromOpenAI}
	if raw := m.RawJSON(); raw != "" {
		turn.Raw = json.RawMessage(raw)
	}
	for _, tc := range m.ToolCalls {
		if tc.Type != "function" {
			continue // custom tools are never offered
		}
		turn.Calls = append(turn.Calls, ToolCall{
			ID:    tc.ID,
			Name:  tc.Function.Name,
			Input: json.RawMessage(tc.Function.Arguments),
		})
	}

	stop := openAIStop(c.FinishReason)
	if m.Refusal != "" {
		stop = StopRefusal
		if turn.Text == "" {
			turn.Text = m.Refusal
		}
	}
	return ToolReply{Turn: turn, Stop: stop, Usage: u}
}

func openAIStop(s string) string {
	switch s {
	case "stop":
		return StopEnd
	case "tool_calls":
		return StopToolUse
	case "length":
		return StopMaxTokens
	default:
		return s
	}
}
