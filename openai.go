package llm

import (
	"context"
	"errors"
	"strings"

	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// OpenAIProvider implements Provider for OpenAI models.
type OpenAIProvider struct {
	client openai.Client
	model  openai.ChatModel
}

func newOpenAIProvider(apiKey, model string) (*OpenAIProvider, error) {
	if strings.TrimSpace(apiKey) == "" {
		return nil, errors.New("openai: apiKey is empty")
	}
	return &OpenAIProvider{
		client: openai.NewClient(option.WithAPIKey(apiKey)),
		model:  openai.ChatModel(model),
	}, nil
}

func (p *OpenAIProvider) Name() string { return "openai:" + string(p.model) }

func (p *OpenAIProvider) buildMessages(systemPrompt string, messages []Message) []openai.ChatCompletionMessageParamUnion {
	out := make([]openai.ChatCompletionMessageParamUnion, 0, 1+len(messages))
	if sp := strings.TrimSpace(systemPrompt); sp != "" {
		out = append(out, openai.SystemMessage(sp))
	}
	for _, m := range messages {
		switch strings.ToLower(strings.TrimSpace(m.Role)) {
		case "system":
			out = append(out, openai.SystemMessage(m.Content))
		case "assistant":
			out = append(out, openai.AssistantMessage(m.Content))
		default:
			out = append(out, openai.UserMessage(m.Content))
		}
	}
	return out
}

func (p *OpenAIProvider) Chat(ctx context.Context, systemPrompt string, messages []Message) (string, Usage, error) {
	msgs := p.buildMessages(systemPrompt, messages)
	resp, err := p.client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:    p.model,
		Messages: msgs,
	})
	if err != nil {
		return "", Usage{}, err
	}
	if len(resp.Choices) == 0 {
		return "", Usage{}, errors.New("openai: empty response choices")
	}
	u := Usage{
		InputTokens:  int(resp.Usage.PromptTokens),
		OutputTokens: int(resp.Usage.CompletionTokens),
	}
	return resp.Choices[0].Message.Content, u, nil
}

func (p *OpenAIProvider) ChatStream(
	ctx context.Context,
	systemPrompt string,
	messages []Message,
	onDelta func(string) error,
) (string, Usage, error) {
	msgs := p.buildMessages(systemPrompt, messages)
	stream := p.client.Chat.Completions.NewStreaming(ctx, openai.ChatCompletionNewParams{
		Model:         p.model,
		Messages:      msgs,
		StreamOptions: openai.ChatCompletionStreamOptionsParam{IncludeUsage: openai.Bool(true)},
	})

	var sb strings.Builder
	var u Usage
	for stream.Next() {
		chunk := stream.Current()
		if chunk.Usage.PromptTokens > 0 || chunk.Usage.CompletionTokens > 0 {
			u.InputTokens = int(chunk.Usage.PromptTokens)
			u.OutputTokens = int(chunk.Usage.CompletionTokens)
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		d := chunk.Choices[0].Delta.Content
		if d == "" {
			continue
		}
		sb.WriteString(d)
		if onDelta != nil {
			if err := onDelta(d); err != nil {
				return sb.String(), u, err
			}
		}
	}
	if err := stream.Err(); err != nil {
		return sb.String(), u, err
	}
	return sb.String(), u, nil
}
