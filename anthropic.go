package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const (
	anthropicAPIBase = "https://api.anthropic.com/v1/messages"
	anthropicVersion = "2023-06-01"
	defaultMaxOutput = 4096
)

// AnthropicProvider implements Provider for Anthropic Claude models.
type AnthropicProvider struct {
	apiKey          string
	model           string
	maxOutputTokens int
	http            *http.Client
}

// New constructs an AnthropicProvider. apiKey must be non-empty.
func newAnthropicProvider(apiKey, model string, maxOutputTokens int) (*AnthropicProvider, error) {
	if strings.TrimSpace(apiKey) == "" {
		return nil, errors.New("anthropic: apiKey is empty")
	}
	if maxOutputTokens <= 0 {
		maxOutputTokens = defaultMaxOutput
	}
	return &AnthropicProvider{
		apiKey:          apiKey,
		model:           model,
		maxOutputTokens: maxOutputTokens,
		http: &http.Client{
			Timeout: 0,
			Transport: &http.Transport{
				Proxy:                 http.ProxyFromEnvironment,
				ForceAttemptHTTP2:     false,
				ResponseHeaderTimeout: 30 * time.Second,
			},
		},
	}, nil
}

func (p *AnthropicProvider) Name() string { return "anthropic:" + p.model }

type anthropicMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicReq struct {
	Model     string         `json:"model"`
	MaxTokens int            `json:"max_tokens"`
	System    string         `json:"system,omitempty"`
	Messages  []anthropicMsg `json:"messages"`
	Stream    bool           `json:"stream,omitempty"`
}

type anthropicContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthropicResp struct {
	Content []anthropicContent `json:"content"`
	Usage   anthropicUsage     `json:"usage"`
	Error   *anthropicError    `json:"error,omitempty"`
}

type anthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type anthropicStreamEvent struct {
	Type  string `json:"type"`
	Delta *struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"delta,omitempty"`
	Message *struct {
		Usage anthropicUsage `json:"usage"`
	} `json:"message,omitempty"`
	Usage *anthropicUsage `json:"usage,omitempty"`
	Error *anthropicError `json:"error,omitempty"`
}

func (p *AnthropicProvider) buildReq(systemPrompt string, messages []Message, stream bool) ([]byte, error) {
	msgs := make([]anthropicMsg, 0, len(messages))
	for _, m := range messages {
		role := strings.ToLower(strings.TrimSpace(m.Role))
		if role == "system" {
			continue
		}
		if role == "" {
			role = "user"
		}
		msgs = append(msgs, anthropicMsg{Role: role, Content: m.Content})
	}
	r := anthropicReq{
		Model:     p.model,
		MaxTokens: p.maxOutputTokens,
		Messages:  msgs,
		Stream:    stream,
	}
	if sp := strings.TrimSpace(systemPrompt); sp != "" {
		r.System = sp
	}
	return json.Marshal(r)
}

func (p *AnthropicProvider) newRequest(ctx context.Context, body []byte) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, "POST", anthropicAPIBase, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", p.apiKey)
	req.Header.Set("anthropic-version", anthropicVersion)
	return req, nil
}

func (p *AnthropicProvider) checkHTTPError(resp *http.Response) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	b, _ := io.ReadAll(resp.Body)
	var e struct {
		Error anthropicError `json:"error"`
	}
	if json.Unmarshal(b, &e) == nil && e.Error.Message != "" {
		return fmt.Errorf("anthropic API error %d: %s", resp.StatusCode, e.Error.Message)
	}
	return fmt.Errorf("anthropic API error %d: %s", resp.StatusCode, string(b))
}

func (p *AnthropicProvider) Chat(ctx context.Context, systemPrompt string, messages []Message) (string, Usage, error) {
	body, err := p.buildReq(systemPrompt, messages, false)
	if err != nil {
		return "", Usage{}, err
	}
	req, err := p.newRequest(ctx, body)
	if err != nil {
		return "", Usage{}, err
	}
	resp, err := p.http.Do(req)
	if err != nil {
		return "", Usage{}, err
	}
	defer resp.Body.Close()
	if err := p.checkHTTPError(resp); err != nil {
		return "", Usage{}, err
	}
	var out anthropicResp
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", Usage{}, err
	}
	if out.Error != nil {
		return "", Usage{}, errors.New(out.Error.Message)
	}
	if len(out.Content) == 0 {
		return "", Usage{}, errors.New("anthropic: empty response content")
	}
	u := Usage{
		InputTokens:  out.Usage.InputTokens,
		OutputTokens: out.Usage.OutputTokens,
	}
	return out.Content[0].Text, u, nil
}

func (p *AnthropicProvider) ChatStream(
	ctx context.Context,
	systemPrompt string,
	messages []Message,
	onDelta func(string) error,
) (string, Usage, error) {
	body, err := p.buildReq(systemPrompt, messages, true)
	if err != nil {
		return "", Usage{}, err
	}
	req, err := p.newRequest(ctx, body)
	if err != nil {
		return "", Usage{}, err
	}
	req.Header.Set("Accept", "text/event-stream")

	resp, err := p.http.Do(req)
	if err != nil {
		return "", Usage{}, err
	}
	defer resp.Body.Close()
	if err := p.checkHTTPError(resp); err != nil {
		return "", Usage{}, err
	}

	var sb strings.Builder
	var u Usage
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)

	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var event anthropicStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}
		if event.Error != nil {
			return sb.String(), u, errors.New(event.Error.Message)
		}
		if event.Type == "message_start" && event.Message != nil {
			u.InputTokens = event.Message.Usage.InputTokens
		}
		if event.Type == "message_delta" && event.Usage != nil {
			u.OutputTokens = event.Usage.OutputTokens
		}
		if event.Type == "content_block_delta" && event.Delta != nil && event.Delta.Text != "" {
			sb.WriteString(event.Delta.Text)
			if onDelta != nil {
				if err := onDelta(event.Delta.Text); err != nil {
					return sb.String(), u, err
				}
			}
		}
	}
	if err := sc.Err(); err != nil {
		return sb.String(), u, err
	}
	return sb.String(), u, nil
}
