package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"
)

// OllamaProvider implements Provider for locally-hosted Ollama models.
type OllamaProvider struct {
	host  string
	model string
	http  *http.Client
}

func newOllamaProvider(host, model string) (*OllamaProvider, error) {
	host = strings.TrimRight(strings.TrimSpace(host), "/")
	if host == "" {
		host = "http://localhost:11434"
	}
	if strings.TrimSpace(model) == "" {
		return nil, errors.New("ollama: model is empty")
	}
	return &OllamaProvider{
		host:  host,
		model: model,
		http: &http.Client{
			Timeout: 0,
			Transport: &http.Transport{
				Proxy:             http.ProxyFromEnvironment,
				ForceAttemptHTTP2: false,
				// Local models (especially larger ones) can take >30s on cold start
				// before producing the first response byte. 120s is generous enough
				// for most hardware while still catching genuine hangs.
				ResponseHeaderTimeout: 120 * time.Second,
			},
		},
	}, nil
}

func (p *OllamaProvider) Name() string { return "ollama:" + p.model }

type ollamaMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaChatReq struct {
	Model    string      `json:"model"`
	Messages []ollamaMsg `json:"messages"`
	Stream   bool        `json:"stream"`
	Think    bool        `json:"think"` // false disables thinking/reasoning mode (e.g. Qwen3)
}

type ollamaChatResp struct {
	Message         ollamaMsg `json:"message"`
	Done            bool      `json:"done"`
	Error           string    `json:"error,omitempty"`
	PromptEvalCount int       `json:"prompt_eval_count"`
	EvalCount       int       `json:"eval_count"`
}

func (p *OllamaProvider) buildReq(systemPrompt string, messages []Message, stream bool) ([]byte, error) {
	reqMsgs := make([]ollamaMsg, 0, 1+len(messages))
	if sp := strings.TrimSpace(systemPrompt); sp != "" {
		reqMsgs = append(reqMsgs, ollamaMsg{Role: "system", Content: sp})
	}
	for _, m := range messages {
		role := strings.ToLower(strings.TrimSpace(m.Role))
		if role == "" {
			role = "user"
		}
		reqMsgs = append(reqMsgs, ollamaMsg{Role: role, Content: m.Content})
	}
	return json.Marshal(ollamaChatReq{Model: p.model, Messages: reqMsgs, Stream: stream})
}

func (p *OllamaProvider) Chat(ctx context.Context, systemPrompt string, messages []Message) (string, Usage, error) {
	body, err := p.buildReq(systemPrompt, messages, false)
	if err != nil {
		return "", Usage{}, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", p.host+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return "", Usage{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.http.Do(req)
	if err != nil {
		return "", Usage{}, err
	}
	defer resp.Body.Close()

	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", Usage{}, err
	}
	var out ollamaChatResp
	if err := json.Unmarshal(b, &out); err != nil {
		return "", Usage{}, err
	}
	if out.Error != "" {
		return "", Usage{}, errors.New(out.Error)
	}
	u := Usage{
		InputTokens:  out.PromptEvalCount,
		OutputTokens: out.EvalCount,
	}
	return out.Message.Content, u, nil
}

func (p *OllamaProvider) ChatStream(
	ctx context.Context,
	systemPrompt string,
	messages []Message,
	onDelta func(string) error,
) (string, Usage, error) {
	body, err := p.buildReq(systemPrompt, messages, true)
	if err != nil {
		return "", Usage{}, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", p.host+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return "", Usage{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.http.Do(req)
	if err != nil {
		return "", Usage{}, err
	}
	defer resp.Body.Close()

	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64*1024), 1024*1024)

	var sb strings.Builder
	var u Usage
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var chunk ollamaChatResp
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			continue
		}
		if chunk.Error != "" {
			return sb.String(), u, errors.New(chunk.Error)
		}
		d := chunk.Message.Content
		if d != "" {
			sb.WriteString(d)
			if onDelta != nil {
				if err := onDelta(d); err != nil {
					return sb.String(), u, err
				}
			}
		}
		if chunk.Done {
			u.InputTokens = chunk.PromptEvalCount
			u.OutputTokens = chunk.EvalCount
			break
		}
	}
	if err := sc.Err(); err != nil {
		return sb.String(), u, err
	}
	return sb.String(), u, nil
}
