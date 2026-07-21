// Package llm provides a shared LLM provider interface and factory.
// It is importable by lore, c2, brief, and arc.
package llm

import (
	"context"
	"fmt"
	"strings"
)

// Usage holds token counts for a single LLM call.
type Usage struct {
	InputTokens  int
	OutputTokens int
	Estimated    bool
}

// Provider is the interface all LLM backends implement.
type Provider interface {
	Name() string
	Chat(ctx context.Context, systemPrompt string, messages []Message) (string, Usage, error)
	ChatStream(ctx context.Context, systemPrompt string, messages []Message, onDelta func(string) error) (string, Usage, error)
}

// ProviderConfig is the minimal configuration needed to construct any provider.
// Callers map their own app config type to this struct before calling New().
// API keys are passed explicitly — constructors in this module do not read env vars.
type ProviderConfig struct {
	Provider        string // "anthropic" | "openai" | "ollama"
	Model           string
	Host            string // Ollama only
	APIKey          string // resolved by the caller from env or config
	MaxOutputTokens int
	Think           bool // Ollama only: enable thinking/reasoning mode (e.g. Qwen3, DeepSeek-R1)
}

// New constructs a Provider from a ProviderConfig.
func New(cfg ProviderConfig) (Provider, error) {
	switch strings.ToLower(strings.TrimSpace(cfg.Provider)) {
	case "anthropic":
		return newAnthropicProvider(cfg.APIKey, cfg.Model, cfg.MaxOutputTokens)
	case "openai":
		return newOpenAIProvider(cfg.APIKey, cfg.Model)
	case "ollama":
		return newOllamaProvider(cfg.Host, cfg.Model, cfg.Think)
	default:
		return nil, fmt.Errorf("llm: unknown provider %q", cfg.Provider)
	}
}
