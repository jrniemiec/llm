// Package llm provides a shared LLM provider interface and factory.
// It is importable by lore, c2, brief, and arc.
package llm

import (
	"context"
	"fmt"
	"strings"
	"time"
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
	Timeout         time.Duration // response header timeout; 0 = provider default (120s for Anthropic)
	Think           bool          // Ollama only: enable thinking/reasoning mode (e.g. Qwen3, DeepSeek-R1)
	Thinking        string        // Anthropic only: "" (omit), ThinkingDisabled, or ThinkingAdaptive
}

// Anthropic thinking modes for ProviderConfig.Thinking. The zero value ""
// omits the `thinking` request parameter entirely, preserving the wire format
// used before adaptive thinking existed.
//
// Note this is distinct from ProviderConfig.Think, which is the Ollama
// reasoning-mode flag.
const (
	ThinkingDisabled = "disabled"
	ThinkingAdaptive = "adaptive"
)

// New constructs a Provider from a ProviderConfig.
func New(cfg ProviderConfig) (Provider, error) {
	switch strings.ToLower(strings.TrimSpace(cfg.Provider)) {
	case "anthropic":
		return newAnthropicProvider(cfg.APIKey, cfg.Model, cfg.MaxOutputTokens, cfg.Timeout, cfg.Thinking)
	case "openai":
		return newOpenAIProvider(cfg.APIKey, cfg.Model)
	case "ollama":
		return newOllamaProvider(cfg.Host, cfg.Model, cfg.Think)
	default:
		return nil, fmt.Errorf("llm: unknown provider %q", cfg.Provider)
	}
}
