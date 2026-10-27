// Package llm provides a shared LLM provider interface and factory.
// It is importable by lore, c2, brief, and arc.
package llm

import (
	"context"
	"errors"
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

// ErrContextFull is returned when the prompt doesn't fit the model's context
// window: by Anthropic and OpenAI, and by Ollama with ProviderConfig.NoTruncate.
// The error includes the provider's message.
var ErrContextFull = errors.New("prompt longer than the context window")

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
	NumCtx          int           // Ollama only: context window in tokens (options.num_ctx); 0 = Ollama's default, often 4096

	// Ollama only: a prompt longer than the context window is an error
	// (ErrContextFull) instead of being cut from the start, and generation
	// stops at the window instead of dropping the oldest tokens.
	NoTruncate bool
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
		return newOllamaProvider(cfg)
	default:
		return nil, fmt.Errorf("llm: unknown provider %q", cfg.Provider)
	}
}
