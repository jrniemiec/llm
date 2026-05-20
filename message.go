package llm

import "time"

const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleNote      = "note" // personal note, never sent to LLM
)

// Message is a single turn in a conversation.
type Message struct {
	Role    string    `json:"role"`
	Content string    `json:"content"`
	Time    time.Time `json:"time,omitempty"`
	Profile string    `json:"profile,omitempty"` // profile code that produced this message (assistant only)
}

// History is the full append-only message log for a topic.
type History struct {
	Msgs []Message `json:"messages"`
}

func NewHistory() *History {
	return &History{Msgs: []Message{}}
}

func (h *History) Append(role, content string) {
	h.Msgs = append(h.Msgs, Message{
		Role:    role,
		Content: content,
		Time:    time.Now(),
	})
}

// AppendAssistant appends an assistant message with an explicit timestamp and
// profile code. The caller must pass time.Now() captured once so that the
// usage log entry can share the exact same timestamp for later matching.
func (h *History) AppendAssistant(content, profile string, ts time.Time) {
	h.Msgs = append(h.Msgs, Message{
		Role:    RoleAssistant,
		Content: content,
		Time:    ts,
		Profile: profile,
	})
}

// NonNoteMsgs returns all messages except notes, for use in LLM context building.
func (h *History) NonNoteMsgs() []Message {
	out := make([]Message, 0, len(h.Msgs))
	for _, m := range h.Msgs {
		if m.Role != RoleNote {
			out = append(out, m)
		}
	}
	return out
}

// ToMessages returns messages covering the last maxUserMessages user turns,
// excluding note messages which are never sent to the LLM.
func (h *History) ToMessages(maxUserMessages int) []Message {
	if h == nil || len(h.Msgs) == 0 {
		return nil
	}
	userCount := 0
	start := 0
	for i := len(h.Msgs) - 1; i >= 0; i-- {
		if h.Msgs[i].Role == RoleUser {
			userCount++
			if userCount >= maxUserMessages {
				start = i
				break
			}
		}
	}
	msgs := h.Msgs[start:]
	out := make([]Message, 0, len(msgs))
	for _, m := range msgs {
		if m.Role != RoleNote {
			out = append(out, m)
		}
	}
	return out
}
