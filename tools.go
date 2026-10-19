package llm

import (
	"context"
	"encoding/json"
)

// ToolProvider is implemented by providers that support tool calling.
// It is separate from Provider so that providers without tool support
// still satisfy Provider; check for it with a type assertion.
type ToolProvider interface {
	// ChatTools sends the conversation and tool definitions and returns the
	// model's reply as one assistant Turn. It does not stream.
	ChatTools(ctx context.Context, systemPrompt string, turns []Turn, tools []ToolDef) (ToolReply, error)
}

// ToolDef describes a tool the model may call.
type ToolDef struct {
	Name        string
	Description string
	InputSchema json.RawMessage // JSON Schema for the arguments, e.g. {"type":"object","properties":{...}}
}

// ToolCall is the model's request to run a tool.
type ToolCall struct {
	ID    string // assigned by the provider, e.g. "toolu_01A…"; echoed in ToolResult.CallID
	Name  string
	Input json.RawMessage // JSON object; not validated against the schema
}

// ToolResult is the outcome of one ToolCall, sent back to the model.
type ToolResult struct {
	CallID  string
	Content string
	IsError bool // the tool failed; Content describes the error
}

// Turn is one message in a tool-using conversation.
//
// A user turn carries Text, Results, or both. All results for one assistant
// turn's calls belong in the next user turn. An assistant turn carries Text,
// Calls, or both, plus Raw.
type Turn struct {
	Role    string // RoleUser or RoleAssistant
	Text    string
	Calls   []ToolCall   // assistant only
	Results []ToolResult // user only

	// Raw is the provider's original content for an assistant turn, and
	// RawFrom names the provider family that produced it (e.g. "anthropic").
	// The same family sends Raw back verbatim, which keeps blocks that Text
	// and Calls cannot represent, such as thinking blocks and their
	// signatures. Other providers ignore Raw and rebuild from Text and Calls.
	Raw     json.RawMessage
	RawFrom string
}

// Stop reasons reported in ToolReply.Stop. Providers map their own values to
// these; an unrecognised value is passed through unchanged.
const (
	StopEnd       = "end"        // the model finished its answer
	StopToolUse   = "tool_use"   // the model wants Calls run
	StopMaxTokens = "max_tokens" // cut off at the output limit; Calls may be incomplete and must not be run
	StopRefusal   = "refusal"    // the model declined to answer
)

// ToolReply is the result of ChatTools.
type ToolReply struct {
	Turn  Turn // the assistant turn, ready to append to the conversation
	Stop  string
	Usage Usage
}
