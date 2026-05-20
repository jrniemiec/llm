package llm

import "fmt"

// ApproxTokens estimates token count as ceil(len(s)/4).
func ApproxTokens(s string) int {
	return (len(s) + 3) / 4
}

// FormatTokens formats a token count with K/M suffix.
func FormatTokens(n int) string {
	switch {
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	case n >= 1_000:
		return fmt.Sprintf("%.1fK", float64(n)/1_000)
	default:
		return fmt.Sprintf("%d", n)
	}
}
