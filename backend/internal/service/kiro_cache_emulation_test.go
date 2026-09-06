package service

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/anthropictokenizer"
	kiropkg "github.com/Wei-Shaw/sub2api/internal/pkg/kiro"
	"github.com/stretchr/testify/require"
)

func TestKiroCacheEmulationGroupDefaultsAndNonKiro(t *testing.T) {
	kiro := &Group{Platform: PlatformKiro, KiroCacheEmulationEnabled: true, KiroCacheEmulationRatio: 0.5}
	if !kiro.EffectiveKiroCacheEmulationEnabled() {
		t.Fatal("kiro group should enable cache emulation")
	}
	if got := kiro.EffectiveKiroCacheEmulationRatio(); got != 0.5 {
		t.Fatalf("ratio = %v, want 0.5", got)
	}
	nonKiro := &Group{Platform: PlatformAnthropic, KiroCacheEmulationEnabled: true, KiroCacheEmulationRatio: 1}
	NormalizeGroupRuntimeFields(nonKiro)
	if nonKiro.KiroCacheEmulationEnabled || nonKiro.KiroCacheEmulationRatio != 0 {
		t.Fatalf("non-kiro fields were not normalized: %+v", nonKiro)
	}
}

func TestKiroInputTokenEstimateIgnoresClientMetadata(t *testing.T) {
	bodyWithoutMetadata := []byte(`{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"hello world"}]}`)
	bodyWithMetadata := []byte(`{"model":"claude-sonnet-4-6","metadata":{"input_tokens":999999},"messages":[{"role":"user","content":"hello world"}]}`)
	withoutMetadata := estimateKiroInputTokens(context.Background(), bodyWithoutMetadata)
	withMetadata := estimateKiroInputTokens(context.Background(), bodyWithMetadata)
	if withMetadata == 999999 {
		t.Fatal("client metadata.input_tokens must not be trusted")
	}
	if withMetadata <= 0 || withoutMetadata <= 0 || withMetadata > withoutMetadata*2 {
		t.Fatalf("unexpected estimates without=%d with=%d", withoutMetadata, withMetadata)
	}
}

func TestKiroTokenCountersMatchReferenceRules(t *testing.T) {
	if got := anthropictokenizer.CountTokens("abc def"); got != 1 {
		t.Fatalf("english tokens = %d, want 1", got)
	}
	if got := anthropictokenizer.CountTokens("你好世界"); got != 1 {
		t.Fatalf("cjk tokens = %d, want 1", got)
	}
	if kiroTokensPerTool != 150 {
		t.Fatalf("tool tokens = %d, want 150", kiroTokensPerTool)
	}
	if got := countKiroMessageContentTokens(context.Background(), map[string]any{"thinking": "abc def"}); got != 1 {
		t.Fatalf("thinking tokens = %d, want 1", got)
	}
	if got := countKiroMessageContentTokens(context.Background(), map[string]any{"input": map[string]any{"path": "/tmp/a.txt"}}); got <= 0 {
		t.Fatalf("tool input tokens should be positive, got %d", got)
	}
	if got := countKiroMessageContentTokens(context.Background(), map[string]any{"content": []any{map[string]any{"text": "abc"}, map[string]any{"text": "你好"}}}); got != 2 {
		t.Fatalf("tool result content tokens = %d, want 2", got)
	}
}

func TestKiroInputTokenEstimateSeparatesVisualTokensFromBase64(t *testing.T) {
	dataURL := kiroPNGDataURL(t, 512, 512, color.RGBA{R: 37, G: 89, B: 151, A: 255})
	body := []byte(fmt.Sprintf(`{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":%q}}]}]}`, strings.TrimPrefix(dataURL, "data:image/png;base64,")))

	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	sanitized, imageTokens := sanitizeKiroImagesForTokenEstimate(context.Background(), payload["messages"])
	canonical, err := canonicalJSON(sanitized)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(canonical, []byte(strings.TrimPrefix(dataURL, "data:image/png;base64,"))) {
		t.Fatal("sanitized token payload must not retain image base64")
	}
	if imageTokens != 350 {
		t.Fatalf("image tokens = %d, want 350", imageTokens)
	}

	got := estimateKiroInputTokens(context.Background(), body)
	want := anthropictokenizer.CountTokens("describe") + imageTokens
	if got < want || got > want+50 {
		t.Fatalf("input token estimate = %d, expected visual-aware estimate near %d", got, want)
	}
}

func TestKiroImageTokenSourcesSupportAnthropicAndOpenAIShapes(t *testing.T) {
	dataURL := kiroPNGDataURL(t, 200, 200, color.RGBA{A: 255})
	base64Data := strings.TrimPrefix(dataURL, "data:image/png;base64,")
	tests := []map[string]any{
		{"type": "image", "source": map[string]any{"type": "base64", "media_type": "image/png", "data": base64Data}},
		{"type": "image_url", "image_url": map[string]any{"url": dataURL}},
		{"type": "input_image", "image_url": dataURL},
	}
	for _, block := range tests {
		if got := countKiroMessageContentTokens(context.Background(), block); got != 54 {
			t.Fatalf("image block %#v tokens = %d, want 54", block, got)
		}
	}
}

func resetKiroCacheTracker() { globalKiroDynamicTracker = newKiroDynamicTracker(time.Now) }

func kiroPNGDataURL(t *testing.T, width, height int, fill color.RGBA) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.SetRGBA(x, y, fill)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

func kiroCacheImageRequestBody(t *testing.T, text string, fill color.RGBA) []byte {
	t.Helper()
	dataURL := kiroPNGDataURL(t, 200, 200, fill)
	return []byte(fmt.Sprintf(`{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":%q},{"type":"image","source":{"type":"base64","media_type":"image/png","data":%q},"cache_control":{"type":"ephemeral"}}]}]}`, text, strings.TrimPrefix(dataURL, "data:image/png;base64,")))
}

// kiroMinimumCacheableTokens 决定「一个前缀至少要多少 token 才值得记进缓存」，直接
// 影响模拟出的 cache_creation / cache_read 分布，进而影响账单。这里把每个 Kiro 实际
// 暴露的模型的阈值钉死：GPT-5.6 三兄弟必须是 1024（对齐 OpenAI 官方最小缓存粒度），
// opus 系必须是 4096。特例断言写字面量、默认档断言写常量名，这样任何一侧被单独改动
// 都会让测试失败，而不是静默漂移。
func TestKiroMinimumCacheableTokens(t *testing.T) {
	t.Parallel()

	// GPT-5.6 是本用例的核心诉求：1024 是显式契约，不是「恰好等于默认档」。
	for _, model := range []string{"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"} {
		require.Equal(t, 1024, kiroMinimumCacheableTokens(model), model)
	}

	// opus 系走 4096，含 -thinking 变体与带日期后缀的 4.5。
	for _, model := range []string{
		"claude-opus-5", "claude-opus-5-thinking",
		"claude-opus-4-8", "claude-opus-4-8-thinking",
		"claude-opus-4-5-20251101", "claude-opus-4-5-20251101-thinking",
	} {
		require.Equal(t, 4096, kiroMinimumCacheableTokens(model), model)
	}

	// 非 opus 的 Claude 走默认档。断言常量而非字面量：默认档本身允许调整，
	// 但调整时必须同步 GPT 的显式 case（GPT 那侧断言的是字面量 1024）。
	for _, model := range []string{
		"claude-sonnet-5", "claude-sonnet-4-6", "claude-haiku-4-5-20251001",
	} {
		require.Equal(t, kiroCacheMinTokensDefault, kiroMinimumCacheableTokens(model), model)
	}

	// 遍历 Kiro 实际暴露的全量模型，确保没有未归类的漏网之鱼。上游新增模型时，
	// 这里会立刻因缺少 expected 条目而失败，迫使新模型被显式归类。
	expected := map[string]int{
		"gpt-5.6-sol":                         1024,
		"gpt-5.6-terra":                       1024,
		"gpt-5.6-luna":                        1024,
		"claude-opus-4-8":                     4096,
		"claude-opus-4-8-thinking":            4096,
		"claude-opus-4-7":                     4096,
		"claude-opus-4-7-thinking":            4096,
		"claude-opus-4-6":                     4096,
		"claude-opus-4-6-thinking":            4096,
		"claude-opus-5":                       4096,
		"claude-opus-5-thinking":              4096,
		"claude-opus-4-5-20251101":            4096,
		"claude-opus-4-5-20251101-thinking":   4096,
		"claude-sonnet-5":                     1024,
		"claude-sonnet-5-thinking":            1024,
		"claude-sonnet-4-6":                   1024,
		"claude-sonnet-4-6-thinking":          1024,
		"claude-sonnet-4-5-20250929":          1024,
		"claude-sonnet-4-5-20250929-thinking": 1024,
		"claude-haiku-4-5-20251001":           1024,
		"claude-haiku-4-5-20251001-thinking":  1024,
	}
	for _, model := range kiropkg.DefaultModels {
		want, ok := expected[model.ID]
		require.Truef(t, ok, "model %q 未在本测试中归类，请为其显式指定最小可缓存 token 数", model.ID)
		require.Equal(t, want, kiroMinimumCacheableTokens(model.ID), model.ID)
	}
}

func kiroCacheGroup(ratio float64) *Group {
	return &Group{ID: 12, Platform: PlatformKiro, KiroCacheEmulationEnabled: true, KiroCacheEmulationRatio: ratio}
}

func kiroCacheAccount(id int64, refreshToken string, accessToken string) *Account {
	return &Account{ID: id, Platform: PlatformKiro, Type: AccountTypeOAuth, Credentials: map[string]any{
		"client_id":     "client-id",
		"refresh_token": refreshToken,
		"access_token":  accessToken,
	}}
}

func kiroCacheRequestBody(label string, oneHour bool) []byte {
	ttl := ""
	if oneHour {
		ttl = `,"ttl":"1h"`
	}
	return []byte(fmt.Sprintf(`{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":%q,"cache_control":{"type":"ephemeral"%s}}]}]}`, strings.Repeat("cacheable prompt chunk "+label+" ", 512), ttl))
}

func kiroCacheMultiMessageBody(prefixLabel, tailLabel string) []byte {
	prefix := strings.Repeat("cacheable prompt chunk "+prefixLabel+" ", 512)
	tail := strings.Repeat("conversation growth chunk "+tailLabel+" ", 160)
	return []byte(fmt.Sprintf(`{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":[{"type":"text","text":%q,"cache_control":{"type":"ephemeral"}}]},{"role":"user","content":[{"type":"text","text":%q}]}]}`, prefix, tail))
}

func kiroChatCompletionsConversationBody(messages []string) []byte {
	items := make([]string, 0, len(messages)+1)
	items = append(items, `{"role":"system","content":"You are a precise assistant."}`)
	for _, message := range messages {
		items = append(items, fmt.Sprintf(`{"role":"user","content":%q}`, message))
	}
	return []byte(fmt.Sprintf(`{"model":"gpt-5","tool_choice":"auto","tools":[{"type":"function","function":{"name":"lookup","description":"lookup data","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}],"messages":[%s]}`, strings.Join(items, ",")))
}

func kiroResponsesCacheRequestBody(label, promptCacheKey, previousResponseID string) []byte {
	return kiroResponsesCacheRequestBodyWithOptions(label, promptCacheKey, previousResponseID, "gpt-5", "auto", "medium", `{"type":"json_object"}`, "lookup")
}

func kiroResponsesCacheRequestBodyWithOptions(label, promptCacheKey, previousResponseID, model, toolChoice, effort, textFormat, toolName string) []byte {
	prompt := strings.Repeat("cacheable responses prompt chunk "+label+" ", 512)
	return []byte(fmt.Sprintf(`{"model":%q,"instructions":"You are a precise assistant.","prompt_cache_key":%q,"previous_response_id":%q,"tool_choice":%q,"reasoning":{"effort":%q},"text":{"format":%s},"tools":[{"type":"function","name":%q,"description":"lookup data","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}],"input":[{"role":"user","content":[{"type":"input_text","text":%q}]}]}`, model, promptCacheKey, previousResponseID, toolChoice, effort, textFormat, toolName, prompt))
}

func kiroResponsesConversationRequestBody(promptCacheKey string, messages []string) []byte {
	items := make([]string, 0, len(messages))
	for _, message := range messages {
		items = append(items, fmt.Sprintf(`{"type":"message","role":"user","content":[{"type":"input_text","text":%q}]}`, message))
	}
	return []byte(fmt.Sprintf(`{"model":"gpt-5","instructions":"You are a precise assistant.","prompt_cache_key":%q,"tool_choice":"auto","tools":[{"type":"function","name":"lookup","description":"lookup data","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}],"input":[%s]}`, promptCacheKey, strings.Join(items, ",")))
}

func kiroResponsesImageCacheRequestBody(t *testing.T, label string, fill color.RGBA) []byte {
	prompt := strings.Repeat("cacheable responses visual prompt "+label+" ", 512)
	imageURL := kiroPNGDataURL(t, 384, 256, fill)
	return []byte(fmt.Sprintf(`{"model":"gpt-5","instructions":"Describe visual changes precisely.","prompt_cache_key":"workspace-image","previous_response_id":"resp-image","input":[{"role":"user","content":[{"type":"input_text","text":%q},{"type":"input_image","image_url":%q}]}]}`, prompt, imageURL))
}

// 客户端完全不下发 cache_control 时（curl、Cherry Studio 等 OpenAI 风格客户端），
// Anthropic 路径必须兜底补断点。否则该请求零 cache_read 却仍计入命中率分母。
func kiroToolHeavyShortRequestBody() []byte {
	desc := strings.Repeat("Detailed behavioural contract for this tool. ", 40)
	tools := make([]string, 0, 15)
	for i := 0; i < 15; i++ {
		tools = append(tools, fmt.Sprintf(
			`{"name":"tool_%d","description":%q,"input_schema":{"type":"object","properties":{"path":{"type":"string","description":%q}},"required":["path"]}}`,
			i, desc, desc))
	}
	return []byte(fmt.Sprintf(
		`{"model":"claude-opus-4-8","system":[{"type":"text","text":%q}],"tools":[%s],"messages":[{"role":"user","content":[{"type":"text","text":%q,"cache_control":{"type":"ephemeral"}}]}]}`,
		strings.Repeat("short system. ", 8), strings.Join(tools, ","),
		strings.Repeat("brief question. ", 10)))
}

// kiroClaudeCodeConversationBody 构造贴近真实 Claude Code 长会话的 Anthropic 请求体。
//
// 形态要点（决定这个用例是否具备判别力，两个约束互相拉扯，必须同时满足）：
//   - 轮数要足够多，让 messages 在总量中占主导。若 system 相对 messages 过大，逐块计数与
//     整体 JSON 计数之间的口径差异会被稀释，用例就测不出问题了。
//   - 但 system 也不能过小。单轮命中率上限 = (base+g)/(base+2g)，base 为 system+tools、
//     g 为每轮增量；base/g < 8 时前几轮天然到不了 90%。这是 prompt caching 的冷启动爬坡，
//     真实 Anthropic 缓存同样如此，不是缺陷。真实 Claude Code 的 system + CLAUDE.md + tools
//     通常 10-25k token，每轮增量 200-2000，base/g 约 10-50:1；这里取 ~3k token 的 system
//     使 base/g ≈ 10，兼顾上述两个约束。
//   - 每轮 assistant 带 text + 2 个 tool_use，user 回 2 个 tool_result，即大量小块。
//     tool_use 只有 input 被计数、tool_result 只有 content 被计数，而 type/id/name 等
//     结构字段一律计 0——块越多越碎，缺口越大，正是线上流量的形态。
//   - cache_control 只打在 system 末块，模拟客户端最保守的断点策略；后续每个消息末尾的
//     断点由 buildKiroCacheProfileFromBlocks 的传播逻辑补齐。
func kiroClaudeCodeConversationBody(turns int) []byte {
	system := strings.Repeat("You are a coding agent operating in a real repository. ", 220)
	tools := `[{"name":"read_file","description":"Read a file from disk","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},{"name":"edit_file","description":"Apply an edit","input_schema":{"type":"object","properties":{"path":{"type":"string"},"patch":{"type":"string"}},"required":["path","patch"]}},{"name":"grep","description":"Search contents","input_schema":{"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}}]`

	messages := make([]string, 0, turns*2)
	messages = append(messages, fmt.Sprintf(`{"role":"user","content":[{"type":"text","text":%q}]}`,
		strings.Repeat("Please refactor the billing module carefully. ", 12)))
	for turn := 1; turn <= turns; turn++ {
		messages = append(messages, fmt.Sprintf(`{"role":"assistant","content":[{"type":"text","text":%q},{"type":"tool_use","id":"toolu_%d_a","name":"read_file","input":{"path":"internal/service/billing_%d.go"}},{"type":"tool_use","id":"toolu_%d_b","name":"grep","input":{"pattern":"CalculateCost_%d"}}]}`,
			strings.Repeat("Inspecting the next call site. ", 6), turn, turn, turn, turn))
		messages = append(messages, fmt.Sprintf(`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_%d_a","content":[{"type":"text","text":%q}]},{"type":"tool_result","tool_use_id":"toolu_%d_b","content":[{"type":"text","text":%q}]}]}`,
			turn, strings.Repeat("func CalculateCost(tokens UsageTokens) float64 { return 0 } ", 5),
			turn, strings.Repeat("billing_service.go:120: CalculateCost invoked here ", 5)))
	}

	return []byte(fmt.Sprintf(`{"model":"claude-sonnet-4-6","system":[{"type":"text","text":%q,"cache_control":{"type":"ephemeral"}}],"tools":%s,"messages":[%s]}`,
		system, tools, strings.Join(messages, ",")))
}

// Kiro 分组模拟缓存在多轮追加式会话下的稳态命中率必须 >= 90%。
//
// 命中率口径与前端面板一致（TokenUsageTrend.vue）：
//
//	cache_read / (input + cache_read + cache_creation)
//
// 而 prepareKiroCacheEmulationPlanFromProfile 令三者之和恒等于 inputTokens，故等价于
// cache_read / inputTokens。这要求断点累计值与 inputTokens 处于同一 token 空间，即
// profile.scaleBreakpointsToInputTokens 必须为 true——该标志缺失时，分子只统计正文、
// 分母含整体 JSON 结构开销，命中率会被系统性压到 90% 以下。
