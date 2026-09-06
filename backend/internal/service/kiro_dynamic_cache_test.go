package service

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"hash/crc32"
	"image/color"
	"io"
	"math"
	"strings"
	"sync"
	"testing"
	"time"

	kiropkg "github.com/Wei-Shaw/sub2api/internal/pkg/kiro"
	"github.com/stretchr/testify/require"
)

func dynamicFrame(t *testing.T, event string, payload any) []byte {
	t.Helper()
	b, err := json.Marshal(payload)
	require.NoError(t, err)
	var headers bytes.Buffer
	headers.WriteByte(byte(len(":event-type")))
	headers.WriteString(":event-type")
	headers.WriteByte(7)
	require.NoError(t, binary.Write(&headers, binary.BigEndian, uint16(len(event))))
	headers.WriteString(event)
	var frame bytes.Buffer
	require.NoError(t, binary.Write(&frame, binary.BigEndian, uint32(16+headers.Len()+len(b))))
	require.NoError(t, binary.Write(&frame, binary.BigEndian, uint32(headers.Len())))
	require.NoError(t, binary.Write(&frame, binary.BigEndian, crc32.ChecksumIEEE(frame.Bytes())))
	frame.Write(headers.Bytes())
	frame.Write(b)
	require.NoError(t, binary.Write(&frame, binary.BigEndian, crc32.ChecksumIEEE(frame.Bytes())))
	return frame.Bytes()
}
func dynamicEvents(t *testing.T, credits float64, output int) []byte {
	t.Helper()
	b := dynamicFrame(t, "assistantResponseEvent", map[string]any{"content": "hello"})
	b = append(b, dynamicFrame(t, "metadataEvent", map[string]any{"outputTokens": output})...)
	return append(b, dynamicFrame(t, "meteringEvent", map[string]any{"usage": credits, "unit": "credit"})...)
}
func dynamicUsage(t *testing.T, credits float64, output int) kiropkg.Usage {
	t.Helper()
	result, err := kiropkg.ParseNonStreamingEventStreamWithContext(bytes.NewReader(dynamicEvents(t, credits, output)), "claude-opus-5", kiropkg.KiroRequestContext{})
	require.NoError(t, err)
	return result.Usage
}
func dynamicRequest(t *testing.T, model, conversation string) kiropkg.KiroRequestContext {
	t.Helper()
	payload := kiropkg.KiroPayload{ProfileArn: "test-profile-a", ConversationState: kiropkg.KiroConversationState{
		ConversationID: conversation, ChatTriggerType: "MANUAL",
		History:        []kiropkg.KiroHistoryMessage{{UserInputMessage: &kiropkg.KiroUserInputMessage{ModelID: model, Origin: "AI_EDITOR", Content: strings.Repeat("stable semantic history ", 2500)}}},
		CurrentMessage: kiropkg.KiroCurrentMessage{UserInputMessage: kiropkg.KiroUserInputMessage{ModelID: model, Origin: "AI_EDITOR", Content: "next question"}},
	}}
	b, err := json.Marshal(payload)
	require.NoError(t, err)
	return kiropkg.KiroRequestContext{CachePayload: b, CacheEndpoint: "https://native.test/generate"}
}
func mutateDynamicRequest(t *testing.T, request kiropkg.KiroRequestContext, change func(*kiropkg.KiroPayload)) kiropkg.KiroRequestContext {
	t.Helper()
	var p kiropkg.KiroPayload
	require.NoError(t, json.Unmarshal(request.CachePayload, &p))
	change(&p)
	b, err := json.Marshal(p)
	require.NoError(t, err)
	request.CachePayload = b
	return request
}
func dynamicSetup(t *testing.T) (*Account, *Group, *time.Time) {
	t.Helper()
	old := globalKiroDynamicTracker
	now := time.Date(2026, 9, 6, 0, 0, 0, 0, time.UTC)
	globalKiroDynamicTracker = newKiroDynamicTracker(func() time.Time { return now })
	t.Cleanup(func() { globalKiroDynamicTracker = old })
	return &Account{ID: 1, Platform: PlatformKiro, Type: AccountTypeOAuth}, kiroCacheGroup(1), &now
}
func dynamicPrepare(t *testing.T, a *Account, g *Group, r kiropkg.KiroRequestContext) *kiroCacheEmulationPlan {
	t.Helper()
	p := prepareKiroDynamicCache(a, g, r, 10000)
	require.NotNil(t, p)
	return p
}
func dynamicTrain(t *testing.T, a *Account, g *Group, r kiropkg.KiroRequestContext) {
	t.Helper()
	for _, credits := range []float64{10, 2, 2} {
		p := dynamicPrepare(t, a, g, r)
		require.Nil(t, p.result())
		p.complete(dynamicUsage(t, credits, 20), true)
	}
}
func TestKiroDynamicCalibrationAndFrozenUsage(t *testing.T) {
	a, g, _ := dynamicSetup(t)
	r := dynamicRequest(t, "claude-opus-5", "a")
	dynamicTrain(t, a, g, r)
	p := dynamicPrepare(t, a, g, r)
	require.Equal(t, 8000, p.result().CacheReadInputTokens)
	require.Equal(t, 2000, p.result().InputTokens)
	require.Zero(t, p.result().CacheCreationInputTokens)
	frozen := *p.result()
	p.complete(dynamicUsage(t, 8, 20), true)
	require.Equal(t, frozen, *p.result())
	p.complete(dynamicUsage(t, 1, 20), true) // idempotent late observer
	next := dynamicPrepare(t, a, g, r)
	require.InDelta(t, 2000, next.result().CacheReadInputTokens, 1)
	next.complete(kiropkg.Usage{}, false)
	// A bounded window can recover from low/negative observations.
	p = dynamicPrepare(t, a, g, r)
	p.complete(dynamicUsage(t, 20, 20), true)
	p = dynamicPrepare(t, a, g, r)
	require.Nil(t, p.result())
	p.complete(kiropkg.Usage{}, false)
	for i := 0; i < 8; i++ {
		p = dynamicPrepare(t, a, g, r)
		p.complete(dynamicUsage(t, 2, 20), true)
	}
	p = dynamicPrepare(t, a, g, r)
	require.Equal(t, 8000, p.result().CacheReadInputTokens)
	p.complete(kiropkg.Usage{}, false)
	require.Len(t, p.scope.pairs, 8)
	require.Zero(t, p.scope.active)
}
func TestKiroDynamicModelSessionAndSlidingTTL(t *testing.T) {
	for _, model := range []string{"claude-opus-5", "gpt-5.6-terra"} {
		t.Run(model, func(t *testing.T) {
			a, g, now := dynamicSetup(t)
			r := dynamicRequest(t, model, "a")
			dynamicTrain(t, a, g, r)
			other := mutateDynamicRequest(t, r, func(p *kiropkg.KiroPayload) { p.ConversationState.ConversationID = "b" })
			p := dynamicPrepare(t, a, g, other)
			if model == "claude-opus-5" {
				require.Equal(t, 10000, p.matched)
				require.NotNil(t, p.result())
			} else {
				require.Zero(t, p.matched)
				require.Nil(t, p.result())
			}
			p.complete(kiropkg.Usage{}, false)
			ttl := 5 * time.Minute
			if model == "gpt-5.6-terra" {
				ttl = 30 * time.Minute
			}
			require.Equal(t, ttl, kiroDynamicTTL(model))
			*now = now.Add(ttl - time.Second)
			p = dynamicPrepare(t, a, g, r)
			require.Equal(t, 10000, p.matched)
			p.complete(dynamicUsage(t, 2, 20), true)
			*now = now.Add(ttl - time.Second)
			p = dynamicPrepare(t, a, g, r)
			require.Equal(t, 10000, p.matched)
			p.complete(kiropkg.Usage{}, false)
			// A prepare/abort does not slide TTL.
			*now = now.Add(time.Second)
			p = dynamicPrepare(t, a, g, r)
			require.Zero(t, p.matched)
			require.Nil(t, p.result())
			p.complete(kiropkg.Usage{}, false)
		})
	}
}
func TestKiroDynamicScopeIsolationAndCredentialRotation(t *testing.T) {
	a, g, _ := dynamicSetup(t)
	r := dynamicRequest(t, "claude-opus-5", "a")
	dynamicTrain(t, a, g, r)
	a.Credentials = map[string]any{"access_token": "rotated-access", "refresh_token": "rotated-refresh"}
	p := dynamicPrepare(t, a, g, r)
	require.Equal(t, 10000, p.matched)
	p.complete(kiropkg.Usage{}, false)
	for _, change := range []string{"account", "principal", "profile", "endpoint", "model"} {
		t.Run(change, func(t *testing.T) {
			other := *a
			other.Credentials = map[string]any{}
			request := r
			switch change {
			case "account":
				other.ID = 2
			case "principal":
				other.Credentials["principal_id"] = "new-principal"
			case "profile":
				request = mutateDynamicRequest(t, r, func(p *kiropkg.KiroPayload) { p.ProfileArn = "test-profile-b" })
			case "endpoint":
				request.CacheEndpoint = "https://fallback.test/generate"
			case "model":
				request = dynamicRequest(t, "gpt-5.6-terra", "a")
			}
			p := dynamicPrepare(t, &other, g, request)
			require.Zero(t, p.matched)
			require.Nil(t, p.result())
			p.complete(kiropkg.Usage{}, false)
		})
	}
	// API-key replacement without a profile is a principal replacement.
	a.Type = AccountTypeAPIKey
	a.Credentials = map[string]any{"api_key": "fake-key-a", "kiro_api_key": "unused-legacy-key"}
	r = mutateDynamicRequest(t, r, func(p *kiropkg.KiroPayload) { p.ProfileArn = "" })
	dynamicTrain(t, a, g, r)
	a.Credentials["api_key"] = "fake-key-b"
	p = dynamicPrepare(t, a, g, r)
	require.Zero(t, p.matched)
	p.complete(kiropkg.Usage{}, false)
}
func TestKiroDynamicUnknownDisabledAndFixedRatiosIgnored(t *testing.T) {
	a, g, _ := dynamicSetup(t)
	for _, model := range []string{"claude-sonnet-4-6", "gpt-5", "gpt-5.6-sol", "claude-opus-5-unknown"} {
		require.Nil(t, prepareKiroDynamicCache(a, g, dynamicRequest(t, model, "a"), 10000))
	}
	r := dynamicRequest(t, "claude-opus-5", "a")
	dynamicTrain(t, a, g, r)
	g.KiroCacheEmulationRatio = 0
	g.KiroCacheReadEmulationRatio = 0
	g.KiroCacheCreationEmulationRatio = 0
	p := dynamicPrepare(t, a, g, r)
	require.Equal(t, 8000, p.result().CacheReadInputTokens)
	p.complete(kiropkg.Usage{}, false)
	g.KiroCacheEmulationEnabled = false
	require.Nil(t, prepareKiroDynamicCache(a, g, r, 10000))
}
func TestKiroDynamicInvalidCreditsFailuresAndOutputVariation(t *testing.T) {
	a, g, now := dynamicSetup(t)
	r := dynamicRequest(t, "claude-opus-5", "a")
	valid := dynamicUsage(t, 10, 20)
	for _, credits := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		p := dynamicPrepare(t, a, g, r)
		usage := valid
		usage.KiroCredits = credits
		p.complete(usage, true)
		require.Empty(t, p.scope.prefixes)
		require.Zero(t, p.scope.active)
	}
	p := dynamicPrepare(t, a, g, r)
	p.complete(valid, false)
	require.Empty(t, p.scope.prefixes)
	p = dynamicPrepare(t, a, g, r)
	p.complete(valid, true)
	for _, output := range []int{21, 500, 0} {
		p = dynamicPrepare(t, a, g, r)
		p.complete(dynamicUsage(t, 2, output), true)
		require.Empty(t, p.scope.pairs)
	}
	p = dynamicPrepare(t, a, g, r)
	*now = now.Add(5 * time.Minute)
	p.complete(valid, true) // slow response must not resurrect expired cache
	require.Zero(t, p.scope.active)
	p = dynamicPrepare(t, a, g, r)
	require.Zero(t, p.matched)
	p.complete(kiropkg.Usage{}, false)
	*now = now.Add(31 * time.Minute)
	globalKiroDynamicTracker.mu.Lock()
	globalKiroDynamicTracker.prune(*now)
	require.Empty(t, globalKiroDynamicTracker.scopes)
	globalKiroDynamicTracker.mu.Unlock()
}
func TestKiroDynamicConcurrentPreparationAndCompletion(t *testing.T) {
	a, g, _ := dynamicSetup(t)
	r := dynamicRequest(t, "claude-opus-5", "a")
	usage := dynamicUsage(t, 10, 20)
	plans := make([]*kiroCacheEmulationPlan, 32)
	var wg sync.WaitGroup
	for i := range plans {
		wg.Add(1)
		go func(i int) { defer wg.Done(); plans[i] = prepareKiroDynamicCache(a, g, r, 10000) }(i)
	}
	wg.Wait()
	for _, p := range plans {
		require.NotNil(t, p)
		require.Zero(t, p.matched)
		wg.Add(1)
		go func(p *kiroCacheEmulationPlan) { defer wg.Done(); p.complete(usage, true); p.complete(usage, true) }(p)
	}
	wg.Wait()
	require.Zero(t, plans[0].scope.active)
	require.Empty(t, plans[0].scope.samples)
	require.Empty(t, plans[0].scope.pairs)
}
func TestKiroDynamicOutboundSemanticPrefixAndToolOrder(t *testing.T) {
	a, g, _ := dynamicSetup(t)
	r := dynamicRequest(t, "claude-opus-5", "a")
	dynamicTrain(t, a, g, r)
	changed := mutateDynamicRequest(t, r, func(p *kiropkg.KiroPayload) {
		p.ConversationState.CurrentMessage.UserInputMessage.Content = "changed final turn"
	})
	p := dynamicPrepare(t, a, g, changed)
	require.Greater(t, p.matched, 0)
	require.Less(t, p.matched, 10000)
	require.LessOrEqual(t, p.result().CacheReadInputTokens, p.matched)
	p.complete(kiropkg.Usage{}, false)
	changed = mutateDynamicRequest(t, r, func(p *kiropkg.KiroPayload) {
		p.ConversationState.History[0].UserInputMessage.Content += "current time changed"
	})
	p = dynamicPrepare(t, a, g, changed)
	require.Zero(t, p.matched)
	p.complete(kiropkg.Usage{}, false)
	tools := []kiropkg.KiroToolWrapper{{ToolSpecification: kiropkg.KiroToolSpecification{Name: "first"}}, {ToolSpecification: kiropkg.KiroToolSpecification{Name: "second"}}}
	toolRequest := mutateDynamicRequest(t, r, func(p *kiropkg.KiroPayload) {
		p.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext = &kiropkg.KiroUserInputMessageContext{Tools: tools}
	})
	p = dynamicPrepare(t, a, g, toolRequest)
	p.complete(dynamicUsage(t, 10, 20), true)
	swapped := mutateDynamicRequest(t, toolRequest, func(p *kiropkg.KiroPayload) {
		c := p.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext
		c.Tools[0], c.Tools[1] = c.Tools[1], c.Tools[0]
	})
	p = dynamicPrepare(t, a, g, swapped)
	require.Zero(t, p.matched)
	p.complete(kiropkg.Usage{}, false)
	// Native content never passes through billing-header or presentation filters.
	prof, _, ok := buildKiroOutboundCacheProfile(r.CachePayload, 10000)
	require.True(t, ok)
	require.Greater(t, len(prof.blocks), 1)
	var obj map[string]any
	require.NoError(t, json.Unmarshal(r.CachePayload, &obj))
	// Reorder root fields and add whitespace; native semantics remain identical.
	raw, err := json.MarshalIndent(obj, "", "  ")
	require.NoError(t, err)
	equivalent, _, ok := buildKiroOutboundCacheProfile(raw, 10000)
	require.True(t, ok)
	require.Equal(t, prof.blocks, equivalent.blocks)
	advanced := mutateDynamicRequest(t, r, func(p *kiropkg.KiroPayload) {
		current := p.ConversationState.CurrentMessage.UserInputMessage
		p.ConversationState.History = append(p.ConversationState.History, kiropkg.KiroHistoryMessage{UserInputMessage: &current}, kiropkg.KiroHistoryMessage{AssistantResponseMessage: &kiropkg.KiroAssistantResponseMessage{Content: "answer"}})
		p.ConversationState.CurrentMessage.UserInputMessage.Content = "next turn"
	})
	advancedProfile, _, ok := buildKiroOutboundCacheProfile(advanced.CachePayload, 12000)
	require.True(t, ok)
	require.Equal(t, prof.blocks, advancedProfile.blocks[:len(prof.blocks)])
}

func TestKiroDynamicCapacityAndCleanup(t *testing.T) {
	a, g, now := dynamicSetup(t)
	r := dynamicRequest(t, "claude-opus-5", "a")
	profile, _, ok := buildKiroOutboundCacheProfile(r.CachePayload, 10000)
	require.True(t, ok)
	tracker := globalKiroDynamicTracker
	keyFor := func(n int) [32]byte { var k [32]byte; binary.BigEndian.PutUint64(k[:8], uint64(n)); return k }
	// Per-account quota rejects new namespaces without evicting any other account.
	for i := 1; i <= kiroDynamicMaxAccountScopes; i++ {
		p := tracker.prepare(a.ID, keyFor(i), keyFor(i), profile, 5*time.Minute)
		require.NotNil(t, p)
		p.complete(kiropkg.Usage{}, false)
	}
	require.Nil(t, tracker.prepare(a.ID, keyFor(10000), keyFor(1), profile, 5*time.Minute))
	for i := kiroDynamicMaxAccountScopes + 1; i <= kiroDynamicMaxScopes; i++ {
		p := tracker.prepare(int64(i), keyFor(i), keyFor(i), profile, 5*time.Minute)
		require.NotNil(t, p)
		p.complete(kiropkg.Usage{}, false)
	}
	require.Len(t, tracker.scopes, kiroDynamicMaxScopes)
	require.Nil(t, tracker.prepare(99999, keyFor(10001), keyFor(1), profile, 5*time.Minute))
	*now = now.Add(31 * time.Minute)
	tracker.mu.Lock()
	tracker.prune(*now)
	require.Empty(t, tracker.scopes)
	tracker.mu.Unlock()
	// Many disjoint native shapes within one namespace hit prefix and sample caps.
	usage := dynamicUsage(t, 10, 20)
	for i := 1; i <= kiroDynamicMaxPrefixes+10; i++ {
		copyProfile := *profile
		copyProfile.blocks = append([]kiroCacheBlock(nil), profile.blocks...)
		for j := range copyProfile.blocks {
			copyProfile.blocks[j].prefixFingerprint = keyFor(10000*i + j)
		}
		p := tracker.prepare(a.ID, keyFor(1), keyFor(i), &copyProfile, 5*time.Minute)
		require.NotNil(t, p)
		p.complete(usage, true)
	}
	scope := tracker.scopes[keyFor(1)]
	require.Len(t, scope.prefixes, kiroDynamicMaxPrefixes)
	require.Len(t, scope.samples, kiroDynamicMaxSamples)
	*now = now.Add(5 * time.Minute)
	tracker.mu.Lock()
	tracker.prune(*now)
	require.Empty(t, scope.prefixes)
	require.Len(t, scope.samples, kiroDynamicMaxSamples)
	tracker.mu.Unlock()
	*now = now.Add(26 * time.Minute)
	tracker.mu.Lock()
	tracker.prune(*now)
	require.Empty(t, tracker.scopes)
	tracker.mu.Unlock()
	// Expired pair evidence does not survive merely because the scope is touched.
	dynamicTrain(t, a, g, r)
	p := dynamicPrepare(t, a, g, r)
	p.complete(kiropkg.Usage{}, false)
	*now = now.Add(31 * time.Minute)
	p = dynamicPrepare(t, a, g, r)
	require.Nil(t, p.result())
	require.Empty(t, p.scope.pairs)
	p.complete(kiropkg.Usage{}, false)
}

func TestKiroDynamicZeroRatiosAuthSnapshot(t *testing.T) {
	a, g, _ := dynamicSetup(t)
	for _, mode := range []string{KiroCacheEmulationModeUniform, KiroCacheEmulationModeIndependent} {
		g.KiroCacheEmulationMode = mode
		g.KiroCacheEmulationRatio = 0
		g.KiroCacheReadEmulationRatio = 0
		g.KiroCacheCreationEmulationRatio = 0
		NormalizeGroupRuntimeFields(g)
		require.True(t, g.EffectiveKiroCacheEmulationEnabled())
		svc := &APIKeyService{}
		snapshot := svc.snapshotFromAPIKey(context.Background(), &APIKey{ID: 1, User: &User{ID: 1}, Group: g})
		require.NotNil(t, snapshot)
		require.True(t, snapshot.Group.KiroCacheEmulationEnabled)
		restored := svc.snapshotToAPIKey("test-key", snapshot)
		require.True(t, restored.Group.EffectiveKiroCacheEmulationEnabled())
		p := dynamicPrepare(t, a, restored.Group, dynamicRequest(t, "claude-opus-5", "a"))
		p.complete(kiropkg.Usage{}, false)
	}
}

func TestKiroDynamicNativeCompletionGates(t *testing.T) {
	good := dynamicEvents(t, 10, 20)
	corrupt := append([]byte(nil), good...)
	corrupt[len(corrupt)-1] ^= 1
	missingMeter := dynamicFrame(t, "assistantResponseEvent", map[string]any{"content": "partial"})
	exception := append(dynamicFrame(t, "internalServerException", map[string]any{"message": "failed"}), good...)
	invalidMeter := append(dynamicFrame(t, "meteringEvent", map[string]any{"usage": "NaN"}), good...)
	malformed := append(dynamicFrame(t, "assistantResponseEvent", []string{"invalid event object"}), good...)
	for name, raw := range map[string][]byte{"complete": good, "missing-meter": missingMeter, "crc": corrupt, "exception": exception, "invalid-meter": invalidMeter, "malformed": malformed, "truncated": append(append([]byte(nil), good...), 1)} {
		for _, stream := range []bool{false, true} {
			t.Run(name+map[bool]string{false: "-buffered", true: "-stream"}[stream], func(t *testing.T) {
				a, g, _ := dynamicSetup(t)
				p := dynamicPrepare(t, a, g, dynamicRequest(t, "claude-opus-5", "a"))
				var usage kiropkg.Usage
				var err error
				if stream {
					var result *kiropkg.StreamResult
					result, err = kiropkg.StreamEventStreamAsAnthropicWithContext(context.Background(), bytes.NewReader(raw), io.Discard, "claude-opus-5", 10000, kiropkg.KiroRequestContext{})
					if result != nil {
						usage = result.Usage
					}
				} else {
					var result *kiropkg.ParseResult
					result, err = kiropkg.ParseNonStreamingEventStreamWithContext(bytes.NewReader(raw), "claude-opus-5", kiropkg.KiroRequestContext{})
					if result != nil {
						usage = result.Usage
					}
				}
				p.complete(usage, err == nil)
				if name == "complete" {
					require.NotEmpty(t, p.scope.prefixes)
				} else {
					require.Empty(t, p.scope.prefixes)
					require.Empty(t, p.scope.samples)
				}
				require.Zero(t, p.scope.active)
			})
		}
	}
}

func TestKiroDynamicNativeImagesNumbersAndModelVisibleTime(t *testing.T) {
	_, _, _ = dynamicSetup(t)
	request := dynamicRequest(t, "claude-opus-5", "a")
	imageA := strings.SplitN(kiroPNGDataURL(t, 80, 80, color.RGBA{R: 255, A: 255}), ",", 2)[1]
	imageB := strings.SplitN(kiroPNGDataURL(t, 80, 80, color.RGBA{B: 255, A: 255}), ",", 2)[1]
	withImage := func(data string) kiropkg.KiroRequestContext {
		return mutateDynamicRequest(t, request, func(p *kiropkg.KiroPayload) {
			p.ConversationState.History[0].UserInputMessage.Images = []kiropkg.KiroImage{{Format: "png", Source: kiropkg.KiroImageSource{Bytes: data}}}
		})
	}
	pa, _, ok := buildKiroOutboundCacheProfile(withImage(imageA).CachePayload, 10000)
	require.True(t, ok)
	pb, _, ok := buildKiroOutboundCacheProfile(withImage(imageB).CachePayload, 10000)
	require.True(t, ok)
	require.NotEqual(t, pa.blocks[0].prefixFingerprint, pb.blocks[0].prefixFingerprint)
	require.Equal(t, pa.blocks[0].cumulativeTokens, pb.blocks[0].cumulativeTokens, "same dimensions use visual tokens, not base64 length")
	// Distinct integer tool arguments beyond float64 precision remain distinct.
	numeric := func(number string) kiropkg.KiroRequestContext {
		return mutateDynamicRequest(t, request, func(p *kiropkg.KiroPayload) {
			p.ConversationState.History[0] = kiropkg.KiroHistoryMessage{AssistantResponseMessage: &kiropkg.KiroAssistantResponseMessage{Content: strings.Repeat("history ", 5000), ToolUses: []kiropkg.KiroToolUse{{ToolUseID: "tool-a", Name: "lookup", Input: map[string]any{"id": json.Number(number)}}}}}
		})
	}
	pa, _, ok = buildKiroOutboundCacheProfile(numeric("9007199254740992").CachePayload, 10000)
	require.True(t, ok)
	pb, _, ok = buildKiroOutboundCacheProfile(numeric("9007199254740993").CachePayload, 10000)
	require.True(t, ok)
	require.NotEqual(t, pa.blocks[0].prefixFingerprint, pb.blocks[0].prefixFingerprint)
	temporal := func(value string) kiropkg.KiroRequestContext {
		return mutateDynamicRequest(t, request, func(p *kiropkg.KiroPayload) {
			p.ConversationState.History[0].UserInputMessage.Content = "[Context: Current time is " + value + "]\n" + p.ConversationState.History[0].UserInputMessage.Content
		})
	}
	pa, _, ok = buildKiroOutboundCacheProfile(temporal("2026-09-06 01:00:00 UTC").CachePayload, 10000)
	require.True(t, ok)
	pb, _, ok = buildKiroOutboundCacheProfile(temporal("2026-09-06 01:00:01 UTC").CachePayload, 10000)
	require.True(t, ok)
	require.NotEqual(t, pa.blocks[0].prefixFingerprint, pb.blocks[0].prefixFingerprint)
}

func TestKiroDynamicGrowingConversationCalibratesWithoutReplay(t *testing.T) {
	a, g, _ := dynamicSetup(t)
	base := mutateDynamicRequest(t, dynamicRequest(t, "claude-opus-5", "a"), func(p *kiropkg.KiroPayload) { p.InferenceConfig = &kiropkg.KiroInferenceConfig{MaxTokens: 1000} })
	grow := func(r kiropkg.KiroRequestContext, turn string) kiropkg.KiroRequestContext {
		return mutateDynamicRequest(t, r, func(p *kiropkg.KiroPayload) {
			current := p.ConversationState.CurrentMessage.UserInputMessage
			p.ConversationState.History = append(p.ConversationState.History, kiropkg.KiroHistoryMessage{UserInputMessage: &current}, kiropkg.KiroHistoryMessage{AssistantResponseMessage: &kiropkg.KiroAssistantResponseMessage{Content: "previous answer"}})
			p.ConversationState.CurrentMessage.UserInputMessage.Content = strings.Repeat(turn+" new input ", 50)
		})
	}
	p := dynamicPrepare(t, a, g, base)
	p.complete(dynamicUsage(t, 10, 20), true)
	second := grow(base, "second")
	p = dynamicPrepare(t, a, g, second)
	require.Nil(t, p.result())
	require.Greater(t, p.matched, 0)
	require.Less(t, p.matched, 10000)
	p.complete(dynamicUsage(t, 3, 20), true) // warm base cost2 plus appended uncached cost1
	require.Len(t, p.scope.pairs, 1)
	require.InDelta(t, .7, p.scope.pairs[0].ratio, 1e-9)
	third := grow(second, "third")
	p = dynamicPrepare(t, a, g, third)
	require.Nil(t, p.result())
	p.complete(dynamicUsage(t, 4, 20), true)
	require.Len(t, p.scope.pairs, 2)
	require.InDelta(t, .6, p.scope.pairs[1].ratio, 1e-9)
	fourth := grow(third, "fourth")
	p = dynamicPrepare(t, a, g, fourth)
	require.NotNil(t, p.result())
	require.Equal(t, int(float64(p.matched)*.6), p.result().CacheReadInputTokens)
	p.complete(kiropkg.Usage{}, false)
	require.Len(t, p.scope.samples, 1, "learning used the original cold prefix without full replay")
	for _, kind := range []string{"output", "config", "prefix"} {
		t.Run(kind, func(t *testing.T) {
			a, g, _ := dynamicSetup(t)
			p := dynamicPrepare(t, a, g, base)
			p.complete(dynamicUsage(t, 10, 20), true)
			request := second
			output := 20
			switch kind {
			case "output":
				output = 21
			case "config":
				request = mutateDynamicRequest(t, request, func(p *kiropkg.KiroPayload) { p.InferenceConfig.MaxTokens = 2000 })
			case "prefix":
				request = mutateDynamicRequest(t, request, func(p *kiropkg.KiroPayload) {
					p.ConversationState.History[0].UserInputMessage.Content += "changed interior system content"
				})
			}
			p = dynamicPrepare(t, a, g, request)
			p.complete(dynamicUsage(t, 3, output), true)
			require.Empty(t, p.scope.pairs)
		})
	}
}

func TestKiroDynamicUsesActualOutboundAPIKeyHash(t *testing.T) {
	a, g, _ := dynamicSetup(t)
	a.Type = AccountTypeAPIKey
	a.Credentials = map[string]any{"api_key": "fake-key-a"}
	r := mutateDynamicRequest(t, dynamicRequest(t, "claude-opus-5", "a"), func(p *kiropkg.KiroPayload) { p.ProfileArn = "" })
	r.CachePrincipalHash = sha256.Sum256([]byte("fake-key-a"))
	dynamicTrain(t, a, g, r)
	// Account storage can change after dispatch; the accepted attempt still used A.
	a.Credentials["api_key"] = "fake-key-b"
	p := dynamicPrepare(t, a, g, r)
	require.Equal(t, 10000, p.matched)
	p.complete(kiropkg.Usage{}, false)
	r.CachePrincipalHash = sha256.Sum256([]byte("fake-key-b"))
	p = dynamicPrepare(t, a, g, r)
	require.Zero(t, p.matched)
	p.complete(kiropkg.Usage{}, false)
}
