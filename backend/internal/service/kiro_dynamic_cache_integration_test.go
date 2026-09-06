//go:build unit

package service

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	kiropkg "github.com/Wei-Shaw/sub2api/internal/pkg/kiro"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// The runtime still constructs its real outbound endpoint and translated body;
// this transport routes it exclusively to httptest, without network credentials.
type dynamicLocalUpstream struct {
	server    *httptest.Server
	mu        sync.Mutex
	endpoints []string
	payloads  [][]byte
}

func (u *dynamicLocalUpstream) Do(req *http.Request, proxy string, account int64, concurrency int) (*http.Response, error) {
	return u.DoWithTLS(req, proxy, account, concurrency, nil)
}
func (u *dynamicLocalUpstream) DoWithTLS(req *http.Request, _ string, _ int64, _ int, _ *tlsfingerprint.Profile) (*http.Response, error) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		return nil, err
	}
	u.mu.Lock()
	u.endpoints = append(u.endpoints, req.URL.String())
	u.payloads = append(u.payloads, body)
	u.mu.Unlock()
	local, err := http.NewRequestWithContext(req.Context(), req.Method, u.server.URL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	local.Header.Set("Content-Type", "application/json")
	return u.server.Client().Do(local)
}
func dynamicService(upstream *dynamicLocalUpstream) *GatewayService {
	return &GatewayService{cfg: &config.Config{Gateway: config.GatewayConfig{MaxLineSize: defaultMaxLineSize}}, httpUpstream: upstream, kiroCooldownStore: noopKiroChatCooldownStore{}, tlsFPProfileService: &TLSFingerprintProfileService{}, rateLimitService: &RateLimitService{}}
}
func dynamicProtocolBody(t *testing.T, protocol string, stream bool) []byte {
	t.Helper()
	body := map[string]any{"model": "claude-opus-5", "stream": stream, "max_tokens": 1000}
	content := strings.Repeat("stable native system and history ", 2500)
	if protocol == "responses" {
		body["input"] = []any{map[string]any{"role": "user", "content": content}}
	} else {
		body["messages"] = []any{map[string]any{"role": "user", "content": content}}
	}
	if protocol == "chat" && stream {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	raw, err := json.Marshal(body)
	require.NoError(t, err)
	return raw
}
func dynamicForward(t *testing.T, svc *GatewayService, a *Account, g *Group, protocol string, stream bool) (*ForwardResult, *httptest.ResponseRecorder, error) {
	t.Helper()
	body := dynamicProtocolBody(t, protocol, stream)
	format := protocol
	if protocol == "chat" {
		format = "chat_completions"
	}
	if protocol == "anthropic" {
		format = "anthropic"
	}
	parsed, err := ParseGatewayRequest(NewRequestBodyRef(body), format)
	require.NoError(t, err)
	parsed.Group = g
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/test", bytes.NewReader(body))
	var result *ForwardResult
	switch protocol {
	case "anthropic":
		result, err = svc.forwardKiroMessages(context.Background(), c, a, parsed, time.Now())
	case "chat":
		result, err = svc.ForwardAsChatCompletions(context.Background(), c, a, body, parsed)
	case "responses":
		result, err = svc.ForwardAsResponses(context.Background(), c, a, body, parsed)
	}
	return result, rec, err
}

// Recursive literal field/type sets cover the full public envelope, including
// all emitted SSE event variants. IDs, timestamps and token numbers are values,
// not schema. No fixture-update switch is exposed in normal test execution.
func dynamicSchema(t *testing.T, raw string, stream bool) []string {
	t.Helper()
	set := map[string]bool{}
	var walk func(any, string)
	walk = func(v any, path string) {
		switch x := v.(type) {
		case map[string]any:
			set[path+":object"] = true
			for k, v := range x {
				walk(v, path+"."+k)
			}
		case []any:
			set[path+":array"] = true
			for _, v := range x {
				walk(v, path+"[]")
			}
		case string:
			set[path+":string"] = true
		case float64:
			set[path+":number"] = true
		case bool:
			set[path+":boolean"] = true
		case nil:
			set[path+":null"] = true
		}
	}
	decode := func(raw string) {
		var v map[string]any
		require.NoError(t, json.Unmarshal([]byte(raw), &v))
		prefix := "$"
		if kind, ok := v["type"].(string); ok {
			prefix = kind
		}
		walk(v, prefix)
	}
	if stream {
		for _, line := range strings.Split(raw, "\n") {
			if !strings.HasPrefix(line, "data:") {
				continue
			}
			data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			if data != "[DONE]" {
				decode(data)
			}
		}
	} else {
		decode(raw)
	}
	keys := make([]string, 0, len(set))
	for key := range set {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
func TestKiroDynamicProtocolSchemasLocalIntegration(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("SUB2API_KIRO_TIME_CONTEXT", "")
	for _, protocol := range []string{"anthropic", "chat", "responses"} {
		for _, stream := range []bool{false, true} {
			name := protocol + "_buffered"
			if stream {
				name = protocol + "_stream"
			}
			t.Run(name, func(t *testing.T) {
				a, g, _ := dynamicSetup(t)
				a.Type = AccountTypeAPIKey
				a.Credentials = map[string]any{"api_key": "ksk_test_local_only"}
				cold := dynamicEvents(t, 10, 20)
				warm := dynamicEvents(t, 2, 20)
				var calls atomic.Int64
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					call := calls.Add(1)
					w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
					if call == 1 {
						_, _ = w.Write(cold)
					} else {
						_, _ = w.Write(warm)
					}
				}))
				defer server.Close()
				upstream := &dynamicLocalUpstream{server: server}
				svc := dynamicService(upstream)
				for i := 0; i < 4; i++ {
					result, rec, err := dynamicForward(t, svc, a, g, protocol, stream)
					require.NoError(t, err)
					require.Equal(t, http.StatusOK, rec.Code)
					require.NotNil(t, result)
					require.Zero(t, result.Usage.CacheCreationInputTokens)
					if i < 3 {
						require.Zero(t, result.Usage.CacheReadInputTokens)
					} else {
						require.Greater(t, result.Usage.CacheReadInputTokens, 0)
						total := result.Usage.InputTokens + result.Usage.CacheReadInputTokens
						require.InDelta(t, .8, float64(result.Usage.CacheReadInputTokens)/float64(total), .0001)
					}
					require.InDelta(t, []float64{10, 2, 2, 2}[i], result.Usage.KiroCredits, 1e-9)
					for _, forbidden := range []string{"_sub2api", "kiro_native_credits", "reuse_index"} {
						require.NotContains(t, rec.Body.String(), forbidden)
					}
					if i == 0 || i == 3 {
						state := "cold"
						if i == 3 {
							state = "warm"
						}
						fixture := filepath.Join("testdata", "kiro_dynamic_"+name+"_"+state+".schema.json")
						actual := dynamicSchema(t, rec.Body.String(), stream)
						expected, readErr := os.ReadFile(fixture)
						require.NoError(t, readErr)
						var keys []string
						require.NoError(t, json.Unmarshal(expected, &keys))
						require.Equal(t, keys, actual)
					}
				}
				require.Equal(t, int64(4), calls.Load())
				for _, raw := range upstream.payloads {
					var payload kiropkg.KiroPayload
					require.NoError(t, json.Unmarshal(raw, &payload))
					require.Equal(t, "claude-opus-5", payload.ConversationState.CurrentMessage.UserInputMessage.ModelID)
				}
				globalKiroDynamicTracker.mu.Lock()
				for _, scope := range globalKiroDynamicTracker.scopes {
					require.Zero(t, scope.active)
					require.Len(t, scope.pairs, 3)
				}
				globalKiroDynamicTracker.mu.Unlock()
			})
		}
	}
}

func TestKiroDynamicFinalFallbackAndParserFailureLocalIntegration(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("SUB2API_KIRO_TIME_CONTEXT", "")
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "buffered", true: "stream"}[stream], func(t *testing.T) {
			a, g, _ := dynamicSetup(t)
			a.Credentials = map[string]any{"auth_method": "idc", "access_token": "test-local-access", "profile_arn": "test-local-profile"}
			g.KiroEndpointMode = KiroEndpointModeAuto
			good := dynamicEvents(t, 10, 20)
			broken := append(append([]byte(nil), good...), 1, 2, 3)
			var calls atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				call := calls.Add(1)
				if call == 1 {
					w.WriteHeader(http.StatusForbidden)
					_, _ = io.WriteString(w, `{"message":"User is not authorized to make this call"}`)
					return
				}
				w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
				_, _ = w.Write(broken)
			}))
			defer server.Close()
			upstream := &dynamicLocalUpstream{server: server}
			svc := dynamicService(upstream)
			_, _, err := dynamicForward(t, svc, a, g, "anthropic", stream)
			require.Error(t, err)
			require.Equal(t, int64(2), calls.Load())
			require.Len(t, upstream.endpoints, 2)
			require.NotEqual(t, upstream.endpoints[0], upstream.endpoints[1])
			// The only attempted cache namespace belongs to the successful fallback HTTP
			// attempt. Failed native parsing neither warms it nor pins its active count.
			require.Len(t, globalKiroDynamicTracker.scopes, 1)
			req := kiropkg.KiroRequestContext{CachePayload: upstream.payloads[1], CacheEndpoint: upstream.endpoints[1]}
			input := estimateKiroInputTokens(context.Background(), dynamicProtocolBody(t, "anthropic", stream))
			p := prepareKiroDynamicCache(a, g, req, input)
			require.NotNil(t, p)
			require.Len(t, globalKiroDynamicTracker.scopes, 1)
			require.Zero(t, p.matched)
			require.Equal(t, 1, p.scope.active)
			require.Empty(t, p.scope.samples)
			p.complete(kiropkg.Usage{}, false)
			req.CacheEndpoint = upstream.endpoints[0]
			p = prepareKiroDynamicCache(a, g, req, input)
			require.NotNil(t, p)
			require.Zero(t, p.matched)
			p.complete(kiropkg.Usage{}, false)
			globalKiroDynamicTracker.mu.Lock()
			for _, scope := range globalKiroDynamicTracker.scopes {
				require.Zero(t, scope.active)
			}
			globalKiroDynamicTracker.mu.Unlock()
			finalURL, parseErr := url.Parse(upstream.endpoints[1])
			require.NoError(t, parseErr)
			require.Equal(t, "q.us-east-1.amazonaws.com", finalURL.Host)
		})
	}
}

func TestKiroDynamicWebSearchGenerationsLocalIntegration(t *testing.T) {
	t.Setenv("SUB2API_KIRO_TIME_CONTEXT", "")
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "buffered", true: "stream"}[stream], func(t *testing.T) {
			a, g, _ := dynamicSetup(t)
			a.Type = AccountTypeAPIKey
			a.Credentials = map[string]any{"api_key": "ksk_test_local_only", "api_region": "us-east-1"}
			var body map[string]any
			require.NoError(t, json.Unmarshal(dynamicProtocolBody(t, "anthropic", stream), &body))
			body["tools"] = []any{map[string]any{"name": "web_search", "description": "Search the web", "input_schema": map[string]any{"type": "object"}}}
			raw, err := json.Marshal(body)
			require.NoError(t, err)
			endpoint := kiropkg.BuildMcpEndpoint("us-east-1")
			kiroWebSearchDescCache.Store(endpoint, "Search the web")
			defer kiroWebSearchDescCache.Delete(endpoint)
			first := dynamicFrame(t, "assistantResponseEvent", map[string]any{"content": "searching", "toolUses": []any{map[string]any{"toolUseId": "next-search", "name": "web_search", "input": map[string]any{"query": "followup"}}}})
			first = append(first, dynamicFrame(t, "meteringEvent", map[string]any{"usage": 10, "unit": "credit"})...)
			second := dynamicEvents(t, 3, 20)
			var generations atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request map[string]any
				if json.NewDecoder(r.Body).Decode(&request) != nil {
					w.WriteHeader(400)
					return
				}
				if request["conversationState"] == nil {
					w.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(w, `{"jsonrpc":"2.0","result":{"content":[{"type":"text","text":"{\"results\":[]}"}]}}`)
					return
				}
				generation := generations.Add(1)
				w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
				if generation == 1 {
					_, _ = w.Write(first)
				} else {
					_, _ = w.Write(second)
				}
			}))
			defer server.Close()
			upstream := &dynamicLocalUpstream{server: server}
			svc := dynamicService(upstream)
			input := estimateKiroInputTokens(context.Background(), raw)
			if stream {
				var out bytes.Buffer
				err = svc.streamKiroWebSearchAsAnthropicWithReady(context.Background(), a, raw, "claude-opus-5", "claude-opus-5", "test-local-token", input, nil, &out, g, nil)
			} else {
				_, err = svc.executeKiroWebSearch(context.Background(), a, g, raw, "claude-opus-5", "claude-opus-5", "test-local-token", nil)
			}
			require.NoError(t, err)
			require.Equal(t, int64(2), generations.Load())
			require.Len(t, globalKiroDynamicTracker.scopes, 1)
			var scope *kiroDynamicScope
			for _, s := range globalKiroDynamicTracker.scopes {
				scope = s
			}
			require.Zero(t, scope.active)
			observed := 0
			for _, payload := range upstream.payloads {
				profile, _, ok := buildKiroOutboundCacheProfile(payload, 10000)
				if !ok {
					continue
				}
				observed++
				require.Contains(t, scope.prefixes, profile.blocks[len(profile.blocks)-1].prefixFingerprint)
			}
			require.Equal(t, 2, observed, "both successfully completed native generations warm their own authoritative prefix")
		})
	}
}
