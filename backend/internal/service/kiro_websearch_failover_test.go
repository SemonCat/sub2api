package service

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	kiropkg "github.com/Wei-Shaw/sub2api/internal/pkg/kiro"
	"github.com/Wei-Shaw/sub2api/internal/pkg/kirocooldown"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/stretchr/testify/require"
)

type kiroWebSearchTestUpstream struct {
	responses []*http.Response
	requests  []*http.Request
}

type kiroWebSearchTestCooldownStore struct{}

func (*kiroWebSearchTestCooldownStore) CheckCooldown(context.Context, string) error {
	return nil
}

func (*kiroWebSearchTestCooldownStore) MarkSuccess(context.Context, string) error {
	return nil
}

func (*kiroWebSearchTestCooldownStore) Mark429(context.Context, string) (time.Duration, error) {
	return time.Minute, nil
}

func (*kiroWebSearchTestCooldownStore) MarkSuspended(context.Context, string) (time.Duration, error) {
	return time.Minute, nil
}

func (*kiroWebSearchTestCooldownStore) GetState(context.Context, string) (*kirocooldown.State, error) {
	return nil, nil
}

func (*kiroWebSearchTestCooldownStore) ClearEarliestTransientCooldown(context.Context, []string) (bool, error) {
	return false, nil
}

func (u *kiroWebSearchTestUpstream) Do(_ *http.Request, _ string, _ int64, _ int) (*http.Response, error) {
	return nil, fmt.Errorf("unexpected Do call")
}

func (u *kiroWebSearchTestUpstream) DoWithTLS(req *http.Request, _ string, _ int64, _ int, _ *tlsfingerprint.Profile) (*http.Response, error) {
	u.requests = append(u.requests, req)
	if len(u.responses) == 0 {
		return nil, fmt.Errorf("no mocked response")
	}
	resp := u.responses[0]
	u.responses = u.responses[1:]
	return resp, nil
}

func newKiroWebSearchTestResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func newKiroWebSearchEventStreamResponse(t *testing.T, events ...[]byte) *http.Response {
	t.Helper()
	body := bytes.NewBuffer(nil)
	for _, event := range events {
		_, _ = body.Write(event)
	}
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/vnd.amazon.eventstream"}},
		Body:       io.NopCloser(body),
	}
}

func kiroWebSearchEventStreamFrame(t *testing.T, eventType string, payload any) []byte {
	t.Helper()
	payloadBytes, err := json.Marshal(payload)
	require.NoError(t, err)

	headers := bytes.NewBuffer(nil)
	_ = headers.WriteByte(byte(len(":event-type")))
	_, _ = headers.WriteString(":event-type")
	_ = headers.WriteByte(7)
	require.NoError(t, binary.Write(headers, binary.BigEndian, uint16(len(eventType))))
	_, _ = headers.WriteString(eventType)

	totalLength := uint32(12 + headers.Len() + len(payloadBytes) + 4)
	frame := bytes.NewBuffer(nil)
	require.NoError(t, binary.Write(frame, binary.BigEndian, totalLength))
	require.NoError(t, binary.Write(frame, binary.BigEndian, uint32(headers.Len())))
	require.NoError(t, binary.Write(frame, binary.BigEndian, uint32(0)))
	_, _ = frame.Write(headers.Bytes())
	_, _ = frame.Write(payloadBytes)
	require.NoError(t, binary.Write(frame, binary.BigEndian, uint32(0)))
	return frame.Bytes()
}

func TestOpenKiroAnthropicStreamResponsePropagatesWebSearchFailoverBeforeSynthetic200(t *testing.T) {
	body := kiroCacheRequestBody("stream failover", false)
	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	payload["tools"] = []any{map[string]any{
		"name":         "web_search",
		"description":  "Search the web",
		"input_schema": map[string]any{"type": "object"},
	}}
	body, err := json.Marshal(payload)
	require.NoError(t, err)

	endpoint := kiropkg.BuildMcpEndpoint("us-east-1")
	kiroWebSearchDescCache.Store(endpoint, "Search the web")
	t.Cleanup(func() { kiroWebSearchDescCache.Delete(endpoint) })

	upstream := &kiroWebSearchTestUpstream{responses: []*http.Response{
		newKiroWebSearchTestResponse(http.StatusOK, `{"jsonrpc":"2.0","result":{"content":[{"type":"text","text":"{\"results\":[]}"}]}}`),
		newKiroWebSearchTestResponse(http.StatusPaymentRequired, `{"message":"payment required"}`),
	}}
	svc := &GatewayService{
		httpUpstream:        upstream,
		kiroCooldownStore:   &kiroWebSearchTestCooldownStore{},
		tlsFPProfileService: &TLSFingerprintProfileService{},
	}
	account := &Account{
		ID:          992,
		Platform:    PlatformKiro,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{"api_key": "ksk_test", "api_region": "us-east-1"},
	}

	resp, _, openErr := svc.openKiroAnthropicStreamResponse(
		context.Background(),
		account,
		nil,
		body,
		"gpt-5.6-sol",
		"gpt-5.6-sol",
		nil,
		kiroCacheGroup(1),
		nil,
	)
	if resp != nil {
		_, _ = io.ReadAll(resp.Body)
		_ = resp.Body.Close()
	}

	var failoverErr *UpstreamFailoverError
	require.True(t, errors.As(openErr, &failoverErr), "got error %v", openErr)
	require.Equal(t, http.StatusPaymentRequired, failoverErr.StatusCode)
	require.Nil(t, resp, "failover must surface before a synthetic HTTP 200 is returned")
	require.Len(t, upstream.requests, 2)
}

func TestOpenKiroAnthropicStreamResponsePreservesWebSearchStreamAfterPreflight(t *testing.T) {
	body := kiroCacheRequestBody("successful stream", false)
	var payload map[string]any
	require.NoError(t, json.Unmarshal(body, &payload))
	payload["tools"] = []any{map[string]any{
		"name":         "web_search",
		"description":  "Search the web",
		"input_schema": map[string]any{"type": "object"},
	}}
	body, err := json.Marshal(payload)
	require.NoError(t, err)

	endpoint := kiropkg.BuildMcpEndpoint("us-east-1")
	kiroWebSearchDescCache.Store(endpoint, "Search the web")
	t.Cleanup(func() { kiroWebSearchDescCache.Delete(endpoint) })

	upstream := &kiroWebSearchTestUpstream{responses: []*http.Response{
		newKiroWebSearchTestResponse(http.StatusOK, `{"jsonrpc":"2.0","result":{"content":[{"type":"text","text":"{\"results\":[]}"}]}}`),
		newKiroWebSearchEventStreamResponse(
			t,
			kiroWebSearchEventStreamFrame(t, "assistantResponseEvent", map[string]any{
				"assistantResponseEvent": map[string]any{"content": "REAL_STREAM_OK"},
			}),
			kiroWebSearchEventStreamFrame(t, "meteringEvent", map[string]any{
				"meteringEvent": map[string]any{"unit": "credit", "usage": 0.1},
			}),
		),
	}}
	svc := &GatewayService{
		httpUpstream:        upstream,
		kiroCooldownStore:   &kiroWebSearchTestCooldownStore{},
		tlsFPProfileService: &TLSFingerprintProfileService{},
	}
	account := &Account{
		ID:          993,
		Platform:    PlatformKiro,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{"api_key": "ksk_test", "api_region": "us-east-1"},
	}

	resp, _, openErr := svc.openKiroAnthropicStreamResponse(
		context.Background(),
		account,
		nil,
		body,
		"gpt-5.6-sol",
		"gpt-5.6-sol",
		nil,
		kiroCacheGroup(1),
		nil,
	)
	require.NoError(t, openErr)
	require.NotNil(t, resp)
	require.Equal(t, http.StatusOK, resp.StatusCode)

	responseBody, readErr := io.ReadAll(resp.Body)
	require.NoError(t, readErr)
	require.NoError(t, resp.Body.Close())
	require.Contains(t, string(responseBody), "event: message_start")
	require.Contains(t, string(responseBody), "REAL_STREAM_OK")
	require.Len(t, upstream.requests, 2)
}
