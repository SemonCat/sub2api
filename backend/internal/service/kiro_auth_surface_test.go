//go:build unit

package service

import (
	"context"
	"errors"
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
)

type kiroSurfaceRefreshSpy struct {
	stubKiroAccountTokenRefresher
	calls int
}

func (s *kiroSurfaceRefreshSpy) RefreshAccountToken(context.Context, *Account) (*KiroTokenInfo, error) {
	s.calls++
	return nil, errors.New("test refresh unavailable")
}

func TestKiroAuthSurfacePassthrough(t *testing.T) {
	tests := []struct {
		name     string
		status   int
		body     string
		category string
	}{
		{"bad request", 400, `{"message":"Invalid model ID"}`, kiroErrorBadRequestInvalidModel},
		{"not found", 404, `{"message":"Not found"}`, kiroErrorUpstreamTransient},
		{"too large", 413, `{"message":"Request too large"}`, kiroErrorUpstreamTransient},
		{"quota", 403, `{"message":"Quota exceeded"}`, kiroErrorQuotaExhausted},
		{"profile", 403, `{"message":"profileArn is required"}`, kiroErrorProfileError},
		{"invalid profile", 403, `{"message":"Invalid profile"}`, kiroErrorProfileError},
		// The existing classifier's broad "invalid" match is not proof of token failure.
		{"model", 403, `{"message":"Invalid model ID","reason":"INVALID_MODEL_ID"}`, kiroErrorAuthError},
		{"content policy", 403, `{"message":"Content policy violation"}`, kiroErrorUsageForbidden},
		{"invalid content", 403, `{"message":"Invalid content: policy violation"}`, kiroErrorAuthError},
		{"monthly limit", 403, `{"message":"Monthly limit reached","reason":"MONTHLY_REQUEST_COUNT"}`, kiroErrorUsageForbidden},
		{"unknown forbidden", 403, `{"message":"Forbidden"}`, kiroErrorUsageForbidden},
		{"generic with policy reason", 403, `{"message":"User is not authorized to make this call.","reason":"CONTENT_POLICY_VIOLATION"}`, kiroErrorUsageForbidden},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			require.Equal(t, tt.category, classifyKiroHTTPError(tt.status, tt.body).Category)
			for _, gateway := range []bool{true, false} {
				name := "account-test"
				if gateway {
					name = "gateway"
				}
				t.Run(name, func(t *testing.T) {
					account := &Account{ID: 44, Platform: PlatformKiro, Type: AccountTypeOAuth, Concurrency: 1,
						Credentials: map[string]any{"auth_method": "idc", "profile_arn": "arn:aws:codewhisperer:us-east-1:123456789012:profile/TEST", "refresh_token": "test-refresh"}}
					repo := &recordingKiroTempUnschedRepo{mockAccountRepoForGemini: mockAccountRepoForGemini{accountsByID: map[int64]*Account{account.ID: account}}}
					first := newJSONResponse(tt.status, tt.body)
					first.Header.Set("x-request-id", "original-error")
					upstream := &queuedHTTPUpstream{responses: []*http.Response{first,
						newJSONResponse(403, `{"message":"Invalid bearer token"}`),
						newJSONResponse(403, `{"message":"Invalid bearer token"}`)}}
					refresh := &kiroSurfaceRefreshSpy{}
					provider := NewKiroTokenProvider(repo, nil, nil)
					provider.kiroOAuthService = refresh
					body := []byte(`{"model":"gpt-5.6-luna","messages":[{"role":"user","content":"hello"}]}`)
					var resp *http.Response
					var err error
					if gateway {
						svc := &GatewayService{accountRepo: repo, httpUpstream: upstream, kiroTokenProvider: provider, kiroCooldownStore: &stubKiroCooldownStore{}, tlsFPProfileService: &TLSFingerprintProfileService{}}
						resp, _, err = svc.executeKiroUpstreamWithParsed(context.Background(), account, &ParsedRequest{Group: &Group{Platform: PlatformKiro, KiroEndpointMode: KiroEndpointModeAuto}}, body, "gpt-5.6-luna", "gpt-5.6-luna", "test-token", nil)
					} else {
						svc := &AccountTestService{httpUpstream: upstream, kiroTokenProvider: provider, tlsFPProfileService: &TLSFingerprintProfileService{}}
						resp, err = svc.executeKiroTestUpstream(context.Background(), account, body, "gpt-5.6-luna", "test-token")
					}
					require.NoError(t, err)
					require.NotNil(t, resp)
					got, err := io.ReadAll(resp.Body)
					require.NoError(t, err)
					require.NoError(t, resp.Body.Close())
					// Non-fatal assertions expose both masking and the scheduler side effect on red.
					assert := require.New(t)
					if repo.called {
						t.Errorf("healthy account marked unavailable: %s", repo.reason)
					}
					if refresh.calls != 0 {
						t.Errorf("deterministic response refreshed credentials %d times", refresh.calls)
					}
					assert.Equal(tt.status, resp.StatusCode)
					assert.Equal(tt.body, string(got))
					assert.Equal("original-error", resp.Header.Get("x-request-id"))
					assert.Len(upstream.requests, 1)
				})
			}
		})
	}
}

func TestKiroAccountTestAuthSurfaceFallback(t *testing.T) {
	for _, first := range []struct {
		name   string
		status int
		body   string
	}{
		{"generic mismatch", 403, `{"message":"User is not authorized to make this call.","reason":null}`},
		{"expired token", 403, `{"message":"token expired"}`},
		{"unauthorized", 401, `{"message":"Unauthorized"}`},
	} {
		t.Run(first.name, func(t *testing.T) {
			account := &Account{ID: 44, Platform: PlatformKiro, Type: AccountTypeOAuth, Concurrency: 1, Credentials: map[string]any{"auth_method": "idc", "profile_arn": "arn:aws:codewhisperer:us-east-1:123456789012:profile/TEST"}}
			upstream := &queuedHTTPUpstream{responses: []*http.Response{newJSONResponse(first.status, first.body), newJSONResponse(403, `{"message":"User is not authorized to make this call."}`), newJSONResponse(200, `{"ok":true}`)}}
			svc := &AccountTestService{httpUpstream: upstream, tlsFPProfileService: &TLSFingerprintProfileService{}}
			resp, err := svc.executeKiroTestUpstream(context.Background(), account, []byte(`{"messages":[{"role":"user","content":"hello"}]}`), "gpt-5.6-luna", "test-token")
			require.NoError(t, err)
			require.NoError(t, resp.Body.Close())
			require.Equal(t, 200, resp.StatusCode)
			require.Len(t, upstream.requests, 3)
			require.Equal(t, "codewhisperer.us-east-1.amazonaws.com", upstream.requests[0].URL.Host)
			require.Equal(t, "q.us-east-1.amazonaws.com", upstream.requests[1].URL.Host)
			require.Equal(t, "runtime.us-east-1.kiro.dev", upstream.requests[2].URL.Host)
		})
	}
}
