package service

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"io"
	"math"
	"sync"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/anthropictokenizer"
	kiropkg "github.com/Wei-Shaw/sub2api/internal/pkg/kiro"
)

const (
	kiroDynamicMaxScopes        = 1024
	kiroDynamicMaxAccountScopes = 32
	kiroDynamicMaxPrefixes      = 256
	kiroDynamicMaxSamples       = 64
	kiroDynamicCalibrationTTL   = 30 * time.Minute
)

type kiroDynamicSampleKey struct {
	shape  [32]byte // inference configuration discriminator
	prefix [32]byte
	output int
}
type kiroDynamicSample struct {
	cold    float64
	expires time.Time
}
type kiroDynamicPair struct {
	ratio   float64
	expires time.Time
}

type kiroDynamicScope struct {
	account  int64
	expires  time.Time
	prefixes map[[32]byte]time.Time
	samples  map[kiroDynamicSampleKey]kiroDynamicSample
	pairs    []kiroDynamicPair
	active   int
	epoch    uint64
}
type kiroDynamicTracker struct {
	mu     sync.Mutex
	now    func() time.Time
	scopes map[[32]byte]*kiroDynamicScope
}

func newKiroDynamicTracker(now func() time.Time) *kiroDynamicTracker {
	return &kiroDynamicTracker{now: now, scopes: make(map[[32]byte]*kiroDynamicScope)}
}

var globalKiroDynamicTracker = newKiroDynamicTracker(time.Now)

type kiroCacheEmulationPlan struct {
	usage     *kiroCacheEmulationUsage
	tracker   *kiroDynamicTracker
	key       [32]byte
	scope     *kiroDynamicScope
	profile   *kiroCacheProfile
	shape     [32]byte
	ttl       time.Duration
	started   time.Time
	matched   int
	epoch     uint64
	exclusive bool
	once      sync.Once
}

func (p *kiroCacheEmulationPlan) result() *kiroCacheEmulationUsage {
	if p == nil {
		return nil
	}
	return p.usage
}
func kiroDynamicTTL(model string) time.Duration {
	switch model {
	case "claude-opus-5":
		return 5 * time.Minute
	case "gpt-5.6-terra":
		return 30 * time.Minute
	default:
		return 0
	}
}

// This profile hashes the native translated semantics, in order. Current and
// historical user messages use the same wrapper so advancing a turn preserves
// the prefix ladder. Tools are a prelude, not sorted or rewritten. No inbound
// metadata, raw byte prefixes, token credentials or generated conversation IDs
// (except Terra's actual session scope) participate.
func buildKiroOutboundCacheProfile(payload []byte, input int) (*kiroCacheProfile, kiropkg.KiroPayload, bool) {
	var native kiropkg.KiroPayload
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	if decoder.Decode(&native) != nil {
		return nil, native, false
	}
	var trailing any
	if decoder.Decode(&trailing) != io.EOF {
		return nil, native, false
	}
	model := native.ConversationState.CurrentMessage.UserInputMessage.ModelID
	ttl := kiroDynamicTTL(model)
	if ttl == 0 || input <= 0 {
		return nil, native, false
	}
	var blocks []kiroPendingBlock
	add := func(v any) {
		tokenValue := v
		imageTokens := 0
		if history, ok := v.(kiropkg.KiroHistoryMessage); ok && history.UserInputMessage != nil && len(history.UserInputMessage.Images) > 0 {
			user := *history.UserInputMessage
			for _, image := range user.Images {
				imageTokens += kiropkg.EstimateImageTokens(context.Background(), "image/"+image.Format, image.Source.Bytes)
			}
			user.Images = nil
			history.UserInputMessage = &user
			tokenValue = history
		}
		raw, err := canonicalJSON(tokenValue)
		if err != nil {
			return
		}
		blocks = append(blocks, kiroPendingBlock{value: v, tokens: anthropictokenizer.CountTokens(string(raw)) + imageTokens, breakpointTTL: &ttl})
	}
	current := native.ConversationState.CurrentMessage.UserInputMessage
	if current.UserInputMessageContext != nil {
		for _, tool := range current.UserInputMessageContext.Tools {
			add(map[string]any{"tool": tool})
		}
	}
	// All model-visible native message fields are retained. Tools in the current
	// context are already represented in the ordered prelude.
	for _, history := range native.ConversationState.History {
		add(history)
	}
	if current.UserInputMessageContext != nil {
		c := *current.UserInputMessageContext
		c.Tools = nil
		current.UserInputMessageContext = &c
		if len(c.ToolResults) == 0 {
			current.UserInputMessageContext = nil
		}
	}
	add(kiropkg.KiroHistoryMessage{UserInputMessage: &current})
	prelude := map[string]any{"model": model, "fields": native.AdditionalModelRequestFields,
		"task": native.ConversationState.AgentTaskType, "trigger": native.ConversationState.ChatTriggerType,
		"continuation": native.ConversationState.AgentContinuationID}
	profile, ok := buildKiroCacheProfileFromBlocks(model, input, prelude, blocks)
	if ok {
		profile.scaleBreakpointsToInputTokens = true
	}
	return profile, native, ok
}

func prepareKiroDynamicCache(account *Account, group *Group, request kiropkg.KiroRequestContext, input int) *kiroCacheEmulationPlan {
	// Ratio/mode fields remain accepted by config/API/UI for compatibility, but
	// are deprecated and deliberately ignored by this estimator.
	if account == nil || account.ID <= 0 || group == nil || !group.EffectiveKiroCacheEmulationEnabled() || request.CacheEndpoint == "" {
		return nil
	}
	profile, native, ok := buildKiroOutboundCacheProfile(request.CachePayload, input)
	if !ok {
		return nil
	}
	model := native.ConversationState.CurrentMessage.UserInputMessage.ModelID
	session := ""
	if model == "gpt-5.6-terra" {
		session = native.ConversationState.ConversationID
		if session == "" {
			return nil
		}
	}
	// Account identity is stable across OAuth access/refresh rotation. A stable
	// provider principal, when supplied, further isolates account replacement.
	var principalHash [32]byte
	if native.ProfileArn == "" && account.Type == AccountTypeAPIKey {
		principalHash = request.CachePrincipalHash
		if principalHash == ([32]byte{}) {
			// Matches GatewayService.GetAccessToken's credential selection. Runtime
			// supplies the hash of the actual outbound key, including retries.
			principal := account.GetCredential("api_key")
			if principal == "" {
				return nil
			}
			principalHash = sha256.Sum256([]byte(principal))
		}
	}
	scopeJSON, _ := canonicalJSON([]any{account.ID, principalHash, account.GetCredential("user_id"), account.GetCredential("principal_id"), native.ProfileArn, request.CacheEndpoint, model, session})
	shapeJSON, _ := canonicalJSON(native.InferenceConfig)
	return globalKiroDynamicTracker.prepare(account.ID, sha256.Sum256(scopeJSON), sha256.Sum256(shapeJSON), profile, kiroDynamicTTL(model))
}

func (t *kiroDynamicTracker) prepare(account int64, key, shape [32]byte, profile *kiroCacheProfile, ttl time.Duration) *kiroCacheEmulationPlan {
	now := t.now()
	t.mu.Lock()
	defer t.mu.Unlock()
	t.prune(now)
	scope := t.scopes[key]
	if scope == nil {
		count := 0
		for _, s := range t.scopes {
			if s.account == account {
				count++
			}
		}
		// Reject new scopes when full; one account cannot evict another's warmth.
		if len(t.scopes) >= kiroDynamicMaxScopes || count >= kiroDynamicMaxAccountScopes {
			return nil
		}
		scope = &kiroDynamicScope{account: account, expires: now.Add(kiroDynamicCalibrationTTL), prefixes: make(map[[32]byte]time.Time), samples: make(map[kiroDynamicSampleKey]kiroDynamicSample)}
		t.scopes[key] = scope
	}
	matched := 0
	for _, bp := range profile.cacheableBreakpoints() {
		if scope.prefixes[profile.blocks[bp.blockIndex].prefixFingerprint].After(now) {
			matched = max(matched, profile.cacheTokensForBreakpoint(bp.cumulativeTokens))
		}
	}
	scope.epoch++
	plan := &kiroCacheEmulationPlan{tracker: t, key: key, scope: scope, profile: profile, shape: shape, ttl: ttl, started: now, matched: matched, epoch: scope.epoch, exclusive: scope.active == 0}
	scope.active++
	if len(scope.pairs) >= 2 && matched > 0 {
		// Lower envelope of the last eight matched pairs. Downward evidence
		// applies immediately; old outliers age out so calibration can recover.
		ratio := .9
		for _, pair := range scope.pairs {
			ratio = math.Min(ratio, pair.ratio)
		}
		read := int(float64(matched) * ratio)
		read = min(max(read, 0), profile.totalInputTokens)
		if read > 0 {
			plan.usage = &kiroCacheEmulationUsage{InputTokens: profile.totalInputTokens - read, CacheReadInputTokens: read}
		}
	}
	return plan
}

// complete is idempotent. The caller invokes it only after native EOF, parser
// success and successful writes. Failed/aborted requests never renew warmth or
// calibrate. Frozen plan.usage is never mutated, including by late metering.
func (p *kiroCacheEmulationPlan) complete(usage kiropkg.Usage, success bool) {
	if p == nil {
		return
	}
	p.once.Do(func() {
		t := p.tracker
		now := t.now()
		t.mu.Lock()
		defer t.mu.Unlock()
		s := p.scope
		s.active--
		if !success || !usage.CacheObservationComplete() || usage.KiroCredits <= 0 || math.IsNaN(usage.KiroCredits) || math.IsInf(usage.KiroCredits, 0) || usage.OutputTokens < 0 {
			return
		}
		if t.scopes[p.key] != s {
			return
		}
		// TTL is anchored at observation start, conservatively excluding long streams.
		expires := p.started.Add(p.ttl)
		if !expires.After(now) {
			return
		}
		if p.exclusive && p.epoch == s.epoch {
			k := kiroDynamicSampleKey{shape: p.shape, output: usage.OutputTokens, prefix: p.profile.blocks[len(p.profile.blocks)-1].prefixFingerprint}
			sample, exists := s.samples[k]
			if exists && !sample.expires.After(now) {
				delete(s.samples, k)
				exists = false
			}
			if p.matched == 0 {
				if !exists || usage.KiroCredits < sample.cold {
					sample = kiroDynamicSample{cold: usage.KiroCredits, expires: now.Add(kiroDynamicCalibrationTTL)}
				}
				if exists || len(s.samples) < kiroDynamicMaxSamples {
					s.samples[k] = sample
				}
			} else {
				// Pair only a canonical cold terminal prefix that occurs on
				// this request's warm ladder, with identical output count and
				// inference configuration. Appended uncached input increases
				// later credits, so ignoring its cost UNDERestimates savings;
				// no input/output price slope is assumed or fitted.
				breakpoints := p.profile.cacheableBreakpoints()
				for i := len(breakpoints) - 1; i >= 0; i-- {
					bp := breakpoints[i]
					if p.profile.cacheTokensForBreakpoint(bp.cumulativeTokens) > p.matched {
						continue
					}
					k.prefix = p.profile.blocks[bp.blockIndex].prefixFingerprint
					cold, found := s.samples[k]
					if !found || !cold.expires.After(now) {
						continue
					}
					ratio := math.Min(.9, math.Max(0, (cold.cold-usage.KiroCredits)/cold.cold))
					s.pairs = append(s.pairs, kiroDynamicPair{ratio: ratio, expires: now.Add(kiroDynamicCalibrationTTL)})
					if len(s.pairs) > 8 {
						s.pairs = s.pairs[len(s.pairs)-8:]
					}
					break
				}
			}
		}
		s.expires = now.Add(kiroDynamicCalibrationTTL)
		for _, bp := range p.profile.cacheableBreakpoints() {
			fp := p.profile.blocks[bp.blockIndex].prefixFingerprint
			if old, exists := s.prefixes[fp]; exists {
				if expires.After(old) {
					s.prefixes[fp] = expires
				}
			} else if len(s.prefixes) < kiroDynamicMaxPrefixes {
				s.prefixes[fp] = expires
			}
		}
	})
}

func (t *kiroDynamicTracker) prune(now time.Time) {
	for key, s := range t.scopes {
		if !s.expires.After(now) && s.active == 0 {
			delete(t.scopes, key)
			continue
		}
		for fp, expires := range s.prefixes {
			if !expires.After(now) {
				delete(s.prefixes, fp)
			}
		}
		for k, sample := range s.samples {
			if !sample.expires.After(now) {
				delete(s.samples, k)
			}
		}
		pairs := s.pairs[:0]
		for _, pair := range s.pairs {
			if pair.expires.After(now) {
				pairs = append(pairs, pair)
			}
		}
		s.pairs = pairs
	}
}
