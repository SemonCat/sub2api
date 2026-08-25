package migrations

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// TestUserPlatformQuotasCNProvidersMigration 校验 224 号迁移把 kimi/zhipu/deepseek
// 加入 user_platform_quotas.platform 的 CHECK 约束，同时保留此前已允许的所有平台。
// 特别是数据库可能已有 kiro 配额行；重建约束时漏掉 kiro 会使迁移自身失败，后续迁移
// 无法执行。
func TestUserPlatformQuotasCNProvidersMigration(t *testing.T) {
	content, err := FS.ReadFile("224_user_platform_quotas_add_cn_providers.sql")
	require.NoError(t, err)

	sql := strings.Join(strings.Fields(string(content)), " ")
	require.Contains(t, sql, "DROP CONSTRAINT IF EXISTS user_platform_quotas_platform_check")
	require.Contains(t, sql,
		"CHECK (platform IN ('anthropic', 'openai', 'gemini', 'antigravity', 'grok', 'kimi', 'zhipu', 'deepseek', 'kiro'))")
}
