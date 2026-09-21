<template>
  <div
    v-if="ad"
    class="ad-panel"
    :class="{ 'ad-panel--outside': !ad.inside_ad }"
    role="status"
    :aria-label="ariaLabel"
  >
    <div class="ad-row">
      <span class="ad-title">Applicability Domain</span>
      <span
        class="ad-badge"
        :class="ad.inside_ad ? 'ad-badge--inside' : 'ad-badge--outside'"
      >
        {{ ad.inside_ad ? 'INSIDE' : 'OUTSIDE' }}
      </span>
      <span class="ad-score">
        <span class="ad-score-label">p_ensemble_all</span>
        <span class="ad-score-value">{{ formatProb(ad.p_ensemble_all) }}</span>
        <span class="ad-threshold">threshold {{ formatProb(ad.ad_threshold) }}</span>
      </span>
    </div>

    <div class="ad-channels" aria-label="Orbital channel probabilities">
      <span
        v-for="ch in channels"
        :key="ch.key"
        class="ad-channel"
        :class="[`ad-channel--${ch.key}`, { 'ad-channel--inactive': ch.inactive }]"
        :title="ch.inactive ? `${ch.key}-channel inactive (element-block policy)` : `${ch.key}-channel ensemble probability`"
      >
        <span class="ad-channel-orb">{{ ch.key }}</span>
        <span class="ad-channel-val">{{ formatProb(ch.value) }}</span>
      </span>
    </div>

    <p v-if="ad.f_element_warning" class="ad-f-warning">
      f-channel calibration is sparse for f-block elements. Treat f-orbital AD scores with caution.
    </p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { formatAdProbability, type AdResult } from '../ad'

const props = defineProps<{
  ad: AdResult | null
}>()

const formatProb = formatAdProbability

const channels = computed(() => {
  const ad = props.ad
  if (!ad) return []
  return [
    { key: 's', value: ad.p_ensemble_s, inactive: false },
    { key: 'p', value: ad.p_ensemble_p, inactive: false },
    { key: 'd', value: ad.p_ensemble_d, inactive: !ad.active_d },
    { key: 'f', value: ad.p_ensemble_f, inactive: !ad.active_f },
  ]
})

const ariaLabel = computed(() => {
  const ad = props.ad
  if (!ad) return 'Applicability domain unavailable'
  const status = ad.inside_ad ? 'inside' : 'outside'
  return `Applicability domain: ${status}, p_ensemble_all ${formatProb(ad.p_ensemble_all)}`
})
</script>


<style scoped>
.ad-panel {
  padding: 0.65rem 1.5rem 0.75rem;
  border-bottom: 1px solid var(--border-color);
  background: var(--bg-tertiary);
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.ad-panel--outside {
  background: rgba(245, 158, 11, 0.08);
  border-bottom-color: rgba(245, 158, 11, 0.4);
}

.ad-row {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-wrap: wrap;
}

.ad-title {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
}

.ad-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
}

.ad-badge--inside {
  background: rgba(16, 185, 129, 0.18);
  color: var(--success);
  border: 1px solid rgba(16, 185, 129, 0.45);
}

.ad-badge--outside {
  background: rgba(245, 158, 11, 0.2);
  color: var(--warning);
  border: 1px solid rgba(245, 158, 11, 0.55);
}

.ad-score {
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
}

.ad-score-label {
  font-size: 0.7rem;
  color: var(--text-muted);
}

.ad-score-value {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-primary);
}

.ad-threshold {
  font-size: 0.7rem;
  color: var(--text-secondary);
}

.ad-channels {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.ad-channel {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  border: 1px solid var(--border-color);
  background: var(--bg-secondary);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
}

.ad-channel-orb {
  font-weight: 700;
  text-transform: lowercase;
}

.ad-channel--s .ad-channel-orb { color: #3b82f6; }
.ad-channel--p .ad-channel-orb { color: #10b981; }
.ad-channel--d .ad-channel-orb { color: #f59e0b; }
.ad-channel--f .ad-channel-orb { color: #ef4444; }

.ad-channel-val {
  color: var(--text-primary);
}

.ad-channel--inactive {
  opacity: 0.55;
}

.ad-f-warning {
  margin: 0;
  font-size: 0.75rem;
  line-height: 1.4;
  color: var(--warning);
}

@media (max-width: 720px) {
  .ad-score {
    margin-left: 0;
  }
}
</style>
