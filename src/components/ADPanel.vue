<template>
  <div
    v-if="ad"
    class="ad-panel"
    :class="{ 'ad-panel--outside': !ad.inside_ad }"
    role="status"
    :aria-label="ariaLabel"
  >
    <div class="ad-row">
      <span class="ad-verdict" :title="verdictDetail">{{ verdict }}</span>

      <span class="ad-channels" aria-label="Orbital confidence">
        <span
          v-for="ch in readout.scored"
          :key="ch.key"
          class="ad-channel"
          :class="`ad-channel--${ch.key}`"
          :title="`${ch.key}-orbital confidence`"
        >
          <span class="ad-channel-orb">{{ ch.key }}</span>
          <span class="ad-channel-val">{{ ch.label }}</span>
        </span>
        <span
          v-if="readout.absentLabel"
          class="ad-absent"
          :title="readout.absentDetail ?? undefined"
        >
          <span class="ad-absent-mark" aria-hidden="true">·</span>
          {{ readout.absentLabel }}
        </span>
      </span>

      <span class="ad-score" :title="confidenceDetail">
        <span class="ad-score-value">{{ confidence }}</span>
        <span class="ad-score-label">confidence</span>
        <span class="ad-threshold">cutoff {{ cutoff }}</span>
      </span>
    </div>

    <p v-if="ad.f_element_warning" class="ad-f-warning">
      f-orbital confidence is weakly calibrated for this element. Treat it with caution.
    </p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import {
  adChannelReadout,
  adConfidenceDetail,
  adVerdict,
  adVerdictDetail,
  formatAdPercent,
  type AdResult,
} from '../ad'

const props = defineProps<{
  ad: AdResult | null
}>()

const verdict = computed(() => (props.ad ? adVerdict(props.ad.inside_ad) : ''))
const verdictDetail = computed(() => (props.ad ? adVerdictDetail(props.ad.inside_ad) : ''))
const readout = computed(() => (props.ad ? adChannelReadout(props.ad) : { scored: [], absentLabel: null, absentDetail: null }))
const confidence = computed(() => (props.ad ? formatAdPercent(props.ad.p_ensemble_all) : ''))
const cutoff = computed(() => (props.ad ? formatAdPercent(props.ad.ad_threshold) : ''))
const confidenceDetail = computed(() => adConfidenceDetail(Boolean(readout.value.absentLabel)))

const ariaLabel = computed(() => {
  const ad = props.ad
  if (!ad) return 'Prediction reliability unavailable'
  const orbitals = readout.value.scored.map(ch => `${ch.key} ${ch.label}`).join(', ')
  const absent = readout.value.absentLabel ? `, ${readout.value.absentLabel}` : ''
  return `${verdict.value}. ${orbitals}${absent}. Confidence ${confidence.value}, cutoff ${cutoff.value}.`
})
</script>

<style scoped>
.ad-panel {
  padding: 0.7rem 1.5rem;
  border: 1px solid var(--border-color);
  background: var(--bg-tertiary);
  border-radius: 12px;
  flex-shrink: 0;
}

.ad-panel--outside {
  background: rgba(245, 158, 11, 0.1);
  border-color: rgba(245, 158, 11, 0.55);
}

.ad-row {
  display: flex;
  align-items: center;
  gap: 1rem 1.25rem;
  flex-wrap: wrap;
}

.ad-verdict {
  font-size: 0.92rem;
  font-weight: 600;
  color: var(--success);
  flex-shrink: 0;
}

.ad-panel--outside .ad-verdict {
  color: var(--warning);
}

.ad-channels {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem;
  min-width: 0;
}

.ad-channel {
  display: inline-flex;
  align-items: baseline;
  gap: 0.35rem;
  padding: 0.12rem 0.45rem;
  border-radius: 4px;
  background: var(--bg-secondary);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
}

.ad-channel-orb {
  font-weight: 700;
}

.ad-channel--s .ad-channel-orb { color: #3b82f6; }
.ad-channel--p .ad-channel-orb { color: #10b981; }
.ad-channel--d .ad-channel-orb { color: #f59e0b; }
.ad-channel--f .ad-channel-orb { color: #ef4444; }

.ad-channel-val {
  color: var(--text-primary);
}

.ad-absent {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.ad-absent-mark {
  margin-right: 0.35rem;
  color: var(--text-secondary);
}

.ad-score {
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
  margin-left: auto;
  font-family: 'JetBrains Mono', monospace;
}

.ad-score-value {
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--text-primary);
}

.ad-score-label,
.ad-threshold {
  font-family: 'Outfit', sans-serif;
  font-size: 0.75rem;
  color: var(--text-secondary);
}

.ad-f-warning {
  margin: 0.45rem 0 0;
  font-size: 0.75rem;
  line-height: 1.4;
  color: var(--warning);
}

@media (max-width: 860px) {
  .ad-score {
    margin-left: 0;
  }
}
</style>
