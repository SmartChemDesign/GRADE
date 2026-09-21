export interface AdResult {
  inside_ad: boolean
  ad_threshold: number
  p_ensemble_all: number
  p_ensemble_s: number
  p_ensemble_p: number
  p_ensemble_d: number
  p_ensemble_f: number
  raw_knn: number
  raw_kde: number
  raw_lof: number
  active_d: boolean
  active_f: boolean
  contains_f_element: boolean
  f_element_warning: boolean
}

export const AD_JSON_FIELDS = [
  'inside_ad',
  'ad_threshold',
  'p_ensemble_all',
  'p_ensemble_s',
  'p_ensemble_p',
  'p_ensemble_d',
  'p_ensemble_f',
  'raw_knn',
  'raw_kde',
  'raw_lof',
  'active_d',
  'active_f',
  'contains_f_element',
  'f_element_warning',
] as const

const boolCsv = (value: boolean): string => (value ? 'true' : 'false')

const ORBITALS = ['s', 'p', 'd', 'f'] as const
export type OrbitalKey = (typeof ORBITALS)[number]

export function formatAdProbability(value: number): string {
  if (!Number.isFinite(value)) return '—'
  return value.toFixed(2)
}

export function formatAdPercent(value: number): string {
  if (!Number.isFinite(value)) return '—'
  return `${Math.round(value * 100)}%`
}

export function adVerdict(inside: boolean): string {
  return inside ? "In the model's range" : "Beyond the model's range"
}

export function adVerdictDetail(inside: boolean): string {
  if (inside) {
    return 'Applicability domain: this crystal is close to structures the model was calibrated on.'
  }
  return 'Applicability domain: this crystal is far from structures the model was calibrated on. Treat the DOS as a rough estimate.'
}

export interface AdChannelScore {
  key: OrbitalKey
  label: string
}

export interface AdChannelReadout {
  scored: AdChannelScore[]
  absentLabel: string | null
  absentDetail: string | null
}

function orbitalActive(ad: AdResult, key: OrbitalKey): boolean {
  if (key === 'd') return ad.active_d
  if (key === 'f') return ad.active_f
  return true
}

function orbitalValue(ad: AdResult, key: OrbitalKey): number {
  if (key === 's') return ad.p_ensemble_s
  if (key === 'p') return ad.p_ensemble_p
  if (key === 'd') return ad.p_ensemble_d
  return ad.p_ensemble_f
}

export function absentOrbitalsLabel(keys: readonly string[]): string | null {
  if (keys.length === 0) return null
  if (keys.length === 1) return `no ${keys[0]}`
  if (keys.length === 2) return `no ${keys[0]} or ${keys[1]}`
  return `no ${keys.slice(0, -1).join(', ')}, or ${keys[keys.length - 1]}`
}

export function absentOrbitalsDetail(keys: readonly string[]): string | null {
  if (keys.length === 0) return null
  if (keys.length === 1) {
    return `${keys[0]} orbitals are not in this crystal, so they are not scored`
  }
  if (keys.length === 2) {
    return `${keys[0]} and ${keys[1]} orbitals are not in this crystal, so they are not scored`
  }
  const listed = `${keys.slice(0, -1).join(', ')}, and ${keys[keys.length - 1]}`
  return `${listed} orbitals are not in this crystal, so they are not scored`
}

export function adChannelReadout(ad: AdResult): AdChannelReadout {
  const scored: AdChannelScore[] = []
  const absent: OrbitalKey[] = []
  for (const key of ORBITALS) {
    if (orbitalActive(ad, key)) {
      scored.push({ key, label: formatAdPercent(orbitalValue(ad, key)) })
    } else {
      absent.push(key)
    }
  }
  return {
    scored,
    absentLabel: absentOrbitalsLabel(absent),
    absentDetail: absentOrbitalsDetail(absent),
  }
}

export function adConfidenceDetail(hasAbsent: boolean): string {
  const base = 'Average confidence that this DOS is accurate for a crystal like this one.'
  if (!hasAbsent) return base
  return `${base} Orbitals not in this crystal count as certain and are included in this number.`
}

export function adSummaryRows(ad: AdResult | null | undefined): (string | number)[][] {
  if (!ad) {
    return [
      [''],
      ['Applicability Domain', 'unavailable'],
    ]
  }
  return [
    [''],
    ['Applicability Domain'],
    ['Inside AD', boolCsv(ad.inside_ad)],
    ['AD threshold', ad.ad_threshold],
    ['p_ensemble_all', ad.p_ensemble_all],
    ['f-element warning', boolCsv(ad.f_element_warning)],
  ]
}

export function adExportRows(ad: AdResult | null | undefined): (string | number)[][] {
  if (!ad) {
    return [
      ['Applicability Domain'],
      [''],
      ['Field', 'Value'],
      ['ad', 'null'],
      ['note', 'AD scores were not returned; DOS prediction is unaffected'],
    ]
  }
  const rows: (string | number)[][] = [
    ['Applicability Domain'],
    [''],
    ['Field', 'Value'],
  ]
  for (const field of AD_JSON_FIELDS) {
    const value = ad[field]
    rows.push([field, typeof value === 'boolean' ? boolCsv(value) : value])
  }
  return rows
}
