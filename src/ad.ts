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

export function formatAdProbability(value: number): string {
  if (!Number.isFinite(value)) return '—'
  return value.toFixed(2)
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
