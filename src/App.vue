<template>
  <div class="app">
    <header class="header">
      <div class="logo">
        <img src="./assets/logo.png" alt="Smart Chemical Design" class="logo-image" />
        <span class="logo-text">GRADE</span>
      </div>
      <div class="header-subtitle">Graph-based Representation for Atomic DOS Estimation</div>
      <div class="header-actions" v-if="hasData">
        <button class="btn-export" @click="exportToZip">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7,10 12,15 17,10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          Export Data
        </button>
      </div>
    </header>

    <main class="main-content">
      <!-- Upload Section -->
      <div v-if="!hasData" class="upload-section fade-in">
        <FileUpload @file-selected="handleFileSelect" :loading="loading" />
      </div>

      <!-- Main Visualization -->
      <div v-else class="visualization-container fade-in">
        <ADPanel :ad="dosResult?.ad ?? null" />
        <div class="visualization-panels">
        <div class="panel crystal-panel">
          <div class="panel-header">
            <h2>Crystal Structure</h2>
            <button class="btn-reset" @click="resetView">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
                <path d="M3 3v5h5"/>
              </svg>
              New Structure
            </button>
          </div>
          <CrystalViewer 
            :cif-content="cifContent"
            :selected-atoms="selectedAtoms"
            :element-symbols="dosResult?.element_symbols"
            @atom-clicked="handleAtomClick"
            @atoms-loaded="handleAtomsLoaded"
          />
          <div class="atom-legend">
            <div v-for="(symbol, idx) in uniqueElements" :key="idx" class="legend-item">
              <span class="legend-color" :style="{ background: getElementColor(symbol) }"></span>
              <span>{{ symbol }}</span>
            </div>
          </div>
        </div>

        <div class="panel dos-panel">
          <div class="panel-header">
            <h2>Density of States</h2>
            <div class="dos-controls">
              <button 
                :class="['tab-btn', { active: dosMode === 'crystal' }]"
                @click="dosMode = 'crystal'; selectedAtoms = []; multiSelectMode = false"
              >
                Total Crystal
              </button>
              <div class="atom-selector">
                <select 
                  @change="(e) => selectAtomFromDropdown(Number((e.target as HTMLSelectElement).value))"
                  class="atom-dropdown"
                >
                  <option value="" disabled selected>Select Atom</option>
                  <option 
                    v-for="(symbol, idx) in dosResult?.element_symbols" 
                    :key="idx" 
                    :value="idx"
                  >
                    {{ symbol }} ({{ idx }})
                  </option>
                </select>
              </div>
              <button 
                :class="['tab-btn', 'multi-btn', { active: multiSelectMode }]"
                @click="toggleMultiSelect"
                title="Select multiple atoms to sum their DOS"
              >
                {{ multiSelectMode ? 'Exit Multi' : 'Multi Select' }}
              </button>
            </div>
          </div>
          <DOSChart 
            :key="`dos-${dosMode}-${selectedAtoms.join('-')}`"
            :dos-data="currentDOSData"
            :energy-grid="dosResult?.energy_grid"
            :title="dosChartTitle"
          />
          <!-- Single-select mode: Element, Index, Integral -->
          <div class="dos-info" v-if="dosMode === 'per_atom' && selectedAtoms.length === 1 && !multiSelectMode">
            <div class="info-item">
              <span class="info-label">Element</span>
              <span class="info-value">{{ dosResult?.element_symbols[selectedAtoms[0]] }}</span>
            </div>
            <div class="info-item">
              <span class="info-label">Atom Index</span>
              <span class="info-value">{{ selectedAtoms[0] }}</span>
            </div>
            <div class="info-item">
              <span class="info-label">Integral</span>
              <span class="info-value">{{ getAtomIntegral(selectedAtoms[0]).toFixed(2) }}</span>
            </div>
          </div>
          <!-- Multi-select mode: list of selected atoms -->
          <div class="dos-info" v-if="dosMode === 'per_atom' && (selectedAtoms.length > 1 || multiSelectMode)">
            <div class="info-item">
              <span class="info-label">Selected Atoms</span>
              <span class="info-value selected-atoms-list">
                <span 
                  v-for="idx in selectedAtoms" 
                  :key="idx" 
                  class="selected-atom-tag"
                  @click="selectedAtoms.splice(selectedAtoms.indexOf(idx), 1)"
                >
                  {{ dosResult?.element_symbols[idx] }}({{ idx }})
                  <span class="remove-atom">×</span>
                </span>
              </span>
            </div>
          </div>
        </div>
        </div>
      </div>
    </main>

    <!-- Loading Overlay -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-spinner"></div>
      <p>Processing structure...</p>
    </div>

    <!-- Error Message -->
    <div v-if="errorMessage" class="error-overlay" @click="errorMessage = ''">
      <div class="error-card" @click.stop>
        <div class="error-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="15" y1="9" x2="9" y2="15"/>
            <line x1="9" y1="9" x2="15" y2="15"/>
          </svg>
        </div>
        <h3>Error</h3>
        <p>{{ errorMessage }}</p>
        <button class="btn-close" @click="errorMessage = ''">Close</button>
      </div>
    </div>

    <!-- Warning Dialog -->
    <div v-if="warningMessage" class="warning-overlay">
      <div class="warning-card">
        <div class="warning-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/>
            <line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
        </div>
        <h3>Warning</h3>
        <p>{{ warningMessage }}</p>
        <p class="warning-distance">Minimum distance: <strong>{{ warningMinDistance.toFixed(3) }} Å</strong></p>
        <div class="warning-buttons">
          <button class="btn-secondary" @click="handleWarningNewStructure">New Structure</button>
          <button class="btn-primary" @click="handleWarningContinue">Continue</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { open, save } from '@tauri-apps/plugin-dialog'
import { writeFile } from '@tauri-apps/plugin-fs'
import JSZip from 'jszip'
import FileUpload from './components/FileUpload.vue'
import CrystalViewer from './components/CrystalViewer.vue'
import DOSChart from './components/DOSChart.vue'
import ADPanel from './components/ADPanel.vue'
import { adExportRows, adSummaryRows, type AdResult } from './ad'

interface PredictionData {
  num_atoms: number
  elements: number[]
  element_symbols: string[]
  energy_grid: number[]
  dos_s: number[][]
  dos_p: number[][]
  dos_d: number[][]
  dos_f: number[][]
  total_atomic_dos: number[][]
  total_crystal_dos: number[]
  cif_content: string
  ad?: AdResult | null
  ad_reason?: string | null
}

interface SidecarResult {
  success: boolean
  data?: PredictionData
  error?: string
  traceback?: string
  warning?: string
  min_distance?: number
  cif_content?: string
}

const loading = ref(false)
const cifContent = ref<string | null>(null)
const dosResult = ref<PredictionData | null>(null)
const selectedAtoms = ref<number[]>([])
const multiSelectMode = ref(false)
const dosMode = ref<'crystal' | 'atomic_sum' | 'per_atom'>('crystal')
const errorMessage = ref('')
// Elements list from 3Dmol (stays in sync with the 3D viewer)
const viewerElements = ref<string[]>([])

// Warning dialog state
const warningMessage = ref('')
const warningMinDistance = ref(0)
const pendingCifPath = ref<string | null>(null)

const hasData = computed(() => cifContent.value && dosResult.value)

const uniqueElements = computed(() => {
  if (!dosResult.value?.element_symbols) return []
  return [...new Set(dosResult.value.element_symbols)]
})

const currentDOSData = computed(() => {
  if (!dosResult.value) return null
  
  if (dosMode.value === 'crystal') {
    const numPoints = dosResult.value.dos_s[0]?.length || 0
    const numAtoms = dosResult.value.dos_s.length
    
    // Sum DOS across all atoms for each orbital
    const sumS = new Array(numPoints).fill(0)
    const sumP = new Array(numPoints).fill(0)
    const sumD = new Array(numPoints).fill(0)
    const sumF = new Array(numPoints).fill(0)
    
    for (let i = 0; i < numAtoms; i++) {
      dosResult.value.dos_s[i]?.forEach((v, j) => { sumS[j] += v })
      dosResult.value.dos_p[i]?.forEach((v, j) => { sumP[j] += v })
      dosResult.value.dos_d[i]?.forEach((v, j) => { sumD[j] += v })
      dosResult.value.dos_f[i]?.forEach((v, j) => { sumF[j] += v })
    }
    
    return {
      s: sumS,
      p: sumP,
      d: sumD,
      f: sumF,
      total: dosResult.value.total_crystal_dos,
      showComponents: true
    }
  }
  
  if (dosMode.value === 'atomic_sum') {
    const numPoints = dosResult.value.total_atomic_dos[0]?.length || 0
    const sumDOS = new Array(numPoints).fill(0)
    dosResult.value.total_atomic_dos.forEach(atomDos => {
      atomDos.forEach((val, i) => { sumDOS[i] += val })
    })
    return {
      total: sumDOS,
      showComponents: false
    }
  }
  
  if (dosMode.value === 'per_atom' && selectedAtoms.value.length > 0) {
    const indices = selectedAtoms.value
    const numPoints = dosResult.value.dos_s[0]?.length || 0
    
    // Sum DOS over the selected atoms
    const sumS = new Array(numPoints).fill(0)
    const sumP = new Array(numPoints).fill(0)
    const sumD = new Array(numPoints).fill(0)
    const sumF = new Array(numPoints).fill(0)
    const sumTotal = new Array(numPoints).fill(0)
    
    indices.forEach(idx => {
      dosResult.value!.dos_s[idx]?.forEach((v, i) => { sumS[i] += v })
      dosResult.value!.dos_p[idx]?.forEach((v, i) => { sumP[i] += v })
      dosResult.value!.dos_d[idx]?.forEach((v, i) => { sumD[i] += v })
      dosResult.value!.dos_f[idx]?.forEach((v, i) => { sumF[i] += v })
      dosResult.value!.total_atomic_dos[idx]?.forEach((v, i) => { sumTotal[i] += v })
    })
    
    return {
      s: sumS,
      p: sumP,
      d: sumD,
      f: sumF,
      total: sumTotal,
      showComponents: true
    }
  }
  
  return null
})

const dosChartTitle = computed(() => {
  if (dosMode.value === 'crystal') return 'Total Crystal DOS'
  if (dosMode.value === 'atomic_sum') return 'Total Atomic DOS (Sum)'
  if (dosMode.value === 'per_atom' && selectedAtoms.value.length > 0) {
    if (selectedAtoms.value.length === 1) {
      const idx = selectedAtoms.value[0]
      const symbol = viewerElements.value[idx]
      return `${symbol} (${idx}) DOS`
    } else {
      return `Selected ${selectedAtoms.value.length} atoms DOS`
    }
  }
  return 'Select an atom'
})

const handleFileSelect = async () => {
  const selected = await open({
    multiple: false,
    filters: [
      { name: 'Crystal Structure', extensions: ['cif', 'vasp', 'poscar'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  })
  
  if (!selected) return
  
  await runPrediction(selected as string, false)
}

const runPrediction = async (cifPath: string, force: boolean) => {
  loading.value = true
  errorMessage.value = ''
  warningMessage.value = ''
  
  try {
    // Call Rust backend which runs Python
    const result = await invoke<SidecarResult>('predict_dos', {
      cifPath,
      force
    })
    
    if (result.success && result.data) {
      dosResult.value = result.data
      cifContent.value = result.data.cif_content
      dosMode.value = 'crystal'
      selectedAtoms.value = []
      multiSelectMode.value = false
      pendingCifPath.value = null
    } else if (result.warning) {
      // Short distance warning - show dialog
      warningMessage.value = result.warning
      warningMinDistance.value = result.min_distance || 0
      pendingCifPath.value = cifPath
    } else {
      errorMessage.value = result.error || 'Unknown error occurred'
      if (result.traceback) {
        console.error('Traceback:', result.traceback)
      }
    }
  } catch (error) {
    errorMessage.value = `Error: ${error}`
  } finally {
    loading.value = false
  }
}

const handleWarningContinue = async () => {
  if (pendingCifPath.value) {
    warningMessage.value = ''
    await runPrediction(pendingCifPath.value, true)
  }
}

const handleWarningNewStructure = () => {
  warningMessage.value = ''
  pendingCifPath.value = null
}

const handleAtomClick = (atomIndex: number) => {
  if (multiSelectMode.value) {
    // Multi-select mode: toggle atom in the selection
    const idx = selectedAtoms.value.indexOf(atomIndex)
    if (idx === -1) {
      selectedAtoms.value.push(atomIndex)
    } else {
      selectedAtoms.value.splice(idx, 1)
    }
  } else {
    // Single-select: replace the selection
    selectedAtoms.value = [atomIndex]
  }
  dosMode.value = 'per_atom'
}

const toggleMultiSelect = () => {
  multiSelectMode.value = !multiSelectMode.value
  if (!multiSelectMode.value && selectedAtoms.value.length > 1) {
    // Leaving multi-select: keep only the first atom
    selectedAtoms.value = [selectedAtoms.value[0]]
  }
}

const handleAtomsLoaded = (elements: string[]) => {
  viewerElements.value = elements
}

const resetView = () => {
  cifContent.value = null
  dosResult.value = null
  selectedAtoms.value = []
  multiSelectMode.value = false
  dosMode.value = 'crystal'
}

const selectAtomFromDropdown = (idx: number) => {
  if (multiSelectMode.value) {
    // Multi-select mode: add atom if not already selected
    if (!selectedAtoms.value.includes(idx)) {
      selectedAtoms.value.push(idx)
    }
  } else {
    selectedAtoms.value = [idx]
  }
  dosMode.value = 'per_atom'
}

const getAtomIntegral = (idx: number) => {
  if (!dosResult.value || idx === null || idx === undefined) return 0
  const dos = dosResult.value.total_atomic_dos?.[idx]
  if (!dos) return 0
  return dos.reduce((a, b) => a + b, 0) * 0.05
}

const getElementColor = (symbol: string) => {
  // Must stay in sync with elementColors in CrystalViewer.vue
  const colors: Record<string, string> = {
    'H': '#ffffff', 'He': '#d9ffff', 'Li': '#cc80ff', 'Be': '#c2ff00',
    'B': '#ffb5b5', 'C': '#909090', 'N': '#3050f8', 'O': '#ff0d0d',
    'F': '#90e050', 'Ne': '#b3e3f5', 'Na': '#ab5cf2', 'Mg': '#8aff00',
    'Al': '#bfa6a6', 'Si': '#f0c8a0', 'P': '#ff8000', 'S': '#ffff30',
    'Cl': '#1ff01f', 'Ar': '#80d1e3', 'K': '#8f40d4', 'Ca': '#3dff00',
    'Sc': '#e6e6e6', 'Ti': '#bfc2c7', 'V': '#a6a6ab', 'Cr': '#8a99c7',
    'Mn': '#9c7ac7', 'Fe': '#e06633', 'Co': '#0000ff', 'Ni': '#50d050',
    'Cu': '#c88033', 'Zn': '#7d80b0', 'Ga': '#c28f8f', 'Ge': '#668f8f',
    'As': '#bd80e3', 'Se': '#ffa100', 'Br': '#a62929', 'Kr': '#5cb8d1',
    'Rb': '#702eb0', 'Sr': '#00ff00', 'Y': '#94ffff', 'Zr': '#94e0e0',
    'Nb': '#73c2c9', 'Mo': '#54b5b5', 'Ru': '#248f8f', 'Rh': '#0a7d8c',
    'Pd': '#006985', 'Ag': '#c0c0c0', 'Cd': '#ffd98f', 'In': '#a67573',
    'Sn': '#668080', 'Sb': '#9e63b5', 'Te': '#d47a00', 'I': '#940094',
    'Cs': '#57178f', 'Ba': '#00c900', 'La': '#70d4ff', 'Ce': '#ffffc7',
    'Pr': '#d9ffc7', 'Nd': '#c7ffc7', 'Sm': '#8fffc7', 'Eu': '#61ffc7',
    'Gd': '#45ffc7', 'Tb': '#30ffc7', 'Dy': '#1fffc7', 'Ho': '#00ff9c',
    'Er': '#00e675', 'Tm': '#00d452', 'Yb': '#00bf38', 'Lu': '#00ab24',
    'Hf': '#4dc2ff', 'Ta': '#4da6ff', 'W': '#2194d6', 'Re': '#267dab',
    'Os': '#266696', 'Ir': '#175487', 'Pt': '#d0d0e0', 'Au': '#ffd123',
    'Hg': '#b8b8d0', 'Tl': '#a6544d', 'Pb': '#575961', 'Bi': '#9e4fb5',
    'default': '#ff69b4'
  }
  return colors[symbol] || colors.default
}

const arrayToCSV = (data: (string | number)[][]): string => {
  return data.map(row =>
    row.map(cell => {
      const str = String(cell)
      // Quote field if it contains a comma, quote, or newline
      if (str.includes(',') || str.includes('"') || str.includes('\n')) {
        return `"${str.replace(/"/g, '""')}"`
      }
      return str
    }).join(',')
  ).join('\n')
}

const exportToZip = async () => {
  if (!dosResult.value) return
  
  const data = dosResult.value
  const zip = new JSZip()
  
  // 1. Summary.csv
  const summaryData: (string | number)[][] = [
    ['GRADE Export'],
    [''],
    ['Number of Atoms', data.num_atoms],
    ['Elements', data.element_symbols.join('; ')],
    ['Unique Elements', [...new Set(data.element_symbols)].join('; ')],
    ['Energy Points', data.energy_grid.length],
    ['Energy Range', `${data.energy_grid[0].toFixed(2)} to ${data.energy_grid[data.energy_grid.length - 1].toFixed(2)} eV`],
    [''],
    ['Atom List:'],
    ['Index', 'Element'],
    ...data.element_symbols.map((sym, idx) => [idx, sym]),
    ...adSummaryRows(data.ad),
  ]
  zip.file('Summary.csv', arrayToCSV(summaryData))
  zip.file('Applicability_Domain.csv', arrayToCSV(adExportRows(data.ad)))
  
  // 2. Crystal_Total.csv
  const crystalData: (string | number)[][] = [
    ['Energy (eV)', 'Total DOS (states/eV)'],
    ...data.energy_grid.map((e, i) => [e, data.total_crystal_dos[i]])
  ]
  zip.file('Crystal_Total.csv', arrayToCSV(crystalData))
  
  // 3. Individual atom CSV files
  for (let i = 0; i < data.num_atoms; i++) {
    const symbol = data.element_symbols[i]
    const fileName = `Atom_${i}_${symbol}.csv`
    
    const atomData: (string | number)[][] = [
      [`Atom ${i}: ${symbol}`],
      [''],
      ['Energy (eV)', 's-orbital', 'p-orbital', 'd-orbital', 'f-orbital', 'Total'],
      ...data.energy_grid.map((e, j) => [
        e,
        data.dos_s[i][j],
        data.dos_p[i][j],
        data.dos_d[i][j],
        data.dos_f[i][j],
        data.total_atomic_dos[i][j]
      ])
    ]
    
    zip.file(fileName, arrayToCSV(atomData))
  }
  
  const formula = [...new Set(data.element_symbols)]
    .map(el => {
      const count = data.element_symbols.filter(s => s === el).length
      return count > 1 ? `${el}${count}` : el
    })
    .join('')
  
  const defaultFilename = `DOS_${formula}.zip`
  
  try {
    const filePath = await save({
      defaultPath: defaultFilename,
      filters: [{ name: 'ZIP Archive', extensions: ['zip'] }]
    })

    if (!filePath) {
      console.log('Save cancelled')
      return
    }

    const zipContent = await zip.generateAsync({ type: 'uint8array' })
    await writeFile(filePath, zipContent)
    
    console.log('ZIP file saved:', filePath)
  } catch (error) {
    console.error('Error saving ZIP:', error)
    errorMessage.value = `Failed to save ZIP: ${error}`
  }
}
</script>

<style scoped>
.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  padding: 1.5rem 2rem;
  display: flex;
  align-items: center;
  gap: 2rem;
  border-bottom: 1px solid var(--border-color);
  background: var(--bg-secondary);
}

.logo {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-shrink: 0;
}

.logo-image {
  height: 36px;
  width: auto;
  filter: brightness(166%);
}

.logo-icon {
  width: 36px;
  height: 36px;
  background: var(--gradient-1);
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.logo-icon svg {
  width: 20px;
  height: 20px;
}

.logo-text {
  font-size: 1.5rem;
  font-weight: 700;
  background: var(--gradient-1);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-family: 'JetBrains Mono', monospace;
}

.header-subtitle {
  color: var(--text-secondary);
  font-size: 1.1rem;
  flex: 1;
  min-width: 0;
  line-height: 1.35;
}

.header-actions {
  display: flex;
  gap: 0.5rem;
}

.btn-export {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  border: none;
  border-radius: 8px;
  color: white;
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
}

.btn-export:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
}

.btn-export svg {
  width: 16px;
  height: 16px;
}

.main-content {
  flex: 1;
  padding: 2rem;
}

.upload-section {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: calc(100vh - 200px);
}

.visualization-container {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  height: calc(100vh - 150px);
  min-height: 0;
}

.visualization-panels {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  flex: 1;
  min-height: 0;
}

.panel {
  background: var(--bg-card);
  border-radius: 16px;
  border: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-height: 0;
}

.panel-header {
  padding: 1rem 1.5rem;
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--bg-tertiary);
}

.panel-header h2 {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.btn-reset {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: transparent;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  color: var(--text-secondary);
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-reset:hover {
  background: var(--bg-secondary);
  color: var(--text-primary);
  border-color: var(--accent-primary);
}

.btn-reset svg {
  width: 16px;
  height: 16px;
}

.dos-controls {
  display: flex;
  gap: 0.5rem;
}

.tab-btn {
  padding: 0.4rem 0.8rem;
  background: transparent;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-secondary);
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s;
}

.tab-btn:hover:not(:disabled) {
  border-color: var(--accent-primary);
}

.tab-btn.active {
  background: var(--accent-primary);
  border-color: var(--accent-primary);
  color: white;
}

.tab-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.atom-selector {
  position: relative;
}

.atom-dropdown {
  padding: 0.4rem 2rem 0.4rem 0.8rem;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 0.8rem;
  cursor: pointer;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%239ca3af' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 0.5rem center;
  min-width: 120px;
}

.atom-dropdown:hover {
  border-color: var(--accent-primary);
}

.atom-dropdown:focus {
  outline: none;
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
}

.atom-dropdown option {
  background: var(--bg-card);
  color: var(--text-primary);
  padding: 0.5rem;
}

.atom-legend {
  padding: 0.75rem 1.5rem;
  display: flex;
  gap: 1rem;
  flex-wrap: wrap;
  border-top: 1px solid var(--border-color);
  background: var(--bg-tertiary);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
}

.legend-color {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  border: 1px solid rgba(255,255,255,0.2);
}

.dos-info {
  padding: 1rem 1.5rem;
  display: flex;
  gap: 2rem;
  border-top: 1px solid var(--border-color);
  background: var(--bg-tertiary);
  flex-shrink: 0;
}

.info-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.info-label {
  font-size: 0.75rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.info-value {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
  font-family: 'JetBrains Mono', monospace;
}

.multi-btn {
  background: transparent;
  border-style: dashed;
}

.multi-btn.active {
  background: rgba(16, 185, 129, 0.2);
  border-color: #10b981;
  border-style: solid;
  color: #10b981;
}

.selected-atoms-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.selected-atom-tag {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
  padding: 0.2rem 0.5rem;
  background: rgba(99, 102, 241, 0.2);
  border: 1px solid rgba(99, 102, 241, 0.4);
  border-radius: 4px;
  font-size: 0.85rem;
  color: var(--text-primary);
}

.selected-atom-tag .remove-atom {
  cursor: pointer;
  color: var(--text-muted);
  font-size: 1rem;
  line-height: 1;
  margin-left: 0.2rem;
}

.selected-atom-tag .remove-atom:hover {
  color: #ef4444;
}

.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(10, 10, 15, 0.9);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1.5rem;
  z-index: 1000;
}

.loading-spinner {
  width: 48px;
  height: 48px;
  border: 3px solid var(--border-color);
  border-top-color: var(--accent-primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

.loading-overlay p {
  color: var(--text-secondary);
  font-size: 1rem;
}

.error-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(10, 10, 15, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.error-card {
  background: var(--bg-card);
  border: 1px solid var(--error);
  border-radius: 16px;
  padding: 2rem;
  max-width: 500px;
  text-align: center;
}

.error-icon {
  width: 48px;
  height: 48px;
  margin: 0 auto 1rem;
  color: var(--error);
}

.error-icon svg {
  width: 100%;
  height: 100%;
}

.error-card h3 {
  color: var(--error);
  margin-bottom: 1rem;
}

.error-card p {
  color: var(--text-secondary);
  margin-bottom: 1.5rem;
  word-break: break-word;
}

.btn-close {
  padding: 0.5rem 1.5rem;
  background: var(--error);
  border: none;
  border-radius: 8px;
  color: white;
  cursor: pointer;
  transition: opacity 0.2s;
}

.btn-close:hover {
  opacity: 0.8;
}

/* Warning Dialog */
.warning-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(10, 10, 15, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.warning-card {
  background: var(--bg-card);
  border: 1px solid #f59e0b;
  border-radius: 16px;
  padding: 2rem;
  max-width: 500px;
  text-align: center;
}

.warning-icon {
  width: 48px;
  height: 48px;
  margin: 0 auto 1rem;
  color: #f59e0b;
}

.warning-icon svg {
  width: 100%;
  height: 100%;
}

.warning-card h3 {
  color: #f59e0b;
  margin-bottom: 1rem;
}

.warning-card p {
  color: var(--text-secondary);
  margin-bottom: 0.5rem;
  word-break: break-word;
}

.warning-distance {
  color: var(--text-primary);
  font-size: 1.1rem;
  margin-bottom: 1.5rem;
}

.warning-distance strong {
  color: #f59e0b;
  font-family: 'JetBrains Mono', monospace;
}

.warning-buttons {
  display: flex;
  gap: 1rem;
  justify-content: center;
}

.btn-secondary {
  padding: 0.6rem 1.5rem;
  background: transparent;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  color: var(--text-secondary);
  cursor: pointer;
  transition: all 0.2s;
  font-size: 0.95rem;
}

.btn-secondary:hover {
  background: var(--bg-secondary);
  color: var(--text-primary);
  border-color: var(--accent-primary);
}

.btn-primary {
  padding: 0.6rem 1.5rem;
  background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
  border: none;
  border-radius: 8px;
  color: white;
  cursor: pointer;
  transition: all 0.2s;
  font-size: 0.95rem;
  font-weight: 500;
}

.btn-primary:hover {
  opacity: 0.9;
  transform: translateY(-1px);
}
</style>
