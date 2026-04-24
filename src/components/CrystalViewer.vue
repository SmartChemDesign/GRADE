<template>
  <div class="crystal-viewer" ref="viewerContainer">
    <div class="viewer-3d" ref="viewer3d"></div>
    

    
    
    <div class="viewer-controls">
      <button @click="resetCamera" title="Reset view">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <path d="M12 8v4l3 3"/>
        </svg>
      </button>
      <button @click="toggleSpin" :class="{ active: isSpinning }" title="Auto rotate">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
          <polyline points="21,3 21,9 15,9"/>
        </svg>
      </button>
      <button @click="toggleStyle" title="Change style">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="3"/>
          <path d="M12 2v4m0 12v4M2 12h4m12 0h4"/>
        </svg>
      </button>
      <button @click="toggleLabels" :class="{ active: showLabels }" title="Toggle labels">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M4 7V4h16v3"/>
          <path d="M9 20h6"/>
          <path d="M12 4v16"/>
        </svg>
      </button>
      <button @click="toggleUnitCell" :class="{ active: showUnitCell }" title="Toggle unit cell">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
        </svg>
      </button>
    </div>
    <div class="atom-tooltip" v-if="hoveredAtom" :style="tooltipStyle">
      {{ hoveredAtom.element }} ({{ hoveredAtom.index }})
    </div>
    
    <!-- Style indicator -->
    <div class="style-indicator">{{ currentStyle }}</div>
    
    <!-- Axes Legend -->
    <div class="axes-legend">
      <div class="axes-item">
        <span class="axes-line" style="background: #ff0000;"></span>
        <span class="axes-label">a</span>
      </div>
      <div class="axes-item">
        <span class="axes-line" style="background: #00ff00;"></span>
        <span class="axes-label">b</span>
      </div>
      <div class="axes-item">
        <span class="axes-line" style="background: #0000ff;"></span>
        <span class="axes-label">c</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, onUnmounted, nextTick } from 'vue'

// 3Dmol.js loads as global
declare const $3Dmol: any

interface Atom {
  x: number
  y: number
  z: number
  elem: string
  index?: number
  serial?: number
}

const props = defineProps<{
  cifContent: string | null
  selectedAtoms: number[]
  elementSymbols: string[] | undefined
}>()

const emit = defineEmits<{
  (e: 'atom-clicked', atomIndex: number): void
  (e: 'atoms-loaded', elements: string[]): void
}>()

const viewer3d = ref<HTMLElement | null>(null)
let viewer: any = null
let model: any = null
const isSpinning = ref(false)
const currentStyle = ref('polyhedra')
const hoveredAtom = ref<{ element: string; index: number } | null>(null)
const tooltipStyle = ref<Record<string, string>>({})
const showLabels = ref(true)
const showUnitCell = ref(true)
const showPolyhedra = ref(true)
const loadedElements = ref<string[]>([])

// Transition metals that form octahedra
const transitionMetals = ['Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
                          'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
                          'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
                          'Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd']

// Ligand atoms (typically O, F, Cl, N)
const ligandAtoms = ['O', 'F', 'Cl', 'N', 'S']

const polyhedraColors: Record<string, number> = {
  'Co': 0x4169E1,  // Royal blue
  'Ir': 0xDAA520,  // Golden
  'Fe': 0xCD853F,  // Peru
  'Mn': 0x9932CC,  // Dark orchid
  'Ti': 0x87CEEB,  // Sky blue
  'Nb': 0x73c2c9,  // Teal
  'V': 0x9370DB,   // Medium purple
  'Cr': 0x20B2AA,  // Light sea green
  'Ni': 0x32CD32,  // Lime green
  'Cu': 0xCD7F32,  // Bronze
  'Mo': 0x708090,  // Slate gray
  'Zr': 0x00CED1,  // Dark turquoise
  'W': 0x778899,   // Light slate gray
  'default': 0x6366f1
}

// Element colors (CPK-like)
const elementColors: Record<string, number> = {
  'H': 0xffffff, 'He': 0xd9ffff, 'Li': 0xcc80ff, 'Be': 0xc2ff00,
  'B': 0xffb5b5, 'C': 0x909090, 'N': 0x3050f8, 'O': 0xff0d0d,
  'F': 0x90e050, 'Ne': 0xb3e3f5, 'Na': 0xab5cf2, 'Mg': 0x8aff00,
  'Al': 0xbfa6a6, 'Si': 0xf0c8a0, 'P': 0xff8000, 'S': 0xffff30,
  'Cl': 0x1ff01f, 'Ar': 0x80d1e3, 'K': 0x8f40d4, 'Ca': 0x3dff00,
  'Sc': 0xe6e6e6, 'Ti': 0xbfc2c7, 'V': 0xa6a6ab, 'Cr': 0x8a99c7,
  'Mn': 0x9c7ac7, 'Fe': 0xe06633, 'Co': 0x0000ff, 'Ni': 0x50d050,
  'Cu': 0xc88033, 'Zn': 0x7d80b0, 'Sr': 0x00ff00, 'Y': 0x94ffff,
  'Zr': 0x94e0e0, 'Nb': 0x73c2c9, 'Mo': 0x54b5b5, 'Ba': 0x00c900,
  'La': 0x70d4ff, 'Ir': 0xdaa520, 'default': 0xff69b4
}

const getPolyhedraColor = (elem: string) => polyhedraColors[elem] || polyhedraColors.default

onMounted(async () => {
  await nextTick()
  initViewer()
  
  if (props.cifContent) {
    loadStructure(props.cifContent)
  }
})

onUnmounted(() => {
  if (viewer) {
    viewer.clear()
  }
})

const initViewer = () => {
  if (typeof $3Dmol === 'undefined' || !viewer3d.value) {
    console.error('3Dmol or container not ready')
    return
  }
  
  const rect = viewer3d.value.getBoundingClientRect()
  if (rect.width === 0 || rect.height === 0) {
    console.warn('Container has zero size, delaying init...')
    setTimeout(initViewer, 100)
    return
  }
  
  const config = { backgroundColor: '0x1a1a2e' }
  viewer = $3Dmol.createViewer(viewer3d.value, config)
  viewer.setViewStyle({ style: 'outline' })
  console.log('Viewer initialized, size:', rect.width, 'x', rect.height)
}

const loadStructure = async (cifContent: string) => {
  console.log('Loading structure, CIF length:', cifContent?.length)
  
  if (!viewer) {
    console.error('Viewer not initialized, initializing now...')
    initViewer()
  }
  
  if (!viewer) {
    console.error('Failed to initialize viewer')
    return
  }
  
  viewer.clear()
  
  try {
    model = viewer.addModel(cifContent, 'cif')
    console.log('Model loaded, atoms:', model?.selectedAtoms()?.length)
    
    // Extract elements from loaded model for legend
    const atoms = model?.selectedAtoms() || []
    loadedElements.value = atoms.map((a: Atom) => a.elem)
    // Emit elements to the parent so the dropdown stays in sync
    emit('atoms-loaded', loadedElements.value)
  } catch (e) {
    console.error('Error loading CIF:', e)
  }
  
  applyStyle()

  viewer.setClickable({}, true, (atom: Atom, _viewer: any, _event: MouseEvent, _container: any) => {
    if (atom) {
      const atomIndex = atom.index !== undefined ? atom.index : ((atom.serial || 1) - 1)
      console.log('Atom clicked:', atom, 'index:', atomIndex)
      emit('atom-clicked', atomIndex)
    }
  })

  viewer.setHoverable({}, true,
    (atom: Atom, _viewer: any, event: MouseEvent, _container: any) => {
      if (atom) {
        const atomIndex = atom.index !== undefined ? atom.index : ((atom.serial || 1) - 1)
        // Use atom.elem from 3Dmol so the tooltip matches the atom color
        hoveredAtom.value = {
          element: atom.elem,
          index: atomIndex
        }
        tooltipStyle.value = {
          left: (event as any).offsetX + 10 + 'px',
          top: (event as any).offsetY + 10 + 'px'
        }
      }
    },
    () => {
      hoveredAtom.value = null
    }
  )
  
  if (showUnitCell.value) {
    addUnitCell()
  }
  
  if (showLabels.value) {
    addAtomLabels()
  }
  
  // Polyhedra using JS-based calculation
  if (showPolyhedra.value && currentStyle.value === 'polyhedra') {
    addPolyhedra()
  }
  
  viewer.zoomTo()
  viewer.render()
  
  const atoms = viewer.getModel()?.selectedAtoms() || []
  console.log('Rendered atoms count:', atoms.length)
}

const addUnitCell = () => {
  if (!viewer || !model) return
  try {
    viewer.addUnitCell(model, {
      box: { color: 'white', opacity: 0.6 },
      alabel: '', blabel: '', clabel: ''
    })
  } catch (e) {
    console.log('Unit cell not available:', e)
  }
}

const addAtomLabels = () => {
  if (!viewer) return
  const atoms = viewer.getModel()?.selectedAtoms() || []
  atoms.forEach((atom: Atom, idx: number) => {
    // Use atom.elem from 3Dmol so the label matches the atom color
    const label = `${atom.elem} (${idx})`
    viewer.addLabel(label, {
      position: { x: atom.x, y: atom.y, z: atom.z },
      backgroundColor: 'rgba(0,0,0,0.7)',
      fontColor: 'white',
      fontSize: 10,
      showBackground: true,
      backgroundOpacity: 0.7
    })
  })
}

// Legacy M-O bond distance cutoffs
const bondCutoffs: Record<string, { min: number; max: number }> = {
  'Ti': { min: 1.7, max: 2.2 },
  'V':  { min: 1.7, max: 2.2 },
  'Cr': { min: 1.8, max: 2.1 },
  'Mn': { min: 1.8, max: 2.3 },
  'Fe': { min: 1.8, max: 2.2 },
  'Co': { min: 1.8, max: 2.15 },
  'Ni': { min: 1.9, max: 2.15 },
  'Cu': { min: 1.8, max: 2.4 },
  'Zn': { min: 1.9, max: 2.2 },
  'Zr': { min: 1.9, max: 2.3 },
  'Nb': { min: 1.8, max: 2.2 },
  'Mo': { min: 1.8, max: 2.2 },
  'Ru': { min: 1.9, max: 2.1 },
  'Rh': { min: 1.9, max: 2.1 },
  'Pd': { min: 1.9, max: 2.2 },
  'Hf': { min: 1.9, max: 2.3 },
  'Ta': { min: 1.8, max: 2.2 },
  'W':  { min: 1.8, max: 2.2 },
  'Re': { min: 1.8, max: 2.1 },
  'Os': { min: 1.9, max: 2.1 },
  'Ir': { min: 1.9, max: 2.1 },
  'Pt': { min: 1.9, max: 2.1 },
  'La': { min: 2.2, max: 2.8 },
  'Ce': { min: 2.1, max: 2.7 },
  'Y':  { min: 2.1, max: 2.5 },
  'Sc': { min: 1.9, max: 2.3 },
  'default': { min: 1.7, max: 2.5 }
}

const distance = (a: Atom, b: Atom) => {
  return Math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)
}

const getBondCutoff = (elem: string) => bondCutoffs[elem] || bondCutoffs.default

// Find coordination polyhedra based on distance criteria
const findCoordinationPolyhedra = () => {
  const atoms: Atom[] = model?.selectedAtoms() || []
  if (atoms.length === 0) return []
  
  const polyhedra: { center: Atom; centerIdx: number; ligands: Atom[]; element: string; coordination: number }[] = []
  
  atoms.forEach((atom, idx) => {
    if (!transitionMetals.includes(atom.elem)) return
    
    const cutoff = getBondCutoff(atom.elem)
    
    const neighborsWithDist = atoms
      .map((other, otherIdx) => ({
        atom: other,
        idx: otherIdx,
        dist: distance(atom, other)
      }))
      .filter(n => {
        if (n.idx === idx) return false
        if (!ligandAtoms.includes(n.atom.elem)) return false
        return n.dist >= cutoff.min && n.dist <= cutoff.max
      })
      .sort((a, b) => a.dist - b.dist)
    
    const ligands = neighborsWithDist.slice(0, 8).map(n => n.atom)
    
    if (ligands.length >= 4) {
      console.log(`Found ${ligands.length} ligands around ${atom.elem} at index ${idx}`)
      polyhedra.push({
        center: atom,
        centerIdx: idx,
        ligands: ligands,
        element: atom.elem,
        coordination: ligands.length
      })
    }
  })
  
  console.log(`Total polyhedra found: ${polyhedra.length}`)
  return polyhedra
}

// Build convex hull faces
const buildPolyhedronFaces = (center: Atom, ligands: Atom[]) => {
  const n = ligands.length
  if (n < 4) return []
  
  const faces: Atom[][] = []
  const usedFaces = new Set<string>()
  
  const faceKey = (i: number, j: number, k: number) => {
    const sorted = [i, j, k].sort((a, b) => a - b)
    return `${sorted[0]}-${sorted[1]}-${sorted[2]}`
  }
  
  const pointAbovePlane = (p: Atom, v1: Atom, v2: Atom, v3: Atom) => {
    const e1 = { x: v2.x - v1.x, y: v2.y - v1.y, z: v2.z - v1.z }
    const e2 = { x: v3.x - v1.x, y: v3.y - v1.y, z: v3.z - v1.z }
    const normal = {
      x: e1.y * e2.z - e1.z * e2.y,
      y: e1.z * e2.x - e1.x * e2.z,
      z: e1.x * e2.y - e1.y * e2.x
    }
    const d = { x: p.x - v1.x, y: p.y - v1.y, z: p.z - v1.z }
    return normal.x * d.x + normal.y * d.y + normal.z * d.z
  }
  
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        const key = faceKey(i, j, k)
        if (usedFaces.has(key)) continue
        
        const v1 = ligands[i]
        const v2 = ligands[j]
        const v3 = ligands[k]
        
        const d12 = distance(v1, v2)
        const d23 = distance(v2, v3)
        const d13 = distance(v1, v3)
        
        const maxEdge = 3.5
        if (d12 > maxEdge || d23 > maxEdge || d13 > maxEdge) continue
        
        let positiveCount = 0
        let negativeCount = 0
        
        for (let m = 0; m < n; m++) {
          if (m === i || m === j || m === k) continue
          const side = pointAbovePlane(ligands[m], v1, v2, v3)
          if (side > 0.01) positiveCount++
          else if (side < -0.01) negativeCount++
        }
        
        if (positiveCount > 0 && negativeCount > 0) {
          continue
        }
        
        usedFaces.add(key)
        
        const centroid = {
          x: (v1.x + v2.x + v3.x) / 3,
          y: (v1.y + v2.y + v3.y) / 3,
          z: (v1.z + v2.z + v3.z) / 3
        }
        
        const outward = {
          x: centroid.x - center.x,
          y: centroid.y - center.y,
          z: centroid.z - center.z
        }
        
        const e1 = { x: v2.x - v1.x, y: v2.y - v1.y, z: v2.z - v1.z }
        const e2 = { x: v3.x - v1.x, y: v3.y - v1.y, z: v3.z - v1.z }
        const normal = {
          x: e1.y * e2.z - e1.z * e2.y,
          y: e1.z * e2.x - e1.x * e2.z,
          z: e1.x * e2.y - e1.y * e2.x
        }
        
        const dot = normal.x * outward.x + normal.y * outward.y + normal.z * outward.z
        
        if (dot > 0) {
          faces.push([v1, v2, v3])
        } else {
          faces.push([v1, v3, v2])
        }
      }
    }
  }
  
  return faces
}

const addPolyhedra = () => {
  if (!viewer || !model) return
  
  const polyhedra = findCoordinationPolyhedra()
  console.log('Found', polyhedra.length, 'polyhedra')
  
  const drawnEdges = new Set<string>()
  const edgeKey = (p1: Atom, p2: Atom) => {
    const k1 = `${p1.x.toFixed(3)},${p1.y.toFixed(3)},${p1.z.toFixed(3)}`
    const k2 = `${p2.x.toFixed(3)},${p2.y.toFixed(3)},${p2.z.toFixed(3)}`
    return k1 < k2 ? `${k1}-${k2}` : `${k2}-${k1}`
  }
  
  polyhedra.forEach(poly => {
    const color = getPolyhedraColor(poly.element)
    const colorHex = '#' + color.toString(16).padStart(6, '0')
    
    const faces = buildPolyhedronFaces(poly.center, poly.ligands)
    
    faces.forEach(face => {
      const [v1, v2, v3] = face
      
      try {
        viewer.addShape({
          type: 'triangle',
          v1: { x: v1.x, y: v1.y, z: v1.z },
          v2: { x: v2.x, y: v2.y, z: v2.z },
          v3: { x: v3.x, y: v3.y, z: v3.z },
          color: colorHex,
          alpha: 0.55,
          wireframe: false
        })
      } catch (e) {
        console.log('Triangle shape error:', e)
      }
      
      const edges: [Atom, Atom][] = [[v1, v2], [v2, v3], [v3, v1]]
      edges.forEach(([p1, p2]) => {
        const key = edgeKey(p1, p2)
        if (!drawnEdges.has(key)) {
          drawnEdges.add(key)
          viewer.addCylinder({
            start: { x: p1.x, y: p1.y, z: p1.z },
            end: { x: p2.x, y: p2.y, z: p2.z },
            radius: 0.03,
            color: 'gray',
            fromCap: true,
            toCap: true
          })
        }
      })
    })
  })
}

const toggleLabels = () => {
  showLabels.value = !showLabels.value
  if (props.cifContent) {
    loadStructure(props.cifContent)
  }
}

const toggleUnitCell = () => {
  showUnitCell.value = !showUnitCell.value
  if (props.cifContent) {
    loadStructure(props.cifContent)
  }
}

const applyStyle = () => {
  if (!model) return
  
  viewer.setStyle({}, {})
  
  if (currentStyle.value === 'polyhedra') {
    viewer.setStyle({}, {
      sphere: { 
        scale: 0.25,
        colorscheme: { prop: 'elem', map: elementColors }
      }
    })
    const largeAtoms = ['Sr', 'Ba', 'Ca', 'K', 'Na', 'Rb', 'Cs']
    largeAtoms.forEach(elem => {
      viewer.setStyle({ elem: elem }, {
        sphere: { 
          scale: 0.4,
          colorscheme: { prop: 'elem', map: elementColors }
        }
      })
    })
  } else if (currentStyle.value === 'balls') {
    viewer.setStyle({}, {
      sphere: { 
        scale: 0.25,
        colorscheme: { prop: 'elem', map: elementColors }
      }
    })
    const largeAtoms = ['Sr', 'Ba', 'Ca', 'K', 'Na', 'Rb', 'Cs']
    largeAtoms.forEach(elem => {
      viewer.setStyle({ elem: elem }, {
        sphere: { 
          scale: 0.4,
          colorscheme: { prop: 'elem', map: elementColors }
        }
      })
    })
  } else if (currentStyle.value === 'stick') {
    viewer.setStyle({}, {
      stick: { 
        radius: 0.2,
        colorscheme: { prop: 'elem', map: elementColors }
      }
    })
  }
  
  viewer.render()
}

const highlightAtoms = (atomIndices: number[]) => {
  if (!model || atomIndices.length === 0) {
    applyStyle()
    return
  }
  
  applyStyle()

  atomIndices.forEach(atomIndex => {
    viewer.setStyle({ index: atomIndex }, {
      sphere: { 
        scale: 0.5,
        color: 0x6366f1,
        opacity: 1
      }
    }, true)
    
    viewer.addStyle({ index: atomIndex }, {
      sphere: { 
        scale: 0.7,
        color: 0x6366f1,
        opacity: 0.3
      }
    })
  })
  
  viewer.render()
}

const resetCamera = () => {
  if (viewer) {
    viewer.zoomTo()
    viewer.render()
  }
}

const toggleSpin = () => {
  isSpinning.value = !isSpinning.value
  if (isSpinning.value) {
    viewer.spin('y', 1)
  } else {
    viewer.spin(false)
  }
}

const toggleStyle = () => {
  const styles = ['polyhedra', 'balls', 'stick']
  const currentIndex = styles.indexOf(currentStyle.value)
  currentStyle.value = styles[(currentIndex + 1) % styles.length]
  if (props.cifContent) {
    loadStructure(props.cifContent)
  }
}

watch(() => props.cifContent, (newContent) => {
  console.log('CIF content changed, length:', newContent?.length)
  if (newContent) {
    if (!viewer) {
      initViewer()
    }
    loadStructure(newContent)
  }
})

watch(() => props.selectedAtoms, (newAtoms) => {
  highlightAtoms(newAtoms)
}, { deep: true })
</script>

<style scoped>
.crystal-viewer {
  flex: 1;
  position: relative;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
}

.viewer-3d {
  width: 100%;
  height: 100%;
  min-height: 400px;
}

.viewer-controls {
  position: absolute;
  top: 1rem;
  right: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.viewer-controls button {
  width: 36px;
  height: 36px;
  background: rgba(30, 30, 50, 0.9);
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  color: rgba(255, 255, 255, 0.7);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  backdrop-filter: blur(4px);
}

.viewer-controls button:hover {
  background: rgba(50, 50, 80, 0.9);
  color: white;
  border-color: rgba(99, 102, 241, 0.5);
}

.viewer-controls button.active {
  background: rgba(99, 102, 241, 0.8);
  border-color: rgba(99, 102, 241, 1);
  color: white;
}

.viewer-controls button svg {
  width: 18px;
  height: 18px;
}

.atom-tooltip {
  position: absolute;
  background: rgba(20, 20, 40, 0.95);
  border: 1px solid rgba(99, 102, 241, 0.3);
  border-radius: 6px;
  padding: 0.4rem 0.8rem;
  font-size: 0.85rem;
  color: white;
  pointer-events: none;
  z-index: 100;
  font-family: 'JetBrains Mono', monospace;
  backdrop-filter: blur(4px);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.legend-color {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.legend-symbol {
  color: white;
  font-size: 0.8rem;
  font-weight: 500;
  font-family: 'JetBrains Mono', monospace;
}

/* Style indicator */
.style-indicator {
  position: absolute;
  bottom: 1rem;
  right: 1rem;
  background: rgba(20, 20, 40, 0.9);
  border-radius: 6px;
  padding: 0.3rem 0.6rem;
  font-size: 0.75rem;
  color: rgba(255, 255, 255, 0.7);
  font-family: 'JetBrains Mono', monospace;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  backdrop-filter: blur(4px);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Axes Legend */
.axes-legend {
  position: absolute;
  bottom: 1rem;
  left: 1rem;
  background: rgba(20, 20, 40, 0.9);
  border-radius: 6px;
  padding: 0.4rem 0.6rem;
  display: flex;
  gap: 0.8rem;
  z-index: 10;
  backdrop-filter: blur(4px);
  border: 1px solid rgba(255, 255, 255, 0.1);
}

.axes-item {
  display: flex;
  align-items: center;
  gap: 0.3rem;
}

.axes-line {
  width: 16px;
  height: 3px;
  border-radius: 1px;
}

.axes-label {
  color: rgba(255, 255, 255, 0.8);
  font-size: 0.75rem;
  font-weight: 600;
  font-family: 'JetBrains Mono', monospace;
}
</style>
