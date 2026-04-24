<template>
  <div class="dos-chart" ref="chartContainer"></div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import Plotly from 'plotly.js-dist-min'

interface DOSData {
  s?: number[]
  p?: number[]
  d?: number[]
  f?: number[]
  total: number[]
  showComponents: boolean
}

const props = defineProps<{
  dosData: DOSData | null
  energyGrid: number[] | undefined
  title: string
}>()

const chartContainer = ref<HTMLElement | null>(null)

const colors = {
  s: '#3b82f6',    // blue
  p: '#10b981',    // green
  d: '#f59e0b',    // amber
  f: '#ef4444',    // red
  total: '#8b5cf6' // purple
}

const hasNonZeroValues = (arr: number[] | undefined): boolean => {
  if (!arr || arr.length === 0) return false
  return arr.some(v => Math.abs(v) > 1e-10)
}

onMounted(() => {
  if (props.dosData && props.energyGrid) {
    renderChart()
  }
})

watch(() => [props.dosData, props.energyGrid, props.title], () => {
  if (props.dosData && props.energyGrid) {
    renderChart()
  }
}, { deep: true })

const renderChart = () => {
  if (!chartContainer.value || !props.dosData || !props.energyGrid) return

  Plotly.purge(chartContainer.value)
  
  const traces: Plotly.Data[] = []
  
  if (props.dosData.showComponents) {
    // Show individual orbital components only if they have non-zero values
    if (hasNonZeroValues(props.dosData.s)) {
      traces.push({
        x: props.energyGrid,
        y: props.dosData.s,
        name: 's-orbital',
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        fillcolor: 'rgba(59, 130, 246, 0.1)',
        line: { color: colors.s, width: 2 }
      })
    }
    
    if (hasNonZeroValues(props.dosData.p)) {
      traces.push({
        x: props.energyGrid,
        y: props.dosData.p,
        name: 'p-orbital',
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        fillcolor: 'rgba(16, 185, 129, 0.1)',
        line: { color: colors.p, width: 2 }
      })
    }
    
    if (hasNonZeroValues(props.dosData.d)) {
      traces.push({
        x: props.energyGrid,
        y: props.dosData.d,
        name: 'd-orbital',
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        fillcolor: 'rgba(245, 158, 11, 0.1)',
        line: { color: colors.d, width: 2 }
      })
    }
    
    if (hasNonZeroValues(props.dosData.f)) {
      traces.push({
        x: props.energyGrid,
        y: props.dosData.f,
        name: 'f-orbital',
        type: 'scatter',
        mode: 'lines',
        fill: 'tozeroy',
        fillcolor: 'rgba(239, 68, 68, 0.1)',
        line: { color: colors.f, width: 2 }
      })
    }
    
    traces.push({
      x: props.energyGrid,
      y: props.dosData.total,
      name: 'Total',
      type: 'scatter',
      mode: 'lines',
      line: { color: colors.total, width: 2.5, dash: 'dot' }
    })
  } else {
    traces.push({
      x: props.energyGrid,
      y: props.dosData.total,
      name: 'Total DOS',
      type: 'scatter',
      mode: 'lines',
      fill: 'tozeroy',
      fillcolor: 'rgba(139, 92, 246, 0.15)',
      line: { color: colors.total, width: 2.5 }
    })
  }
  
  // Fermi level line at E = 0
  traces.push({
    x: [0, 0],
    y: [0, Math.max(...(props.dosData.total || [1])) * 1.1],
    name: 'E_F',
    type: 'scatter',
    mode: 'lines',
    line: { color: '#ef4444', width: 1.5, dash: 'dash' },
    showlegend: true
  })
  
  const layout: Partial<Plotly.Layout> = {
    title: {
      text: props.title,
      font: { color: '#e8e8ed', size: 14, family: 'Outfit' },
      x: 0.02,
      xanchor: 'left'
    },
    xaxis: {
      title: { text: 'Energy (eV)', font: { color: '#a0a0b0', size: 12 } },
      color: '#a0a0b0',
      gridcolor: '#2a2a3a',
      zerolinecolor: '#3a3a4a',
      tickfont: { family: 'JetBrains Mono', size: 10 },
      automargin: true
    },
    yaxis: {
      title: { text: 'DOS (states/eV)', font: { color: '#a0a0b0', size: 12 } },
      color: '#a0a0b0',
      gridcolor: '#2a2a3a',
      zerolinecolor: '#3a3a4a',
      tickfont: { family: 'JetBrains Mono', size: 10 },
      rangemode: 'tozero',
      automargin: true
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(18, 18, 26, 0.5)',
    font: { color: '#e8e8ed', family: 'Outfit' },
    legend: {
      x: 1,
      y: 1,
      xanchor: 'right',
      bgcolor: 'rgba(22, 22, 31, 0.9)',
      bordercolor: '#2a2a3a',
      borderwidth: 1,
      font: { size: 11 }
    },
    margin: { l: 50, r: 10, t: 35, b: 70 },
    autosize: true,
    hovermode: 'x unified',
    hoverlabel: {
      bgcolor: '#16161f',
      bordercolor: '#2a2a3a',
      font: { family: 'JetBrains Mono', size: 11, color: '#e8e8ed' }
    }
  }
  
  const config: Partial<Plotly.Config> = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    displaylogo: false
  }
  
  Plotly.newPlot(chartContainer.value, traces, layout, config)
}
</script>

<style scoped>
.dos-chart {
  flex: 1;
  min-height: 0;
  max-height: 100%;
  padding: 0.5rem;
  overflow: hidden;
  position: relative;
}

.dos-chart :deep(.js-plotly-plot),
.dos-chart :deep(.plot-container),
.dos-chart :deep(.plotly) {
  width: 100% !important;
  height: 100% !important;
}

.dos-chart :deep(.main-svg) {
  max-height: 100%;
}
</style>
