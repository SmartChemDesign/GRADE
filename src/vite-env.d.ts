/// <reference types="vite/client" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

// 3Dmol.js global
declare const $3Dmol: any

// Plotly.js
declare module 'plotly.js-dist-min' {
  import Plotly from 'plotly.js'
  export default Plotly
  export * from 'plotly.js'
}
