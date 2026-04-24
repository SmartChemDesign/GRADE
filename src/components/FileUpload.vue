<template>
  <div 
    class="upload-container"
    :class="{ dragging: isDragging }"
    @dragover.prevent="isDragging = true"
    @dragleave="isDragging = false"
    @drop.prevent="handleDrop"
  >   
    <h2>Upload Crystal Structure</h2>
    <p class="upload-description">
      Click to browse for a CIF file
    </p>
    
    <button class="upload-btn" @click="$emit('file-selected')" :disabled="loading">
      <span>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17,8 12,3 7,8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
        Select File
      </span>
    </button>
    
    <div class="supported-formats">
      <span class="format-tag">.cif</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

defineProps<{
  loading: boolean
}>()

const emit = defineEmits<{
  (e: 'file-selected'): void
}>()

const isDragging = ref(false)

const handleDrop = () => {
  isDragging.value = false
  // Drag-and-drop in Tauri requires a dedicated path; fall back to the file dialog for now.
  emit('file-selected')
}
</script>

<style scoped>
.upload-container {
  background: var(--bg-card);
  border: 2px dashed var(--border-color);
  border-radius: 24px;
  padding: 4rem;
  text-align: center;
  transition: all 0.3s;
  max-width: 500px;
  width: 100%;
}

.upload-container:hover,
.upload-container.dragging {
  border-color: var(--accent-primary);
  background: rgba(99, 102, 241, 0.05);
  box-shadow: var(--shadow-glow);
}

.upload-icon {
  width: 80px;
  height: 80px;
  margin: 0 auto 1.5rem;
  background: var(--gradient-1);
  border-radius: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
}

.upload-icon svg {
  width: 40px;
  height: 40px;
}

h2 {
  font-size: 1.5rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: var(--text-primary);
}

.upload-description {
  color: var(--text-secondary);
  margin-bottom: 2rem;
}

.upload-btn {
  display: inline-flex;
  cursor: pointer;
  background: none;
  border: none;
  padding: 0;
}

.upload-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.upload-btn span {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.875rem 2rem;
  background: var(--gradient-1);
  border-radius: 12px;
  color: white;
  font-weight: 500;
  font-size: 1rem;
  transition: all 0.3s;
  box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
}

.upload-btn:hover:not(:disabled) span {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
}

.upload-btn svg {
  width: 20px;
  height: 20px;
}

.supported-formats {
  margin-top: 2rem;
  display: flex;
  justify-content: center;
  gap: 0.5rem;
}

.format-tag {
  padding: 0.35rem 0.75rem;
  background: var(--bg-tertiary);
  border-radius: 6px;
  font-size: 0.8rem;
  color: var(--text-muted);
  font-family: 'JetBrains Mono', monospace;
}
</style>
