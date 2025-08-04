import { createPinia } from 'pinia'

const pinia = createPinia()

export default pinia

// 导出所有store
export * from './user'
export * from './system'
export * from './data'
export * from './websocket'