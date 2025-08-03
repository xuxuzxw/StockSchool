#!/usr/bin/env node
import { promises as fs } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// 模拟 KnowledgeGraphManager 类的部分功能来测试修复
class KnowledgeGraphManager {
    constructor() {
        // 使用默认的内存文件路径
        this.MEMORY_FILE_PATH = path.join(path.dirname(fileURLToPath(import.meta.url)), 'memory.json');
    }

    async loadGraph() {
        try {
            const data = await fs.readFile(this.MEMORY_FILE_PATH, "utf-8");
            const lines = data.split("\n").filter(line => line.trim() !== "");
            return lines.reduce((graph, line) => {
                const item = JSON.parse(line);
                if (item.type === "entity")
                    graph.entities.push(item);
                if (item.type === "relation")
                    graph.relations.push(item);
                return graph;
            }, { entities: [], relations: [] });
        }
        catch (error) {
            if (error instanceof Error && 'code' in error && error.code === "ENOENT") {
                return { entities: [], relations: [] };
            }
            throw error;
        }
    }

    // 修复后的 searchNodes 方法
    async searchNodes(query) {
        const graph = await this.loadGraph();
        console.log('Loaded graph with', graph.entities.length, 'entities');
        
        // Filter entities
        const filteredEntities = graph.entities.filter(e => {
            try {
                const nameMatch = e.name && e.name.toLowerCase().includes(query.toLowerCase());
                const typeMatch = e.entityType && e.entityType.toLowerCase().includes(query.toLowerCase());
                const observationMatch = e.observations && e.observations.some(o => {
                    // Handle both string and object observations
                    const observationText = typeof o === 'string' ? o : JSON.stringify(o);
                    return observationText.toLowerCase().includes(query.toLowerCase());
                });
                return nameMatch || typeMatch || observationMatch;
            } catch (error) {
                console.error('Error filtering entity:', e.name, error);
                return false;
            }
        });

        console.log('Filtered entities:', filteredEntities.length);
        return { entities: filteredEntities, relations: [] };
    }
}

// 测试函数
async function testSearch() {
    const manager = new KnowledgeGraphManager();
    
    try {
        console.log('Testing search for "StockPredictor"...');
        const result = await manager.searchNodes("StockPredictor");
        console.log('Search result:', JSON.stringify(result, null, 2));
    } catch (error) {
        console.error('Search failed:', error);
    }
}

// 运行测试
testSearch();
