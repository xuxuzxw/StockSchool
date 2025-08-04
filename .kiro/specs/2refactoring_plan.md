# 全局代码重构与优化计划

## 1. 核心问题

经过对 `data-sync-enhancement`, `factor-calculation-engine`, `ai-strategy-system`, `monitoring-alert-system` 四个核心模块的审查，发现当前代码库普遍存在以下关键问题：

- **技术债务严重**：多个模块存在新旧两套实现并存的情况，导致代码冗余、逻辑混乱、维护成本高。
- **硬编码普遍存在**：数据库查询（SQL）、业务逻辑参数（因子特征、模型超参数）、告警规则等被大量硬编码在业务逻辑中，严重影响了系统的灵活性、可维护性和安全性。
- **缺乏统一规范**：配置管理、数据库访问、日志记录等基础组件在不同模块中的使用方式不统一，增加了集成的复杂性。
- **工程质量有待提高**：部分模块违反了单一职责原则，存在循环依赖的风险，且缺乏足够的单元测试和集成测试。

## 2. 重构目标

本次重构旨在解决上述问题，提升代码库的整体质量，使其达到生产级别的标准。具体目标如下：

- **消除技术债务**：清理所有冗余和废弃的代码，统一各模块的实现逻辑。
- **配置驱动开发**：将所有可变部分（数据库连接、SQL查询、业务参数、告警规则等）外部化到配置文件或数据库中。
- **建立统一规范**：引入并强制使用统一的数据库访问层（ORM）、配置加载器和结构化日志系统。
- **提升代码质量**：优化代码结构，遵循设计原则，并补充必要的测试用例。

## 3. 分模块重构计划

### 3.1. 数据同步 (`data-sync-enhancement`)

- **任务**：引入ORM（如SQLAlchemy），替换所有硬编码的SQL查询。
- **步骤**：
  1. 定义所有数据表的ORM模型。
  2. 重构 `TushareSynchronizer`, `AkshareSynchronizer`, `IndustryClassificationManager` 中的数据读写逻辑，使用ORM进行数据库操作。
  3. 将数据表名、API密钥等配置移入统一的配置文件。

### 3.2. 因子计算 (`factor-calculation-engine`)

- **任务**：清理旧的 `FundamentalFactorEngine`，统一使用新版本。
- **步骤**：
  1. 全局搜索并替换所有对 `src.compute.factor_engine.FundamentalFactorEngine` 的引用，改为 `src.compute.fundamental_factor_engine.FundamentalFactorEngine`。
  2. 删除 `src/compute/factor_engine.py` 文件中的 `FundamentalFactorEngine` 类定义。
  3. **(可选优化)** 参照新引擎的设计，重构 `TechnicalFactorEngine`，使其也采用“引擎+计算器”的组合模式。

### 3.3. AI策略与回测 (`ai-strategy-system`)

- **任务**：消除硬编码，实现配置驱动的模型训练。
- **步骤**：
  1. 将 `ai_model.py` 中的SQL查询逻辑抽象成独立的数据获取层，并使用ORM实现。
  2. 将模型超参数、特征列、目标列等移入配置文件。
  3. 建立模型版本管理机制，将训练好的模型及标准化器（Scaler）与版本号关联存储。

### 3.4. 监控与告警 (`monitoring-alert-system`)

- **任务**：废弃旧的告警系统，全面采用 `alerts.py` 中的新设计。
- **步骤**：
  1. 删除 `logger.py` 中的 `Alert`, `AlertManager`, `EmailNotifier`, `WebhookNotifier` 等与告警相关的类。
  2. 重构 `logger.py`，使其专注于提供统一的结构化日志记录功能。
  3. 将 `alerts.py` 作为唯一的告警中心，并将其规则存储从SQLite迁移到主数据库中，方便统一管理。
  4. 梳理并迁移所有现存的告警逻辑，使用新系统的可配置规则进行重新定义。

## 4. 实施建议

- **分步进行**：建议按照上述模块顺序，逐一进行重构，每完成一个模块的重构后进行充分的测试。
- **版本控制**：为每次重构创建一个独立的特性分支，通过Pull Request进行代码审查。
- **补充测试**：在重构过程中，为修改和新增的功能补充单元测试和集成测试，确保代码质量和功能正确性。