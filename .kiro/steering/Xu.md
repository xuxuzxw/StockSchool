说简体中文

我使用的是windows10系统，python3.11

当要获取时间，请用`mcp.time`获取当前时间

当要建立新代码，新库，或代码遇到问题，使用`mcp.context7`

复杂任务或开放性问题，请细化推理过程顺序思考`mcp.Sequential Thinking`



善于使用网络搜索来解决疑难杂症

生成代码期间遇到任何问题,或者需要讨论商议的地方.必须等待用户回复

请保持谨慎和稳妥的行为,充分考虑代码 功能性（正确性与健壮性） 可维护性（理解与修改成本） 效率（资源使用与性能） 设计质量（扩展与复用能力） 工程实践（协作与长期维护） 降低长期维护成本，提升系统生命力

当代码任务告一段落,考虑记录 说明内容 日志内容

`mcp.Memory`就是你的记忆,合理利用。

**实体规范**
- 命名 ：驼峰命名（类名首字母大写，方法名首字母小写），文件用路径，无特殊字符
- 类型 ：核心类型(Class、Method、File、Factor、Config)，辅助类型(Function、Variable等)
- 版本 ：添加版本号和时间戳，重大变更升级主版本
- 标识 ：完全限定名唯一标识，避免同名不同类型 
**关系规范**
- 命名 ：动词/动词短语（如calls、defines），建立关系词典
- 方向 ：明确from/to方向，避免循环关系
- 属性 ：重要关系添加属性（如调用频率）
- 约束 ：明确源实体和目标实体类型约束 
**观察值规范**
- 格式 ：简洁文本，避免行号、临时变量等瞬时信息
- 分类 ：功能描述、特性说明、实现细节、注意事项
- 更新 ：定期验证，代码变更后及时更新
- 冗余 ：避免重复，合并相似观察值

**知识图谱构建模式 (Schema)**:
* **实体 (Entities)**: `File`, `Module`, `Class`, `Function`, `Method`, `Interface`, `Variable`, `Parameter`, `Commit`, `Issue`, `BugReport`, `UserStory`, `APIEndpoint`, `DesignPattern`, `Library`, `Framework`
* **关系 (Relations)**: `imports`, `exports`, `inheritsFrom`, `implements`, `contains`, `calls`, `returns`, `reads`, `writesTo`, `modifies`, `fixes`, `authoredBy`, `documents`, `isInstanceOf`
* **观察值 (Observations)**: 为`Function`或`Method`记录`returnType`, `isAsync`, `isDeprecated`, `purpose`描述；为`Variable`记录`dataType`, `scope`；为`Commit`记录`hash`, `timestamp`, `commitMessage`。

**操作流程 (请严格按此顺序执行)**:

2.  **数据质量检查 (Quality Check)**: 确保图谱中的数据符合我们定义的架构，并检查是否存在任何错误、重复或过时内容。

3.  **分析需求 (Analyze Requirements)**: 利用`MCP工具`，分析项目代码，提取出所有的函数调用关系、类定义、模块导入等信息。这将为后续的修复和去重操作提供基础。

4.  **修复与去重 (Fix & Deduplicate)**:
    * **修复错误**: 重新分析项目代码，验证图谱中已有实体和关系的准确性。如果发现函数A实际已不再调用函数B，但图中仍有此关系，请修正。
    * **合并重复**: 查找是否存在代表同一个代码元素的重复实体（例如，因大小写不同而创建了两个实体），并提议合并它们。
    * **处理过时内容**: 识别并提议删除那些在当前代码库中已不存在的实体（Orphan Nodes）。

5.  **清理与简化 (Clean & Simplify)**:
    * **删除瞬时数据**: 移除所有代表瞬时状态的属性特别是代码行号和单次运行的变量值。
    * **删除冗余内容**: 删除那些仅重复代码自身信息的注释类观察值。
    * **简化底层细节**: 合并或删除过于细碎、对理解架构无益的实体（如无意义的临时变量）。
    * **临时或测试代码**: 删除所有临时代码，标记所有测试代码，与正常代码进行区分。

7.  **增补缺失 (Add Missing)**:
    * 在完成上述清理后，再次全面扫描项目，查找是否有符合我们构建模式，但尚未被收录的重要实体或关系，并进行添加。

8.	请优先考虑一次看多个内容,然后使用批量处理知识图谱