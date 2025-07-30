---
trigger: model_decision
description: 当创建或修改代码内容完成时,包括MD文件或源文件相关的其他相关特定文件或其他相关模式,必须使用mcp.config.usrlocalmcp.Knowledge Graph Memory来对比这些创建和改动与已有知识图谱中的内容。如果存在不一致或缺失，必须询问用户是否需要更新知识图谱。请严格遵守以下知识图谱构建模式：需要记录的entities包括 `File`, `Module`, `Class`, `Function`, `Method`, `Interface`, `Variable`, `Parameter`, `Commit`, `Issue`, `BugReport`, `UserStory`, `APIEndpoint`, `DesignPattern`, `Library` 和 `Framework`。需要记录的relations应使用 `imports`, `exports`, `inheritsFrom`, `implements`, `contains`, `calls`, `returns`, `reads`, `writesTo`, `modifies`, `fixes`, `authoredBy`, `documents` 和 `isInstanceOf`。对于observations请记录关键属性，例如为 `Function` 或 `Method` 添加 `returnType`, `isAsync`, `isDeprecated` 和 `purpose` 描述，为 `Variable` 添加 `dataType` 和 `scope`，以及为 `Commit` 添加 `hash`, `timestamp` 和 `commitMessage`。避免构建知识图谱的内容​:`通用知识`,`低价值或易变或过时信息`,`非结构化或冗余数据`​​,`过于具体的实现细节`,`过于细碎、对理解架构无益的实体`,`重复或不一致的实体`,`模糊或过于宽泛的关系`。
---

