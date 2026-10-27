# Welding Motion Planner / 焊接运动规划代码

面向双机械臂与变位机的离线焊接规划研究代码。包含姿态几何筛选、FANUC 解析逆解接口、完整动作候选构建、FCL 碰撞检查、经典 TrajOpt 风格序列凸优化、候选与焊点顺序联合选择、运动时间约束及 MuJoCo 场景回放。

这是代码公开版本。工业 CAD/JT/GLB 模型、焊点数据、场景标定、候选库、结果 NPZ、私有交接资料及连接凭据不包含在本仓库。克隆后可以阅读算法、运行无需工业数据的测试；复现完整工业场景需要自行准备有权使用的数据。本仓库不会从原私人仓库自动下载数据。

## 环境与测试

建议 Ubuntu 22.04、Python 3.10 或更高版本。规划主要运行在 CPU，现有实现不使用 V100 CUDA 加速。

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

默认发布测试套件由 `pytest.ini` 明确列出；21 项需要未公开场景的测试明确跳过，准备数据后用 `--run-private-integration` 启用。完整历史实验目录没有统一测试入口，不属于默认发布检查。实际检查及结果见 `VALIDATION.md`。固定依赖版本记录自原项目环境，其他系统可能需要调整安装方式。视频导出还需要系统 `ffmpeg`，GUI 回放需要 OpenGL 显示环境，离屏渲染可使用 EGL。

## 代码入口

| 目录 | 内容 |
| --- | --- |
| `experimental/welding_candidate_catalog_v1` | 采样候选目录、存储与验证 |
| `experimental/welding_candidate_catalog_analytic_v1` | 解析逆解与候选合并 |
| `experimental/trajopt_transfer_v1` | 经典 TrajOpt 风格算法实现与转移规划；不是官方 TrajOpt 库 |
| `experimental/welding_full_candidate_selection_v1` | 全候选联合选择 |
| `experimental/welding_retreat300_library_v1` | 重建 300 mm 完整动作候选 |
| `experimental/welding_retreat300_anchored_v3` | 固定六阶段边界下联合选择候选与焊点顺序 |
| `experimental/welding_retreat300_timing_v4` | 时间约束与计时修正 |
| `experimental/dual_arm_parallel_v1` | 双臂时序实验与回放；最新 300 mm 基线没有要求双臂同时焊接 |
| `examples/site_specific` | 最终结果审计、边界规范化、重计时与视频导出操作脚本 |
| `tests` | 算法及验证测试 |

从几何筛选、逆解及碰撞验证构建候选，随后对候选和顺序联合选择，再逐段规划并复核转移。有限搜索预算下未找到解，不代表数学上证明不可行。300 mm 规划器中的焊点编号、角度分组与场景绑定是本次工程配置，迁移新工件时需要修改。

入口参数可在准备好对应场景后查看，例如：

```bash
.venv/bin/python -m experimental.welding_retreat300_library_v1.run --help
.venv/bin/python -m experimental.welding_retreat300_anchored_v3.run --help
```

库生成需要已有候选池和种子动作；联合选择需要新库及基准轨迹。这些输入不是仓库内置示例。`examples/site_specific` 保留原运行目录布局和 SHA 绑定以说明审计链，不能直接当成通用一键启动脚本。

## 本次工业场景结果

同一场景、21 个焊点：原基线 459.45 s，重新生成 0.3 m 动作库并联合重排后的结果为 349.83 s，减少约 23.9%；变位机仍转动 5 次。候选池 784 行全部尝试，其中 659 个动作通过，加上 21 个种子形成 680 个可选动作；125 行搜索未解决。

这些是未公开场景中的离线实验结果，不是通用性能保证。碰撞复核使用项目原有接触例外及离散采样规则，继承转移依赖封存基线；不构成机器人实体安全、控制器或动力学认证。硬件执行需另行验证。

## 许可证与第三方

项目新增代码采用 MIT，见 `LICENSE`。`vendor/JTReader` 保留其作者和 MIT 许可证；外部 Python 依赖各自遵循其许可证。MIT 授权不包含未公开工业模型、工件数据及其他第三方资产。解析逆解接口及转换器的来源信息保留在相应源文件中。
