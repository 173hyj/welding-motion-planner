# 21焊点：从零规划实测复现指南

更新：2026-11-02。推荐Ubuntu 22.04、Python 3.10，至少16GB内存和20GB可用磁盘。规划使用CPU及Linux fork，不依赖V100 CUDA；Windows请使用WSL2。

## 实际验证结果与边界

本次在独立公开代码目录，从原始场景重新生成动作和路径，没有将349.83秒最终轨迹或原680节点候选库作为本轮规划输入：

| 检查 | 实测结果 |
|---|---|
| 原始焊点 / 几何筛选保留 | 29 / 21 |
| 新生成300mm完整动作 | 21/21通过 |
| 必需点间转移 | 20/20通过 |
| 导出15维轨迹状态 | 63,274 |
| 新路线展示时间 | **402.2446465924116秒** |
| 完整工厂回放 | 源FK驱动，已观察到实体运动 |
| 发布测试 | 148通过，21个依赖特定工业数据的原集成测试默认跳过 |

动作搜索约337.63秒，成功的转移搜索约950.58秒，属于计算耗时。较低预算首轮只通过19/20条转移，失败报告保留，不能计为成功。

**这证明能够从零规划完整21点，但尚未从零重现349.83秒的原优化成绩。** 原349.83秒路线包含全候选联合选择、历史六阶段边界和继承转移，本次冷入口先解决完整可行路线，未重演原优化历史。不能把两者称为同一优化结果的算法复现。

## 下载锁定版本

仓库：https://github.com/173hyj/welding-motion-planner

完整源码与数据：https://github.com/173hyj/welding-motion-planner/releases/tag/reproduction-data-20261102

必须下载：

- `welding-planner-reproduction-source-20261102-v3.zip`：含300mm搜索、反馈修复、完整轨迹导出、回放、测试及环境锁定。
- `public-reproduction-assets-20261102.tar.gz`：原始工程、模型、装配场景及运行时必需模块。
- `REPRODUCTION_SHA256SUMS.txt`：SHA256清单。

可选下载`cold300-clean-validation-20261102.tar.gz`核对本次证据，**不能将它作为从零规划输入**。

请优先使用Release源码附件；Git main的部分新修复曾受传输错误影响，不保证旧main包含全部修复。原始数据大小1,088,863,250字节，SHA256为`331e0aa5f6a8c358884da89885788023207d9dc03ab097e2cc4c885c40fb8185`，已与GitHub服务器资产digest核对一致。

## 安装和解压

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip unzip libgl1 libglfw3 ffmpeg
mkdir welding-motion-planner
cd welding-motion-planner
# 把Release的三个必需文件下载到此目录。
grep -E ' (welding-planner-reproduction-source-20261102-v3.zip|public-reproduction-assets-20261102.tar.gz)$' REPRODUCTION_SHA256SUMS.txt | sha256sum -c -
unzip welding-planner-reproduction-source-20261102-v3.zip
tar -xzf public-reproduction-assets-20261102.tar.gz
python3 -m venv .venv
.venv/bin/pip install -r ENVIRONMENT_LOCK.txt
test -f build/scene_placement_corrected_v1.json
test -f source/psz/EngData/engData_ver2.xml
```

build中的场景、预处理网格和坐标标定是模型输入，必须保留。不要复制原设备的虚拟环境。

## 从零规划命令

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
RUN="build/cold300_$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python -u experimental/welding_retreat_transfer_v1/run_fresh_pipeline_cold300.py \
  --out "$RUN" --workers 3 --directions 64 --poses 96 --configurations 4 \
  --point-seconds 240 --angle-seconds 20 \
  --transfer-iterations 1000 --edge-seconds 60 --edge-attempts 2 \
  --reselect-seconds 120 --reselect-per-edge 3 --reorder-first-rounds 2 --ordering-rounds 8
```

必须使用全新输出目录。入口先生成新动作，再生成新转移，不传入旧route、incumbent、pose-hints或seed-snapshot。反馈重选只使用本轮新生成的其他有效动作证据。

上述参数对应本次成功阶段的配置。由于有限时间预算和进程调度，不保证其他设备逐字节得到相同构型、顺序或402.24秒。低预算首轮实测失败，不能把旧12秒/200次迭代参数写成“保证完整”的命令。

## 验收、导出和回放

先检查：

1. `$RUN/pipeline_report.json`中`complete=true`，两个阶段退出码为0。
2. `$RUN/joint_actions/report.json`中21个动作全部通过，要求距离300mm。
3. `$RUN/transfers/report.json`中`all_required_points_route_planned=true`、`selected_passed_edges=20`。
4. 焊点ID集合与原交接包21点一致，不能只比较数量。

全部通过后：

```bash
.venv/bin/python experimental/welding_retreat_transfer_v1/export_route_cold300.py \
  --transfers "$RUN/transfers" --out "$RUN/replay"
.venv/bin/python experimental/welding_retreat_transfer_v1/replay_factory_cold300.py \
  --route "$RUN/replay" --speed 1 --paused
```

得到`route.npz`和`audit.json`。states列0:6和6:12是两臂关节角，列12为变位机角度，列13/14为两枪开合。角度rad、开枪mm、times为s、TCP为源工程世界坐标mm，机器人UUID及访问顺序见audit.json。空格播放/暂停、R重放、N/B切换焊点、T显示路径、+/-改速度，鼠标调整视角。

本次先在`build/cold300-clean-20261102/joint_actions`生成动作，随后经过未通过试验和反馈修复，成功转移保存为`transfers-extended`。因此实测证据中的目录名与统一入口的新输出`transfers`不同。证据附件保留各阶段实际命令、完整报告和GUI检查，没有把失败pipeline_report改成成功。

## 原349.83秒成果

原交接包的最终V6轨迹为21点、62244状态、349.830037133138秒，其SHA256：

```text
a7b94ab501b33e8a87c119dff0ee5a75c08401786debd46ec27714dd1f6c7137
```

解压原交接ZIP后，可用本次源码附件的入口核对并回放：

```bash
.venv/bin/python replay_handoff21.py --bundle /absolute/path/to/unpacked-handoff --verify-only
.venv/bin/python replay_handoff21.py --bundle /absolute/path/to/unpacked-handoff --paused --speed 1
```

这只是封存结果回放，不能冒充从零重算。若要求从零重新达到349.83秒及同样的全候选联合优化成果，目前仍需继续接通和验证该优化链；本指南不能对此作已完成保证。

## 检查范围

碰撞检查沿用原装配接触例外及离散采样规则。“从零启动”表示重新计算动作及路径，不等于另一实现的独立冷碰撞复核，报告中的`independent_cold_performed=false`应保留原义。展示计时使用既有速度模型，不构成实体控制器或完整动力学认证，不包含尚未规划的home到首动作路径。

工业数据按项目所有者要求公开分享。代码MIT许可不会自动改变第三方模型既有权利。本包无SSH密码、私钥或GitHub令牌。
