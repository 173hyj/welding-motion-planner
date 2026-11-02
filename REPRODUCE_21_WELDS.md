# 21焊点项目：从零规划与349.83秒封存结果

更新：2026-11-02。

## 两种复现必须区分

1. **封存结果回放**：读取最终V6轨迹，复现21点、349.830037133138秒、62244个状态。不是重新规划。
2. **从零重新规划**：从原始场景加载几何，重新进行5°焊枪姿态筛选、机械臂逆解、开枪及300mm退枪联合搜索、焊点排序、转移搜索与检查。必须写入全新输出目录，不能把已有route.npz当作计算输入。

此前公开版本只有代码及部分场景描述，不能据此宣称完整冷启动已经通过。此次新增入口为 `experimental/welding_retreat_transfer_v1/run_fresh_pipeline_cold300.py`，原1000mm入口保持不变。

**当前验收状态：全量冷启动验证正在运行，尚不能承诺新运行得到21点或349.83秒。** 原349.83秒结果利用了历史六阶段边界、候选种子和继承转移；从零重算的路线与时间可能不同。有限预算未找到解不能当作无解，未验收的新结果不能替换封存基线。

## 仓库和数据

代码仓库：<https://github.com/173hyj/welding-motion-planner>。

新的原始数据资产名：`public-reproduction-assets-20261102.tar.gz`。

- 文件字节数：1088863250。
- SHA256：`331e0aa5f6a8c358884da89885788023207d9dc03ab097e2cc4c885c40fb8185`。
- 内容：完整source原始工程、build根目录场景JSON、工厂/机械臂/工件/夹具网格、变位机描述，以及历史build目录中实际被运行时导入的少量源码。
- 不含SSH密钥、GitHub令牌或虚拟环境。
- 数据将放在GitHub Release；Release草稿不是可供他人下载的正式发布。

## 新Ubuntu设备准备

建议Ubuntu 22.04、Python 3.10、至少16GB内存与20GB可用磁盘。CPU承担当前规划，V100 GPU不是算法必需条件。多进程使用Linux fork，Windows请通过WSL2 Ubuntu运行规划。

```bash
git clone https://github.com/173hyj/welding-motion-planner.git
cd welding-motion-planner
# 必须切换到包含cold300入口的发布提交；不能默认旧main已包含它。
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
sudo apt-get install libgl1 libglfw3 ffmpeg
```

下载正式Release的数据资产到仓库目录，先检查字节数和SHA256，再解压：

```bash
echo '331e0aa5f6a8c358884da89885788023207d9dc03ab097e2cc4c885c40fb8185  public-reproduction-assets-20261102.tar.gz' | sha256sum -c -
tar -xzf public-reproduction-assets-20261102.tar.gz
test -f build/scene_placement_corrected_v1.json
test -f source/psz/EngData/engData_ver2.xml
```

请勿使用原始ZIP中的虚拟环境，也不要将V100的绝对路径拷贝到本机直接运行。

## 全量冷规划命令

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
RUN="build/cold300_$(date -u +%Y%m%dT%H%M%SZ)"
.venv/bin/python -u experimental/welding_retreat_transfer_v1/run_fresh_pipeline_cold300.py \
  --out "$RUN" --workers 3 --directions 64 --poses 96 --configurations 4 \
  --point-seconds 240 --angle-seconds 20 \
  --transfer-iterations 200 --edge-seconds 12 --edge-attempts 2 \
  --reselect-seconds 45 --reorder-first-rounds 2
```

该入口新建输出目录，先运行新300mm完整动作搜索，再运行新转移搜索。不要传入旧候选、pose-hints、incumbent或最终V6轨迹。上述预算只是首轮验证参数，尚未声称已足以从零重现全部21点。

## 验收条件

- 程序正常结束，`pipeline_report.json`中的`complete`为true，各阶段退出码为0。
- `joint_actions/report.json`中所有要求点均有完整动作，长度统一300mm，保留全部失败/未解决记录。
- `transfers/report.json`中的`all_required_points_route_planned`为true，完成规划所需全部转移。
- 必须核对焊点ID集合与交接包21个ID一致，不能只比较数量。
- 将来还需接入本次全候选联合选择和计时验收，冷启动输出通过前不能称为349.83秒成果的完整算法复现。
- 模型检查沿用原接触例外和采样规则；设备执行还需控制器及动力学验证。

## 已封存成果的独立核对

最终轨迹SHA256为 `a7b94ab501b33e8a87c119dff0ee5a75c08401786debd46ec27714dd1f6c7137`。

配套交接ZIP的SHA256为 `853f5756d9f0ffa54dd9a322fb91192c09a9dd55a48934a1fc851500db695a8f`。解压交接包后，使用本次补充的 `replay_handoff21.py`：

```bash
.venv/bin/python replay_handoff21.py --bundle /absolute/path/to/unpacked-handoff --verify-only
.venv/bin/python replay_handoff21.py --bundle /absolute/path/to/unpacked-handoff --paused --speed 1
```

`--verify-only`已在独立交接目录实测通过：全部清单哈希、21焊点、62244个状态、349.830037133138秒和源模型FK启动。该检查不等同重新执行碰撞规划。

**不要把本文当前版本作为“完整从零规划已验收”的证明。只有全量冷运行和后续联合选择的实际报告通过，才能更新这一结论。**
