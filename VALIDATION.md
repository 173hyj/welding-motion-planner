# Release validation — 2026-10-27

环境：Ubuntu 22.04，Python 3.10，原项目虚拟环境。测试从不含私有模型的新发布目录执行，没有链接原工程 build/source 数据。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

发布套件：146 passed, 21 skipped。OSQP 产生 18 条未来默认参数变动警告。默认目录由 pytest.ini 定义，包含 tests 和五个 300 mm 实验包。21 项工业集成测试的精确名称在 PRIVATE_INTEGRATION_TESTS.json；默认跳过原因为未公开工业模型、场景和源文件，提供数据后使用 --run-private-integration。没有将这些跳过项计为通过。

首次无数据运行：128 项通过，19 项失败、2 项 setup 错误，全部对应未分发的 build/scene.json、build/robot.json、build/gunR1_detail.json、source/psz/StandaloneStudy_PsState.xml 或依赖私有输入的子进程。为公开发布显式区分这些集成检查，测试断言未改动。

历史 experimental 全目录不能单次直接收集：旧脚本存在同名测试模块、相对导入和独立启动要求。此发布不宣称完整历史目录全部测试通过。

新增包声明仅用于正确收集 300 mm 测试，没有修改规划算法。所有公开文本在提交前扫描常见令牌、私钥及硬编码密码；没有包含 .git 历史、模型、结果二进制、交接文档或连接凭据。扫描不是形式化安全证明。

工业结果来源：此前封存审计记录，349.830037133138 s；本发布没有重新运行工业场景，也不声称克隆代码后可直接复现该工件。
