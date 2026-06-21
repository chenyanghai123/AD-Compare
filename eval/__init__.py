"""AD-Compare Evaluation Pipeline.

自动 Reference 匹配评估框架：
1. 抽取 OK 池 embedding (01_extract_ok_features.py)
2. 为 NG 检索 top-1 OK reference (02_retrieve_reference.py)
3. 批量 grounding 推理 (03_infer_grounding.py)
4. 坐标对齐 + mAP 计算 (04_compute_map.py)
5. 抽样可视化 (05_visualize_samples.py)
6. 生成评估报告 (06_make_report.py)
"""
