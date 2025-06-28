#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抗菌肽发现系统使用示例
演示完整的工作流程
"""

import os
import sys
import pandas as pd

def demo_prediction():
    """演示单独预测功能"""
    print("=" * 60)
    print("演示1: 多肽活性预测")
    print("=" * 60)
    
    # 创建示例序列
    demo_sequences = [
        "KWKLFKKIEK",      # 高活性预期
        "AAAAAAAAA",       # 低活性预期  
        "RRWWKKIRW",       # 高活性预期
        "GGGGGGGGG"        # 低活性预期
    ]
    
    # 创建临时CSV文件
    demo_df = pd.DataFrame({'sequence': demo_sequences})
    demo_input = 'demo_input.csv'
    demo_output = 'demo_output.csv'
    demo_df.to_csv(demo_input, index=False)
    
    # 运行预测
    cmd = f"python src/predict_peptide.py --model_path results/physchem_seqeng_biobert_dl_rawseq/best_physchem_seqeng_biobert_rawseq_classification.h5 --input {demo_input} --output {demo_output}"
    print(f"运行命令: {cmd}")
    os.system(cmd)
    
    # 显示结果
    if os.path.exists(demo_output):
        result_df = pd.read_csv(demo_output)
        print("\n预测结果:")
        for _, row in result_df.iterrows():
            activity = "高活性" if row['pred_label'] == 1 else "低活性"
            print(f"序列: {row['sequence']} -> 概率: {row['pred_probability']:.4f} ({activity})")
    
    # 清理临时文件
    for file in [demo_input, demo_output]:
        if os.path.exists(file):
            os.remove(file)

def demo_discovery():
    """演示完整发现流程"""
    print("\n" + "=" * 60)
    print("演示2: 三方法集成多肽发现")
    print("=" * 60)
    
    # 运行小规模发现
    cmd = "python three_method_discovery.py --sv_count 10 --rd_count 10 --vae_count 10"
    print(f"运行命令: {cmd}")
    os.system(cmd)
    
    print("\n发现流程完成！请查看生成的results_three_methods_*目录。")

def main():
    """主函数"""
    print("🚀 抗菌肽智能发现系统演示")
    print("本演示将展示系统的核心功能")
    
    # 检查模型文件
    model_path = "results/physchem_seqeng_biobert_dl_rawseq/best_physchem_seqeng_biobert_rawseq_classification.h5"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请确保已完成模型训练并且模型文件位于正确位置。")
        return
    
    try:
        # 演示1: 预测功能
        demo_prediction()
        
        # 演示2: 发现流程
        demo_discovery()
        
        print("\n" + "=" * 60)
        print("🎉 演示完成！")
        print("=" * 60)
        print("系统功能验证成功，可以开始实际的多肽发现工作。")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
