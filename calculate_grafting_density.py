#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算多肽接枝密度脚本
将200nm/spot转换为μg/cm²，其中spot为直径6mm的圆
使用BioPython进行精确的分子量计算
"""

import pandas as pd
import numpy as np
import math

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPYTHON_AVAILABLE = True
    print("使用BioPython进行精确分子量计算")
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("警告: BioPython未安装，使用备用计算方法")
    print("建议安装: pip install biopython")

def calculate_peptide_molecular_weight(sequence):
    """
    使用BioPython或备用方法计算多肽分子量 (Da)
    
    BioPython方法更精确，考虑了:
    - 标准氨基酸分子量
    - 肽键形成时的脱水反应
    - 分子式的精确计算
    """
    sequence = sequence.upper().strip()
    
    if BIOPYTHON_AVAILABLE:
        try:
            # 使用BioPython的ProteinAnalysis进行精确计算
            protein_analysis = ProteinAnalysis(sequence, monoisotopic=False)
            molecular_weight = protein_analysis.molecular_weight()
            return molecular_weight
        except Exception as e:
            print(f"BioPython计算失败 '{sequence}': {e}")
            print("使用备用方法计算...")

def calculate_grafting_density(molecular_weight, density_nm_per_spot=200, spot_diameter_mm=6):
    """
    计算接枝密度 (μg/cm²)
    
    参数:
    - molecular_weight: 多肽分子量 (Da)
    - density_nm_per_spot: 接枝密度 (nm/spot)
    - spot_diameter_mm: spot直径 (mm)
    
    返回:
    - grafting_density: 接枝密度 (μg/cm²)
    """
    # 计算spot面积 (cm²)
    spot_radius_cm = (spot_diameter_mm / 2) / 10  # mm转cm
    spot_area_cm2 = math.pi * (spot_radius_cm ** 2)    # 密度转换
    # 200 nmol/spot = 200 × 10^-9 mol/spot 
    moles_per_spot = density_nm_per_spot * 1e-9

    # 转换为质量 (g)
    # 质量 = 摩尔数 × 分子量 (g/mol)
    # 注意：分子量单位为Da，需要转换为g/mol (1 Da = 1 g/mol)
    mass_per_spot_g = moles_per_spot * molecular_weight
    
    # 转换为μg
    mass_per_spot_ug = mass_per_spot_g * 1e6
    
    # 计算接枝密度 (μg/cm²)
    grafting_density_ug_cm2 = mass_per_spot_ug / spot_area_cm2
    
    return grafting_density_ug_cm2

def main():
    # 读取预测结果文件
    input_file = 'top15.csv'
    output_file = 'top15.csv'
    
    print("读取预测结果文件...")
    df = pd.read_csv(input_file)
    
    print(f"处理 {len(df)} 个多肽序列...")
    
    # 计算每个多肽的分子量
    molecular_weights = []
    grafting_densities = []
    
    for i, sequence in enumerate(df['sequence']):
        print(f"处理序列 {i+1}/{len(df)}: {sequence}")
        
        # 计算分子量
        mw = calculate_peptide_molecular_weight(sequence)
        molecular_weights.append(mw)
        
        # 计算接枝密度
        if mw is not None:
            gd = calculate_grafting_density(mw)
            grafting_densities.append(gd)
            print(f"  分子量: {mw:.2f} Da")
            print(f"  接枝密度: {gd:.4f} μg/cm²")
        else:
            grafting_densities.append(None)
            print(f"  无法计算分子量")
    
    # 添加新列到DataFrame
    df['molecular_weight_Da'] = molecular_weights
    df['grafting_density_ug_cm2'] = grafting_densities
    
    # 保存结果
    df.to_csv(output_file, index=False)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 显示统计信息
    valid_mw = [mw for mw in molecular_weights if mw is not None]
    valid_gd = [gd for gd in grafting_densities if gd is not None]
    
    if valid_mw:
        print(f"\n统计信息:")
        print(f"分子量范围: {min(valid_mw):.2f} - {max(valid_mw):.2f} Da")
        print(f"平均分子量: {np.mean(valid_mw):.2f} Da")
        print(f"接枝密度范围: {min(valid_gd):.4f} - {max(valid_gd):.4f} μg/cm²")
        print(f"平均接枝密度: {np.mean(valid_gd):.4f} μg/cm²")

if __name__ == '__main__':
    main()
