#!/usr/bin/env python3
"""
检查 PyTorch 检查点文件的内容
用于查看 .pth.tar 文件中包含的具体信息
"""
import torch
import os
import argparse

def check_checkpoint(checkpoint_path):
    """
    检查检查点文件的内容
    
    Args:
        checkpoint_path: 检查点文件路径
    """
    print(f"正在检查检查点文件: {checkpoint_path}")
    print("-" * 50)
    
    try:
        # 加载检查点文件
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print("检查点文件内容:")
        print(f"文件类型: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print("\n包含的键值:")
            for key, value in checkpoint.items():
                print(f"  {key}: {type(value)}")
                
                # 对于特定的键，显示更多信息
                if key == 'epoch':
                    print(f"    当前训练轮次: {value}")
                elif key == 'net_type':
                    print(f"    网络类型: {value}")
                elif key == 'actor_type':
                    print(f"    Actor类型: {value}")
                elif key == 'net' and isinstance(value, dict):
                    print(f"    网络参数数量: {len(value)} 个层")
                    # 显示前几个层的名称
                    layer_names = list(value.keys())[:5]
                    print(f"    前几个层: {layer_names}")
                elif key == 'optimizer' and isinstance(value, dict):
                    print(f"    优化器状态键: {list(value.keys())}")
                elif key == 'stats' and isinstance(value, dict):
                    print(f"    训练统计信息: {list(value.keys())}")
        
        print("\n检查点文件检查完成!")
        
    except Exception as e:
        print(f"加载检查点文件时出错: {e}")

def extract_model_weights(checkpoint_path, output_path):
    """
    从检查点文件中提取模型权重并保存为 .pth 文件
    
    Args:
        checkpoint_path: 检查点文件路径
        output_path: 输出的权重文件路径
    """
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'net' in checkpoint:
            # 提取模型权重
            model_weights = checkpoint['net']
            
            # 保存为 .pth 文件
            torch.save(model_weights, output_path)
            print(f"模型权重已保存到: {output_path}")
        else:
            print("检查点文件中没有找到 'net' 键")
            
    except Exception as e:
        print(f"提取模型权重时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='检查 PyTorch 检查点文件')
    parser.add_argument('checkpoint', help='检查点文件路径')
    parser.add_argument('--extract', '-e', help='提取模型权重并保存为指定路径的 .pth 文件')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"错误: 文件 {args.checkpoint} 不存在")
        return
    
    # 检查检查点文件
    check_checkpoint(args.checkpoint)
    
    # 如果指定了提取选项，则提取模型权重
    if args.extract:
        extract_model_weights(args.checkpoint, args.extract)

if __name__ == "__main__":
    main()
