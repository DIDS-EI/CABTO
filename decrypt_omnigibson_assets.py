#!/usr/bin/env python3
"""
解密 OmniGibson 所有资产到指定文件夹的脚本

使用方法:
    python decrypt_omnigibson_assets.py
"""

import os
import sys
from pathlib import Path
from cryptography.fernet import Fernet

# 尝试导入 tqdm，如果没有安装则使用简单的迭代
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=""):
        """简单的 tqdm 替代，如果没有安装 tqdm"""
        print(f"{desc}...")
        return iterable

# 添加项目路径以便导入 global_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dexrl import global_config


def decrypt_file(encrypted_filename, decrypted_filename, key_path):
    """
    解密单个文件
    
    Args:
        encrypted_filename: 加密文件路径
        decrypted_filename: 解密后文件保存路径
        key_path: 密钥文件路径
    """
    with open(key_path, "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)

    with open(encrypted_filename, "rb") as enc_f:
        encrypted = enc_f.read()

    decrypted = fernet.decrypt(encrypted)

    # 确保目标目录存在
    os.makedirs(os.path.dirname(decrypted_filename), exist_ok=True)
    
    with open(decrypted_filename, "wb") as decrypted_file:
        decrypted_file.write(decrypted)


def find_all_encrypted_files(source_dir):
    """
    查找所有加密的 .encrypted.usd 文件
    
    Args:
        source_dir: OmniGibson 资产根目录
        
    Returns:
        list: 所有加密文件路径的列表
    """
    encrypted_files = []
    source_path = Path(source_dir)
    
    # 查找所有 .encrypted.usd 文件
    for encrypted_file in source_path.rglob("*.encrypted.usd"):
        encrypted_files.append(encrypted_file)
    
    return encrypted_files


def get_relative_path(full_path, base_path):
    """
    获取相对于基础路径的相对路径
    
    Args:
        full_path: 完整路径
        base_path: 基础路径
        
    Returns:
        str: 相对路径
    """
    return os.path.relpath(full_path, base_path)


def decrypt_all_assets(source_dir, target_dir, key_path):
    """
    解密所有 OmniGibson 资产到目标文件夹
    
    Args:
        source_dir: OmniGibson 资产源目录
        target_dir: 解密后资产保存的目标目录 (应该是 og_dataset/objects/ 目录)
        key_path: 密钥文件路径
    """
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误: 源目录不存在: {source_dir}")
        return
    
    # 检查密钥文件是否存在
    if not os.path.exists(key_path):
        print(f"错误: 密钥文件不存在: {key_path}")
        return
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 查找所有加密文件
    print(f"正在扫描加密文件: {source_dir}")
    encrypted_files = find_all_encrypted_files(source_dir)
    
    if not encrypted_files:
        print("未找到任何加密文件")
        return
    
    print(f"找到 {len(encrypted_files)} 个加密文件")
    
    # 确定基础路径（用于计算相对路径）
    # 源目录结构: {source_dir}/og_dataset/objects/{category}/{object_name}/usd/{object_name}.encrypted.usd
    # 我们需要提取 og_dataset/objects/ 之后的部分
    source_path = Path(source_dir)
    
    # 查找 og_dataset/objects 在源目录中的位置
    og_dataset_objects_path = None
    source_parts = source_path.parts
    
    for i in range(len(source_parts)):
        if source_parts[i] == "og_dataset" and i + 1 < len(source_parts) and source_parts[i + 1] == "objects":
            og_dataset_objects_path = Path(*source_parts[:i + 2])
            break
    
    if og_dataset_objects_path is None:
        # 如果没有找到，尝试直接在源目录下查找 og_dataset/objects
        potential_path = Path(source_dir) / "og_dataset" / "objects"
        if potential_path.exists():
            og_dataset_objects_path = potential_path
        else:
            print("警告: 无法确定 og_dataset/objects 路径，将使用完整路径结构")
            og_dataset_objects_path = Path(source_dir)
    
    # 解密所有文件
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    print(f"\n开始解密文件到: {target_dir}")
    
    for encrypted_file in tqdm(encrypted_files, desc="解密进度"):
        try:
            encrypted_path = Path(encrypted_file)
            
            # 计算相对于 og_dataset/objects 的路径
            try:
                relative_path = encrypted_path.relative_to(og_dataset_objects_path)
            except ValueError:
                # 如果无法计算相对路径，尝试从完整路径中提取
                # 查找 og_dataset/objects 在完整路径中的位置
                parts = encrypted_path.parts
                for i in range(len(parts)):
                    if parts[i] == "og_dataset" and i + 1 < len(parts) and parts[i + 1] == "objects":
                        relative_path = Path(*parts[i + 2:])
                        break
                else:
                    relative_path = encrypted_path.name
            
            # 构建目标文件路径
            # 将文件名从 {object_name}.encrypted.usd 改为 {object_name}.usd
            # relative_path 应该是: {category}/{object_name}/usd/{object_name}.encrypted.usd
            # 目标路径应该是: {target_dir}/{category}/{object_name}/usd/{object_name}.usd
            target_file = Path(target_dir) / relative_path
            # 替换文件名，去掉 .encrypted
            target_file = target_file.parent / target_file.name.replace(".encrypted.usd", ".usd")
            
            # 如果目标文件已存在，跳过
            if target_file.exists():
                skip_count += 1
                continue
            
            # 解密文件
            decrypt_file(str(encrypted_file), str(target_file), key_path)
            success_count += 1
            
        except Exception as e:
            print(f"\n解密失败 {encrypted_file}: {e}")
            fail_count += 1
    
    print(f"\n解密完成!")
    print(f"成功: {success_count} 个文件")
    print(f"跳过: {skip_count} 个文件 (已存在)")
    print(f"失败: {fail_count} 个文件")
    print(f"目标目录: {target_dir}")


def main():
    """主函数"""
    # OmniGibson 资产源目录
    source_dir = os.path.join(global_config.data_path, "Assets/OmniGibson")
    
    # 目标目录
    target_dir = "/media/cys/disk1/OmniBT-Data/Assets/OmniGibson/og_dataset/objects/"
    
    # 密钥文件路径
    key_path = os.path.join(global_config.data_path, "Assets/OmniGibson/omnigibson.key")
    
    print("=" * 60)
    print("OmniGibson 资产解密工具")
    print("=" * 60)
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"密钥文件: {key_path}")
    print("=" * 60)
    
    # 执行解密
    decrypt_all_assets(source_dir, target_dir, key_path)


if __name__ == "__main__":
    main()

