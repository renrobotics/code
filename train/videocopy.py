import os
import shutil
from typing import List

def copy_videos(video_list_file: str, source_dir: str, destination_dir: str) -> None:
    """
    根据文件名列表，将视频从源目录复制到目标目录。

    Args:
        video_list_file (str): 包含要复制的视频名称的 .txt 文件的路径。
        source_dir (str): 包含所有视频文件的目录的路径。
        destination_dir (str): 将视频复制到的目录的路径。
    """
    # 确保目标目录存在，如果不存在则创建
    os.makedirs(destination_dir, exist_ok=True)

    # 读取视频文件列表
    try:
        with open(video_list_file, 'r', encoding='utf-8') as f:
            # 使用列表推导式读取文件名，并去除每行末尾的换行符和空白
            video_names = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"错误：找不到文件 {video_list_file}。请检查路径是否正确。")
        return
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return

    print(f"在列表文件中找到 {len(video_names)} 个视频。")

    copied_count = 0
    not_found_count = 0
    not_found_list: List[str] = []

    # 遍历视频名称并进行复制
    for video_name in video_names:
        source_path = os.path.join(source_dir, video_name)
        destination_path = os.path.join(destination_dir, video_name)

        if os.path.exists(source_path):
            try:
                print(f"正在复制 {video_name} 到 {destination_dir}...")
                shutil.copy2(source_path, destination_path)
                copied_count += 1
            except Exception as e:
                print(f"复制文件 {video_name} 时发生错误: {e}")
        else:
            print(f"警告: 在源文件夹 {source_dir} 中未找到 {video_name}")
            not_found_count += 1
            not_found_list.append(video_name)

    # 打印总结信息
    print("\n--- 复制任务总结 ---")
    print(f"计划复制的总视频数: {len(video_names)}")
    print(f"成功复制的视频数: {copied_count}")
    print(f"未在源文件夹中找到的视频数: {not_found_count}")
    if not_found_list:
        print("\n未找到的文件列表:")
        for name in not_found_list:
            print(f"- {name}")
    print("----------------------\n")


if __name__ == "__main__":
    # --- 请在这里配置您的路径 ---
    # 1. 包含视频文件名的txt文件的完整路径
    #    Windows 示例: "C:\\Users\\YourUser\\Desktop\\video_list.txt"
    VIDEO_LIST_TXT = "C:\Workspace\emp\label.txt"

    # 2. 存放您所有视频的源文件夹的路径
    #    Windows 示例: "D:\\MyVideos\\All_Videos"
    SOURCE_FOLDER = "C:\Workspace\emp\sourcevideo"

    # 3. 您希望将视频复制到的目标文件夹的路径
    #    如果这个文件夹不存在，程序会自动创建
    #    Windows 示例: "D:\\MyVideos\\Selected_Videos"
    DESTINATION_FOLDER = "C:\Workspace\emp\desvideo"
    # ------------------------------------

    # 检查路径是否已被修改
    if "path\\to\\your" in VIDEO_LIST_TXT or "path\\to\\your" in SOURCE_FOLDER or "path\\to\\your" in DESTINATION_FOLDER:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! 请先修改脚本中的三个路径（VIDEO_LIST_TXT, SOURCE_FOLDER, DESTINATION_FOLDER），然后再运行! !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        copy_videos(VIDEO_LIST_TXT, SOURCE_FOLDER, DESTINATION_FOLDER)
