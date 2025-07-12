# import os
# import numpy as np
#
#
# def process_xyz_files_to_single_txt(input_dirs, output_filepath):
#     # 打开最终的输出文件
#     with open(output_filepath, 'w') as output_file:
#         # 遍历每个输入文件夹
#         for input_dir in input_dirs:
#             # 获取文件夹中的所有 .xyz 文件
#             for filename in os.listdir(input_dir):
#                 if filename.endswith(".xyz"):
#                     input_filepath = os.path.join(input_dir, filename)
#
#                     # 读取 .xyz 文件
#                     points = np.loadtxt(input_filepath, dtype=np.float32, ndmin=2)
#
#                     # 为每个点添加固定颜色，并写入到输出文件
#                     fixed_color = (100.000000, 100.000000, 100.000000)
#                     for point in points:
#                         # 检查点的长度是否为3（x, y, z）
#                         if len(point) < 3:
#                             continue
#                         x, y, z = point[:3]
#                         output_file.write(
#                             f"{x:.6f} {y:.6f} {z:.6f} {fixed_color[0]:.6f} {fixed_color[1]:.6f} {fixed_color[2]:.6f}\n")
#
#                     print(f"Processed {input_filepath} into {output_filepath}")
#
# def rename_folders_to_area_format(directory):
#     # 获取指定目录中的所有文件夹名称并排序
#     folders = sorted([f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))])
#
#     # 按顺序重命名
#     for i, folder in enumerate(folders, start=1):
#         old_path = os.path.join(directory, folder)
#         new_folder_name = f"Area_{i}"
#         new_path = os.path.join(directory, new_folder_name)
#
#         # 重命名文件夹
#         os.rename(old_path, new_path)
#         print(f"Renamed '{folder}' to '{new_folder_name}'")
#
# # 指定文件夹路径
# directory_path = r"E:\CCFA\project\model\2024.11.11\MMAUD" # 替换为实际路径
#
# # 调用函数
# rename_folders_to_area_format(directory_path)


# 输入多个文件夹路径和输出文件夹路径
# input_dirs = [os.path.join(r"E:\CCFA\project\dataset\MMAUD", str(i), "lidar_360_xyz"), os.path.join(r"E:\CCFA\project\dataset\MMAUD", "livox_avia_xyz")]  # 替换为实际路径
# output_dir = "path/to/output_folder"  # 替换为实际输出路径

# 调用函数处理文件
# process_xyz_files(input_dirs, output_dir)

# for i in range(92, 102):
    # process_xyz_files_to_single_txt([os.path.join(r"E:\CCFA\project\dataset\MMAUD", str(i), "lidar_360_xyz"),
    #                    os.path.join(r"E:\CCFA\project\dataset\MMAUD", str(i), "livox_avia_xyz")],
    #                   os.path.join(r"E:\CCFA\project\model\2024.11.11\MMAUD", "seq"+str(i), "ALlPoints.txt"))


# import os
# import numpy as np
#
#
# def load_points_from_txt(file_path):
#     """Load points from a txt file with x y z r g b format."""
#     return np.loadtxt(file_path)
#
#
# def save_points_to_txt(file_path, points):
#     """Save points to a txt file with x y z r g b format."""
#     # 确保父文件夹存在
#     parent_dir = os.path.dirname(file_path)
#     if not os.path.exists(parent_dir):
#         os.makedirs(parent_dir)
#     # 保存点数据到文件
#     np.savetxt(file_path, points, fmt="%.6f %.6f %.6f %.6f %.6f %.6f")
#
# def remove_points(a_file, b_file, c_file):
#     """Remove points from a_file that exist in b_file and save result to c_file."""
#     # Load points from a and b
#     points_a = load_points_from_txt(a_file)
#     points_b = load_points_from_txt(b_file)
#
#     # Remove points from a that exist in b
#     mask = ~np.in1d(points_a.view([('', points_a.dtype)] * points_a.shape[1]),
#                     points_b.view([('', points_b.dtype)] * points_b.shape[1]))
#     filtered_points = points_a[mask]
#
#     # Save the filtered points to the c file
#     save_points_to_txt(c_file, filtered_points)
#     print(f"Processed {a_file}: saved result to {c_file}")
#
#
# # 使用示例
# for i in range(1, 102):
#     a_file = os.path.join(
#         r"E:\CCFA\report_poster\Experiments\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version",
#         f"Area_{i}", "ALlPoints.txt")
#     b_file = os.path.join(
#         r"E:\CCFA\report_poster\Experiments\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version",
#         f"Area_{i}", r"ALlPoints\Annotations\drone.txt")
#     c_file = os.path.join(
#         r"E:\CCFA\report_poster\Experiments\mmdetection3d\data\s3dis\Stanford3dDataset_v1.2_Aligned_Version",
#         f"Area_{i}", r"ALlPoints\Annotations\background.txt")
#
#     if os.path.exists(a_file) and os.path.exists(b_file):
#         remove_points(a_file, b_file, c_file)
#     else:
#         print(f"Skipping Area_{i}: Missing a_file or b_file")

import mmengine
# 加载 .pkl 文件
pkl_path = r'E:\CCFA\report_poster\Experiments\mmdetection3d\data/s3dis/s3dis_infos_Area_5.pkl'
data = mmengine.load(pkl_path)
# 将 data 转换为字典结构而不是列表（假设列表中只有一个字典）
if isinstance(data, list) and len(data) == 1:
    data = data[0]  # 提取第一个元素，去掉外层的 []
# 保存修改后的数据
output_path = r"E:\CCFA\report_poster\Experiments\mmdetection3d\data\s3dis\s3dis_infos_Area_5.pkl"
mmengine.dump(data, output_path)
print("已去除列表外层的 [] 并保存到新文件中。")


