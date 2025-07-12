import os
import numpy as np


class XYZConverter:
    def __init__(self, root_path, output_file_path, sequence):
        self.root_path = root_path
        self.output_file_path = output_file_path
        self.sequence = sequence

    def _write_xyz_file(self, data_array, output_file):
        # 检查数据是否为空或全为零
        if data_array.size == 0 or np.all(data_array == 0):
            print(f"Skipping saving {output_file} as it contains no valid points.")
            return

        with open(output_file, 'w') as f:
            if data_array.ndim == 1 and len(data_array) == 3:  # 确保 data_array 是 [x, y, z] 格式
                x, y, z = data_array
                if not np.all([x, y, z] == 0):  # 检查点不全为 0
                    f.write(f"{x} {y} {z}\n")
            elif data_array.ndim == 2:  # 处理包含多个点的情况
                valid_points = 0  # 计数有效点
                for point in data_array:
                    if len(point) == 3 and not np.all(point == 0):  # 确保是 [x, y, z] 格式并且不是全为 0
                        x, y, z = point
                        f.write(f"{x} {y} {z}\n")
                        valid_points += 1
                if valid_points == 0:
                    print(f"No valid points in {output_file}. File will not be saved.")
                    f.close()
                    os.remove(output_file)  # 删除空文件
            else:
                print(f"Unexpected data format: {data_array}")

    def _convert_to_xyz(self, folder_name):
        folder_path = os.path.join(self.root_path, folder_name)
        file_list = os.listdir(folder_path)
        output_folder = os.path.join(self.output_file_path, str(self.sequence))

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file in file_list:
            data_array = np.load(os.path.join(folder_path, file))
            if data_array.size == 0 or np.all(data_array == 0):  # 检查文件是否包含有效点
                print(f"Skipping {file} as it contains no valid points.")
                continue

            filename = os.path.splitext(file)[0] + ".xyz"
            output_file = os.path.join(output_folder, filename)
            self._write_xyz_file(data_array, output_file)

        print(f"Sequence {self.sequence}: {folder_name} to {subfolder_name} xyz file conversion complete.")

    # def convert_ground_truth(self):
    #     self._convert_to_xyz("ground_truth", "gt_xyz")

    # def convert_lidar_360(self):
    #     self._convert_to_xyz("lidar_360", "lidar_360_xyz")

    def convert_livox_avia(self):
        self._convert_to_xyz("radar_enhance_pcl")

    def filter_lidar_360(self, height_threshold=0):
        l360_path = os.path.join(self.root_path, "lidar_360")
        file_list = os.listdir(l360_path)
        output_folder = os.path.join(self.output_file_path, str(self.sequence), "lidar_360_filtered")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file in file_list:
            data_array = np.load(os.path.join(l360_path, file))
            filtered_points = [point for point in data_array if
                               len(point) == 3 and not np.all(point == 0) and point[2] > height_threshold]

            if len(filtered_points) == 0:  # 如果没有符合条件的点，则跳过文件保存
                print(f"No valid points met the threshold in {file}, skipping file creation.")
                continue

            filename = os.path.splitext(file)[0] + ".xyz"
            output_file = os.path.join(output_folder, filename)

            self._write_xyz_file(np.array(filtered_points), output_file)

        print(f"Sequence {self.sequence}: Lidar 360 xyz file filtering complete.")
        return output_folder





# converter = XYZConverter(root_path=root_path, output_file_path="path_to_output", sequence=1)
# # converter.convert_ground_truth()
# # converter.convert_lidar_360()
# converter.convert_livox_avia()
# filtered_lidar_path = converter.filter_lidar_360(height_threshold=0.7)

sequence = 99
root_path = r"G:\Anti-UAV\MMUAV_dataset\Anti_UAV_data\train\seq" + str(sequence)
output_file_path = r"G:"
result_path = os.path.join(output_file_path, str(sequence))
#
converter = XYZConverter(root_path=root_path, output_file_path=output_file_path, sequence=sequence)
converter.convert_livox_avia()