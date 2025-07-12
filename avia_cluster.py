# import os
# import numpy as np
# from sklearn.cluster import DBSCAN
#
#
# class AviaPointCluster:
#     def __init__(self, eps=2, min_samples=5):
#         self.eps = eps
#         self.min_samples = min_samples
#
#     def load_points(self, filename):
#         points = []
#         with open(filename, 'r') as f:
#             for line in f:
#                 x, y, z = map(float, line.strip().split())
#                 points.append([x, y, z])
#         return np.array(points)
#
#     def save_points(self, points, filename):
#         with open(filename, 'w') as f:
#             for point in points:
#                 f.write(f"{point[0]} {point[1]} {point[2]}\n")
#
#     def cluster_and_save(self, input_filename, output_filename):
#         points = self.load_points(input_filename)
#         if len(points) == 0:
#             print(f"No points found in {input_filename}.")
#             return
#
#         # Perform DBSCAN clustering
#         #  = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
#         # labels = clustering.labels_
#
#         clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
#         labels = clustering.fit_predict(points)
#
#         clustered_points = points[labels != -1]
#
#         if len(clustered_points) > 0:
#             self.save_points(clustered_points, output_filename)
#             print(f"Clustered points saved to {output_filename}.")
#         else:
#             print(f"No clustered points found in {input_filename}. File not saved.")
#
#     def cluster_folder(self, folder_path, output_folder):
#         if not os.path.exists(output_folder):
#             os.makedirs(output_folder)
#
#         # 此处移除硬编码的路径部分
#         # folder_path = os.path.join(folder_path, "livox_avia_xyz")
#
#         for file in os.listdir(folder_path):
#             input_file = os.path.join(folder_path, file)
#             output_file = os.path.join(output_folder, file)
#             self.cluster_and_save(input_file, output_file)

import os
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


class AviaPointCluster:
    def __init__(self, eps=2, min_samples=10):
        self.eps = eps
        self.min_samples = min_samples

    def load_points(self, filename):
        points = []
        with open(filename, 'r') as f:
            for line in f:
                x, y, z = map(float, line.strip().split())
                points.append([x, y, z])
        return np.array(points)

    def save_points(self, points, filename):
        with open(filename, 'w') as f:
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

    def merge_clusters(self, points, labels, distance_threshold=5):
        unique_labels = set(labels)
        # if -1 in unique_labels:
        #     unique_labels.remove(-1)  # 移除噪声点

        # 计算聚类质心
        centroids = np.array([points[labels == label].mean(axis=0) for label in unique_labels])

        # 确保 centroids 是二维数组
        centroids = np.atleast_2d(centroids)

        # 计算质心之间的距离
        distances = cdist(centroids, centroids)

        # 合并距离较近的聚类
        for i, label in enumerate(unique_labels):
            close_clusters = np.where(distances[i] < distance_threshold)[0]
            for close_label_index in close_clusters:
                close_label = list(unique_labels)[close_label_index]
                if close_label != label:
                    labels[labels == close_label] = label

        return labels

    def cluster_and_save(self, input_filename, output_filename):
        points = self.load_points(input_filename)
        if len(points) == 0:
            print(f"No points found in {input_filename}.")
            return

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(points)

        # Merge nearby clusters to form a continuous trajectory
        labels = self.merge_clusters(points, labels)

        # 排除噪声
        clustered_points = points[labels != -1]

        clustered_points = points

        if len(clustered_points) > 0:
            self.save_points(clustered_points, output_filename)
            print(f"Clustered points saved to {output_filename}.")
        else:
            print(f"No clustered points found in {input_filename}. File not saved.")

    def cluster_folder(self, folder_path, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for file in os.listdir(folder_path):
            input_file = os.path.join(folder_path, file)
            output_file = os.path.join(output_folder, file)
            self.cluster_and_save(input_file, output_file)
