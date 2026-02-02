# -*- coding: utf-8 -*-
"""
功能: 对 Avia 点云进行 DBSCAN 聚类并保存结果。
输入: .xyz 点云文件与聚类参数。
输出: 聚类后的 .xyz 文件。
运行: 由工具脚本调用。
"""

import os
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


class AviaPointCluster:
    def __init__(self, eps, min_samples, merge_distance_threshold):
        self.eps = eps
        self.min_samples = min_samples
        self.merge_distance_threshold = merge_distance_threshold

    def load_points(self, filename):
        points = []
        with open(filename, "r") as f:
            for line in f:
                x, y, z = map(float, line.strip().split())
                points.append([x, y, z])
        return np.array(points)

    def save_points(self, points, filename):
        with open(filename, "w") as f:
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

    def merge_clusters(self, points, labels):
        unique_labels = set(labels)

        centroids = np.array([points[labels == label].mean(axis=0) for label in unique_labels])
        centroids = np.atleast_2d(centroids)
        distances = cdist(centroids, centroids)

        for i, label in enumerate(unique_labels):
            close_clusters = np.where(distances[i] < self.merge_distance_threshold)[0]
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

        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(points)

        labels = self.merge_clusters(points, labels)

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

