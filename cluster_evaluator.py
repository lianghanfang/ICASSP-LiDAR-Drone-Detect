class ClusterEvaluator:
    @staticmethod
    def calculate_score(cluster_info, label):
        info = cluster_info[label]
        path_length_value = info.get("path_length", 0)
        relative_density_value = info["relative_density"]
        voxel_roi_value = info["voxel_RoI"]
        cluster_info[label]["score"]["relative_density_score"] = relative_density_value
        cluster_info[label]["score"]["voxel_roi_score"] = voxel_roi_value
        cluster_info[label]["score"]["path_length_score"] = path_length_value

    """
    这个方法还有待商榷，跟文件数量有关，能否直接比较两个参数集的得分
    """
    @staticmethod
    def compare_and_filter(prev_scores, current_scores):
        filtered_classes = {}
        for label, current_score in current_scores.items():
            if label in prev_scores:
                prev_score = prev_scores[label]
                if (current_score["relative_density_score"] < prev_score["relative_density_score"] and
                        current_score["voxel_roi_score"] > prev_score["voxel_roi_score"]):
                    filtered_classes[label] = label
        return filtered_classes

    """
    后续修改
    将权重参数导入
    """
    @staticmethod
    def calculate_total_score(cluster_info):
        print("Calculating Total Score...")
        for label, info in cluster_info.items():
            # 假设 total_score 是相对密度得分和 voxel RoI 得分的加权和
            relative_density_score = info.get("relative_density", 0)
            voxel_roi_score = info.get("voxel_RoI", 0)

            # 计算 total_score，例如：相对密度得分占 50%，voxel RoI 占 50%
            info["score"] = {
                "total_score": 0.5 * relative_density_score + 0.5 * voxel_roi_score
            }

            print(f"Total score for label {label}: {info['score']['total_score']}")

        print("Calculate Total Score done")

    """
    按照得分排序，可以在这个中间加上方差判断
    """
    # @staticmethod
    # def filter_top_clusters(cluster_info, top_n=1):
    #     # 按照总得分进行排序，选择得分最高的前 top_n 个聚类
    #     sorted_clusters = sorted(cluster_info.items(), key=lambda item: item[1]["score"]["total_score"], reverse=True)
    #     top_clusters = dict(sorted_clusters[:top_n])
    #     return top_clusters

    @staticmethod
    def filter_top_clusters(cluster_info, top_n=1):
        # 按照 Voxel_RoI 进行排序，选择 Voxel_RoI 最大的前 top_n 个聚类
        sorted_clusters = sorted(cluster_info.items(), key=lambda item: item[1]["voxel_RoI"], reverse=True)
        top_clusters = dict(sorted_clusters[:top_n])
        return top_clusters
