import torch

class NeuralCollapseMetrics:
    def __init__(self, num_classes, feature_dim, device='cpu'):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.reset()

    def reset(self):
        """重置统计量"""
        self.class_counts = torch.zeros(self.num_classes, device=self.device)
        self.class_means = torch.zeros(self.num_classes, self.feature_dim, device=self.device)
        self.class_squares = torch.zeros(self.num_classes, self.feature_dim, device=self.device)
        self.global_count = 0
        self.global_mean = torch.zeros(self.feature_dim, device=self.device)
        self.global_square = torch.zeros(self.feature_dim, device=self.device)

    def update(self, features, labels):
        """逐批次更新类均值和全局均值"""
        batch_size = features.size(0)
        self.global_count += batch_size

        for i in range(self.num_classes):
            class_mask = (labels == i)
            class_features = features[class_mask]
            class_count = class_features.size(0)

            if class_count > 0:
                self.class_counts[i] += class_count

                # 更新类均值和平方和
                self.class_means[i] += class_features.sum(dim=0)
                self.class_squares[i] += (class_features ** 2).sum(dim=0)

        # 更新全局均值和平方和
        self.global_mean += features.sum(dim=0)
        self.global_square += (features ** 2).sum(dim=0)

    def compute_nc1(self):
        """计算 NC1：类内变异性"""
        variances = []
        for i in range(self.num_classes):
            if self.class_counts[i] > 0:
                mean = self.class_means[i] / self.class_counts[i]
                square_mean = self.class_squares[i] / self.class_counts[i]
                variance = torch.mean(square_mean - mean ** 2)
                variances.append(variance)

        return torch.mean(torch.tensor(variances)).item()

    def compute_nc2(self):
        """计算 NC2：中心化后的类均值的余弦相似度和范数相似度"""
        global_mean = self.global_mean / self.global_count
        class_means = self.class_means / self.class_counts.unsqueeze(1)
        centered_class_means = class_means - global_mean  # 对类均值进行中心化处理

        # 计算余弦相似度的均匀性
        cosine_similarities = []
        for i in range(len(centered_class_means)):
            for j in range(i + 1, len(centered_class_means)):
                cosine_similarity = torch.nn.functional.cosine_similarity(
                    centered_class_means[i].unsqueeze(0), centered_class_means[j].unsqueeze(0)
                )
                cosine_similarities.append(cosine_similarity.item())

        cosine_similarity_variance = torch.var(torch.tensor(cosine_similarities)).item()

        # 计算范数的均匀性
        norms = torch.norm(centered_class_means, dim=1)
        norm_variance = torch.var(norms).item()

        return cosine_similarity_variance, norm_variance

    def compute_nc3(self, classifier_weights):
        """计算 NC3：分类器权重与类均值对齐程度"""
        global_mean = self.global_mean / self.global_count
        class_means = self.class_means / self.class_counts.unsqueeze(1)
        class_means = class_means[self.class_counts > 0]
        centered_class_means = class_means - global_mean  # 对类均值进行中心化处理

        # 对齐程度
        alignments = []
        for i, class_mean in enumerate(centered_class_means):
            cosine_similarity = torch.nn.functional.cosine_similarity(class_mean.unsqueeze(0),
                                                                      classifier_weights[i].unsqueeze(0))
            alignments.append(cosine_similarity.item())

        return 1 - torch.mean(torch.tensor(alignments)).item()

    def compute_nc4(self, features, labels):
        """计算 NC4：样本到各类中心的距离"""
        class_means = self.class_means / self.class_counts.unsqueeze(1)
        class_means = class_means[self.class_counts > 0]

        correct_count = 0

        for feature, label in zip(features, labels):
            distances = torch.norm(feature - class_means, dim=1)
            predicted_label = torch.argmin(distances)

            if predicted_label == label:
                correct_count += 1

        nc4_value = correct_count
        return nc4_value

    def compute_all(self, features, labels, classifier_weights=None):
        """计算所有 NC 指标"""
        nc1 = self.compute_nc1()
        nc2_cosine_sim, nc2_norm_var = self.compute_nc2()
        nc3 = self.compute_nc3(classifier_weights) if classifier_weights is not None else None
        nc4 = self.compute_nc4(features, labels)

        return nc1, (nc2_cosine_sim, nc2_norm_var), nc3, nc4
