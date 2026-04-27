import numpy as np

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.clusters = [[] for _ in range(self.k)]

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_sample_indices = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_indices]

        for _ in range(self.max_iters):

            self.clusters = self._create_clusters(self.centroids)
            
        
            centroids_old = self.centroids

            
            self.centroids = self._get_centroids(self.clusters)

            
            if self._is_converged(centroids_old, self.centroids):
                print(f"Thuat toan da hoi tu tai lan lap thu: {_+1}")
                break
            
    def calculate_wcss(self):
        wcss = 0
        for cluster_idx, cluster in enumerate(self.clusters):
            cluster_points = self.X[cluster]
            centroid = self.centroids[cluster_idx]
            wcss += np.sum((cluster_points - centroid) ** 2)
        return wcss
    def calculate_silhouette_score(self, sample_size=500):
        
        indices = np.random.choice(self.n_samples, min(self.n_samples, sample_size), replace=False)
        X_sub = self.X[indices]
        labels = self.predict(X_sub)
        
        s_vals = []
        for i, sample in enumerate(X_sub):
            label_i = labels[i]
            same_cluster = X_sub[labels == label_i]
            a_i = np.mean(np.linalg.norm(same_cluster - sample, axis=1)) if len(same_cluster) > 1 else 0
            
            b_i = float('inf')
            for k in range(self.k):
                if k == label_i: continue
                other_cluster = X_sub[labels == k]
                if len(other_cluster) > 0:
                    b_i = min(b_i, np.mean(np.linalg.norm(other_cluster - sample, axis=1)))
            
            s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
            s_vals.append(s_i)
        return np.mean(s_vals)

    def calculate_dbi(self):
        k = self.k
        labels = self.predict(self.X)
        sigmas = np.zeros(k)
        for i in range(k):
            cluster_points = self.X[labels == i]
            if len(cluster_points) > 0:
                sigmas[i] = np.mean(np.linalg.norm(cluster_points - self.centroids[i], axis=1))

        dbi_sum = 0
        for i in range(k):
            max_ratio = 0
            for j in range(k):
                if i != j:
                    dist_cen = np.linalg.norm(self.centroids[i] - self.centroids[j])
                    ratio = (sigmas[i] + sigmas[j]) / dist_cen if dist_cen > 0 else 0
                    max_ratio = max(max_ratio, ratio)
            dbi_sum += max_ratio
        return dbi_sum / k

    def calculate_calinski_harabasz(self):
        overall_mean = np.mean(self.X, axis=0)
        wcss = self.calculate_wcss()
        bcss = 0
        labels = self.predict(self.X)
        for i in range(self.k):
            n_i = np.sum(labels == i)
            if n_i > 0:
                bcss += n_i * (np.linalg.norm(self.centroids[i] - overall_mean) ** 2)
        
        if wcss == 0 or self.k == 1: return 0
        return (bcss / (self.k - 1)) / (wcss / (self.n_samples - self.k))
    
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
    
        distances = [np.sqrt(np.sum((sample - point)**2)) for point in centroids]
        return np.argmin(distances)

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids_new):
        distances = [np.sqrt(np.sum((centroids_old[i] - centroids_new[i])**2)) for i in range(self.k)]
        return sum(distances) == 0

    def predict(self, X):
    
        labels = []
        for sample in X:
            labels.append(self._closest_centroid(sample, self.centroids))
        return np.array(labels)
    