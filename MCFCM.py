import numpy as np
from HedgeAlgebra import HedgeAlgebra
from scipy.spatial.distance import cdist

class MCFCM:
    def __init__(self, m_l, m_u, data_frame, N, n_clusters, fmSmall, uVery, uMore, length, eps=1e-4, max_count=1e+5):
        self.m = np.zeros(N)  # tham so fuzzy
        self.max_count = max_count  # so vong lap toi da
        self.eps = eps  # epsilon, dieu kien dung
        self.data_frame = data_frame  # data sets
        self.N = N  # kich thuoc data sets
        self.K = n_clusters  # so nhan cum
        self.m_l = m_l
        self.m_u = m_u
        fmSmall, uVery, uMore, length
        self.fmSmall = fmSmall
        self.uVery = uVery
        self.uMore = uMore
        self.length = length
        # self.alpha = alpha
        # self.HAList = [0.072, 0.192, 0.336, 0.496, 0.712, 0.892]
        self.HAList = HedgeAlgebra(self.fmSmall, self.uVery, self.uMore, self.length)

    def DistanceMatrix(self, center, x):
        distanceMatrix = cdist(x, center, 'euclidean').T
        return distanceMatrix

    def predict(self, center,x):
        n, kk = x.shape
        result = np.zeros((n, 1))
        d = self.DistanceMatrix(center, x)
        mind = d.min(axis=0)
        for i in range(n):
            result[i] = (d[:, [i]] == mind[i]).argmax()
        return result

    def dist(self, A, B):
        return np.linalg.norm(A - B, keepdims=False)

    def initializeMembershipMatrix(self):
        """

        Return Matrix K x N
        -------
        TYPE: numpy array
            khoi tao ngay nhien ma tran K x N co cac phan tu thuoc (0, 1)
            tong cac phan tu tren moi cot deu bang 1

        """
        matrix = np.random.rand(self.K, self.N)
        tmp = matrix.sum(axis=0)
        return matrix / tmp

    def updateCenters(self, membership_mat):
        """

        Parameters
        ----------
        membership_mat : numpy array
            ma tran bieu thi tu cach thanh vien cua cac phan tu trong moi cum khac nhau

        Returns
        -------
        center_clusters : numpy array
            mang cac center (K x n) tuong ung voi K cum theo thu tu.

        """
        center_clusters = np.zeros((self.K, self.data_frame.shape[1]))
        for i in range(self.K):
            center = np.zeros(self.data_frame.shape[1])
            s = 0
            for j in range(self.N):
                t = (membership_mat[i][j] ** self.m[j])
                s += t
                tmp1 = t * self.data_frame[j, :]
                center += tmp1
            center_clusters[i, :] = center / s
            # print(center)
        return center_clusters

    def updateMembershipMatrix(self, center_clusters):
        """
        Parameters
        ----------
        center_clusters : numpy array
            mang cac center (K x n) tuong ung voi K cum theo thu tu.

        Returns
        -------
        matrix : numpy array
            ma tran bieu thi tu cach thanh vien cua cac phan tu trong moi cum khac nhau.

        """
        matrix = np.random.rand(self.K, self.N)
        # p = 2 / (self.m - 1)
        matrix_distance = np.zeros((self.N, self.K))
        inverse_matrix_distance = np.zeros((self.N, self.K))
        for i in range(self.N):
            for j in range(self.K):
                matrix_distance[i, j] = self.dist(self.data_frame[i, :], center_clusters[j, :])

        inverse_matrix_distance = 1.00 / matrix_distance

        for j in range(self.K):
            # Cj = center_clusters[j, :]
            for i in range(self.N):
                # Xi = self.data_frame[i, :]
                # dist_Xi_Cj = self.dist(Xi, Cj)
                s = 0
                p = 2 / (self.m[i] - 1)
                for k in range(self.K):
                    # Ck = center_clusters[k, :]
                    t = matrix_distance[i, j] * inverse_matrix_distance[i, k]
                    s += (t ** p)
                matrix[j, i] = float(1) / s
        return matrix

    def findLabels(self, membership_mat):
        """
        Parameters
        ----------
        membership_mat : numpy array
            mma tran bieu thi tu cach thanh vien cua cac phan tu trong moi cum khac nhau.

        Returns
        -------
        labels : numpy array
            mang gom N phan tu bieu thi labels cua phan tu tuong ung trong datasets.
        """
        matrix = (membership_mat == membership_mat.max(axis=0))
        labels = np.arange(self.N)
        for i in range(self.N):
            for j in range(self.K):
                if matrix[j][i] == True:
                    labels[i] = j
        return labels

    def normalizeData(self):
        a = np.max(self.data_frame, axis=0)
        b = np.min(self.data_frame, axis=0)
        for i in range(self.data_frame.shape[0]):
            for j in range(self.data_frame.shape[1]):
                self.data_frame[i][j] = (self.data_frame[i][j] - b[j]) / (a[j] - b[j])

    def computeM(self):
        delta_s = np.zeros(self.N)
        nb = int(self.N/self.K)
        delta_sum = np.zeros(self.N)
        for i in range(self.N):
            for j in range(self.N):
                distance = self.dist(self.data_frame[i, :], self.data_frame[j, :])
                delta_s[j] = distance

            delta_s.sort()
            for l in range(nb):
                delta_sum[i] = delta_sum[i] + delta_s[l]
            # self.m[i] = self.m_l + (self.m_u - self.m_l) * pow(((delta_sum - delta_min)/(delta_max - delta_min)), self.alpha)
            # print(self.m[i])

        for i in range(self.N):
            delta_max = max(delta_sum)
            delta_min = min(delta_sum)
            tmp = (delta_sum[i] - delta_min)/(delta_max - delta_min)
            # tmp = pow(((delta_sum[i] - delta_min) / (delta_max - delta_min)), self.alpha)
            # print(tmp)
            min_subtract = tmp - self.HAList[0]
            min_ind = 0
            for l in range(len(self.HAList)):
                subtract = abs(tmp - self.HAList[l])
                if (min_subtract > subtract):
                    min_subtract = subtract
                    min_ind = l

            xha = self.HAList[min_ind]
            # print(xha)
            self.m[i] = self.m_l + (self.m_u - self.m_l) * xha
            # self.m[i] = self.m_l + (self.m_u - self.m_l) * tmp

    def fit(self):
        # self.normalizeData()
        matrix = self.initializeMembershipMatrix()
        # tmp = matrix
        cluster_centers = np.zeros((self.K, self.data_frame.shape[1]))
        tmp = cluster_centers
        count = 0
        self.computeM()
        while True:
            cluster_centers = self.updateCenters(matrix)
            matrix = self.updateMembershipMatrix(cluster_centers)
            # terminate = (np.linalg.norm(matrix - tmp))
            terminate = (np.linalg.norm(cluster_centers - tmp))
            # tmp = matrix
            tmp = cluster_centers
            if ((terminate < self.eps) or (count >= self.max_count)):
                break
            # print("terminate = ", terminate)
            count += 1

        labels = self.findLabels(matrix)

        return cluster_centers, labels









