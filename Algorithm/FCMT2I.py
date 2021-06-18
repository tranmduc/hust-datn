import numpy as np
from scipy.spatial.distance import cdist

class FCMT2I:
    def __init__(self,x,vs,r,r1,r2,dis='euclidean'):
        self.x = x
        self.vs = vs
        self.r = r
        self.r1 = r1
        self.r2 = r2
        self.dis = dis

    def DistanceMatrix(self, center, x, distance):
        distanceMatrix = cdist(x, center, distance).T
        return distanceMatrix

    def MembershipMatrix1(self, distanceMatrix, r):
        c, n = distanceMatrix.shape
        membershipMatrix = np.zeros((c, n))
        p = 2 / (r - 1)
        [row2, col2] = np.where((np.isnan(distanceMatrix)))
        zs2 = np.size(row2)
        for j in range(zs2):
            distanceMatrix[row2(j), col2(j)] = 1
        [row, col] = np.where(distanceMatrix == 0)
        dp = 1 / (distanceMatrix ** p)
        dsum = np.sum(dp, axis=0)
        for j in range(n):
            membershipMatrix[:, j] = dp[:, j] / dsum[j]
        zs = np.size(row)
        for j in range(zs):
            membershipMatrix[:, col(j)] = np.zeros(c, 1)
            membershipMatrix[row(j), col(j)] = 1
        return membershipMatrix

    def MembershipMatrix(self, center, x, distance, r):
        distanceMatrix = self.DistanceMatrix(center, x, distance)
        membershipMatrix = self.MembershipMatrix1(distanceMatrix, r)
        return membershipMatrix

    def GetCenter(self, x, u, r):
        try:
            n = x.shape[0]
        except IndexError:
            n = 1
        try:
            m = x.shape[1]
        except IndexError:
            m = 1

        ur = u ** r
        center = np.dot(ur, x) / (np.dot(ur, np.ones((n, m))))
        return center

    def KMCL(self, x, ulower, uupper, r):
        e = 1
        l = 1
        L = 10 ** 5

        n, m = x.shape
        u = (ulower + uupper) / 2
        c, tt = u.shape
        v1 = self.GetCenter(x, u, r)
        v2 = np.copy(v1)
        while (l <= L):
            for i in range(m):
                for j in range(c):
                    for k in range(n):
                        if x[k, i] <= v1[j, i]:
                            u[j, k] = uupper[j, k]
                        else:
                            u[j, k] = ulower[j, k]
                v2[:, [i]] = self.GetCenter(x[:, [i]], u, r)
            d = np.sum(np.sum(np.abs(v1 - v2), axis=0), axis=0)

            if d < e:
                v = np.copy(v2)
                break
            else:
                v1 = np.copy(v2)

        return v

    def KMCR(self, x, ulower, uupper, r):
        e = 1
        l = 1
        L = 10 ** 5

        n, m = x.shape
        u = (ulower + uupper) / 2
        c, tt = u.shape
        v1 = self.GetCenter(x, u, r)
        v2 = np.copy(v1)
        while (l <= L):
            for i in range(m):
                for j in range(c):
                    for k in range(n):
                        if x[k, i] >= v1[j, i]:
                            u[j, k] = uupper[j, k]
                        else:
                            u[j, k] = ulower[j, k]
                v2[:, [i]] = self.GetCenter(x[:, [i]], u, r)
            d = np.sum(np.sum(np.abs(v1 - v2)))

            if d < e:
                v = np.copy(v2)
                break
            else:
                v1 = np.copy(v2)

        return v

    def predict(self, center,x):
        n, kk = x.shape
        result = np.zeros((n, 1))
        d = self.DistanceMatrix(center, x, 'euclidean')
        mind = d.min(axis=0)
        for i in range(0, n):
            result[i] = (d[:, [i]] == mind[i]).argmax()
        return result

    def fit(self):
        v1 = self.vs
        e = 10 ** (-4)
        l = 0
        L = 10 ** 5

        while (l <= L):
            u1 = self.MembershipMatrix(v1, self.x, self.dis, self.r1)
            u2 = self.MembershipMatrix(v1, self.x, self.dis, self.r2)
            ulower = np.minimum(u1, u2)
            uupper = np.maximum(u1, u2)

            vl = self.KMCL(self.x, ulower, uupper, self.r)

            vr = self.KMCR(self.x, ulower, uupper, self.r)
            v2 = (vl + vr) / 2
            d = np.sum(np.sum(np.abs(v1 - v2)))

            if d > e:

                v1 = np.copy(v2)
                l = l + 1
            else:
                break
        ve = np.copy(v2)

        ve = v2[v2[:, 0].argsort(),]

        result = self.predict(ve[:], self.x)
        return ve, result.T[0]


    def merge(self, a, b):
        x = np.squeeze(np.stack((a, b), axis=1))
        x = x[x[:, 1].argsort()]
        return x
