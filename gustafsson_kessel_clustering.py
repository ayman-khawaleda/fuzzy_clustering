import numpy as np
import matplotlib.pyplot as plt


class GustafssonKesselClustering:
    def __init__(self, Z, no_c=2, m=2, E=0.001, norm_matrix_type="I"):
        """Norm Matrix Type Could Be One Of Three Values I,D,M
        I: Ones Diagonal Matrix
        D: Diagonal Variance Matrix
        M: Mahalanobis Matrix
        """
        self.centers = np.random.rand(no_c, Z.shape[1])
        self.old_centers = self.centers.copy()
        self.m = m
        self.n = Z.shape[0]
        self.data = Z
        self.featur_names = [str(d) for d in self.data]
        self.Z = np.array(Z)
        self.oldU = np.random.rand(no_c, Z.shape[0])
        self.newU = self.oldU.copy()
        self.E = E
        self.GM = True if (2 / (self.m - 1)) > 1 else False
        self.type = norm_matrix_type
        self.phi = np.ones(self.centers.shape[0])

    def update_centers(self, dataset_indx):
        for i in range(self.centers.shape[0]):
            sum = 0
            for k in range(self.Z.shape[0]):
                u_ik_m = self.oldU[i, k] ** self.m
                z_u_ik_m = self.Z[k, dataset_indx] * u_ik_m
                sum += z_u_ik_m
            self.centers[i, dataset_indx] = sum / np.sum(self.oldU[i] ** self.m)

    def F(self, dataset_indx):
        data = self.Z[:, dataset_indx]
        F = []
        u_ik = self.oldU**self.m
        for i in range(self.centers.shape[0]):
            center = self.centers[i, dataset_indx]
            diff_zc = data - center
            res = diff_zc**2
            sum = np.sum(u_ik[i] * res)
            F.append(sum / np.sum(u_ik[i]))

        return np.array(F)

    def A(self, dataset_indx):
        
        if self.type == "I":
            Ai = np.eye(self.Z.shape[0])
        elif self.type == "D":
            mean = np.sum(self.Z[:, dataset_indx]) / self.n
            variance = np.sum((self.Z[:, dataset_indx] - mean) ** 2) / (self.n - 1)
            Ai = np.eye(self.Z.shape[0]) * variance
            Ai = np.linalg.inv(Ai)
        else:
            mean = np.sum(self.Z[:, dataset_indx]) / self.n
            zk = self.Z[:, dataset_indx] - mean
            sum = np.sum(zk**2) / self.n
            I = np.eye(self.n)
            R = I * sum
            Ai = np.linalg.inv(R)

        F = self.F(dataset_indx)
        A = []
        for i in range(self.centers.shape[0]):
            mat = Ai * F[i]
            det = np.linalg.det(mat)
            det_n = det ** (1 / self.n)
            det_n_phi = self.phi[i] * det_n
            det_n_phi_inv = det_n_phi / F[i]
            A.append(det_n_phi_inv)
        return np.array(A)

    def D(self, dataset_indx):
        A = self.A(dataset_indx)
        data = self.Z[:, dataset_indx]
        D = []
        for i in range(self.centers.shape[0]):
            zc = data - self.centers[i, dataset_indx]
            res = zc.T * A[i] * zc
            D.append(res)
        return np.array(D)

    def update_partitions(self, dataset_indx):
        D = self.D(dataset_indx)
        for i in range(self.centers.shape[0]):
            di = D[i]
            sum = 0
            for j in range(self.centers.shape[0]):
                dj = D[j]
                div = di / dj
                pow = 1 / (self.m - 1)
                div_pow = div**pow
                sum += div_pow
            self.newU[i] = 1 / sum

    def check_Uls(self):
        dist = np.sqrt(np.sum((self.newU - self.oldU) ** 2))
        if dist < self.E:
            return False
        else:
            self.oldU = self.newU.copy()
            return True

    def apply(self):
        cond = True
        for i in range(self.Z.shape[1] - 1):
            while cond:
                self.update_centers(i)
                self.update_partitions(i)
                cond = self.check_Uls()
            cond = True

    def plot(self, feature1, feature2):
        u = np.argmax(self.oldU, axis=0)

        fig, ax = plt.subplots()
        plt.title("Gustafsson Kessel Clustering")
        ax.set_xlabel(self.featur_names[feature1])
        ax.set_ylabel(self.featur_names[feature2])
        colors = {0: "r", 1: "g", 2: "b", 3: "k", 4: "y"}
        for i in range(self.centers.shape[0]):
            ax.scatter(
                self.Z[u == i, feature1], self.Z[u == i, feature2], color=colors[i]
            )
            ax.scatter(
                self.centers[i,feature1],
                self.centers[i,feature2],
                color=colors[i],
                marker="*",
                s=300,
            )
        plt.show()

    def show_statistics(self, samples=3):
        print("The Centers:\n", self.centers)
        print(f"The First {samples} Portion Values: \n", self.oldU[:, :samples:])
        print(
            f"Condetion Where Portion Values Are Equal One For Every Center: \n",
            np.sum(self.oldU[:, :samples], axis=0),
        )
        print(
            f"Sum Of Portion Values For every row of Data Should be in range [0,N]: \n",
            np.sum(self.oldU[:, :], axis=1),
            "\n And The Sum for all Sums is [N] => ",
            np.sum(np.sum(self.oldU[:, :], axis=1)),
        )
