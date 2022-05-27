import numpy as np
import matplotlib.pyplot as plt


class C_MeansClustering:
    def __init__(self, Z, no_c=2, m=2, E=0.001):
        self.centers = np.random.rand(no_c, Z.shape[1])
        self.old_centers = self.centers.copy()
        self.m = m
        self.data = Z
        self.featur_names = [str(d) for d in self.data]
        self.Z = np.array(Z)
        self.oldU = np.random.rand(no_c, Z.shape[0])
        self.newU = self.oldU.copy()
        self.E = E

    def update_centers(self, dataset_indx):
        for i in range(self.centers.shape[0]):
            sum = 0
            for k in range(self.Z.shape[0]):
                u_ik_m = self.oldU[i, k] ** self.m
                z_u_ik_m = self.Z[k, dataset_indx] * u_ik_m
                sum += z_u_ik_m
            self.centers[i, dataset_indx] = sum / np.sum(self.oldU[i] ** self.m)

    def update_partitions(self, dataset_indx):
        for i in range(self.centers.shape[0]):
            current_center = self.centers[i, dataset_indx]
            z_cc = self.Z[:, dataset_indx] - current_center  # The Current Center
            sum = 0
            for k in range(self.centers.shape[0]):
                z_oc = (
                    self.Z[:, dataset_indx] - self.centers[k, dataset_indx]
                )  # The Other Centers
                div_z_centers = (z_cc / z_oc) ** 2
                pow = 1 / (self.m - 1)
                z_pow = div_z_centers.astype(np.float64) ** pow
                sum += z_pow
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
        for i in range(self.Z.shape[1]):
            while cond:
                self.update_centers(i)
                self.update_partitions(i)
                cond = self.check_Uls()
            cond = True

    def plot(self, feature1, feature2):
        u = np.argmax(self.oldU, axis=0)

        fig, ax = plt.subplots()
        ax.set_xlabel(self.featur_names[feature1])
        ax.set_ylabel(self.featur_names[feature2])
        plt.title("C-Means Clustering")
        colors = {0: "r", 1: "g", 2: "b", 3: "k", 4: "y"}
        for i in range(self.centers.shape[0]):
            ax.scatter(
                self.Z[u == i, feature1], self.Z[u == i, feature2], color=colors[i]
            )
            ax.scatter(
                self.centers[i, feature1],
                self.centers[i, feature2],
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
