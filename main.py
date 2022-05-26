from c_means_clustering import C_MeansClustering as CMC
from gustafsson_kessel_clustering import GustafssonKesselClustering as GKC
from sklearn.datasets import load_iris


if __name__ == "__main__":
    irisDS = load_iris(as_frame=True)
    Z = irisDS.data
    no_c = 2

    c_means = CMC(Z, no_c=no_c, m=2, E=0.00001)
    c_means.apply()
    c_means.plot(0, 2)

    gk_clustering = GKC(Z, no_c=no_c, m=2, E=0.00001, norm_matrix_type="H")
    gk_clustering.apply()
    gk_clustering.plot(0, 2)
