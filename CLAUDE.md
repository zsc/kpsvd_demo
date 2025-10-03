写一个 python script，对于给定的图，转灰度，然后跑 kpsvd，然后做k-阶近似。在得到的近似左、右因子上，可以分别加噪声得到两个系列的图。python 将这些图像写入一个 html 进行可视化
kpsvd 指 Kronecker Product SVD / Nearest Kronecker Sum（Van Loan–Pitsianis 方法）：把 M 重新排列为一个“重排矩阵” R(M)，对 R(M) 做（截断）SVD
