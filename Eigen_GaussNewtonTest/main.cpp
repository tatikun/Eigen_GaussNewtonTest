// Include STL
#include <iostream>
#include <fstream>
#include <vector>

// Include Eigen library
#include <Eigen/Dense>

int main()
{
	std::vector<double> x = { 0.038,0.194,0.425,0.626,1.253,2.500,3.740 };
	std::vector<double> y = { 0.050,0.127,0.094,0.2122,0.2729,0.2665,0.3317 };
	const int DATA_COUNT = 7;
	Eigen::MatrixXd J(7, 2); //ヤコビアン
	Eigen::MatrixXd r(7, 1); //残差
	Eigen::MatrixXd B(2, 1); //推定したいパラメータbeta
	B(0, 0) = 0.9; //初期値beta1
	B(1, 0) = 0.2; //初期値beta2
	double residual = 0; //残差
	double rss = 0; //残差平方和

	std::cout << "初期のB:" << std::endl;
	std::cout << B << std::endl;

	for (int j = 0;j < 5;++j) {
		std::cout << j + 1 << "回目" << std::endl;
		rss = 0;
		for (int i = 0;i < DATA_COUNT;++i)
		{
			residual = y[i] - (B(0,0) * x[i]) / (B(1,0) + x[i]);
			rss += std::pow(residual, 2);
			r(i, 0) = residual;
			J(i, 0) = -(x[i] / (B(1,0) + x[i]));
			J(i, 1) = (B(0,0) * x[i]) / (std::pow((B(1,0) + x[i]), 2));
		}

		Eigen::MatrixXd nextB = B - (J.transpose() * J).inverse() * J.transpose() * r;

		std::cout << "推定値B:" << std::endl;
		std::cout << nextB << std::endl;
		std::cout << "残差平方和:" << std::endl;
		std::cout << rss << std::endl;
		std::cout << "====================" << std::endl;
		B = nextB;
	}
	std::cout << "最終推定値B:" << std::endl;
	std::cout << B << std::endl;

}