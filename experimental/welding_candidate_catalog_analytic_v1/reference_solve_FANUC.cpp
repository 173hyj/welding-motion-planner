// 从 PH/robot_inverseKinematicsSolver.cpp:214 原样提取。
// 这是原类的函数片段，依赖原类成员和 T01/T12/T23/T34、inverse_transform、forward 等辅助函数。
// 函数内已包含：J3 联动角还原、弧度转度、J4/J6 的 ±360° 候选扩展、正解位置筛选。

std::vector<std::vector<double>> InverseKinematicsSolver::solve_FANUC(const InverseKinematicsSolver::Matrix4D& M06) {







	double d1 = m_dhParams[0].d;
	double d2 = m_dhParams[1].d;;
	double d3 = m_dhParams[2].d;;
	double d4 = m_dhParams[3].d;;
	double d5 = m_dhParams[4].d;;
	double d6 = m_dhParams[5].d;;

	double a1 = m_dhParams[0].a;
	double a2 = m_dhParams[1].a;
	double a3 = m_dhParams[2].a;
	double a4 = m_dhParams[3].a;
	double a5 = m_dhParams[4].a;
	double a6 = m_dhParams[5].a;

	double alpha1 = 0;
	double alpha2 = -M_PI / 2;
	double alpha3 = M_PI;
	double alpha4 = M_PI / 2;
	double alpha5 = -M_PI / 2;
	double alpha6 = -M_PI / 2;

	std::array<double, 6> theta = { 0, 0, 0, 0, 0, 0 };

	std::vector<std::vector<double>> MDH = { {
	{theta[0],       d1, a1, alpha1},
	{theta[1] + M_PI / 2, d2, a2, alpha2},
	{theta[2],       d3, a3, alpha3},
	{theta[3],       d4, a4, alpha4},
	{theta[4],       d5, a5, alpha5},
	{theta[5] + M_PI, d6, a6, alpha6}
	} };

	/*std::cout << "MDH = \t" << std::endl;
	printMatrix(MDH);*/

	double px = M06(0, 3) - (MDH[5][1]) * M06(0, 2);
	double py = M06(1, 3) - (MDH[5][1]) * M06(1, 2);
	theta[0] = atan2(py, px);

	/*std::cout << "theta[0] = \t" << theta[0] << std::endl;*/

	InverseKinematicsSolver::Matrix4D M01 = T01(theta[0]);//基座到关机1的变换矩阵
	InverseKinematicsSolver::Matrix4D matrix1 = inverse_transform(M01) * M06;

	/*std::cout << "matrix1 = \t" << std::endl;
	matrix1.print();*/
	// 计算第五关节中心在关节1坐标系中的位置
	double temp1 = -(MDH[5][1]) * matrix1(0, 2) + matrix1(0, 3);
	double temp2 = -(MDH[5][1]) * matrix1(2, 2) + matrix1(2, 3);
	// 构建三角方程系数
	double A = -2 * (temp1 - MDH[1][2]) * MDH[2][2];
	double B = 2 * (-MDH[2][2]) * temp2;
	double C = pow(temp1 - MDH[1][2], 2) + pow(-MDH[2][2], 2) + pow(temp2, 2) - pow(-MDH[3][2], 2) - pow(-MDH[3][1], 2);
	// 求解θ?的两个可能值
	//double theta2_1 = atan2(C, sqrt(A * A + B * B - C * C)) - atan2(B, A);
	double temp = A * A + B * B - C * C;
	double sqrt_val = (temp < 0.0) ? 0.0 : sqrt(temp);  // 取实部
	double theta2_1 = atan2(C, sqrt_val) - atan2(B, A);

	//double theta2_2 = atan2(C, -sqrt(A * A + B * B - C * C)) - atan2(B, A);
	double theta2_2 = atan2(C, -sqrt_val) - atan2(B, A);
	// 对每个θ?求解θ?
	A = -MDH[3][1];
	B = -MDH[3][2];
	C = -MDH[2][2] * cos(theta2_1) - temp2;
	temp = A * A + B * B - C * C;
	sqrt_val = (temp < 0.0) ? 0.0 : sqrt(temp);
	double theta3_1_1 = theta2_1 - atan2(C, sqrt_val) - atan2(B, A);
	double theta3_1_2 = theta2_1 - atan2(C, -sqrt_val) - atan2(B, A);

	C = -MDH[2][2] * cos(theta2_2) - temp2;
	temp = A * A + B * B - C * C;
	sqrt_val = (temp < 0.0) ? 0.0 : sqrt(temp);
	double theta3_2_1 = theta2_2 - atan2(C, sqrt_val) - atan2(B, A);
	double theta3_2_2 = theta2_2 - atan2(C, -sqrt_val) - atan2(B, A);

	std::vector<std::vector<double>> theta_list;

	theta_list = {
		{theta[0], theta2_1, theta3_1_1, 0, 0, 0 },
		{theta[0], theta2_1, theta3_1_2, 0, 0, 0 },
		{theta[0], theta2_2, theta3_2_1, 0, 0, 0 },
		{theta[0], theta2_2, theta3_2_2, 0, 0, 0 }
	};

	/*std::cout << "theta_list = \t" << std::endl;
	printMatrix(theta_list);*/

	std::vector<std::vector<double>> new_theta_list;  // 额外的存储空间

	for (size_t i = 0; i < theta_list.size(); ++i) {
		auto& temp_theta = theta_list[i];
		InverseKinematicsSolver::Matrix4D M12 = T12(temp_theta[1], MDH);
		InverseKinematicsSolver::Matrix4D M23 = T23(temp_theta[2], MDH);
		InverseKinematicsSolver::Matrix4D M03 = M01 * M12 * M23;
		InverseKinematicsSolver::Matrix4D matrix2 = inverse_transform(M03) * M06;

		/*std::cout << "matrix2 = \t" << std::endl;
		matrix2.print();*/

		if (std::abs(matrix2(2, 2)) <= 1e-16 && std::abs(matrix2(0, 2)) <= 1e-16) {
			temp_theta[3] = 2 * M_PI * (rand() / (double)RAND_MAX - 0.5);
		}
		// 非奇异位置计算θ?
		else {
			double theta4_1 = atan2(matrix2(2, 2), matrix2(0, 2));
			double theta4_2 = theta4_1 + M_PI;
			temp_theta[3] = theta4_1;
			new_theta_list.push_back({ temp_theta[0], temp_theta[1], temp_theta[2], theta4_2, 0, 0 });
		}
	}
	theta_list.insert(theta_list.end(), new_theta_list.begin(), new_theta_list.end());//合并 `new_theta_list` 到 `theta_list`

	/*std::cout << "theta_list = \t" << std::endl;
	printMatrix(theta_list);*/

	for (size_t k = 0; k < theta_list.size(); ++k) {
		auto& temp_theta = theta_list[k];

		// 计算 M04 矩阵
		InverseKinematicsSolver::Matrix4D M04 = M01 * T12(temp_theta[1], MDH) * T23(temp_theta[2], MDH) * T34(temp_theta[3], MDH);
		InverseKinematicsSolver::Matrix4D matrix3 = inverse_transform(M04) * M06;

		/*std::cout << "matrix3 = \t" << std::endl;
		matrix3.print();*/

		// 计算关节角 theta(5) 和 theta(6)
		temp_theta[4] = atan2(-matrix3(0, 2), -matrix3(2, 2));
		temp_theta[5] = -atan2(matrix3(1, 0), matrix3(1, 1));
		temp_theta[2] = temp_theta[2] - temp_theta[1];
	}

	// 角度转换为度
	for (auto& temp_theta : theta_list) {
		for (double& angle : temp_theta) {
			angle = angle * 180.0 / M_PI;
		}
	}

	std::vector<std::vector<double>> theta_n;
	std::vector<std::vector<double>> theta_n1;
	std::vector<std::vector<double>> theta_n2;
	std::vector<std::vector<double>> theta_n3;

	// 处理 theta(4) > 180
	for (const auto& temp_theta : theta_list) {
		if (temp_theta[3] > 180) {
			theta_n.push_back(temp_theta);  // 先添加 temp_theta
			theta_n.back()[3] -= 360;       // 访问最后一行并修改第4列
		}
	}

	/*std::cout << "theta_n = \t" << std::endl;
	printMatrix(theta_n);*/

	// 处理 theta(4) < -180
	for (const auto& temp_theta : theta_list) {
		if (temp_theta[3] < -180) {
			theta_n1.push_back(temp_theta);  // 先添加 temp_theta
			theta_n1.back()[3] += 360;       // 访问最后一行并修改第4列
		}
	}

	/*std::cout << "theta_n1 = \t" << std::endl;
	printMatrix(theta_n1);*/

	// 处理 theta(6) < -180
	for (const auto& temp_theta : theta_list) {
		if (temp_theta[5] < -180) {
			theta_n2.push_back(temp_theta);
			theta_n2.back()[5] += 360;
		}
	}

	/*std::cout << "theta_n2 = \t" << std::endl;
	printMatrix(theta_n2);*/

	// 处理 theta(6) > 180
	for (const auto& temp_theta : theta_list) {
		if (temp_theta[5] > 180) {
			theta_n3.push_back(temp_theta);
			theta_n3.back()[5] -= 360;
		}
	}

	/*std::cout << "theta_n3 = \t" << std::endl;
	printMatrix(theta_n3);*/

	// 合并所有解集
	theta_list.insert(theta_list.end(), theta_n.begin(), theta_n.end());
	theta_list.insert(theta_list.end(), theta_n1.begin(), theta_n1.end());
	theta_list.insert(theta_list.end(), theta_n2.begin(), theta_n2.end());
	theta_list.insert(theta_list.end(), theta_n3.begin(), theta_n3.end());

	/*std::cout << "theta_list = \t" << std::endl;
	printMatrix(theta_list);*/

	std::vector<std::vector<double>> theta_filtered;
	std::vector<size_t> rr;

	//std::cout << "theta_list size: " << theta_list.size() << std::endl;

	for (size_t ite = 0; ite < theta_list.size(); ++ite) {
		const auto& temp_theta = theta_list[ite];

		InverseKinematicsSolver::Matrix4D T1 = forward(temp_theta, "deg", 1);
		InverseKinematicsSolver::Matrix4D T2 = forward(temp_theta, "deg", 2);
		InverseKinematicsSolver::Matrix4D T3 = forward(temp_theta, "deg", 3);
		InverseKinematicsSolver::Matrix4D T4 = forward(temp_theta, "deg", 4);
		InverseKinematicsSolver::Matrix4D T5 = forward(temp_theta, "deg", 5);
		InverseKinematicsSolver::Matrix4D T6 = forward(temp_theta, "deg", 6);

		InverseKinematicsSolver::Matrix4D T = T1 * T2 * T3 * T4 * T5 * T6;  // 计算整体变换矩阵

		// 计算 T 和 M06 的末三列的欧几里得距离
		Vector3 t_position = { T(0, 3), T(1, 3), T(2, 3) };
		Vector3 M06_position = { M06(0, 3), M06(1, 3), M06(2, 3) };
		double distance = (t_position - M06_position).norm();

		if (distance < 10) {
			rr.push_back(ite); // 记录符合条件的索引
		}
	}

	// 只保留符合条件的 theta 行
	for (size_t index : rr) {
		theta_filtered.push_back(theta_list[index]);
	}

	// 更新 theta_list
	theta_list = theta_filtered;

	return theta_list;
}
