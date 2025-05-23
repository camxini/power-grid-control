请先安装相关库：networkx, pandas, typing, numpy, collections等

代码中可调参数：
1. 修改失负荷危害度系数：779行
2. 修改过负荷危害度系数：785行
3. 修改蒙特卡洛模拟次数：787行
4. 修改联络开关最大负载能力：788行
5. 修改DG输出功率：789行
6. 修改DG输出功率从I到3I的迭代逻辑：795行
	若为第一问，请修改为risk, overload_risk, lostload_prob, overload_prob, r_sys = simulate_and_draw(p_dg, 0.3*p_dg, p_dg, simulation_times, limit, hazard_num, ol_hazard_num)
	若为第二问，请修改为risk, overload_risk, lostload_prob, overload_prob, r_sys = simulate_and_draw(p_dg, 0.3*p_dg, 3*p_dg, simulation_times, limit, hazard_num, ol_hazard_num)