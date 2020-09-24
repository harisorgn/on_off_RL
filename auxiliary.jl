
get_optimal_agent(env::OU_bandit_distribution_outlier_environment) = optimal_bandit_distribution_outlier_agent(env.n_steps, 
																												env.n_sessions)

get_optimal_agent(env::OU_bandit_frequency_outlier_environment) = optimal_bandit_frequency_outlier_agent(env.n_steps, 
																										env.n_sessions, 
																										env.r_outlier, 
																										env.p_outlier_max, 
																										env.η_p_outlier, 
																										env.decay_p_outlier)
function initialise_OU_bandit_distribution_outlier_environment()

	n_steps = 40
	n_sessions = 100
	n_actions = 2

	out1_d = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.8, 		0.15, 		0.05])

	out2_d = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.9, 		0.05, 		0.05])

	μ_OU_bandit_v = [0.0, 0.0]
	σ_OU_bandit_v = [0.5, 0.5]
	γ_OU_bandit_v = [0.01, 0.01]

	env = OU_bandit_distribution_outlier_environment(n_steps, n_sessions, γ_OU_bandit_v, 
													μ_OU_bandit_v, σ_OU_bandit_v, [out1_d, out2_d])
	return env
end

function initialise_OU_bandit_frequency_outlier_environment()

	n_steps = 40
	n_sessions = 100
	n_actions = 2

	μ_OU_bandit_v = [0.0, 0.0]
	σ_OU_bandit_v = [0.5, 0.5]
	γ_OU_bandit_v = [0.01, 0.01]

	r_outlier = 10.0 
	p_outlier_max = 0.2 
	η_p_outlier = 0.05 
	decay_p_outlier = 0.1

	env = OU_bandit_frequency_outlier_environment(n_steps, n_sessions, γ_OU_bandit_v, μ_OU_bandit_v, σ_OU_bandit_v,
												r_outlier, p_outlier_max, η_p_outlier, decay_p_outlier)
	return env
end

function run_example(env, agent_v; plot_flag = false)

	optimal_agent = get_optimal_agent(env)

	random_agent = delta_agent(env.n_steps, env.n_actions, env.n_sessions, 0.0, 0.0,
							offline_bias(10, env.n_actions, env.n_sessions, 10, 0.0, 0.0), 
							ε_greedy_policy(1.0))

	run_environment!(env, optimal_agent)
	run_environment!(env, random_agent)

	c = 1
	for agent in agent_v

		run_environment!(env, agent)

		println("agent $c score : ", (sum(agent.accumulated_r_v) - sum(random_agent.accumulated_r_v)) / 
								 	(sum(optimal_agent.accumulated_r_v) - sum(random_agent.accumulated_r_v)))

		if plot_flag
			plot_timeseries(agent.r_m, agent.bias.b_m, 1:env.n_sessions)
		end

		c += 1
	end
end
