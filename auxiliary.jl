
get_optimal_agent(env::Union{OU_bandit_distribution_outlier_environment, 
							OU_bandit_delay_outlier_environment, 
							OU_bandit_test_environment}) = optimal_bandit_distribution_outlier_agent(env.n_steps, 
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
	n_bandits = 2

	outlier1_distr = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.8, 		0.15, 		0.05])

	outlier2_distr = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.9, 		0.05, 		0.05])

	r_0_v = zeros(n_bandits)
	μ_OU_bandit_v = [0.0, 0.0]
	σ_OU_bandit_v = [0.5, 0.5]
	γ_OU_bandit_v = [0.01, 0.01]

	return OU_bandit_distribution_outlier_environment(n_steps, n_sessions, r_0_v, γ_OU_bandit_v, 
													μ_OU_bandit_v, σ_OU_bandit_v, [outlier1_distr, outlier2_distr])
end

function initialise_OU_bandit_frequency_outlier_environment()

	n_steps = 40
	n_sessions = 100
	n_bandits = 2

	r_0_v = zeros(n_bandits)
	μ_OU_bandit_v = [0.0, 0.0]
	σ_OU_bandit_v = [0.5, 0.5]
	γ_OU_bandit_v = [0.01, 0.01]

	r_outlier = 10.0 
	p_outlier_max = 0.2 
	η_p_outlier = 0.05 
	decay_p_outlier = 0.1

	return OU_bandit_frequency_outlier_environment(n_steps, n_sessions, r_0_v, γ_OU_bandit_v, μ_OU_bandit_v, σ_OU_bandit_v,
													r_outlier, p_outlier_max, η_p_outlier, decay_p_outlier)
end

function initialise_OU_bandit_delay_outlier_environment()

	n_steps = 40
	n_sessions = 100
	n_bandits = 2

	r_0_v = [0.0, 0.0]
	μ_OU_bandit_v = [0.0, 0.0]
	σ_OU_bandit_v = [1.0, 1.0]
	γ_OU_bandit_v = [1.0, 1.0]

	r_outlier = 10.0 
	outlier_delay_distr = Geometric(0.000000005)

	return OU_bandit_delay_outlier_environment(n_steps, n_sessions, r_0_v, γ_OU_bandit_v, μ_OU_bandit_v, σ_OU_bandit_v,
												r_outlier, outlier_delay_distr)
end

function initialise_OU_bandit_test_environment()

	n_steps = 40
	n_sessions = 100
	n_bandits = 2

	r_0_v = [0.0, 0.0]
	μ_OU_bandit_v = [0.0, 0.0]
	σ_OU_bandit_v = [1.0, 1.0]
	γ_OU_bandit_v = [0.5, 0.5]

	r_outlier = 10.0 

	return OU_bandit_test_environment(n_steps, n_sessions, r_0_v, γ_OU_bandit_v, μ_OU_bandit_v, σ_OU_bandit_v, r_outlier)
end

function initialise_ABT_environment()

	n_trials = 20
	n_bandits = 3
	n_sessions = 4

	r_v = [1.0, 1.0, 0.0]
	manipulation_δr_v = [0.0, -5.0, 0.0, -5.0]

	available_action_m = repeat([[1,3] [2,3]], outer = [1, Int(n_sessions / 2)])

	return ABT_environment(n_trials, n_bandits, n_sessions, r_v, manipulation_δr_v, available_action_m)

end

function run_bandit_example(env::abstract_bandit_environment, agent_v::Array{T, 1}; plot_flag = false) where T <:abstract_bandit_agent

	optimal_agent = get_optimal_agent(env)

	random_agent = delta_agent(env.n_steps, env.n_bandits, env.n_sessions, 0.0, 0.0,
							offline_bias(10, env.n_bandits, env.n_sessions, 10, 0.0, 0.0), 
							ε_greedy_policy(1.0))

	run_environment!(env, optimal_agent)
	run_environment!(env, random_agent)

	c = 1
	for agent in agent_v
	
		run_environment!(env, agent)

		println("agent $c score : ", (sum(agent.accumulated_r_v) - sum(random_agent.accumulated_r_v)) / 
								 	(sum(optimal_agent.accumulated_r_v) - sum(random_agent.accumulated_r_v)))

		if plot_flag
			plot_timeseries(agent.r_m, agent.bias.b_m, agent.bias.Δr_v, 1:40)
		end

		c += 1
	end
end

function run_ABT_example(plot_flag = false)

	env = initialise_ABT_environment()

	agent = probabilistic_delta_agent(env.n_trials, env.n_bandits, env.n_sessions, 1.0, 0.01, 0.5, 1.0, 
									offline_bias(10, env.n_bandits, env.n_sessions, 5, 0.1, 0.01),
									softmax_policy(5.5))

	run_environment!(env, agent)

	if plot_flag
		plot_ABT(agent.r_m, agent.bias.b_m, env.available_action_m)
	end
end

function run_opt(env::abstract_bandit_environment)

	optimal_agent = get_optimal_agent(env)

	random_agent = delta_agent(env.n_steps, env.n_bandits, env.n_sessions, 0.0, 0.0,
								offline_bias(10, env.n_bandits, env.n_sessions, 10, 0.0, 0.0), 
								ε_greedy_policy(1.0))

	run_environment!(env, random_agent)
	run_environment!(env, optimal_agent)
	
	#___agent parameter vector = [n_bias_steps, bias_buffer_length, decay_bias, decay_reward]___
	agent_param_v = Any[Int(floor(0.5env.n_steps)), Int(floor(0.2env.n_steps)), 0.01, 0.01]

	#___objective function patameter vector = [sum performance of optimal agent, sum performance of random agent]___
	obj_param_v = [sum(optimal_agent.accumulated_r_v), sum(random_agent.accumulated_r_v)]

	d_agent_x_v = run_delta_agent_opt(env, agent_param_v, obj_param_v)

	prob_d_agent_x_v = run_prob_delta_agent_opt(env, agent_param_v, obj_param_v)

	d_agent = delta_agent(env.n_steps, env.n_bandits, env.n_sessions, d_agent_x_v[2], agent_param_v[4], 
						offline_bias(agent_param_v[1], env.n_bandits, env.n_sessions, agent_param_v[2], d_agent_x_v[1], agent_param_v[3]), 
						ε_greedy_policy(d_agent_x_v[3]))

	prob_d_agent = probabilistic_delta_agent(env.n_steps, env.n_bandits, env.n_sessions, prob_d_agent_x_v[2], agent_param_v[4], 
											prob_d_agent_x_v[3], prob_d_agent_x_v[4],
											offline_bias(agent_param_v[1], env.n_bandits, env.n_sessions, agent_param_v[2], 
														prob_d_agent_x_v[1], agent_param_v[3]),
											ε_greedy_policy(prob_d_agent_x_v[5]))


	run_bandit_example(env, [prob_d_agent] ; plot_flag = true)
end

function run_test(env::abstract_bandit_environment)

	(n_steps, n_bandits, n_sessions) = size(env.r_m)
	
	n_bias_steps = floor(Int(0.5env.n_steps))
	bias_buffer_length = floor(Int(0.2env.n_steps))

	η_b = 0.01
	decay_b = 0.01

	μ_prob_delta = 0.0
	σ_prob_delta = 1.0
	decay_r = 0.01

	η_r_prob_delta = 1.0

	ε = 0.1

	agent = probabilistic_delta_agent(n_steps, n_bandits, n_sessions, 
									η_r_prob_delta, decay_r, μ_prob_delta, σ_prob_delta, 
									offline_bias(n_bias_steps, n_bandits, n_sessions, bias_buffer_length, η_b, decay_b), 
									ε_greedy_policy(ε))

	run_bandit_example(env, [agent] ; plot_flag = true)

end
