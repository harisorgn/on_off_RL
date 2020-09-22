
function run_example(env::OU_bandit_distribution_outlier_environment, agent_v)

	(n_steps, n_actions, n_sessions) = size(env.r_m)

	optimal_agent = optimal_bandit_distribution_outlier_agent(n_steps, n_sessions)

	random_agent = delta_agent(n_steps, n_actions, n_sessions, 0.0, 0.0,
							offline_bias(10, n_actions, n_sessions, 10, 0.0, 0.0), 
							ε_greedy_policy(1.0))

	run_environment!(env, optimal_agent)
	run_environment!(env, random_agent)

	c = 1
	for agent in agent_v

		run_environment!(env, agent)

		println("agent $c score : ", (median(agent.accumulated_r_v) - median(random_agent.accumulated_r_v)) / 
								 	(median(optimal_agent.accumulated_r_v) - median(random_agent.accumulated_r_v)))

		c += 1
	end
end

function run_OU_bandit_distribution_outlier_example()

	out1_d = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.8, 		0.2, 		0.0])

	out2_d = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [1.0, 		0.0, 		0.0])
	
	n_steps = 40
	n_sessions = 300
	n_actions = 2

	μ_OU_bandit_v = [0.0, 0.0]
	σ_OU_bandit_v = [0.5, 0.5]
	γ_OU_bandit_v = [0.01, 0.01]

	env = OU_bandit_distribution_outlier_environment(n_steps, n_sessions, γ_OU_bandit_v, 
													μ_OU_bandit_v, σ_OU_bandit_v, [out1_d, out2_d])
	
	n_bias_steps = 20
	bias_buffer_length = 10

	η_b = 0.01   # or (η_b / n_steps)
	decay_b = 0.01
	bias = offline_bias(n_bias_steps, n_actions, n_sessions, bias_buffer_length, η_b, decay_b)

	β = 2.5
	policy = softmax_policy(β)

	η_r_delta = 0.1
	decay_r = 0.01

	d_agent = delta_agent(n_steps, n_actions, n_sessions, η_r_delta, decay_r, bias, policy)

	μ_prob_delta = 0.0
	σ_prob_delta = 0.5
	η_r_prob_delta = 10.0 * η_r_delta / pdf(Normal(μ_prob_delta, σ_prob_delta), μ_prob_delta)

	@assert(η_r_prob_delta <= 1.0/pdf(Normal(μ_prob_delta, σ_prob_delta), μ_prob_delta), "n_r_prob_delta out of bounds")

	prob_d_agent = probabilistic_delta_agent(n_steps, n_actions, n_sessions, 
											η_r_prob_delta, decay_r, μ_prob_delta, σ_prob_delta, 
											bias, policy)

	run_environment!(env, d_agent)

	run_environment!(env, prob_d_agent)


	optimal_agent = optimal_bandit_distribution_outlier_agent(n_steps, n_sessions)

	random_agent = delta_agent(n_steps, n_actions, n_sessions, 0.0, 0.0,
							offline_bias(n_bias_steps, n_actions, n_sessions, bias_buffer_length, 0.0, 0.0), 
							ε_greedy_policy(1.0))

	run_environment!(env, optimal_agent)
	run_environment!(env, random_agent)

	println("delta agent score : ", (median(d_agent.accumulated_r_v) - median(random_agent.accumulated_r_v)) / 
								 	(median(optimal_agent.accumulated_r_v) - median(random_agent.accumulated_r_v)))

	println("probabilistic delta agent score : ", (median(prob_d_agent.accumulated_r_v) - median(random_agent.accumulated_r_v)) / 
								 			   	  (median(optimal_agent.accumulated_r_v) - median(random_agent.accumulated_r_v)))

	#plot_timeseries(d_agent.r_m, d_agent.bias.b_m, 1:n_sessions)
	#plot_timeseries(prob_d_agent.r_m, prob_d_agent.bias.b_m, 1:n_sessions)

end

function run_OU_bandit_frequency_outlier_example()

	n_steps = 40
	n_sessions = 10
	n_actions = 2

	μ_OU_bandit_v = [0.0, 0.0]
	σ_OU_bandit_v = [0.5, 0.5]
	γ_OU_bandit_v = [1.0, 1.0]

	r_outlier = 10.0 
	p_outlier_max = 0.2 
	η_p_outlier = 0.05 
	decay_p_outlier = 0.1

	env = OU_bandit_frequency_outlier_environment(n_steps, n_sessions, γ_OU_bandit_v, μ_OU_bandit_v, σ_OU_bandit_v,
												r_outlier, p_outlier_max, η_p_outlier, decay_p_outlier)
	
	n_bias_steps = 20
	bias_buffer_length = 10

	η_b = 0.01
	decay_b = 0.01
	bias = offline_bias(n_bias_steps, n_actions, n_sessions, bias_buffer_length, η_b, decay_b)

	β = 2.5
	policy = softmax_policy(β)

	η_r_delta = 0.1
	decay_r = 0.01

	d_agent = delta_agent(n_steps, n_actions, n_sessions, η_r_delta, decay_r, bias, policy)

	μ_prob_delta = 0.0
	σ_prob_delta = 0.5
	η_r_prob_delta = 10.0 * η_r_delta / pdf(Normal(μ_prob_delta, σ_prob_delta), μ_prob_delta)

	@assert(η_r_prob_delta <= 1.0/pdf(Normal(μ_prob_delta, σ_prob_delta), μ_prob_delta), "n_r_prob_delta out of bounds")

	prob_d_agent = probabilistic_delta_agent(n_steps, n_actions, n_sessions, 
											η_r_prob_delta, decay_r, μ_prob_delta, σ_prob_delta, 
											bias, policy)

	run_environment!(env, d_agent)
	run_environment!(env, prob_d_agent)

	optimal_agent = optimal_bandit_frequency_outlier_agent(n_steps, n_sessions, r_outlier, 
														p_outlier_max, η_p_outlier, decay_p_outlier)

	random_agent = delta_agent(n_steps, n_actions, n_sessions, 0.0, 0.0,
							offline_bias(n_bias_steps, n_actions, n_sessions, bias_buffer_length, 0.0, 0.0), 
							ε_greedy_policy(1.0))

	run_environment!(env, optimal_agent)
	run_environment!(env, random_agent)

	println("delta agent score : ", (median(d_agent.accumulated_r_v) - median(random_agent.accumulated_r_v)) / 
								 	(median(optimal_agent.accumulated_r_v) - median(random_agent.accumulated_r_v)))

	println("probabilistic delta agent score : ", (median(prob_d_agent.accumulated_r_v) - median(random_agent.accumulated_r_v)) / 
								 			   	  (median(optimal_agent.accumulated_r_v) - median(random_agent.accumulated_r_v)))

	#plot_timeseries(d_agent.r_m, d_agent.bias.b_m, 1:n_sessions)
	#plot_timeseries(prob_d_agent.r_m, prob_d_agent.bias.b_m, 1:n_sessions)

end