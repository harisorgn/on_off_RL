
get_optimal_agent(env::Union{OU_bandit_environment,
							OU_bandit_distribution_outlier_environment, 
							OU_bandit_delay_outlier_environment, 
							OU_bandit_test_environment}) = optimal_bandit_distribution_outlier_agent(env.n_steps, 
																									env.n_sessions)

get_optimal_agent(env::OU_bandit_frequency_outlier_environment) = optimal_bandit_frequency_outlier_agent(env.n_steps, 
																										env.n_sessions, 
																										env.r_outlier, 
																										env.p_outlier_max, 
																										env.η_p_outlier, 
																										env.decay_p_outlier)

OU_distr(x, γ, σ, t; t_0  = 0.0, x_0 = 0.0) = sqrt(γ / (2.0 * pi * (σ^2.0 / 2.0) * (1.0 - exp(-2.0 * γ * (t - t_0))))) * 
											exp(-(γ * (x - x_0 * exp(-γ * (t - t_0)))^2.0)/
												(2.0 * (σ^2.0 / 2.0) * (1.0 - exp(-2.0 * γ * (t - t_0)))))

function initialise_OU_bandit_environment(n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v)

	r_0_v = zeros(n_bandits)

	return OU_bandit_environment(n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v)
end

function initialise_OU_bandit_distribution_outlier_environment(n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v)

	outlier1_distr = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.8, 		0.15, 		0.05])

	outlier2_distr = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.9, 		0.05, 		0.05])

	r_0_v = zeros(n_bandits)

	return OU_bandit_distribution_outlier_environment(n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v, [outlier1_distr, outlier2_distr])
end

function initialise_OU_bandit_frequency_outlier_environment(n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v)

	r_0_v = zeros(n_bandits)
	
	r_outlier = 10.0 
	p_outlier_max = 0.2 
	η_p_outlier = 0.05 
	decay_p_outlier = 0.1

	return OU_bandit_frequency_outlier_environment(n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v, 
												r_outlier, p_outlier_max, η_p_outlier, decay_p_outlier)
end

function initialise_OU_bandit_delay_outlier_environment(n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v)

	r_0_v = zeros(n_bandits)

	r_outlier = 10.0 
	outlier_delay_distr = Geometric(0.02)

	return OU_bandit_delay_outlier_environment(n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v, r_outlier, outlier_delay_distr)
end

function initialise_OU_bandit_test_environment(n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v)

	r_0_v = zeros(n_bandits)

	r_outlier = 10.0 

	return OU_bandit_test_environment(n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v, r_outlier)
end

function run_bandit_example(env::abstract_bandit_environment, agent_v::Array{T, 1}; plot_flag = false) where T <:abstract_bandit_agent

	optimal_agent = get_optimal_agent(env)

	random_agent = delta_agent(env.n_steps, env.n_bandits, env.n_sessions, 0.0, 0.0,
							offline_bias(10, env.n_bandits, env.n_sessions, 10, 0.0, 0.0), 
							ε_greedy_policy(1.0))

	run_environment!(env, optimal_agent)
	run_environment!(env, random_agent)

	score_v = zeros(length(agent_v))

	c = 1
	for agent in agent_v
	
		run_environment!(env, agent)

		#score_v[c] = (sum(agent.accumulated_r_v) - sum(random_agent.accumulated_r_v)) / 
		#			(sum(optimal_agent.accumulated_r_v) - sum(random_agent.accumulated_r_v))
		score_v[c] = sum(agent.accumulated_r_v)

		if plot_flag
			plot_bandit_results(env, agent, 1:10)
		end

		c += 1
	end

	return score_v
end

function run_opt(env_v::Array{T,1}) where T <: abstract_bandit_environment

	random_obj = 0.0
	optimal_obj = 0.0

	for env in env_v

		optimal_agent = get_optimal_agent(env)

		random_agent = delta_agent(env.n_steps, env.n_bandits, env.n_sessions, 0.0, 0.0,
									offline_bias(10, env.n_bandits, env.n_sessions, 10, 0.0, 0.0), 
									ε_greedy_policy(1.0))

		run_environment!(env, random_agent)
		run_environment!(env, optimal_agent)

		random_obj += sum(random_agent.accumulated_r_v)
		optimal_obj += sum(optimal_agent.accumulated_r_v)

	end

	#___objective function patameter vector = [sum performance of optimal agent, sum performance of random agent]___
	obj_param_v = [optimal_obj, random_obj]

	d_agent_x_v = run_delta_agent_opt(env_v, obj_param_v)

	prob_d_agent_x_v = run_prob_delta_agent_opt(env_v, obj_param_v)

	d_agent = delta_agent(env_v[1].n_steps, env_v[1].n_bandits, env_v[1].n_sessions, d_agent_x_v[2], 0.01, 
						offline_bias(Int(floor(0.5*env_v[1].n_steps)), env_v[1].n_bandits, env_v[1].n_sessions, 
									Int(floor(0.2*env_v[1].n_steps)), d_agent_x_v[1], 0.01), 
						ε_greedy_policy(d_agent_x_v[3]))

	prob_d_agent = probabilistic_delta_agent(env_v[1].n_steps, env_v[1].n_bandits, env_v[1].n_sessions, prob_d_agent_x_v[2], 
											0.01, prob_d_agent_x_v[3], prob_d_agent_x_v[4],
											offline_bias(Int(floor(0.5*env_v[1].n_steps)), env_v[1].n_bandits, env_v[1].n_sessions, 
														Int(floor(0.2*env_v[1].n_steps)), prob_d_agent_x_v[1], 0.01),
											ε_greedy_policy(prob_d_agent_x_v[5]))


	save(string("opt_res_", rand(MersenneTwister(), 1000:9999), ".jld"), 
		"env_v", env_v, "d_agent", d_agent, "prob_d_agent", prob_d_agent)

end

function run_bandit_test()

	env = initialise_OU_bandit_test_environment()

	(n_steps, n_bandits, n_sessions) = size(env.r_m)
	
	n_bias_steps = floor(Int(0.5env.n_steps))
	bias_buffer_length = floor(Int(0.2env.n_steps))

	η_b = 0.01
	decay_b = 0.01

	μ_prob_delta = 0.0
	σ_prob_delta = 0.5
	decay_r = 0.01

	η_r_prob_delta = 1.0

	ε = 0.1

	agent = probabilistic_delta_agent(n_steps, n_bandits, n_sessions, 
									η_r_prob_delta, decay_r, μ_prob_delta, σ_prob_delta, 
									offline_bias(n_bias_steps, n_bandits, n_sessions, bias_buffer_length, η_b, decay_b), 
									ε_greedy_policy(ε))

	run_bandit_example(env, [agent] ; plot_flag = true)

end
