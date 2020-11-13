
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

OU_distr(x, t, γ, σ; t_0  = 0.0, x_0 = 0.0) = sqrt(γ / (2.0 * pi * (σ^2.0 / 2.0) * (1.0 - exp(-2.0 * γ * (t - t_0))))) * 
											exp(-(γ * (x - x_0 * exp(-γ * (t - t_0)))^2.0)/
												(2.0 * (σ^2.0 / 2.0) * (1.0 - exp(-2.0 * γ * (t - t_0)))))

function initialise_OU_bandit_environment(n_warmup_steps, n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v)

	r_0_v = zeros(n_bandits)

	return OU_bandit_environment(n_warmup_steps, n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v)
end

function initialise_OU_bandit_distribution_outlier_environment(n_warmup_steps, n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v)

	outlier1_distr = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.8, 		0.15, 		0.05])

	outlier2_distr = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.9, 		0.05, 		0.05])

	r_0_v = zeros(n_bandits)

	return OU_bandit_distribution_outlier_environment(n_warmup_steps, n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v, 
													[outlier1_distr, outlier2_distr])
end

function initialise_OU_bandit_frequency_outlier_environment(n_warmup_steps, n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v)

	r_0_v = zeros(n_bandits)
	
	r_outlier = 10.0 
	p_outlier_max = 0.2 
	η_p_outlier = 0.05 
	decay_p_outlier = 0.1

	return OU_bandit_frequency_outlier_environment(n_warmup_steps, n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v, 
												r_outlier, p_outlier_max, η_p_outlier, decay_p_outlier)
end

function initialise_OU_bandit_delay_outlier_environment(n_warmup_steps, n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v)

	r_0_v = zeros(n_bandits)

	r_outlier = 10.0 
	outlier_delay_distr = Geometric(0.02)

	return OU_bandit_delay_outlier_environment(n_warmup_steps, n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v, r_outlier, outlier_delay_distr)
end

function initialise_OU_bandit_test_environment(n_warmup_steps, n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v)

	r_0_v = zeros(n_bandits)

	r_outlier = 10.0 

	return OU_bandit_test_environment(n_warmup_steps, n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v, r_outlier)
end

function run_bandit_example(env::abstract_bandit_environment, agent_v::Array{T, 1}; plot_flag = false) where T <:abstract_bandit_agent

	score_v = zeros(length(agent_v))

	c = 1
	for agent in agent_v
	
		run_environment!(env, agent)

		score_v[c] = sum(agent.accumulated_r_v)

		if plot_flag

			println(typeof(agent), " score : ", score_v[c])

			plot_bandit_results(env, agent, 1:10)
		end

		c += 1
	end

	return score_v
end

function run_opt(env_v::Array{T,1}) where T <: abstract_bandit_environment

	#___objective function patameter vector = [decay_bias, decay_reward, γ_ΟU, σ_OU]___
	obj_param_v = [0.01, 0.01, env_v[1].γ_v[1], env_v[1].σ_v[1]]

	d_agent_bias_v = run_delta_agent_opt(env_v, obj_param_v, offline_bias)

	d_agent_bias = delta_agent(env_v[1].n_steps, env_v[1].n_bandits, env_v[1].n_sessions, d_agent_bias_v[2], obj_param_v[2], 
								offline_bias(Int(floor(0.5*env_v[1].n_steps)), env_v[1].n_bandits, env_v[1].n_sessions, 
											Int(floor(0.2*env_v[1].n_steps)), d_agent_bias_v[1], obj_param_v[1]), 
								ε_greedy_policy(d_agent_bias_v[3]))

	d_agent_Q_v = run_delta_agent_opt(env_v, obj_param_v, offline_Q)

	d_agent_Q = delta_agent(env_v[1].n_steps, env_v[1].n_bandits, env_v[1].n_sessions, d_agent_Q_v[2], obj_param_v[2], 
								offline_bias(Int(floor(0.5*env_v[1].n_steps)), env_v[1].n_bandits, env_v[1].n_sessions, 
											Int(floor(0.2*env_v[1].n_steps)), d_agent_Q_v[1], obj_param_v[1]), 
								ε_greedy_policy(d_agent_Q_v[3]))


	d_agent_no_offline_v = run_delta_agent_no_offline_opt(env_v, obj_param_v)

	d_agent_no_offline = delta_agent(env_v[1].n_steps, env_v[1].n_bandits, env_v[1].n_sessions, d_agent_no_offline_v[1], obj_param_v[2], 
								offline_bias(Int(floor(0.5*env_v[1].n_steps)), env_v[1].n_bandits, env_v[1].n_sessions, 
											Int(floor(0.2*env_v[1].n_steps)), 0.0, 0.0), 
								ε_greedy_policy(d_agent_no_offline_v[2]))

	#=
	prob_d_agent_x_v = run_prob_delta_agent_opt(env_v, obj_param_v)

	prob_d_agent = probabilistic_delta_agent(env_v[1].n_steps, env_v[1].n_bandits, env_v[1].n_sessions, prob_d_agent_x_v[2], 
											obj_param_v[2], prob_d_agent_x_v[3], prob_d_agent_x_v[4],
											offline_bias(Int(floor(0.5*env_v[1].n_steps)), env_v[1].n_bandits, env_v[1].n_sessions, 
														Int(floor(0.2*env_v[1].n_steps)), prob_d_agent_x_v[1], obj_param_v[1]),
											ε_greedy_policy(prob_d_agent_x_v[5]))

	ou_agent = OU_agent(env_v[1].n_steps, env_v[1].n_bandits, env_v[1].n_sessions, OU_agent_x_v[2], 
						obj_param_v[2], obj_param_v[3], obj_param_v[4],
						offline_bias(Int(floor(0.5*env_v[1].n_steps)), env_v[1].n_bandits, env_v[1].n_sessions, 
									Int(floor(0.2*env_v[1].n_steps)), OU_agent_x_v[1], obj_param_v[1]),
						ε_greedy_policy(OU_agent_x_v[3]))

	=#

	save(string("opt_res_", rand(MersenneTwister(), 1000:9999), ".jld"), 
		"env_v", env_v, "d_agent_bias", d_agent_bias, "d_agent_Q", d_agent_Q, "d_agent_no_offline", d_agent_no_offline)

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
