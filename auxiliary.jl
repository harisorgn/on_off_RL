
function get_optimal_agent(env::bandit_environment{T, Y}) where {T <: abstract_reward_process, Y <: Union{no_out_process, distribution_out_process, delay_out_process, test_out_process}}

	return optimal_bandit_distribution_out_agent(env.reward_process.n_steps, env.reward_process.n_sessions)
end

function get_optimal_agent(env::bandit_environment{T, frequency_out_process}) where T <: abstract_reward_process

	return optimal_bandit_frequency_outlier_agent(env.reward_process.n_steps, 
												env.reward_process.n_sessions, 
												env.out_process.r_out, 
												env.out_process.p_out_max, 
												env.out_process.η_p_out, 
												env.out_process.decay_p_out)
end

OU_distr(x, t, γ, σ; t_0  = 0.0, x_0 = 0.0) = sqrt(γ / (2.0 * pi * (σ^2.0 / 2.0) * (1.0 - exp(-2.0 * γ * (t - t_0))))) * 
											exp(-(γ * (x - x_0 * exp(-γ * (t - t_0)))^2.0)/
												(2.0 * (σ^2.0 / 2.0) * (1.0 - exp(-2.0 * γ * (t - t_0)))))

function initialise_OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, μ_v, σ_v, γ_v)

	r_0_v = zeros(n_bandits)

	return OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, γ_v, μ_v, σ_v)
end

function initialise_distribution_out_process(n_steps, n_bandits, n_sessions)

	outlier1_distr = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.8, 		0.15, 		0.05])

	outlier2_distr = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.9, 		0.05, 		0.05])

	return distribution_out_process(n_steps, n_bandits, n_sessions, [outlier1_distr, outlier2_distr])
end

function initialise_frequency_out_process(n_steps, n_bandits, n_sessions)
	
	r_out = 10.0 
	p_out_max = 0.2 
	η_p_out = 0.05 
	decay_p_out = 0.1

	return frequency_out_process(n_steps, n_bandits, n_sessions, r_out, p_out_max, η_p_out, decay_p_out)
end

function initialise_delay_out_process(n_steps, n_bandits, n_sessions)

	r_out = 10.0 
	out_delay_distr = Geometric(0.02)

	return delay_out_process(n_steps, n_bandits, n_sessions, r_out, out_delay_distr)
end

function initialise_test_out_process(n_steps, n_bandits, n_sessions)

	r_out = 10.0 

	return test_out_process(n_steps, n_bandits, n_sessions, r_out)
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

	#___objective function patameter vector = [decay_offline, decay_reward, γ_ΟU, σ_OU]___
	obj_param_v = [0.01, 0.01, env_v[1].reward_process.γ_v[1], env_v[1].reward_process.σ_v[1]]

	bias_v = run_delta_agent_opt(env_v, obj_param_v, offline_bias)
 	
	bias_d = Dict("η_offline" => bias_v[1], "η_r" => bias_v[2], "ε" => bias_v[3])

	Q_v = run_delta_agent_opt(env_v, obj_param_v, offline_Q)

	Q_d = Dict("η_offline" => Q_v[1], "η_r" => Q_v[2], "ε" => Q_v[3])

	no_offline_v = run_delta_agent_no_offline_opt(env_v, obj_param_v)

	no_offline_d = Dict("η_r" => no_offline_v[1], "ε" => no_offline_v[2])

	save(string("opt_res_", rand(MersenneTwister(), 1000:9999), ".jld"), 
		"env_v", env_v, "bias_d", bias_d, "Q_d", Q_d, "no_offline_d", no_offline_d)

end

function run_bandit_test()

	n_warmup_steps = 10
	n_steps = 40
	n_sessions = 10
	n_bandits = 2

	r_0_v = [0.0, 0.0]
	μ_v = [0.0, 0.0]
	σ_v = [1.0, 1.0]
	γ_v = [0.5, 0.5]

	r_out = 10.0

	env = bandit_environment(OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, μ_v, σ_v, γ_v),
							test_out_process(n_steps, n_bandits, n_sessions, r_out),
							MersenneTwister())

	(n_steps, n_bandits, n_sessions) = size(env.reward_process.r_m)
	
	n_bias_steps = floor(Int(0.5env.reward_process.n_steps))
	bias_buffer_length = floor(Int(0.2env.reward_process.n_steps))

	η_offline = 0.01
	decay_offline = 0.01

	η_r = 0.1
	decay_r = 0.01

	ε = 0.1

	agent = delta_agent(n_steps, n_bandits, n_sessions, 
						η_r, decay_r, 
						offline_Q(n_bias_steps, n_bandits, n_sessions, bias_buffer_length, η_offline, decay_offline), 
						ε_greedy_policy(ε))

	run_bandit_example(env, [agent] ; plot_flag = true)

end
