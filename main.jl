using Distributions
using Random
using Statistics

include("agents.jl")
include("environments.jl")
include("examples.jl")
include("plot.jl")
include("opt.jl")

function main()

	#run_OU_bandit_distribution_outlier_example()

	#run_OU_bandit_frequency_outlier_example()

	out1_d = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.8, 		0.15, 		0.05])

	out2_d = MixtureModel(Normal, [(0.0, 0.01), (10.0, 0.1), (-10.0, 0.1)], 
								  [0.9, 		0.05, 		0.05])
	
	n_steps = 20
	n_sessions = 200
	n_actions = 2

	μ_OU_bandit_v = [0.0, 0.0]
	σ_OU_bandit_v = [0.5, 0.5]
	γ_OU_bandit_v = [0.01, 0.01]

	env = OU_bandit_distribution_outlier_environment(n_steps, n_sessions, γ_OU_bandit_v, 
													μ_OU_bandit_v, σ_OU_bandit_v, [out1_d, out2_d])

	# agent_param_v = [n_bias_steps, bias_buffer_length, decay_b, decay_r]
	agent_param_v = Any[20, 10, 0.01, 0.01]

	d_agent_x_v = run_delta_agent_opt(env, agent_param_v)
	prob_d_agent_x_v = run_prob_delta_agent_opt(env, agent_param_v)

	d_agent = delta_agent(n_steps, n_actions, n_sessions, d_agent_x_v[2], agent_param_v[4], 
						offline_bias(agent_param_v[1], n_actions, n_sessions, agent_param_v[2], d_agent_x_v[1], agent_param_v[3]), 
						ε_greedy_policy(d_agent_x_v[3]))

	prob_d_agent = probabilistic_delta_agent(n_steps, n_actions, n_sessions, prob_d_agent_x_v[2], agent_param_v[4], 
											prob_d_agent_x_v[3], prob_d_agent_x_v[4],
											offline_bias(agent_param_v[1], n_actions, n_sessions, agent_param_v[2], 
														prob_d_agent_x_v[1], agent_param_v[3]),
											ε_greedy_policy(prob_d_agent_x_v[5]))

	run_example(env, [d_agent, prob_d_agent])
end

main()


