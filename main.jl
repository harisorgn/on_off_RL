using Distributions
using Random
using Statistics

include("agents.jl")
include("environments.jl")
include("auxiliary.jl")
include("plot.jl")
include("opt.jl")

function main()
		
	env = initialise_OU_bandit_frequency_outlier_environment()

	# agent parameter vector = [n_bias_steps, bias_buffer_length, decay_b, decay_r]
	agent_param_v = Any[20, 10, 0.01, 0.01]

	d_agent_x_v = run_delta_agent_opt(env, agent_param_v)
	prob_d_agent_x_v = run_prob_delta_agent_opt(env, agent_param_v)

	d_agent = delta_agent(env.n_steps, env.n_actions, env.n_sessions, d_agent_x_v[2], agent_param_v[4], 
						offline_bias(agent_param_v[1], env.n_actions, env.n_sessions, agent_param_v[2], d_agent_x_v[1], agent_param_v[3]), 
						softmax_policy(d_agent_x_v[3]))

	prob_d_agent = probabilistic_delta_agent(env.n_steps, env.n_actions, env.n_sessions, prob_d_agent_x_v[2], agent_param_v[4], 
											prob_d_agent_x_v[3], prob_d_agent_x_v[4],
											offline_bias(agent_param_v[1], env.n_actions, env.n_sessions, agent_param_v[2], 
														prob_d_agent_x_v[1], agent_param_v[3]),
											softmax_policy(prob_d_agent_x_v[5]))

	run_example(env, [d_agent, prob_d_agent] ; plot_flag = false)
end

main()


