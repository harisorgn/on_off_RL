#using Optim
using NLopt

function obj_func_delta(x::Vector, grad::Vector, 
						env::OU_bandit_distribution_outlier_environment, agent_param_v)
	
	(n_steps, n_actions, n_sessions) = size(env.r_m)

	n_bias_steps = agent_param_v[1]
	bias_buffer_length = agent_param_v[2]

	η_b = x[1]
	decay_b = agent_param_v[3]
	bias = offline_bias(n_bias_steps, n_actions, n_sessions, bias_buffer_length, η_b, decay_b)

	η_r_delta = x[2]
	decay_r = agent_param_v[4]

	ε = x[3]
	policy = ε_greedy_policy(ε)

	d_agent = delta_agent(n_steps, n_actions, n_sessions, η_r_delta, decay_r, bias, policy)

	run_environment!(env, d_agent)

	optimal_agent = optimal_bandit_distribution_outlier_agent(n_steps, n_sessions)

	random_agent = delta_agent(n_steps, n_actions, n_sessions, 0.0, 0.0,
							offline_bias(10, n_actions, n_sessions, 10, 0.0, 0.0), 
							ε_greedy_policy(1.0))

	run_environment!(env, optimal_agent)
	run_environment!(env, random_agent)

	return ((median(d_agent.accumulated_r_v) - median(random_agent.accumulated_r_v)) / 
			(median(optimal_agent.accumulated_r_v) - median(random_agent.accumulated_r_v)))
end

function obj_func_prob_delta(x::Vector, grad::Vector, 
							env::OU_bandit_distribution_outlier_environment, agent_param_v)

	(n_steps, n_actions, n_sessions) = size(env.r_m)
	
	n_bias_steps = agent_param_v[1]
	bias_buffer_length = agent_param_v[2]

	η_b = x[1]
	decay_b = agent_param_v[3]

	bias = offline_bias(n_bias_steps, n_actions, n_sessions, bias_buffer_length, η_b, decay_b)

	μ_prob_delta = x[3]
	σ_prob_delta = x[4]
	decay_r = agent_param_v[4]

	η_r_prob_delta = x[2]

	ε = x[5]
	policy = ε_greedy_policy(ε)

	prob_d_agent = probabilistic_delta_agent(n_steps, n_actions, n_sessions, 
											η_r_prob_delta, decay_r, μ_prob_delta, σ_prob_delta, 
											bias, policy)

	
	run_environment!(env, prob_d_agent)

	optimal_agent = optimal_bandit_distribution_outlier_agent(n_steps, n_sessions)

	random_agent = delta_agent(n_steps, n_actions, n_sessions, 0.0, 0.0,
							offline_bias(10, n_actions, n_sessions, 10, 0.0, 0.0), 
							ε_greedy_policy(1.0))

	run_environment!(env, optimal_agent)
	run_environment!(env, random_agent)

	return ((median(prob_d_agent.accumulated_r_v) - median(random_agent.accumulated_r_v)) / 
			(median(optimal_agent.accumulated_r_v) - median(random_agent.accumulated_r_v)))
end

constraint_prob_delta(x::Vector, grad::Vector) = x[2] - 1.0/pdf(Normal(x[3], x[4]), x[3])

function run_delta_agent_opt(env, agent_param_v)

	# x = [η_b, η_r, ε]

	opt = Opt(:LN_COBYLA, 3)
	opt.lower_bounds = [0.0, 0.0, 0.0]
	opt.upper_bounds = [1.0, 1.0, 1.0]
	opt.ftol_rel = 1e-12
	opt.ftol_abs = 1e-12
	opt.xtol_rel = 1e-16
	opt.xtol_abs = 1e-16

	opt.max_objective = (x, g) -> obj_func_delta(x, g, env, agent_param_v)

	(max_f, max_x, ret) = optimize(opt, [0.01, 0.1, 0.05])
	numevals = opt.numevals 
	println("got $max_f at $max_x after $numevals iterations (returned $ret)")

	return max_x
end

function run_prob_delta_agent_opt(env, agent_param_v)

	# x = [η_b, η_r, μ, σ, ε]

	opt = Opt(:LN_COBYLA, 5)
	opt.lower_bounds = [0.0, 0.0, -Inf, 0.0, 0.0]
	opt.upper_bounds = [1.0, Inf, Inf, Inf, 1.0]
	opt.ftol_rel = 1e-12
	opt.ftol_abs = 1e-12
	opt.xtol_rel = 1e-16
	opt.xtol_abs = 1e-16

	opt.max_objective = (x, g) -> obj_func_prob_delta(x, g, env, agent_param_v)

	inequality_constraint!(opt, constraint_prob_delta, 1e-8)

	(max_f, max_x, ret) = optimize(opt, [0.01, 10.0, 0.0, 0.5, 0.05])
	numevals = opt.numevals 
	println("got $max_f at $max_x after $numevals iterations (returned $ret)")

	return max_x
end