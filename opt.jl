using NLopt

function obj_func_delta(x::Vector, grad::Vector, env, agent_param_v, obj_param_v)
	
	(n_steps, n_actions, n_sessions) = size(env.r_m)

	n_bias_steps = agent_param_v[1]
	bias_buffer_length = agent_param_v[2]

	η_b = x[1]
	decay_b = agent_param_v[3]

	η_r_delta = x[2]
	decay_r = agent_param_v[4]

	ε = x[3]

	agent = delta_agent(n_steps, n_actions, n_sessions, η_r_delta, decay_r,
						offline_bias(n_bias_steps, n_actions, n_sessions, bias_buffer_length, η_b, decay_b), 
						softmax_policy(ε))

	run_environment!(env, agent)

	return ((sum(agent.accumulated_r_v) - obj_param_v[2]) / 
			(obj_param_v[1] - obj_param_v[2]))
end

function obj_func_prob_delta(x::Vector, grad::Vector, env, agent_param_v, obj_param_v)

	(n_steps, n_actions, n_sessions) = size(env.r_m)
	
	n_bias_steps = agent_param_v[1]
	bias_buffer_length = agent_param_v[2]

	η_b = x[1]
	decay_b = agent_param_v[3]

	μ_prob_delta = x[3]
	σ_prob_delta = x[4]
	decay_r = agent_param_v[4]

	η_r_prob_delta = x[2]

	ε = x[5]

	agent = probabilistic_delta_agent(n_steps, n_actions, n_sessions, 
											η_r_prob_delta, decay_r, μ_prob_delta, σ_prob_delta, 
											offline_bias(n_bias_steps, n_actions, n_sessions, bias_buffer_length, η_b, decay_b), 
											softmax_policy(ε))

	
	run_environment!(env, agent)

	return ((sum(agent.accumulated_r_v) - obj_param_v[2]) / 
			(obj_param_v[1] - obj_param_v[2]))
end

constraint_prob_delta(x::Vector, grad::Vector) = x[2] - 1.0/pdf(Normal(x[3], x[4]), x[3])

function run_delta_agent_opt(env, agent_param_v)

	# vector to be optimised = [η_b, η_r, ε]

	optimal_agent = get_optimal_agent(env)

	random_agent = delta_agent(env.n_steps, env.n_actions, env.n_sessions, 0.0, 0.0,
								offline_bias(10, env.n_actions, env.n_sessions, 10, 0.0, 0.0), 
								ε_greedy_policy(1.0))

	run_environment!(env, optimal_agent)
	run_environment!(env, random_agent)

	obj_param_v = [sum(optimal_agent.accumulated_r_v), sum(random_agent.accumulated_r_v)]

	opt = Opt(:LN_COBYLA, 3)
	opt.lower_bounds = [0.0, 0.0, 0.0]
	opt.upper_bounds = [1.0, 1.0, Inf]
	opt.ftol_rel = 1e-16
	opt.ftol_abs = 1e-16
	opt.xtol_rel = 1e-16
	opt.xtol_abs = 1e-16

	opt.max_objective = (x, g) -> obj_func_delta(x, g, env, agent_param_v, obj_param_v)

	(max_f, max_x, ret) = optimize(opt, [0.01, 0.1, 2.0])
	numevals = opt.numevals 
	println("got $max_f at $max_x after $numevals iterations (returned $ret)")

	return max_x
end

function run_prob_delta_agent_opt(env, agent_param_v)

	# vector to be optimised = [η_b, η_r, μ, σ, ε]

	optimal_agent = get_optimal_agent(env)

	random_agent = delta_agent(env.n_steps, env.n_actions, env.n_sessions, 0.0, 0.0,
								offline_bias(10, env.n_actions, env.n_sessions, 10, 0.0, 0.0), 
								ε_greedy_policy(1.0))

	run_environment!(env, optimal_agent)
	run_environment!(env, random_agent)

	obj_param_v = [sum(optimal_agent.accumulated_r_v), sum(random_agent.accumulated_r_v)]

	opt = Opt(:LN_COBYLA, 5)
	opt.lower_bounds = [0.0, 0.0, -Inf, 0.0, 0.0]
	opt.upper_bounds = [1.0, Inf, Inf, Inf, Inf]
	opt.ftol_rel = 1e-16
	opt.ftol_abs = 1e-16
	opt.xtol_rel = 1e-16
	opt.xtol_abs = 1e-16

	opt.max_objective = (x, g) -> obj_func_prob_delta(x, g, env, agent_param_v, obj_param_v)

	inequality_constraint!(opt, constraint_prob_delta, 1e-8)

	(max_f, max_x, ret) = optimize(opt, [0.01, 1.0, 0.0, 0.5, 2.0])
	numevals = opt.numevals 
	println("got $max_f at $max_x after $numevals iterations (returned $ret)")

	return max_x
end