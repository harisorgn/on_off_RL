abstract type abstract_bandit_agent end

abstract type abstract_optimal_bandit_agent end

abstract type abstract_policy end

mutable struct offline_bias 
	η::Float64
	decay::Float64
	b_m::Array{Float64}		# offline steps x actions x sessions
	Δr::Float64
	surprise_v::Array{Float64, 1}
	action_v::Array{Int64, 1}
	rng::AbstractRNG

	offline_bias(n_bias_steps, n_actions, n_sessions, buffer_length, η, decay) = new(η, 
																					decay, 
																					zeros(n_bias_steps + 1, n_actions, n_sessions), 
																					0.0, 
																					zeros(buffer_length), 
																					zeros(buffer_length),
																					MersenneTwister())
end

function (bias::offline_bias)(session)

	weight_v = bias.surprise_v ./ sum(bias.surprise_v)
	
	T_b = size(bias.b_m)[1] - 1

	i = 2
	
	for s in rand(bias.rng, Categorical(weight_v), T_b)

		action = bias.action_v[s]

		bias.b_m[i, action, session] = bias.b_m[i - 1, action, session] + bias.η * bias.Δr/T_b

		bias.b_m[i, 1:size(bias.b_m)[2] .!= action, session] .= (1.0 - bias.decay) * 
																bias.b_m[i - 1, 1:size(bias.b_m)[2] .!= action, session]

		i += 1
	end

	if session < size(bias.b_m)[3]

		bias.b_m[1, :, session + 1] = bias.b_m[end, :, session]
	end

	bias.Δr = 0.0
end

function offline_buffer_update!(bias::offline_bias, current_surprise, current_action)

	min_surprise = minimum(bias.surprise_v)

	if current_surprise > min_surprise

		min_idx = findfirst(x -> x == min_surprise, bias.surprise_v)

		bias.surprise_v[min_idx] = current_surprise
		bias.action_v[min_idx] = current_action
	end
end

struct softmax_policy <: abstract_policy
	β::Float64
	rng::AbstractRNG

	softmax_policy(β) = new(β, MersenneTwister())
end

(policy::softmax_policy)(r_v) = rand(policy.rng, 
								Categorical([exp(policy.β * r_v[i]) / sum(exp.(policy.β * r_v)) for i = 1 : length(r_v)]))

struct ε_greedy_policy <: abstract_policy
	ε::Float64
	rng::AbstractRNG

	ε_greedy_policy(ε) = new(ε, MersenneTwister())
end

(policy::ε_greedy_policy)(r_v) = rand(policy.rng) < (1.0 - policy.ε) ? argmax(r_v) : rand(policy.rng, 1 : length(r_v))

struct delta_agent <: abstract_bandit_agent
	η::Float64
	decay::Float64
	action_m::Array{Int64}	# steps x sessions
	r_m::Array{Float64}		# steps x actions x sessions
	accumulated_r_v::Array{Float64, 1}
	bias::offline_bias
	policy::abstract_policy

	delta_agent(n_steps, n_actions, n_sessions, η, decay, bias, policy) = new(η, 
																			decay, 
																			zeros(n_steps, n_sessions), 
																			zeros(n_steps, n_actions, n_sessions), 
																			zeros(n_sessions), bias, policy)
end

function (agent::delta_agent)(r_environment, cstep, session)

	agent.accumulated_r_v[session] += r_environment

	latest_action = agent.action_m[cstep - 1, session]

	δr = r_environment - agent.r_m[cstep - 1, latest_action, session]

	agent.bias.Δr += δr - agent.bias.b_m[1, latest_action, session]

	surprise = abs(δr)

	offline_buffer_update!(agent.bias, surprise, latest_action)

	agent.r_m[cstep, latest_action, session] = agent.r_m[cstep - 1, latest_action, session] + agent.η * δr

	agent.r_m[cstep, 1:size(agent.r_m)[2] .!= latest_action, session] .= (1.0 - agent.decay) * 
																	agent.r_m[cstep - 1, 1:size(agent.r_m)[2] .!= latest_action, session]

	agent.action_m[cstep, session] = agent.policy(agent.r_m[cstep, :, session] .+ agent.bias.b_m[1, :, session])

	if (cstep == size(agent.r_m)[1]) && (session < size(agent.r_m)[3])

		agent.r_m[1, :, session + 1] = agent.r_m[cstep, :, session]
	end

	return agent.action_m[cstep, session]
end

struct probabilistic_delta_agent <: abstract_bandit_agent
	η::Float64
	decay::Float64
	μ::Float64
	σ::Float64
	action_m::Array{Int64}	# steps x sessions
	r_m::Array{Float64}		# steps x actions x sessions
	accumulated_r_v::Array{Float64, 1}
	bias::offline_bias
	policy::abstract_policy

	probabilistic_delta_agent(n_steps, n_actions, n_sessions, η, decay, μ, σ, bias, policy) = new(η, 
																								decay, 
																								μ, 
																								σ, 
																								zeros(n_steps, n_sessions), 
																								zeros(n_steps, n_actions, n_sessions), 
																								zeros(n_sessions), 
																								bias, 
																								policy)
end

function (agent::probabilistic_delta_agent)(r_environment, cstep, session)

	agent.accumulated_r_v[session] += r_environment

	latest_action = agent.action_m[cstep - 1, session]

	δr = r_environment - agent.r_m[cstep - 1, latest_action, session]

	agent.bias.Δr += δr - agent.bias.b_m[1, latest_action, session]

	surprise = -logpdf(Normal(agent.μ, agent.σ), r_environment)

	offline_buffer_update!(agent.bias, surprise, latest_action)

	agent.r_m[cstep, latest_action, session] = agent.r_m[cstep - 1, latest_action, session] + 
												agent.η * pdf(Normal(agent.μ, agent.σ), r_environment) * δr

	agent.r_m[cstep, 1:size(agent.r_m)[2] .!= latest_action, session] .= (1.0 - agent.decay) * 
																	agent.r_m[cstep - 1, 1:size(agent.r_m)[2] .!= latest_action, session]

	agent.action_m[cstep, session] = agent.policy(agent.r_m[cstep, :, session] .+ agent.bias.b_m[1, :, session])


	if (cstep == size(agent.r_m)[1]) && (session < size(agent.r_m)[3])

		agent.r_m[1, :, session + 1] = agent.r_m[cstep, :, session]
	end

	return agent.action_m[cstep, session]
end

#----------------------------------------------------------------------------------------------------------------------------------
#--------------------------Under development---------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------

struct advantage_delta_agent <: abstract_bandit_agent
	η_r_state::Float64
	η_r_action::Float64
	decay::Float64
	action_m::Array{Int64}			# steps x sessions
	r_state_m::Array{Float64}		# steps x sessions
	r_action_m::Array{Float64}		# steps x actions x sessions
	accumulated_r_v::Array{Float64, 1}
	bias::offline_bias
	policy::abstract_policy

	advantage_delta_agent(n_steps, n_actions, n_sessions, 
						η_r_state, η_r_action, decay, bias, policy) = new(η_r_state, η_r_action, decay, 
																		zeros(n_steps, n_sessions), 
																		zeros(n_steps, n_sessions),
																		zeros(n_steps, n_actions, n_sessions), 
																		zeros(n_sessions), bias, policy)
end

function (agent::advantage_delta_agent)(r_environment, cstep, session)

	agent.accumulated_r_v[session] += r_environment

	latest_action = agent.action_m[cstep - 1, session]

	δr = r_environment - agent.r_action_m[cstep - 1, latest_action, session]

	δr_offline = δr - agent.bias.b_m[1, latest_action, session]

	agent.bias.Δr += δr_offline

	surprise = abs(δr_offline)

	offline_buffer_update!(agent.bias, surprise, latest_action)

	agent.r_state_m[cstep, session] = agent.r_state_m[cstep - 1, session] + 
									agent.η_r_state * (r_environment - agent.r_state_m[cstep - 1, session])

	agent.r_action_m[cstep, latest_action, session] = agent.r_action_m[cstep - 1, latest_action, session] + agent.η_r_action * δr

	agent.r_action_m[cstep, 1:size(agent.r_m)[2] .!= latest_action, session] .= (1.0 - agent.decay) * 
														agent.r_action_m[cstep - 1, 1:size(agent.r_m)[2] .!= latest_action, session]

	agent.action_m[cstep, session] = agent.policy(agent.r_m[cstep, :, session] .+ agent.bias.b_m[1, :, session])

	if (cstep == size(agent.r_m)[1]) && (session < size(agent.r_m)[3])

		agent.r_m[1, :, session + 1] = agent.r_m[cstep, :, session]

	end

	return agent.action_m[cstep, session]
end
#----------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------

struct random_agent <: abstract_bandit_agent
	r_m::Array{Float64}
	accumulated_r_v::Array{Float64, 1}
	policy::abstract_policy
	bias::offline_bias

	random_agent(n_steps, n_actions, n_sessions, bias) = new(zeros(n_steps, n_actions, n_sessions), zeros(n_sessions), 
														ε_greedy_policy(1.0), bias)
end

function (agent::random_agent)(r_environment, cstep, session)

	agent.accumulated_r_v[session] += r_environment

	return agent.policy(1:agent.n_actions)
end

struct optimal_bandit_distribution_outlier_agent <: abstract_optimal_bandit_agent
	action_m::Array{Int64}
	accumulated_r_v::Array{Float64, 1}
	policy::abstract_policy

	optimal_bandit_distribution_outlier_agent(n_steps, n_sessions) = new(zeros(n_steps, n_sessions), 
																		zeros(n_sessions), 
																		ε_greedy_policy(0.0))
end

function (agent::optimal_bandit_distribution_outlier_agent)(r_environment, next_r_environment_v, cstep, session)

	agent.accumulated_r_v[session] += r_environment

	agent.action_m[cstep, session] = agent.policy(next_r_environment_v)

	return agent.action_m[cstep, session]
end

struct optimal_bandit_frequency_outlier_agent <: abstract_optimal_bandit_agent
	action_m::Array{Int64}
	r_outlier::Float64
	p_outlier_max::Float64
	η_p_outlier::Float64
	decay_p_outlier::Float64
	p_outlier_m::Array{Float64}
	accumulated_r_v::Array{Float64, 1}
	policy::abstract_policy

	optimal_bandit_frequency_outlier_agent(n_steps, n_sessions, r_outlier, 
											p_outlier_max, η_p_outlier, decay_p_outlier) = new(zeros(n_steps, n_sessions), 
																								r_outlier,
																								p_outlier_max, 
																								η_p_outlier,
																								decay_p_outlier,
																								zeros(n_steps, n_sessions),
																								zeros(n_sessions),
																								ε_greedy_policy(0.0))
end

function (agent::optimal_bandit_frequency_outlier_agent)(r_environment, next_r_environment_v, cstep, session)

	agent.accumulated_r_v[session] += r_environment

	latest_action = agent.action_m[cstep - 1, session]

	if (latest_action == 1)
		 agent.p_outlier_m[cstep, session] = agent.p_outlier_m[cstep - 1, session] + 
											agent.η_p_outlier * (agent.p_outlier_max - agent.p_outlier_m[cstep - 1, session])
	else
		agent.p_outlier_m[cstep, session] = (1.0 - agent.decay_p_outlier) * agent.p_outlier_m[cstep - 1, session]
	end

	r_outlier_v = zeros(length(next_r_environment_v))

	r_outlier_v[1] += (agent.p_outlier_m[cstep, session] + 
						agent.η_p_outlier * (agent.p_outlier_max - agent.p_outlier_m[cstep, session])) * agent.r_outlier

	agent.action_m[cstep, session] = agent.policy(next_r_environment_v .+ r_outlier_v)

	return agent.action_m[cstep, session]
end
