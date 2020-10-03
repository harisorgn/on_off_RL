abstract type abstract_bandit_agent end

abstract type abstract_optimal_bandit_agent end

abstract type abstract_policy end

mutable struct offline_bias 
	n_steps::Int64
	n_sessions::Int64
	buffer_length::Int64
	η::Float64
	decay::Float64
	b_m::Array{Float64}		# offline steps x actions x sessions
	Δr::Float64
	surprise_v::Array{Float64, 1}
	action_v::Array{Int64, 1}
	rng::AbstractRNG

	offline_bias(n_steps, n_actions, n_sessions, buffer_length, η, decay) = new(n_steps,
																				n_sessions,
																				buffer_length,
																				η, 
																				decay, 
																				zeros(n_steps + 1, n_actions, n_sessions), 
																				0.0, 
																				zeros(buffer_length), 
																				zeros(buffer_length),
																				MersenneTwister())
end

function (bias::offline_bias)(session)

	weight_v = bias.surprise_v ./ sum(bias.surprise_v)
	
	i = 2
	
	for s in rand(bias.rng, Categorical(weight_v), bias.n_steps)

		action = bias.action_v[s]

		bias.b_m[i, action, session] = bias.b_m[i - 1, action, session] + bias.η * bias.Δr/bias.n_steps

		bias.b_m[i, 1:size(bias.b_m)[2] .!= action, session] .= (1.0 - bias.decay) * 
																bias.b_m[i - 1, 1:size(bias.b_m)[2] .!= action, session]

		i += 1
	end

	if session < bias.n_sessions

		bias.b_m[1, :, session + 1] = bias.b_m[end, :, session]
		
	end

	reset_offline_buffer!(bias)
end

function update_offline_buffer!(bias::offline_bias, current_surprise, current_action)

	min_surprise = minimum(bias.surprise_v)

	if current_surprise > min_surprise

		min_idx = findfirst(x -> x == min_surprise, bias.surprise_v)

		bias.surprise_v[min_idx] = current_surprise
		bias.action_v[min_idx] = current_action
	end
end

function reset_offline_buffer!(bias::offline_bias)

	bias.Δr = 0.0
	bias.surprise_v[:] = zeros(bias.buffer_length)
	bias.action_v[:] = zeros(bias.buffer_length)

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
	n_steps::Int64
	n_actions::Int64
	n_sessions::Int64
	η::Float64
	decay::Float64
	action_m::Array{Int64}	# steps x sessions
	r_m::Array{Float64}		# steps x actions x sessions
	accumulated_r_v::Array{Float64, 1}
	bias::offline_bias
	policy::abstract_policy

	delta_agent(n_steps, n_actions, n_sessions, η, decay, bias, policy) = new(n_steps,
																			n_actions,
																			n_sessions,
																			η, 
																			decay, 
																			zeros(n_steps, n_sessions), 
																			zeros(n_steps, n_actions, n_sessions), 
																			zeros(n_sessions), bias, policy)
end

function (agent::delta_agent)(r_environment, cstep, session, available_action_v)

	agent.accumulated_r_v[session] += r_environment

	latest_action = agent.action_m[cstep - 1, session]

	δr = r_environment - agent.r_m[cstep - 1, latest_action, session]

	agent.bias.Δr += δr - agent.bias.b_m[1, latest_action, session]

	surprise = abs(δr)

	update_offline_buffer!(agent.bias, surprise, latest_action)

	agent.r_m[cstep, latest_action, session] = agent.r_m[cstep - 1, latest_action, session] + agent.η * δr

	agent.r_m[cstep, setdiff(1:agent.n_actions, latest_action), session] .= (1.0 - agent.decay) * 
															agent.r_m[cstep - 1, setdiff(1:agent.n_actions, latest_action), session]

	agent.action_m[cstep, session] = available_action_v[agent.policy(agent.r_m[cstep, available_action_v, session] .+ 
																	agent.bias.b_m[1, available_action_v, session])]

	if (cstep == agent.n_steps) && (session < agent.n_sessions)

		agent.r_m[1, :, session + 1] = agent.r_m[cstep, :, session]
	end

	return agent.action_m[cstep, session]
end

struct probabilistic_delta_agent <: abstract_bandit_agent
	n_steps::Int64
	n_actions::Int64
	n_sessions::Int64
	η::Float64
	decay::Float64
	μ::Float64
	σ::Float64
	action_m::Array{Int64}	# steps x sessions
	r_m::Array{Float64}		# steps x actions x sessions
	accumulated_r_v::Array{Float64, 1}
	bias::offline_bias
	policy::abstract_policy

	probabilistic_delta_agent(n_steps, n_actions, n_sessions, η, decay, μ, σ, bias, policy) = new(n_steps,
																								n_actions,
																								n_sessions,
																								η, 
																								decay, 
																								μ, 
																								σ, 
																								zeros(n_steps, n_sessions), 
																								zeros(n_steps, n_actions, n_sessions), 
																								zeros(n_sessions), 
																								bias, 
																								policy)
end

function (agent::probabilistic_delta_agent)(r_environment, cstep, session, available_action_v)

	agent.accumulated_r_v[session] += r_environment

	latest_action = agent.action_m[cstep - 1, session]

	δr = r_environment - agent.r_m[cstep - 1, latest_action, session]

	agent.bias.Δr += δr - agent.bias.b_m[1, latest_action, session]

	surprise = -logpdf(Normal(agent.μ, agent.σ), r_environment)

	update_offline_buffer!(agent.bias, surprise, latest_action)

	agent.r_m[cstep, latest_action, session] = agent.r_m[cstep - 1, latest_action, session] + 
												agent.η * pdf(Normal(agent.μ, agent.σ), r_environment) * δr

	agent.r_m[cstep, setdiff(1:agent.n_actions, latest_action), session] .= (1.0 - agent.decay) * 
															agent.r_m[cstep - 1, setdiff(1:agent.n_actions, latest_action), session]

	agent.action_m[cstep, session] = available_action_v[agent.policy(agent.r_m[cstep, available_action_v, session] .+ 
																	agent.bias.b_m[1, available_action_v, session])]

	if (cstep == agent.n_steps) && (session < agent.n_sessions)

		agent.r_m[1, :, session + 1] = agent.r_m[cstep, :, session]
	end

	return agent.action_m[cstep, session]
end

#----------------------------------------------------------------------------------------------------------------------------------
#--------------------------Under development---------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------

struct advantage_delta_agent <: abstract_bandit_agent
	n_steps::Int64
	n_actions::Int64
	n_sessions::Int64
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
						η_r_state, η_r_action, decay, bias, policy) = new(n_steps, n_actions, n_sessions,
																		η_r_state, η_r_action, decay, 
																		zeros(n_steps, n_sessions), 
																		zeros(n_steps, n_sessions),
																		zeros(n_steps, n_actions, n_sessions), 
																		zeros(n_sessions), bias, policy)
end

function (agent::advantage_delta_agent)(r_environment, cstep, session, available_action_v)

	agent.accumulated_r_v[session] += r_environment

	latest_action = agent.action_m[cstep - 1, session]

	δr = r_environment - agent.r_action_m[cstep - 1, latest_action, session]

	δr_offline = δr - agent.bias.b_m[1, latest_action, session]

	agent.bias.Δr += δr_offline

	surprise = abs(δr_offline)

	update_offline_buffer!(agent.bias, surprise, latest_action)

	agent.r_state_m[cstep, session] = agent.r_state_m[cstep - 1, session] + 
									agent.η_r_state * (r_environment - agent.r_state_m[cstep - 1, session])

	agent.r_action_m[cstep, latest_action, session] = agent.r_action_m[cstep - 1, latest_action, session] + agent.η_r_action * δr

	agent.r_action_m[cstep, setdiff(1:agent.n_actions, latest_action), session] .= (1.0 - agent.decay) * 
															agent.r_action_m[cstep - 1, setdiff(1:agent.n_actions, latest_action), session]

	agent.action_m[cstep, session] = available_action_v[agent.policy(agent.r_m[cstep, available_action_v, session] .+ 
														agent.bias.b_m[1, available_action_v, session])]

	if (cstep == agent.n_steps) && (session < agent.n_sessions)

		agent.r_m[1, :, session + 1] = agent.r_m[cstep, :, session]

	end

	return agent.action_m[cstep, session]
end
#----------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------

struct optimal_bandit_distribution_outlier_agent <: abstract_optimal_bandit_agent
	action_m::Array{Int64}
	accumulated_r_v::Array{Float64, 1}
	policy::abstract_policy

	optimal_bandit_distribution_outlier_agent(n_steps, n_sessions) = new(zeros(n_steps, n_sessions), 
																		zeros(n_sessions), 
																		ε_greedy_policy(0.0))
end

function (agent::optimal_bandit_distribution_outlier_agent)(r_environment, next_r_environment_v, cstep, session, available_action_v)

	agent.accumulated_r_v[session] += r_environment

	agent.action_m[cstep, session] = available_action_v[agent.policy(next_r_environment_v)]

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

function (agent::optimal_bandit_frequency_outlier_agent)(r_environment, next_r_environment_v, cstep, session, available_action_v)

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

	agent.action_m[cstep, session] = available_action_v[agent.policy(next_r_environment_v .+ r_outlier_v)]

	return agent.action_m[cstep, session]
end

function reset_agent!(agent::Union{abstract_bandit_agent, abstract_optimal_bandit_agent})

	agent.accumulated_r_v[:] = zeros(length(agent.accumulated_r_v))

end
