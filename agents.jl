abstract type abstract_bandit_agent end

abstract type abstract_optimal_bandit_agent end

abstract type  abstract_offline_learning end

abstract type abstract_policy end

mutable struct offline_bias <: abstract_offline_learning
	n_steps::Int64
	n_actions::Int64
	n_sessions::Int64
	buffer_length::Int64
	η::Float64
	decay::Float64
	r_m::Array{Float64}		# offline steps x actions x sessions
	Δr_v::Array{Float64, 1}
	surprise_v::Array{Float64, 1}
	action_v::Array{Int64, 1}
	rng::AbstractRNG

	offline_bias(n_steps, n_actions, n_sessions, buffer_length, η, decay) = new(n_steps,
																				n_actions,
																				n_sessions,
																				buffer_length,
																				η, 
																				decay, 
																				zeros(n_steps + 1, n_actions, n_sessions), 
																				zeros(n_sessions), 
																				zeros(buffer_length), 
																				zeros(buffer_length),
																				MersenneTwister())
end

function (off_bias::offline_bias)(session)

	weight_v = off_bias.surprise_v ./ sum(off_bias.surprise_v)
	
	i = 2
	
	for s in rand(off_bias.rng, Categorical(weight_v), off_bias.n_steps)

		action = off_bias.action_v[s]

		off_bias.r_m[i, action, session] = off_bias.r_m[i - 1, action, session] + 
										off_bias.η * off_bias.Δr_v[session] / off_bias.n_steps

		off_bias.r_m[i, setdiff(1:off_bias.n_actions, action), session] = (1.0 - off_bias.decay) * 
																	off_bias.r_m[i - 1, setdiff(1:off_bias.n_actions, action), session]

		i += 1
	end

	if session < off_bias.n_sessions

		off_bias.r_m[1, :, session + 1] = off_bias.r_m[end, :, session]
		
	end

	reset_offline_buffer!(off_bias)
end

function update_offline_buffer!(off_bias::offline_bias, surprise, action, δr, session)

	off_bias.Δr_v[session] += δr - off_bias.r_m[1, action, session]

	min_surprise = minimum(off_bias.surprise_v)

	if surprise > min_surprise

		min_idx = findfirst(x -> x == min_surprise, off_bias.surprise_v)
 
		off_bias.surprise_v[min_idx] = surprise
		off_bias.action_v[min_idx] = action
	end
end

function reset_offline_buffer!(off_bias::offline_bias)

	off_bias.Δr_v[:] = zeros(off_bias.n_sessions)
	off_bias.surprise_v[:] = zeros(off_bias.buffer_length)
	off_bias.action_v[:] = zeros(off_bias.buffer_length)

end

mutable struct offline_Q <: abstract_offline_learning
	n_steps::Int64
	n_actions::Int64
	n_sessions::Int64
	buffer_length::Int64
	η::Float64
	decay::Float64
	r_m::Array{Float64}		# offline steps x actions x sessions
	surprise_v::Array{Float64, 1}
	action_v::Array{Int64, 1}
	reward_v::Array{Float64, 1}
	rng::AbstractRNG

	offline_Q(n_steps, n_actions, n_sessions, buffer_length, η, decay) = new(n_steps,
																			n_actions,
																			n_sessions,
																			buffer_length,
																			η, 
																			decay, 
																			zeros(n_steps + 1, n_actions, n_sessions), 
																			zeros(buffer_length), 
																			zeros(buffer_length),
																			zeros(buffer_length),
																			MersenneTwister())
end

function (off_Q::offline_Q)(session)

	weight_v = off_Q.surprise_v ./ sum(off_Q.surprise_v)
	
	i = 2
	
	for s in rand(off_Q.rng, Categorical(weight_v), off_Q.n_steps)

		action = off_Q.action_v[s]

		off_Q.r_m[i, action, session] = off_Q.r_m[i - 1, action, session] + 
										off_Q.η * (off_Q.reward_v[s] - off_Q.r_m[i - 1, action, session])

		off_Q.r_m[i, 1:off_Q.n_actions .!= action, session] .= (1.0 - off_Q.decay) * 
																off_Q.r_m[i - 1, 1:off_Q.n_actions .!= action, session]

		i += 1
	end

	reset_offline_buffer!(off_Q)
end

function update_offline_buffer!(off_Q::offline_Q, surprise, action, reward, session)

	min_surprise = minimum(off_Q.surprise_v)

	if surprise > min_surprise

		min_idx = findfirst(x -> x == min_surprise, off_Q.surprise_v)
 
		off_Q.surprise_v[min_idx] = surprise
		off_Q.action_v[min_idx] = action
		off_Q.reward_v[min_idx] = reward

	end
end

function reset_offline_buffer!(off_Q::offline_Q)

	off_Q.surprise_v[:] = zeros(off_Q.buffer_length)
	off_Q.action_v[:] = zeros(off_Q.buffer_length)
	off_Q.reward_v[:] = zeros(off_Q.buffer_length)

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

(policy::ε_greedy_policy)(r_v) = (rand(policy.rng) < (1.0 - policy.ε)) && !all(x->x == r_v[1], r_v) ? argmax(r_v) : rand(policy.rng, 1 : length(r_v))

struct delta_agent{T <: abstract_offline_learning, Y <: abstract_policy} <: abstract_bandit_agent
	n_steps::Int64
	n_actions::Int64
	n_sessions::Int64
	η::Float64
	decay::Float64
	action_m::Array{Int64}	# steps x sessions
	r_m::Array{Float64}		# steps x actions x sessions
	accumulated_r_v::Array{Float64, 1}
	offline::T
	policy::Y

	function delta_agent(n_steps, n_actions, n_sessions, η, decay, offline::T, policy::Y) where {T <: abstract_offline_learning,
																								Y <: abstract_policy}
		return new{T, Y}(n_steps, n_actions, n_sessions, η, decay, 
						zeros(n_steps, n_sessions), 
						zeros(n_steps, n_actions, n_sessions), 
						zeros(n_sessions), 
						offline, 
						policy)
	end
end

function (agent::delta_agent{offline_bias})(r_environment, cstep, session, available_action_v)

	if cstep > 1

		agent.accumulated_r_v[session] += r_environment

		latest_action = agent.action_m[cstep - 1, session]

		δr = r_environment - agent.r_m[cstep - 1, latest_action, session]

		surprise = abs(δr)

		update_offline_buffer!(agent.offline, surprise, latest_action, δr, session)

		agent.r_m[cstep, latest_action, session] = agent.r_m[cstep - 1, latest_action, session] + agent.η * δr

		agent.r_m[cstep, setdiff(1:agent.n_actions, latest_action), session] .= (1.0 - agent.decay) * 
																agent.r_m[cstep - 1, setdiff(1:agent.n_actions, latest_action), session]

		agent.action_m[cstep, session] = available_action_v[agent.policy(agent.r_m[cstep, available_action_v, session] .+ 
																		agent.offline.r_m[1, available_action_v, session])]

	elseif cstep == 1

		agent.r_m[cstep, :, session] = (session == 1) ? zeros(agent.n_actions) : agent.r_m[end, :, session - 1]

		agent.action_m[cstep, session] = available_action_v[agent.policy(agent.r_m[cstep, available_action_v, session] .+ 
																		agent.offline.r_m[1, available_action_v, session])]

	end

	return agent.action_m[cstep, session]
end

function (agent::delta_agent{offline_Q})(r_environment, cstep, session, available_action_v)

	if cstep > 1

		agent.accumulated_r_v[session] += r_environment

		latest_action = agent.action_m[cstep - 1, session]

		δr = r_environment - agent.r_m[cstep - 1, latest_action, session]

		surprise = abs(δr)

		update_offline_buffer!(agent.offline, surprise, latest_action, r_environment, session)

		agent.r_m[cstep, latest_action, session] = agent.r_m[cstep - 1, latest_action, session] + agent.η * δr

		agent.r_m[cstep, setdiff(1:agent.n_actions, latest_action), session] .= (1.0 - agent.decay) * 
																agent.r_m[cstep - 1, setdiff(1:agent.n_actions, latest_action), session]

		agent.action_m[cstep, session] = available_action_v[agent.policy(agent.r_m[cstep, available_action_v, session])]

		agent.offline.r_m[1, :, session] = agent.r_m[cstep, :, session]

	elseif cstep == 1

		agent.r_m[cstep, :, session] = (session == 1) ? zeros(agent.n_actions) : agent.offline.r_m[end, :, session - 1]

		agent.action_m[cstep, session] = available_action_v[agent.policy(agent.r_m[cstep, available_action_v, session])]

	end

	return agent.action_m[cstep, session]
end


initialise_new_instance(agent::delta_agent{offline_bias}, n_steps, n_actions, n_sessions) = delta_agent(n_steps, 
																						n_actions, 
																						n_sessions, 
																						agent.η, 
																						agent.decay, 
																						offline_bias(Int(floor(0.5*n_steps)), 
																									n_actions, 
																									n_sessions, 
																									Int(floor(0.2*n_steps)), 
																									agent.offline.η, 
																									agent.offline.decay), 
																						agent.policy)

initialise_new_instance(agent::delta_agent{offline_Q}, n_steps, n_actions, n_sessions) = delta_agent(n_steps, 
																						n_actions, 
																						n_sessions, 
																						agent.η, 
																						agent.decay, 
																						offline_Q(Int(floor(0.5*n_steps)), 
																								n_actions, 
																								n_sessions, 
																								Int(floor(0.2*n_steps)), 
																								agent.offline.η, 
																								agent.offline.decay), 
																						agent.policy)

#----------------------------------------------------------------------------------------------------------------------------------
#--------------------------Under development---------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------
#=
struct probabilistic_delta_agent{T} <: abstract_bandit_agent where T <: abstract_offline_learning
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
	offline::T
	policy::abstract_policy

	probabilistic_delta_agent(n_steps, n_actions, n_sessions, η, decay, μ, σ, offline, policy) = new(n_steps,
																								n_actions,
																								n_sessions,
																								η, 
																								decay, 
																								μ, 
																								σ, 
																								zeros(n_steps, n_sessions), 
																								zeros(n_steps, n_actions, n_sessions), 
																								zeros(n_sessions), 
																								offline, 
																								policy)

end

function (agent::probabilistic_delta_agent)(r_environment, cstep, session, available_action_v)

	agent.accumulated_r_v[session] += r_environment

	latest_action = agent.action_m[cstep - 1, session]

	δr = r_environment - agent.r_m[cstep - 1, latest_action, session]

	surprise = -logpdf(Normal(agent.μ, agent.σ), r_environment)

	update_offline_buffer!(agent.offline, surprise, latest_action, 
							r_environment, agent.r_m[cstep - 1, latest_action, session],
							session)

	agent.r_m[cstep, latest_action, session] = agent.r_m[cstep - 1, latest_action, session] + 
												agent.η * pdf(Normal(agent.μ, agent.σ), r_environment) * δr

	agent.r_m[cstep, setdiff(1:agent.n_actions, latest_action), session] .= (1.0 - agent.decay) * 
															agent.r_m[cstep - 1, setdiff(1:agent.n_actions, latest_action), session]

	agent.action_m[cstep, session] = available_action_v[agent.policy(agent.r_m[cstep, available_action_v, session] .+ 
																	agent.offline.r_m[1, available_action_v, session])]

	if (cstep == agent.n_steps) && (session < agent.n_sessions)

		agent.r_m[1, :, session + 1] = agent.r_m[cstep, :, session]
	end

	return agent.action_m[cstep, session]
end

initialise_new_instance(agent::probabilistic_delta_agent, n_steps, n_actions, n_sessions) = probabilistic_delta_agent(n_steps, 
																												n_actions, 
																												n_sessions, 
																												agent.η, 
																												agent.decay, 
																												agent.μ, 
																												agent.σ, 
																												initialise_new_instance(agent.bias, 
																													n_steps, n_actions, n_sessions),
																												agent.policy) 

struct OU_agent{T} <: abstract_bandit_agent where T <: abstract_offline_learning
	n_steps::Int64
	n_actions::Int64
	n_sessions::Int64
	η::Float64
	decay::Float64
	γ::Float64
	σ::Float64
	action_m::Array{Int64}				# steps x sessions
	r_m::Array{Float64}					# steps x actions x sessions
	accumulated_r_v::Array{Float64, 1}
	offline::T
	policy::abstract_policy

	OU_agent(n_steps, n_actions, n_sessions, η, decay, γ, σ, offline, policy) = new(n_steps,
																				n_actions,
																				n_sessions,
																				η, 
																				decay, 
																				γ, 
																				σ, 
																				zeros(n_steps, n_sessions), 
																				zeros(n_steps, n_actions, n_sessions), 
																				zeros(n_sessions), 
																				offline, 
																				policy)

end

function (agent::OU_agent)(r_environment, cstep, session, available_action_v)

	agent.accumulated_r_v[session] += r_environment

	latest_action = agent.action_m[cstep - 1, session]

	δr = r_environment - agent.r_m[cstep - 1, latest_action, session]

	surprise = -logpdf(Normal((1.0 - agent.γ) * agent.r_m[cstep - 1, latest_action, session], agent.σ), r_environment)

	update_offline_buffer!(agent.offline, surprise, latest_action, 
							r_environment, agent.r_m[cstep - 1, latest_action, session],
							session)

	agent.r_m[cstep, latest_action, session] = agent.r_m[cstep - 1, latest_action, session] + 
												agent.η * 
												pdf(Normal((1.0 - agent.γ) * agent.r_m[cstep - 1, latest_action, session], agent.σ),
													r_environment)
												δr

	agent.r_m[cstep, setdiff(1:agent.n_actions, latest_action), session] .= (1.0 - agent.decay) * 
															agent.r_m[cstep - 1, setdiff(1:agent.n_actions, latest_action), session]

	agent.action_m[cstep, session] = available_action_v[agent.policy(agent.r_m[cstep, available_action_v, session] .+ 
																	agent.offline.r_m[1, available_action_v, session])]

	if (cstep == agent.n_steps) && (session < agent.n_sessions)

		agent.r_m[1, :, session + 1] = agent.r_m[cstep, :, session]
	end

	return agent.action_m[cstep, session]
end

initialise_new_instance(agent::OU_agent, n_steps, n_actions, n_sessions) = OU_agent(n_steps, 
																					n_actions, 
																					n_sessions, 
																					agent.η, 
																					agent.decay, 
																					agent.γ, 
																					agent.σ, 
																					initialise_new_instance(agent.bias, 
																							n_steps, n_actions, n_sessions),
																					agent.policy) 

struct advantage_delta_agent <: abstract_bandit_agent where T <: abstract_offline_learning
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
	offline::T
	policy::abstract_policy

	advantage_delta_agent(n_steps, n_actions, n_sessions, 
						η_r_state, η_r_action, decay, offline, policy) = new(n_steps, n_actions, n_sessions,
																		η_r_state, η_r_action, decay, 
																		zeros(n_steps, n_sessions), 
																		zeros(n_steps, n_sessions),
																		zeros(n_steps, n_actions, n_sessions), 
																		zeros(n_sessions), 
																		offline, 
																		policy)
end

function (agent::advantage_delta_agent)(r_environment, cstep, session, available_action_v)

	agent.accumulated_r_v[session] += r_environment

	latest_action = agent.action_m[cstep - 1, session]

	δr = r_environment - agent.r_action_m[cstep - 1, latest_action, session]

	surprise = abs(δr_offline)

	update_offline_buffer!(agent.offline, surprise, latest_action, 
							r_environment, agent.r_m[cstep - 1, latest_action, session],
							session)

	agent.r_state_m[cstep, session] = agent.r_state_m[cstep - 1, session] + 
									agent.η_r_state * (r_environment - agent.r_state_m[cstep - 1, session])

	agent.r_action_m[cstep, latest_action, session] = agent.r_action_m[cstep - 1, latest_action, session] + agent.η_r_action * δr

	agent.r_action_m[cstep, setdiff(1:agent.n_actions, latest_action), session] .= (1.0 - agent.decay) * 
															agent.r_action_m[cstep - 1, setdiff(1:agent.n_actions, latest_action), session]

	agent.action_m[cstep, session] = available_action_v[agent.policy(agent.r_m[cstep, available_action_v, session] .+ 
														agent.offline.r_m[1, available_action_v, session])]

	if (cstep == agent.n_steps) && (session < agent.n_sessions)

		agent.r_m[1, :, session + 1] = agent.r_m[cstep, :, session]

	end

	return agent.action_m[cstep, session]
end
=#
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

	if cstep > 1

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

	elseif cstep == 1

		r_outlier_v = zeros(length(next_r_environment_v))

		r_outlier_v[1] += (agent.p_outlier_m[cstep, session] + 
							agent.η_p_outlier * (agent.p_outlier_max - agent.p_outlier_m[cstep, session])) * agent.r_outlier

		agent.action_m[cstep, session] = available_action_v[agent.policy(next_r_environment_v .+ r_outlier_v)]

	end

	return agent.action_m[cstep, session]
end

function reset_agent!(agent::abstract_bandit_agent)

	agent.accumulated_r_v[:] = zeros(length(agent.accumulated_r_v))
	agent.action_m[:] = zeros(Int64, agent.n_steps, agent.n_sessions)
	agent.r_m[:] = zeros(agent.n_steps, agent.n_actions, agent.n_sessions)

	agent.offline.r_m[:] = zeros(agent.offline.n_steps + 1, agent.n_actions, agent.n_sessions)
end
