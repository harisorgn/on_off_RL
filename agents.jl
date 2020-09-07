
abstract type abstract_agent end

abstract type abstract_policy end

struct offline_bias
	n::Float64
	decay::Float64
	b_m::Array{Float64}		# offline timesteps x actions x sessions
	δr_v::Array{Float64, 1}
	surprise_v::Array{Float64, 1}
	action_v::Array{Int64, 1}

	offline_bias(n_bias_timesteps, n_actions, n_sessions, buffer_length, n, decay) = new(n, decay, 
																			zeros(n_bias_timesteps + 1, n_actions, n_sessions), 
																			zeros(buffer_length), zeros(buffer_length), zeros(buffer_length))
end

function (bias::offline_bias)(session)

	weight_v = bias.surprise_v ./ sum(bias.surprise_v)
	
	Δr = sum(bias.δr_v)

	T_b = size(bias.b_m)[1] - 1

	i = 2
	
	for s in rand(Categorical(weight_v), T_b)

		action = bias.action_v[s]

		bias.b_m[i, action, session] = bias.b_m[i - 1, action, session] + bias.n * (Δr/T_b - bias.b_m[i - 1, action, session])

		bias.b_m[i, 1:size(bias.b_m)[2] .!= action, session] .= (1.0 - bias.decay) * 
																bias.b_m[i - 1, 1:size(bias.b_m)[2] .!= action, session]

		i += 1
	end

	if session < size(bias.b_m)[3]

		bias.b_m[1, :, session + 1] = bias.b_m[end, :, session]
	end
end

function bias_buffer_update!(bias::offline_bias, current_surprise, current_action, current_δr)

	min_surprise = minimum(bias.surprise_v)

	if current_surprise > min_surprise

		min_idx = findfirst(x -> x == min_surprise, bias.surprise_v)

		bias.surprise_v[min_idx] = current_surprise
		bias.action_v[min_idx] = current_action
		bias.δr_v[min_idx] = current_δr
	end
end

struct softmax_policy <: abstract_policy
	β::Float64
end

(policy::softmax_policy)(r_v) = [exp(policy.β * r_v[i]) / sum(exp.(policy.β * r_v)) for i = 1 : length(r_v)]

struct delta_agent <: abstract_agent
	n::Float64
	decay::Float64
	action_m::Array{Int64}	# timesteps x sessions
	r_m::Array{Float64}		# timesteps x actions x sessions
	accumulated_r_v::Array{Float64, 1}
	bias::offline_bias
	policy::abstract_policy

	delta_agent(n_timesteps, n_actions, n_sessions, n, decay, bias, β) = new(n, decay, zeros(n_timesteps, n_sessions), 
																		zeros(n_timesteps, n_actions, n_sessions), 
																		zeros(n_sessions), bias, softmax_policy(β))
end

function (agent::delta_agent)(r_environment, trial, session)

	agent.accumulated_r_v[session] += r_environment

	latest_action = agent.action_m[trial - 1, session]

	δr = r_environment - agent.r_m[trial - 1, latest_action, session]

	δr_offline = δr #- agent.bias.b_m[1, latest_action, session]

	surprise = abs(δr_offline)

	bias_buffer_update!(agent.bias, surprise, latest_action, δr_offline)

	agent.r_m[trial, latest_action, session] = agent.r_m[trial - 1, latest_action, session] + agent.n * δr

	agent.r_m[trial, 1:size(agent.r_m)[2] .!= latest_action, session] .= (1.0 - agent.decay) * 
																	agent.r_m[trial - 1, 1:size(agent.r_m)[2] .!= latest_action, session]

	p_action = agent.policy(agent.r_m[trial, :, session] .+ agent.bias.b_m[1, :, session])

	agent.action_m[trial, session] = rand(Categorical(p_action))


	if (trial == size(agent.r_m)[1]) && (session < size(agent.r_m)[3])

		agent.r_m[1, :, session + 1] = agent.r_m[trial, :, session]
	end

	return agent.action_m[trial, session]
end

struct probabilistic_delta_agent <: abstract_agent
	n::Float64
	decay::Float64
	μ::Float64
	σ::Float64
	action_m::Array{Int64}	# timesteps x sessions
	r_m::Array{Float64}		# timesteps x actions x sessions
	accumulated_r_v::Array{Float64, 1}
	bias::offline_bias
	policy::abstract_policy

	probabilistic_delta_agent(n_timesteps, n_actions, n_sessions, n, decay, μ, σ, bias, β) = new(n, decay, μ, σ, 
																						zeros(n_timesteps, n_sessions), 
																						zeros(n_timesteps, n_actions, n_sessions), 
																						zeros(n_sessions), bias, softmax_policy(β))
end

function (agent::probabilistic_delta_agent)(r_environment, trial, session)

	agent.accumulated_r_v[session] += r_environment

	latest_action = agent.action_m[trial - 1, session]

	δr = r_environment - agent.r_m[trial - 1, latest_action, session]

	δr_offline = δr #- agent.bias.b_m[1, latest_action, session]

	surprise = -logpdf(Normal(agent.μ, agent.σ), r_environment)

	bias_buffer_update!(agent.bias, surprise, latest_action, δr_offline)

	agent.r_m[trial, latest_action, session] = agent.r_m[trial - 1, latest_action, session] + 
												agent.n * pdf(Normal(agent.μ, agent.σ), r_environment) * δr

	agent.r_m[trial, 1:size(agent.r_m)[2] .!= latest_action, session] .= (1.0 - agent.decay) * 
																	agent.r_m[trial - 1, 1:size(agent.r_m)[2] .!= latest_action, session]

	p_action = agent.policy(agent.r_m[trial, :, session] .+ agent.bias.b_m[1, :, session])

	agent.action_m[trial, session] = rand(Categorical(p_action))


	if (trial == size(agent.r_m)[1]) && (session < size(agent.r_m)[3])

		agent.r_m[1, :, session + 1] = agent.r_m[trial, :, session]
	end

	return agent.action_m[trial, session]
end

struct advantage_delta_agent <: abstract_agent
	n_r_state::Float64
	n_r_action::Float64
	decay::Float64
	action_m::Array{Int64}			# timesteps x sessions
	r_state_m::Array{Float64}		# timesteps x sessions
	r_action_m::Array{Float64}		# timesteps x actions x sessions
	accumulated_r_v::Array{Float64, 1}
	bias::offline_bias
	policy::abstract_policy

	advantage_delta_agent(n_timesteps, n_actions, n_sessions, n_r_state, n_r_action, decay, bias, β) = new(n_r_state, n_r_action, decay, 
																							zeros(n_timesteps, n_sessions), 
																							zeros(n_timesteps, n_sessions),
																							zeros(n_timesteps, n_actions, n_sessions), 
																							zeros(n_sessions), bias, softmax_policy(β))
end

function (agent::advantage_delta_agent)(r_environment, trial, session)

	agent.accumulated_r_v[session] += r_environment

	latest_action = agent.action_m[trial - 1, session]

	δr = r_environment - agent.r_action_m[trial - 1, latest_action, session]

	δr_offline = r_environment - agent.r_action_m[trial - 1, latest_action, session] - agent.bias.b_m[1, latest_action, session]

	surprise = abs(δr_offline)

	bias_buffer_update!(agent.bias, surprise, latest_action, δr_offline)

	agent.r_state_m[trial, session] = agent.r_state_m[trial - 1, session] + 
									agent.n_r_state * (r_environment - agent.r_state_m[trial - 1, session])

	agent.r_action_m[trial, latest_action, session] = agent.r_action_m[trial - 1, latest_action, session] + agent.n_r_action * δr

	agent.r_action_m[trial, 1:size(agent.r_m)[2] .!= latest_action, session] .= (1.0 - agent.decay) * 
														agent.r_action_m[trial - 1, 1:size(agent.r_m)[2] .!= latest_action, session]

	p_action = agent.policy(agent.r_m[trial, :, session] .+ agent.bias.b_m[1, :, session])

	agent.action_m[trial, session] = rand(Categorical(p_action))


	if (trial == size(agent.r_m)[1]) && (session < size(agent.r_m)[3])

		agent.r_m[1, :, session + 1] = agent.r_m[trial, :, session]
	end

	return agent.action_m[trial, session]
end
