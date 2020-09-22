abstract type abstract_environment end
abstract type OU_bandit_environment <: abstract_environment end

struct OU_bandit_distribution_outlier_environment <:OU_bandit_environment
	γ_v::Array{Float64, 1}
	μ_v::Array{Float64, 1}
	σ_v::Array{Float64, 1}
	outlier_distr_v::Array{Distribution, 1}
	r_OU_m::Array{Float64}				# steps x actions x sessions
	r_m::Array{Float64}					# steps x actions x sessions
	rng::AbstractRNG

	function OU_bandit_distribution_outlier_environment(n_steps, n_sessions, γ_v, μ_v, σ_v, outlier_distr_v)

		rng = MersenneTwister()

		n_actions = length(μ_v)

		r_OU_m = zeros(n_steps, n_actions, n_sessions)
		r_m = zeros(n_steps, n_actions, n_sessions)

		for i = 1 : n_sessions
			for j = 2 : n_steps

				r_OU_m[j, :, i] = r_OU_m[j - 1, :, i] .+ γ_v .* (μ_v .- r_OU_m[j - 1, :, i]) .+ 
								σ_v .* rand(rng, Normal(0, 1), n_actions)

				r_m[j, :, i] = r_OU_m[j, :, i] .+ [rand(rng, outlier_distr) for outlier_distr in outlier_distr_v]

			end
		end

		new(γ_v, μ_v, σ_v, outlier_distr_v, r_OU_m, r_m, rng)
	end
end

(env::OU_bandit_distribution_outlier_environment)(action, cstep, session) = env.r_m[cstep, action, session]

struct OU_bandit_frequency_outlier_environment <:OU_bandit_environment
	γ_v::Array{Float64, 1}
	μ_v::Array{Float64, 1}
	σ_v::Array{Float64, 1}
	r_outlier::Float64
	p_outlier_max::Float64
	η_p_outlier::Float64
	decay_p_outlier::Float64
	p_outlier_m::Array{Float64}
	r_OU_m::Array{Float64}				# steps x actions x sessions
	r_m::Array{Float64}					# steps x actions x sessions
	rng::AbstractRNG

	function OU_bandit_frequency_outlier_environment(n_steps, n_sessions, γ_v, μ_v, σ_v, 
													r_outlier, p_outlier_max, η_p_outlier, decay_p_outlier)

		rng = MersenneTwister()

		n_actions = length(μ_v)

		r_OU_m = zeros(n_steps, n_actions, n_sessions)
		r_m = zeros(n_steps, n_actions, n_sessions)

		for i = 1 : n_sessions
			for j = 2 : n_steps

				r_OU_m[j, :, i] = r_OU_m[j - 1, :, i] .+ γ_v .* (μ_v .- r_OU_m[j - 1, :, i]) .+ 
								σ_v .* rand(rng, Normal(0, 1), n_actions)

				r_m[j, :, i] = r_OU_m[j, :, i]

			end
		end

		new(γ_v, μ_v, σ_v, r_outlier, p_outlier_max, η_p_outlier, decay_p_outlier, 
			zeros(n_steps + 1, n_sessions), r_OU_m, r_m, rng)
	end
end

function (env::OU_bandit_frequency_outlier_environment)(action, cstep, session)
	
	if (action == 1)
		 env.p_outlier_m[cstep + 1, session] = env.p_outlier_m[cstep, session] + 
												env.η_p_outlier * (env.p_outlier_max - env.p_outlier_m[cstep, session])
	else
		env.p_outlier_m[cstep + 1, session] = (1.0 - env.decay_p_outlier) * env.p_outlier_m[cstep, session]
	end

	return rand(env.rng) < env.p_outlier_m[cstep + 1, session] ? env.r_m[cstep, session] + env.r_outlier : env.r_m[cstep, session]
end


function run_environment!(env::OU_bandit_environment, agent::abstract_bandit_agent)

	(n_steps, n_actions, n_sessions) = size(env.r_m)

	for session = 1 : n_sessions

		agent.action_m[1, session] = agent.policy(agent.r_m[1, :, session] .+ agent.bias.b_m[1, :, session])

		r = env(agent.action_m[1, session], 1, session)

		for cstep = 2 : n_steps

			action = agent(r, cstep, session)

			r = env(action, cstep, session)
		end

		agent.bias(session)
	end	
end

function run_environment!(env::OU_bandit_environment, agent::abstract_optimal_bandit_agent)

	(n_steps, n_actions, n_sessions) = size(env.r_m)

	for session = 1 : n_sessions
		
		agent.action_m[1, session] = agent.policy(env.r_m[1, :, session])

		r = env(agent.action_m[1, session], 1, session)

		for cstep = 2 : n_steps

			action = agent(r, env.r_m[cstep, :, session], cstep, session)

			r = env(action, cstep, session)
		end
	end	
end