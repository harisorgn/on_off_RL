abstract type abstract_environment end
abstract type abstract_bandit_environment <: abstract_environment end

struct OU_bandit_environment <:abstract_bandit_environment
	n_steps::Int64
	n_bandits::Int64
	n_sessions::Int64
	γ_v::Array{Float64, 1}
	μ_v::Array{Float64, 1}
	σ_v::Array{Float64, 1}
	r_m::Array{Float64}						# steps x bandits x sessions
	r_outlier_m::Array{Float64}				# steps x bandits x sessions
	rng::AbstractRNG

	function OU_bandit_environment(n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v)

		rng = MersenneTwister()

		n_bandits = length(μ_v)

		r_m = zeros(n_steps, n_bandits, n_sessions)
		r_outlier_m = zeros(n_steps, n_bandits, n_sessions)

		for session = 1 : n_sessions

			r_m[1, :, session] = r_0_v

			for cstep = 2 : n_steps

				r_m[cstep, :, session] = r_m[cstep - 1, :, session] .+ γ_v .* (μ_v .- r_m[cstep - 1, :, session]) .+ 
										σ_v .* rand(rng, Normal(0, 1), n_bandits)

			end
		end

		new(n_steps, n_bandits, n_sessions, γ_v, μ_v, σ_v, r_m, r_outlier_m, rng)
	end
end

(env::OU_bandit_environment)(action, cstep, session) = env.r_m[cstep, action, session]

initialise_new_instance(env::OU_bandit_environment) = OU_bandit_environment(env.n_steps, 
																			env.n_sessions, 
																			env.r_m[1, :, 1], 
																			env.γ_v, 
																			env.μ_v, 
																			env.σ_v)

struct OU_bandit_distribution_outlier_environment <:abstract_bandit_environment
	n_steps::Int64
	n_bandits::Int64
	n_sessions::Int64
	γ_v::Array{Float64, 1}
	μ_v::Array{Float64, 1}
	σ_v::Array{Float64, 1}
	outlier_distr_v::Array{Distribution, 1}
	r_m::Array{Float64}						# steps x bandits x sessions
	r_outlier_m::Array{Float64}				# steps x bandits x sessions
	rng::AbstractRNG

	function OU_bandit_distribution_outlier_environment(n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v, outlier_distr_v)

		rng = MersenneTwister()

		n_bandits = length(μ_v)

		r_m = zeros(n_steps, n_bandits, n_sessions)
		r_outlier_m = zeros(n_steps, n_bandits, n_sessions)

		for session = 1 : n_sessions

			r_m[1, :, session] = r_0_v

			r_outlier_m[1, :, session] = [rand(rng, outlier_distr) for outlier_distr in outlier_distr_v]

			for cstep = 2 : n_steps

				r_m[cstep, :, session] = r_m[cstep - 1, :, session] .+ γ_v .* (μ_v .- r_m[cstep - 1, :, session]) .+ 
										σ_v .* rand(rng, Normal(0, 1), n_bandits)

				r_outlier_m[cstep, :, session] = [rand(rng, outlier_distr) for outlier_distr in outlier_distr_v]

			end
		end


		new(n_steps, n_bandits, n_sessions, γ_v, μ_v, σ_v, outlier_distr_v, r_m, r_outlier_m, rng)
	end
end

(env::OU_bandit_distribution_outlier_environment)(action, cstep, session) = env.r_m[cstep, action, session] + 
																			env.r_outlier_m[cstep, action, session]

initialise_new_instance(env::OU_bandit_distribution_outlier_environment) = OU_bandit_distribution_outlier_environment(env.n_steps, 
																													env.n_sessions, 
																													env.r_m[1, :, 1], 
																													env.γ_v, 
																													env.μ_v, 
																													env.σ_v, 
																													env.outlier_distr_v)

struct OU_bandit_frequency_outlier_environment <:abstract_bandit_environment
	n_steps::Int64
	n_bandits::Int64
	n_sessions::Int64
	γ_v::Array{Float64, 1}
	μ_v::Array{Float64, 1}
	σ_v::Array{Float64, 1}
	r_outlier::Float64
	p_outlier_max::Float64
	η_p_outlier::Float64
	decay_p_outlier::Float64
	p_outlier_m::Array{Float64}			# steps x sessions
	r_m::Array{Float64}					# steps x bandits x sessions
	r_outlier_m::Array{Float64}			# steps x bandits x sessions
	rng::AbstractRNG

	function OU_bandit_frequency_outlier_environment(n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v, 
													r_outlier, p_outlier_max, η_p_outlier, decay_p_outlier)

		rng = MersenneTwister()

		n_bandits = length(μ_v)

		r_m = zeros(n_steps, n_bandits, n_sessions)

		for session = 1 : n_sessions

			r_m[1, :, session] = r_0_v

			for cstep = 2 : n_steps

				r_m[cstep, :, session] = r_m[cstep - 1, :, session] .+ γ_v .* (μ_v .- r_m[cstep - 1, :, session]) .+ 
										σ_v .* rand(rng, Normal(0, 1), n_bandits)

			end
		end

		new(n_steps, n_bandits, n_sessions, γ_v, μ_v, σ_v, 
			r_outlier, p_outlier_max, η_p_outlier, decay_p_outlier, zeros(n_steps + 1, n_sessions), 
			r_m, zeros(n_steps, n_bandits, n_sessions), rng)
	end
end

function (env::OU_bandit_frequency_outlier_environment)(action, cstep, session)
	
	if (action == 1)
		 env.p_outlier_m[cstep + 1, session] = env.p_outlier_m[cstep, session] + 
												env.η_p_outlier * (env.p_outlier_max - env.p_outlier_m[cstep, session])
	else
		env.p_outlier_m[cstep + 1, session] = (1.0 - env.decay_p_outlier) * env.p_outlier_m[cstep, session]
	end

	if rand(env.rng) < env.p_outlier_m[cstep + 1, session]

		env.r_outlier_m[cstep, :, session] .= env.r_outlier 

	end

	return env.r_m[cstep, action, session] + env.r_outlier_m[cstep, action, session]
end

initialise_new_instance(env::OU_bandit_frequency_outlier_environment) = OU_bandit_frequency_outlier_environment(env.n_steps, 
																											env.n_sessions, 
																											env.r_m[1, :, 1], 
																											env.γ_v, 
																											env.μ_v, 
																											env.σ_v, 
																											env.r_outlier, 
																											env.p_outlier_max, 
																											env.η_p_outlier, 
																											env.decay_p_outlier)

struct OU_bandit_delay_outlier_environment <:abstract_bandit_environment
	n_steps::Int64
	n_bandits::Int64
	n_sessions::Int64
	γ_v::Array{Float64, 1}
	μ_v::Array{Float64, 1}
	σ_v::Array{Float64, 1}
	r_outlier::Float64
	outlier_delay_distr::Distribution
	r_m::Array{Float64}					# steps x bandits x sessions
	r_outlier_m::Array{Float64}			# steps x bandits x sessions
	rng::AbstractRNG

	function OU_bandit_delay_outlier_environment(n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v, 
												r_outlier, outlier_delay_distr)

		rng = MersenneTwister()

		n_bandits = length(μ_v)

		r_m = zeros(n_steps, n_bandits, n_sessions)

		for session = 1 : n_sessions

			r_m[1, :, session] = r_0_v

			for cstep = 2 : n_steps

				r_m[cstep, :, session] = r_m[cstep - 1, :, session] .+ γ_v .* (μ_v .- r_m[cstep - 1, :, session]) .+ 
										σ_v .* rand(rng, Normal(0, 1), n_bandits)

			end
		end

		new(n_steps, n_bandits, n_sessions, γ_v, μ_v, σ_v, 
			r_outlier, outlier_delay_distr, r_m, zeros(n_steps, n_bandits, n_sessions), rng)
	end
end

function (env::OU_bandit_delay_outlier_environment)(action, cstep, session)
	
	if action == 1

		delayed_step_outlier = rand(env.rng, env.outlier_delay_distr) + 1 	# when env.outlier_delay_distr is Geometric 
																			# number of failures of outlier occurence is being sampled 
																			# so +1 is required

		if cstep + delayed_step_outlier <= env.n_steps
			env.r_outlier_m[cstep + delayed_step_outlier, :, session] .= env.r_outlier
		end
	end
	
	return env.r_m[cstep, action, session] + env.r_outlier_m[cstep, action, session]
end

initialise_new_instance(env::OU_bandit_delay_outlier_environment) = OU_bandit_delay_outlier_environment(env.n_steps, 
																										env.n_sessions, 
																										env.r_m[1, :, 1], 
																										env.γ_v, 
																										env.μ_v, 
																										env.σ_v, 
																										env.r_outlier, 
																										env.outlier_delay_distr)

struct OU_bandit_proportional_outlier_environment <:abstract_bandit_environment
	n_steps::Int64
	n_bandits::Int64
	n_sessions::Int64
	γ_v::Array{Float64, 1}
	μ_v::Array{Float64, 1}
	σ_v::Array{Float64, 1}
	r_m::Array{Float64}						# steps x bandits x sessions
	r_outlier_m::Array{Float64}				# steps x bandits x sessions
	rng::AbstractRNG

	function OU_bandit_proportional_outlier_environment(n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v, r_outlier_v)

		rng = MersenneTwister()

		n_bandits = length(μ_v)

		r_m = zeros(n_steps, n_bandits, n_sessions)
		r_outlier_m = zeros(n_steps, n_bandits, n_sessions)

		for session = 1 : n_sessions

			r_m[1, :, session] = r_0_v

			for cstep = 2 : n_steps

				r_m[cstep, :, session] = r_m[cstep - 1, :, session] .+ γ_v .* (μ_v .- r_m[cstep - 1, :, session]) .+ 
										σ_v .* rand(rng, Normal(0, 1), n_bandits)

			end
		end

		new(n_steps, n_bandits, n_sessions, γ_v, μ_v, σ_v, r_m, r_outlier_m, rng)
	end
end

(env::OU_bandit_proportional_outlier_environment)(action, cstep, session) = env.r_m[cstep, action, session]

struct OU_bandit_test_environment <:abstract_bandit_environment
	n_steps::Int64
	n_bandits::Int64
	n_sessions::Int64
	γ_v::Array{Float64, 1}
	μ_v::Array{Float64, 1}
	σ_v::Array{Float64, 1}
	r_outlier::Float64
	r_m::Array{Float64}					# steps x bandits x sessions
	r_outlier_m::Array{Float64}			# steps x bandits x sessions
	rng::AbstractRNG

	function OU_bandit_test_environment(n_steps, n_sessions, r_0_v, γ_v, μ_v, σ_v, r_outlier)

		rng = MersenneTwister()

		n_bandits = length(μ_v)

		r_m = zeros(n_steps, n_bandits, n_sessions)

		for session = 1 : n_sessions

			r_m[1, :, session] = r_0_v

			for cstep = 2 : n_steps

				r_m[cstep, :, session] = r_m[cstep - 1, :, session] .+ γ_v .* (μ_v .- r_m[cstep - 1, :, session]) .+ 
										σ_v .* rand(rng, Normal(0, 1), n_bandits)

			end
		end

		new(n_steps, n_bandits, n_sessions, γ_v, μ_v, σ_v, r_outlier, r_m, zeros(n_steps, n_bandits, n_sessions), rng)
	end
end

function (env::OU_bandit_test_environment)(action, cstep, session)
	
	if cstep == 2 
		env.r_outlier_m[cstep, action, session] = env.r_outlier
	end
	
	return env.r_m[cstep, action, session] + env.r_outlier_m[cstep, action, session]
end

function reset_environment!(env::abstract_bandit_environment)

	env.r_outlier_m[:] = zeros(env.n_steps, env.n_bandits, env.n_sessions)
end

function run_environment!(env::Union{OU_bandit_environment, OU_bandit_distribution_outlier_environment}, agent::abstract_bandit_agent)

	for session = 1 : env.n_sessions

		agent.action_m[1, session] = agent.policy(agent.r_m[1, :, session] .+ agent.bias.b_m[1, :, session])

		r = env(agent.action_m[1, session], 1, session)

		for cstep = 2 : env.n_steps

			action = agent(r, cstep, session, 1:env.n_bandits)

			r = env(action, cstep, session)
		end

		agent.bias(session)
	end	
end

function run_environment!(env::Union{OU_bandit_frequency_outlier_environment, OU_bandit_delay_outlier_environment, OU_bandit_test_environment}, 
						agent::abstract_bandit_agent)

	for session = 1 : env.n_sessions

		agent.action_m[1, session] = agent.policy(agent.r_m[1, :, session] .+ agent.bias.b_m[1, :, session])
		
		r = env(agent.action_m[1, session], 1, session)

		for cstep = 2 : env.n_steps

			action = agent(r, cstep, session, 1:env.n_bandits)
			
			r = env(action, cstep, session)

		end

		agent.bias(session)
	end	

	reset_environment!(env)
end

function run_environment!(env::Union{OU_bandit_environment, OU_bandit_distribution_outlier_environment}, 
						agent::abstract_optimal_bandit_agent)

	for session = 1 : env.n_sessions
		
		agent.action_m[1, session] = agent.policy(env.r_m[1, :, session])

		r = env(agent.action_m[1, session], 1, session)

		for cstep = 2 : env.n_steps

			action = agent(r, env.r_m[cstep, :, session] + env.r_outlier_m[cstep, :, session], cstep, session, 1:env.n_bandits)

			r = env(action, cstep, session)
		end
	end	
end

function run_environment!(env::Union{OU_bandit_frequency_outlier_environment, OU_bandit_delay_outlier_environment, OU_bandit_test_environment}, 
						agent::abstract_optimal_bandit_agent)

	for session = 1 : env.n_sessions
		
		agent.action_m[1, session] = agent.policy(env.r_m[1, :, session] + env.r_outlier_m[1, :, session])

		r = env(agent.action_m[1, session], 1, session)

		for cstep = 2 : env.n_steps

			action = agent(r, env.r_m[cstep, :, session] + env.r_outlier_m[cstep, :, session], cstep, session, 1:env.n_bandits)

			r = env(action, cstep, session)

		end
	end	

	reset_environment!(env)
end
