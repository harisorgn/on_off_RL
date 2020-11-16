abstract type abstract_bandit_environment end
abstract type abstract_reward_process end
abstract type abstract_out_process end

struct bandit_environment{T <: abstract_reward_process, Y <: abstract_out_process} <: abstract_bandit_environment

	reward_process::T
	out_process::Y
	rng::AbstractRNG
end

function (env::bandit_environment)(action, cstep, session)

	update_out_process!(env.out_process, action, cstep, session, env.rng)

	return env.reward_process.r_m[cstep, action, session] + env.out_process.r_m[cstep, action, session]
end

struct OU_process <:abstract_reward_process

	n_warmup_steps::Int64
	n_steps::Int64
	n_bandits::Int64
	n_sessions::Int64
	γ_v::Array{Float64, 1}
	μ_v::Array{Float64, 1}
	σ_v::Array{Float64, 1}
	r_m::Array{Float64}						# steps x bandits x sessions

	function OU_process(n_warmup_steps, n_steps, n_bandits, n_sessions, r_0_v, γ_v, μ_v, σ_v)

		rng = MersenneTwister()

		r_m = zeros(n_steps, n_bandits, n_sessions)

		for session = 1 : n_sessions

			r_v = r_0_v

			for warmup_step = 1 : n_warmup_steps

				r_v = r_v + γ_v .* (μ_v - r_v) + σ_v .* rand(rng, Normal(0, 1), n_bandits)

			end

			r_m[1, :, session] = r_v

			for cstep = 2 : n_steps

				r_m[cstep, :, session] = r_m[cstep - 1, :, session] + γ_v .* (μ_v .- r_m[cstep - 1, :, session]) + 
										σ_v .* rand(rng, Normal(0, 1), n_bandits)

			end
		end

		new(n_warmup_steps, n_steps, n_bandits, n_sessions, γ_v, μ_v, σ_v, r_m)
	end
end

struct no_out_process <: abstract_out_process

	r_m::Array{Float64}						# steps x bandits x sessions

	no_out_process(n_steps, n_bandits, n_sessions) = new(zeros(n_steps, n_bandits, n_sessions))
end

function update_out_process!(process::no_out_process, action, cstep, session, rng)
end

struct distribution_out_process <:abstract_out_process
	
	outlier_distr_v::Array{Distribution, 1}
	r_m::Array{Float64}						# steps x bandits x sessions

	function distribution_out_process(n_steps, n_bandits, n_sessions, outlier_distr_v)

		rng = MersenneTwister()

		r_m = zeros(n_steps, n_bandits, n_sessions)

		for session = 1 : n_sessions
			for cstep = 1 : n_steps

				r_m[cstep, :, session] = [rand(rng, outlier_distr) for outlier_distr in outlier_distr_v]

			end
		end


		new(outlier_distr_v, r_m)
	end
end

function update_out_process!(process::distribution_out_process, action, cstep, session, rng)
end

struct frequency_out_process <: abstract_out_process

	r::Float64
	p_max::Float64
	η_p::Float64
	decay_p::Float64
	p_m::Array{Float64}					# steps x sessions
	r_m::Array{Float64}					# steps x bandits x sessions

	frequency_out_process(n_steps, n_bandits, n_sessions, r, p_max, η_p, decay_p) = new(r, 
																						p_max, 
																						η_p, 
																						decay_p, 
																						zeros(n_steps + 1, n_sessions), 
																						zeros(n_steps, n_bandits, n_sessions))
end

function update_out_process!(process::frequency_out_process, action, cstep, session, rng)
	
	if (action == 1)

		 process.p_m[cstep + 1, session] = process.p_m[cstep, session] + process.η_p * (process.p_max - process.p_m[cstep, session])

	else

		process.p_m[cstep + 1, session] = (1.0 - process.decay_p) * process.p_m[cstep, session]

	end

	if rand(rng) < process.p_m[cstep + 1, session]

		process.r_m[cstep, :, session] .= process.r

	end
end

struct delay_out_process <: abstract_out_process

	n_steps::Int64
	r::Float64
	delay_distr::Distribution
	r_m::Array{Float64}					# steps x bandits x sessions

	delay_out_process(n_steps, n_bandits, n_sessions, r, delay_distr) = new(n_steps, r, delay_distr, 
																			zeros(n_steps, n_bandits, n_sessions))

end

function update_out_process!(process::delay_out_process, action, cstep, session, rng)
	
	if action == 1

		delayed_step = rand(rng, process.delay_distr) + 1 	# when env.outlier_delay_distr is Geometric 
															# number of failures of outlier occurence is being sampled 
															# so +1 is required

		if cstep + delayed_step <= process.n_steps

			process.r_m[cstep + delayed_step, :, session] .= process.r
		end
	end	
end

struct test_out_process <: abstract_out_process
	
	r::Float64
	r_m::Array{Float64}					# steps x bandits x sessions

	test_out_process(n_steps, n_bandits, n_sessions, r) = new(r, zeros(n_steps, n_bandits, n_sessions))
end

function update_out_process!(process::test_out_process, action, cstep, session, rng)
	
	if cstep == 2 

		process.r_m[cstep, action, session] = process.r

	end	
end

function reset_environment!(env::bandit_environment{T, Y}) where {T <: abstract_reward_process, Y <: Union{distribution_out_process, no_out_process}}
end

function reset_environment!(env::bandit_environment{T, Y}) where {T <: abstract_reward_process, Y <: Union{frequency_out_process, delay_out_process, test_out_process}}

	env.out_process.r_m[:] = zeros(env.reward_process.n_steps, env.reward_process.n_bandits, env.reward_process.n_sessions)
end

function run_environment!(env::bandit_environment, agent::abstract_bandit_agent)

	for session = 1 : env.reward_process.n_sessions

		r = 0.0

		for cstep = 1 : env.reward_process.n_steps

			action = agent(r, cstep, session, 1:env.reward_process.n_bandits)

			r = env(action, cstep, session)
		end

		agent.offline(session)
	end	

	reset_environment!(env)
end

function run_environment!(env::bandit_environment, agent::abstract_optimal_bandit_agent)

	for session = 1 : env.reward_process.n_sessions
		
		r = 0.0

		for cstep = 1 : env.reward_process.n_steps

			action = agent(r, env.reward_process.r_m[cstep, :, session] + env.out_process.r_m[cstep, :, session], 
							cstep, session, 1:env.n_bandits)

			r = env(action, cstep, session)
		end
	end	

	reset_environment!(env)
end

