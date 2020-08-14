using Distributions
using Random
using Statistics

include("models.jl")
include("plot.jl")

# Ornstein - Uhlenbeck process step, w_sample ~ N(0, 1)
OU_step(x, m, s, revert_coeff, w_sample) = revert_coeff * (m - x) + s * w_sample

function run()

	rng = MersenneTwister();

	# Outlier statistics & probability of happening  
	m_out0 = 0.0
	s_out0 = 0.1

	m_out1 = 10.0 
	s_out1 = 0.1
	w_out1 = 0.05

	m_out2 = - m_out1
	s_out2 = s_out1
	w_out2 = w_out1

	out_d = MixtureModel(Normal, [(m_out0, s_out0), (m_out1, s_out1), (m_out2, s_out2)], 
								 [1.0 - w_out1 - w_out2, w_out1, w_out2])

	#=
	out1_d = MixtureModel(Normal, [(m_out0, s_out0), (m_out1, s_out1), (m_out2, s_out2)], 
								 [1.0 - 0.2 - 0.05, 0.2, 0.05])

	out2_d = MixtureModel(Normal, [(m_out0, s_out0), (m_out1, s_out1), (m_out2, s_out2)], 
								 [1.0 - 0.05 - 0.2, 0.05, 0.2])
	=#

	out1_d = out_d
	out2_d = out_d

	w_asym = 0.1

	asym_r_d = MixtureModel(Normal, [(0.0, 0.0), (5.0, 0.1)], [1.0 - w_asym, w_asym])

	wiener_d = Normal(0.0, 1.0)	# noise component of OU process

    gamma_d = 0.01	# reversion rate

	# Reward statistics and mean-reverting coefficient for OU process
	m_r1 = 1.0 
	s_r1 = 0.5
	revert_coeff_r1 = gamma_d
		
	m_r2 = 1.0 
	s_r2 = 0.5
	revert_coeff_r2 = gamma_d

    # work out OU statistics :
    n_OU_test = 1000 

    x1 = m_r1
    x2 = m_r2

    x1_out = Array{Float64,1}(undef, n_OU_test)
    x2_out = Array{Float64,1}(undef, n_OU_test)

    for k in 1 : n_OU_test

        x1 += OU_step(x1, m_r1, s_r1, revert_coeff_r1, rand(rng, wiener_d)) 
        x1_out[k] = x1 + rand(rng, out1_d) #+ rand(rng, asym_r_d)

        x2 += OU_step(x2, m_r2, s_r2, revert_coeff_r2, rand(rng, wiener_d)) 
        x2_out[k] = x2 + rand(rng, out2_d)

    end
    
    # statistics for our probabilistic RW rule
    # using offline calculation :
	m_prob_RW = mean([x1_out ; x2_out])
	s_prob_RW = std([x1_out ; x2_out])

	# Learning rates 
	println("Max probabilistc RW learning rate = ", 1.0/gauss_pdf(m_prob_RW, m_prob_RW, s_prob_RW))
	println("Probabilistic RW mean = $m_prob_RW , std = $s_prob_RW")

	n_RW = 0.1			# 0 <= n_RW <= 1
	n_prob_RW = 6.0		# 0 <= n_prob_RW <= 1/p(m_prob_RW)

	# bounds check
	@assert(n_prob_RW <= 1.0/gauss_pdf(m_prob_RW, m_prob_RW, s_prob_RW), "n_prob_RW out of bounds")

    # statistics for Kalman Filter
    m_KF = m_r2
    s_KF = s_r2
    revert_coeff_KF = gamma_d
    s_observe_KF = 0.5

	beta = 2.5			# softmax β	
	decay = 0.1			# value decay when not chosen

	n_sessions = 100	# sessions 
	n_trials = 50		# trials per session
	
	# Offline bias calculations
	off_capacity = 10	# trials to be saved for offline calculations
	n_off_samples = 20	# samples to be drawn from saved trials for bias updates
	n_b = 0.1			# bias learning rate

	# Accumulated reward for each rule for each session
	accum_r_RW_v = zeros(n_sessions)
	accum_r_stoch_RW_v = zeros(n_sessions)
    accum_r_prob_RW_v = zeros(n_sessions)
    accum_random = zeros(n_sessions)
    accum_ideal = zeros(n_sessions)
    accum_r_KF_v = zeros(n_sessions)

	#-------------------
	# Matrix definitions
	#-------------------

	# Rewards
	r1_m = Matrix{Float64}(undef, n_trials, n_sessions)
	r2_m = Matrix{Float64}(undef, n_trials, n_sessions)

	# Rewards + outliers
	r1_out_m = Matrix{Float64}(undef, n_trials, n_sessions)
	r2_out_m = Matrix{Float64}(undef, n_trials, n_sessions)

	# Estimated rewards for classic RW
	r1_RW_m = zeros(n_trials, n_sessions)
	r2_RW_m = zeros(n_trials, n_sessions)

	# Estimated rewards for stochastic RW
	r1_stoch_RW_m = zeros(n_trials, n_sessions)
	r2_stoch_RW_m = zeros(n_trials, n_sessions)

	# Estimated rewards for our probabilistic RW
	r1_prob_RW_m = zeros(n_trials, n_sessions)
	r2_prob_RW_m = zeros(n_trials, n_sessions)

	# Estimated mean rewards and STDs for Kalman Filter
	r1_KF_m = zeros(n_trials, n_sessions)
	r2_KF_m = zeros(n_trials, n_sessions)

	s_r1_KF_m = Matrix{Float64}(undef, n_trials, n_sessions)
	s_r2_KF_m = Matrix{Float64}(undef, n_trials, n_sessions)

	s_r1_KF_m[1, 1] = s_KF 
	s_r2_KF_m[1, 1] = s_KF 

	# actions for each rule
	action_RW_m = Matrix{Int64}(undef, n_trials, n_sessions)
	action_stoch_RW_m = Matrix{Int64}(undef, n_trials, n_sessions)
	action_prob_RW_m = Matrix{Int64}(undef, n_trials, n_sessions)
	action_KF_m = Matrix{Int64}(undef, n_trials, n_sessions)

	# Biases updated offline
	b1_RW_m = zeros(n_off_samples + 1, n_sessions)
	b2_RW_m = zeros(n_off_samples + 1, n_sessions)

	b1_prob_RW_m = zeros(n_off_samples + 1, n_sessions)
	b2_prob_RW_m = zeros(n_off_samples + 1, n_sessions)

	for j = 1 : n_sessions

	asym_r = rand(rng, asym_r_d)

	# Initialise values for session
	r1_m[1, j] = rand(rng, Normal(m_r1, s_r1))
	r1_out_m[1, j] = r1_m[1, j] + rand(rng, out1_d) + asym_r

	r2_m[1, j] = rand(rng, Normal(m_r2, s_r2))
	r2_out_m[1, j] = r2_m[1, j] + rand(rng, out2_d)

	# Buffers for offline calculations
	off_surprise_RW_v = zeros(off_capacity)
	off_action_RW_v = zeros(Int, off_capacity)
	off_dr_RW_v = zeros(off_capacity)

	off_surprise_prob_RW_v = zeros(off_capacity)
	off_action_prob_RW_v = zeros(Int, off_capacity)
	off_dr_prob_RW_v = zeros(off_capacity)

	if j > 1

		# reward expectations update offline :
		#--------------------------------------------
		#=
		r1_RW_m[1, j] = b1_RW_m[end, j - 1]
		r2_RW_m[1, j] = b2_RW_m[end, j - 1]

		r1_prob_RW_m[1, j] = b1_prob_RW_m[end, j - 1]
		r2_prob_RW_m[1, j] = b2_prob_RW_m[end, j - 1]
		=#
		#--------------------------------------------

		# biases update offline : 
		#-----------------------------------------------------
		
		r1_RW_m[1, j] = r1_RW_m[end, j - 1] * (1.0 - decay)
		r2_RW_m[1, j] = r2_RW_m[end, j - 1] * (1.0 - decay)

		r1_prob_RW_m[1, j] = r1_prob_RW_m[end, j - 1] * (1.0 - decay)
		r2_prob_RW_m[1, j] = r2_prob_RW_m[end, j - 1] * (1.0 - decay)

		b1_RW_m[1, j] = b1_RW_m[end, j - 1]
		b2_RW_m[1, j] = b2_RW_m[end, j - 1]

		b1_prob_RW_m[1, j] = b1_prob_RW_m[end, j - 1]
		b2_prob_RW_m[1, j] = b2_prob_RW_m[end, j - 1]
		
		#-----------------------------------------------------

		r1_stoch_RW_m[1, j] = r1_stoch_RW_m[end, j - 1] * (1.0 - decay)
		r2_stoch_RW_m[1, j] = r2_stoch_RW_m[end, j - 1] * (1.0 - decay)

		r1_KF_m[1, j] = r1_KF_m[end, j - 1] * (1.0 - decay)
		r2_KF_m[1, j] = r2_KF_m[end, j - 1] * (1.0 - decay)

		s_r1_KF_m[1, j] = s_r1_KF_m[end, j - 1]
		s_r2_KF_m[1, j] = s_r2_KF_m[end, j - 1] 

	end

	b1_RW = b1_RW_m[1, j]
	b2_RW = b2_RW_m[1, j]

	b1_prob_RW = b1_prob_RW_m[1, j]
	b2_prob_RW = b2_prob_RW_m[1, j]

	for i = 2 : n_trials

		p_r1_ideal = softmax_p(beta, r1_m[i - 1, j] + asym_r, r2_m[i - 1, j])

		# Resolve actions, accumulate reward and update expected reward for each rule

        rand(rng) > 0.5 ? accum_random[j] += r1_out_m[i - 1, j] : accum_random[j] += r2_out_m[i - 1, j]


        p_r1_ideal > rand(rng) ? accum_ideal[j] += r1_out_m[i - 1, j] : accum_ideal[j] += r2_out_m[i - 1, j]
        

        (r1_RW_m[i, j], r2_RW_m[i, j], action_RW_m[i - 1, j], r_trial_RW, dr_RW) = RW_update!(r1_out_m[i - 1, j], 
        																					r2_out_m[i - 1, j],
        																					r1_RW_m[i - 1, j], 
        																					r2_RW_m[i - 1, j], 
        																					n_RW, beta, decay, rng,
        																					b1_RW, b2_RW)

        accum_r_RW_v[j] += r_trial_RW

    	(r1_prob_RW_m[i, j], r2_prob_RW_m[i, j], action_prob_RW_m[i - 1, j], 
    	r_trial_prob_RW, dr_prob_RW, surprise_prob_RW) = prob_RW_update!(r1_out_m[i - 1, j], 
    																	r2_out_m[i - 1, j],
    																	r1_prob_RW_m[i - 1, j], 
    																	r2_prob_RW_m[i - 1, j], 
    																	n_prob_RW, m_prob_RW, s_prob_RW,
    																	beta, decay, rng,
    																	b1_prob_RW, b2_prob_RW)

    	accum_r_prob_RW_v[j] += r_trial_prob_RW

    	(r1_stoch_RW_m[i, j], r2_stoch_RW_m[i, j], 
    	action_stoch_RW_m[i - 1, j], r_trial_stoch_RW, dr_stoch_RW) = stoch_RW_update!(r1_out_m[i - 1, j], 
																					r2_out_m[i - 1, j],
																					r1_stoch_RW_m[i - 1, j], 
																					r2_stoch_RW_m[i - 1, j], 
																					n_RW, beta, decay, rng)

        accum_r_stoch_RW_v[j] += r_trial_stoch_RW
		
		(r1_KF_m[i, j], r2_KF_m[i, j], s_r1_KF_m[i, j], s_r2_KF_m[i, j],
    	action_KF_m[i - 1, j], r_trial_KF, dr_KF) = KF_update!(r1_out_m[i - 1, j], r2_out_m[i - 1, j],
															r1_KF_m[i - 1, j], r2_KF_m[i - 1, j], 
															s_r1_KF_m[i - 1, j], s_r2_KF_m[i - 1, j], 
															m_KF, s_KF, revert_coeff_KF, s_observe_KF,
															beta, decay, rng)

        accum_r_KF_v[j] += r_trial_KF

		# update environment rewards based on OU process and add outliers

		asym_r = rand(rng, asym_r_d)

		r1_m[i, j] = r1_m[i - 1, j] + OU_step(r1_m[i - 1, j], m_r1, s_r1, revert_coeff_r1, rand(rng, wiener_d))

        r1_out_m[i, j] = r1_m[i, j] + rand(rng, out1_d) + asym_r

		r2_m[i, j] = r2_m[i - 1, j] + OU_step(r2_m[i - 1, j], m_r2, s_r2, revert_coeff_r2, rand(rng, wiener_d))

		r2_out_m[i, j] = r2_m[i, j] + rand(rng, out2_d)

        # update buffers for offline calculations

        off_buffer_update!(off_surprise_prob_RW_v, off_action_prob_RW_v, off_dr_prob_RW_v,
						surprise_prob_RW, action_prob_RW_m[i - 1, j], dr_prob_RW)


        off_buffer_update!(off_surprise_RW_v, off_action_RW_v, off_dr_RW_v,
						abs(dr_RW), action_RW_m[i - 1, j], dr_RW)

	end

		# offline updates
		
		# reward expectations update offline :
		#----------------------------------------
		#=
		b1_prob_RW_m[1, j] = r1_prob_RW_m[end, j] 
		b2_prob_RW_m[1, j] = r2_prob_RW_m[end, j]

		b1_RW_m[1, j] = r1_RW_m[end, j]
		b2_RW_m[1, j] = r2_RW_m[end, j]
		=#
		#----------------------------------------

		off_value_update!(off_surprise_prob_RW_v, off_action_prob_RW_v, sum(off_dr_prob_RW_v), 
						b1_prob_RW_m, b2_prob_RW_m, n_b, decay, n_off_samples, rng, j)

		off_value_update!(off_surprise_RW_v, off_action_RW_v, sum(off_dr_RW_v), 
						b1_RW_m, b2_RW_m, n_b, decay, n_off_samples, rng, j)
	end

	med_RW = median(accum_r_RW_v)
	med_stoch_RW = median(accum_r_stoch_RW_v)
	med_prob_RW = median(accum_r_prob_RW_v)
	med_ideal = median(accum_ideal)
	med_random = median(accum_random)
	med_KF = median(accum_r_KF_v)

	sum_RW = sum(accum_r_RW_v)
	sum_stoch_RW = sum(accum_r_stoch_RW_v)
	sum_prob_RW = sum(accum_r_prob_RW_v)
	sum_ideal = sum(accum_ideal)
	sum_random = sum(accum_random)
	sum_KF = sum(accum_r_KF_v)

	println("----------------------------------------------")
	println("Model \t \t \t Median")
	println("----------------------------------------------")
	println("RW : \t \t \t $med_RW")
	println("stochastic RW : \t $med_stoch_RW")
	println("probabilistic RW : \t $med_prob_RW")
	println("Kalman Filter : \t $med_KF")
	println("ideal : \t \t $med_ideal")
    println("random : \t \t $med_random")

    println("----------------------------------------------")
	println("Model \t \t \t Median Score")
	println("----------------------------------------------")
    rw_score=(med_RW-med_random)/(med_ideal-med_random)
    println("RW: \t \t \t $rw_score")
    prob_score=(med_prob_RW-med_random)/(med_ideal-med_random)
    println("probabilistic RW : \t $prob_score")
    KF_score=(med_KF-med_random)/(med_ideal-med_random)
    println("Kalman Filter : \t $KF_score")

    println("----------------------------------------------")
	println("Model \t \t \t Sum Score")
	println("----------------------------------------------")
    rw_score=(sum_RW-sum_random)/(sum_ideal-sum_random)
    println("RW: \t \t \t $rw_score")
    prob_score=(sum_prob_RW-sum_random)/(sum_ideal-sum_random)
    println("probabilistic RW : \t $prob_score")
    KF_score=(sum_KF-sum_random)/(sum_ideal-sum_random)
    println("Kalman Filter : \t $KF_score")

	idx = rand(rng, 1 : length(n_sessions))
	#idx = findfirst(x -> x == maximum(accum_r_RW_v), accum_r_RW_v)
	#idx = findfirst(x -> x == maximum(accum_r_prob_RW_v), accum_r_prob_RW_v)
	#=
	plot_reward_learning(n_RW, n_prob_RW, m_prob_RW, s_prob_RW, 
						r1_m[:,idx], r2_m[:,idx], 
						r1_RW_m[:,idx], r2_RW_m[:,idx],
						r1_prob_RW_m[:,idx], r2_prob_RW_m[:,idx],
						action_RW_m[:,idx], action_prob_RW_m[:,idx])
	=#

	#plot_reward(r1_out_m, r2_out_m, r1_prob_RW_m, r2_prob_RW_m, n_off_samples, 91:100)
	#plot_reward(r1_out_m, r2_out_m, r1_RW_m, r2_RW_m, n_off_samples, 91:100)

	plot_reward_bias(r1_out_m, r2_out_m, b1_prob_RW_m, b2_prob_RW_m, 91:100)
	plot_reward_bias(r1_out_m, r2_out_m, b1_RW_m, b2_RW_m, 91:100)
end

run()

