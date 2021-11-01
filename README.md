# asl
Learning under different affective states

This work was inspired by behavioural results from animal studies of reward learning and decision making (originally in Stuart et al.,2013), where it was shown that the memory of a rewarded event depended upon the affective state of the animal during the same day. The term "affective bias" is commonly used in the literature for observations of this sort. Importantly, the memory of the rewarded event was equally biased by the affective state both when the affective state was manipulated before and after the rewarded action. 

This led us to a reinforcement learning formulation that includes both online learning (during the task) and offline updates, based on memories of the past day. The repository currently holds different versions of customizable multi-armed bandit environments, where learning action-reward contingencies is hard for a traditional delta-rule-type agent.

Please see the deferred_RL repository for a more recent implementation of these ideas. There I switched from custom-made agents and environments to using the ReinforcementLearning.jl package, which offers a great, functional interface for agent comparisons and extensions to environments.

References

Stuart, S., Butler, P., Munafò, M. et al. A Translational Rodent Assay of Affective Biases in Depression and Antidepressant Therapy. Neuropsychopharmacol 38, 1625–1635 (2013). https://doi.org/10.1038/npp.2013.69
