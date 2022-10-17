# README

The core procedure of the paper is simple -- we shuffle the data K times, and take half as training set and half as validation set.

In this code base, we provide two simulated environments that we used and data partitions we used.

- `envs/tutorbot/tutor_env.py` shows how TutorEnv works, which was not described in detail in the appendix.
- `envs/sepsis/create_behavior_policy.py` shows how we generate the dataset for the Sepsis domain.

The BVFT code with our FQE implementation is in the `ope/` folder.

P-MDP code is inside the `pmdp` folder.

POIS code is inside the `pois` folder.

We ran the Robomimic experiments by modifying https://github.com/ARISE-Initiative/robomimic 

- `results` folder contains the aggregate CSV files of our experiment run. These won't work with the `*_evaluation.py` files; though they
can still be processed and examined!

We ran the D4RL experiments by using the [d3rlpy](https://github.com/takuseno/d3rlpy) repository.

For additional inquiries about our experimental details or direct access to our logged results or our original code, please contact the corresponding author of the paper.

We plan to release a more user-friendly version of the code in the near future.
