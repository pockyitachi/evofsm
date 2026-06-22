# archive/ — abandoned research routes

Self-contained archives of research directions the project explored and then
dropped. Kept for reference and possible paper ablations; **none is on the main
line**.

| Subdir | Route | Why archived |
|---|---|---|
| `ppo_prm/` | PPO + Process Reward Model | Main-line RL settled on GRPO; the PPO+PRM sweep was gradient-unstable |
| `rft/` | Rejection / reward fine-tuning | Non-main-line exploration |
| `b4_init_ablation/` | B4 π^pre init-ablation sweeps (zip only) | Completed; raw trajectories archived to reclaim disk |

Each route subdir has its own README. Large trajectory backups are zipped
(`traces_backup.zip` / `*.zip`) and **gitignored**.
