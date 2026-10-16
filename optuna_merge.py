import optuna
from optuna.trial import TrialState

# SQLite example; replace with your PostgreSQL URL if needed.
storage = "sqlite:///optuna.db"

study1 = optuna.load_study(study_name="m8", storage=storage)
study2 = optuna.load_study(study_name="m9", storage=storage)

if study1.directions != study2.directions:
    raise ValueError("Studies must have the same objective directions.")

# Use a new name each time to avoid importing trials twice.
merged = optuna.create_study(
    study_name="m8+m9",
    storage=storage,
    directions=study1.directions,
    sampler=optuna.samplers.TPESampler(seed=42),
)

for study in (study1, study2):
    merged.add_trials(
        study.get_trials(states=[TrialState.COMPLETE])
    )

print("Total completed trials:", len(merged.trials))

# For a single-objective study:
print("Best score:", merged.best_value)
print("Best parameters:", merged.best_params)

# Continue optimization:
# merged.optimize(objective, n_trials=100)