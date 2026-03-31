import runpy

print("=" * 50)
print("STEP 1 — Exploratory Data Analysis")
print("=" * 50)
runpy.run_path("eda.py")

print("\n" + "=" * 50)
print("STEP 2 — Model Training")
print("=" * 50)
runpy.run_path("training.py")

print("\n" + "=" * 50)
print("STEP 3 — Bootstrap & Stability Analysis")
print("=" * 50)
runpy.run_path("bootstrap.py")

print("\nAll steps complete. Outputs saved to output/")