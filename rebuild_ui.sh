rm -rf classiflow-ui/node_modules
rm -rf classiflow-ui/dist
rm -rf classiflow-ui/.vite
rm -rf classiflow-ui/node_modules/.vite

# 3) clean npm cache
npm cache clean --force

# 4) reinstall + rebuild frontend
cd classiflow-ui
npm ci
npm run build
cd ..

# 5) ensure CLI is from this repo/env
conda activate classiflow
python -m pip install -e .

# 6) serve with explicit static dir (important)
classiflow ui serve \
  --projects-root DB_test_classiflow/projects \
  --static-dir "$(pwd)/classiflow-ui/dist"