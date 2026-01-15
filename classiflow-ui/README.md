# Classiflow UI

A read-only web interface for browsing Classiflow ML projects, runs, and artifacts.

## Features

- Browse projects with decision badges and headline metrics
- View runs grouped by phase (technical_validation, independent_test, final_model)
- Explore artifacts: images, reports, metrics, configs
- Add comments and reviews for collaboration
- Local-first: works entirely on filesystem + SQLite

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.10+ with classiflow installed

### Running the UI

1. Start the backend API:
```bash
# From the classiflow repository root
classiflow ui serve --projects-root ./projects

# Or with auto-browser open
classiflow ui serve --projects-root ./projects --open
```

2. Access the UI at http://localhost:8765

### Development Mode

For frontend development with hot reload:

1. Start backend in dev mode (port 8765):
```bash
classiflow ui serve --projects-root ./projects --dev
```

2. In a separate terminal, start the frontend dev server (port 5173):
```bash
cd classiflow-ui
npm install
npm run dev
```

3. Access the UI at http://localhost:5173 (frontend will proxy API calls to backend)

### Building for Production

```bash
cd classiflow-ui
npm run build
```

The built files will be in `classiflow-ui/dist/`. The backend can serve these directly:

```bash
classiflow ui serve --projects-root ./projects --static-dir ./classiflow-ui/dist
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CLASSIFLOW_PROJECTS_ROOT` | Projects directory | `./projects` |
| `CLASSIFLOW_DB_PATH` | SQLite database for comments | `.classiflow/ui.db` |
| `CLASSIFLOW_UI_HOST` | API host | `127.0.0.1` |
| `CLASSIFLOW_UI_PORT` | API port | `8765` |
| `CLASSIFLOW_CORS_ORIGINS` | Allowed CORS origins | `http://localhost:5173,http://localhost:3000` |

### CLI Options

```bash
classiflow ui serve --help

# Options:
#   -p, --projects-root PATH  Root directory containing projects
#   -h, --host TEXT          Host to bind to
#   --port INTEGER           Port to listen on
#   --db-path PATH           SQLite database path
#   --static-dir PATH        Directory containing built frontend
#   --dev                    Run in development mode
#   --open                   Open browser after starting
```

## API Reference

### Health Check
```
GET /api/health
```

### Projects
```
GET /api/projects                          # List projects
GET /api/projects/{project_id}             # Get project details
GET /api/projects/{project_id}/runs        # Get runs by phase
```

### Runs
```
GET /api/runs/{run_key}                    # Get run details (key: project:phase:run_id)
GET /api/runs/{run_key}/artifacts          # List run artifacts
```

### Artifacts
```
GET /api/artifacts/{artifact_id}           # Get artifact metadata
GET /api/artifacts/{artifact_id}/content   # Get artifact content
GET /api/projects/{project_id}/runs/{phase}/{run_id}/artifacts/{path}  # Get by path
```

### Comments
```
GET /api/comments?scope_type=...&scope_id=...
POST /api/comments
DELETE /api/comments/{id}
```

### Reviews
```
GET /api/reviews?scope_type=...&scope_id=...
POST /api/reviews
PATCH /api/reviews/{id}
```

## Storage Modes

### Local (Default)

Uses filesystem for projects/runs/artifacts and SQLite for comments/reviews:
- Projects discovered from `projects_root` directory
- Comments stored in `db_path` SQLite file

### Postgres (Optional)

For shared comments/reviews across teams:
```bash
export CLASSIFLOW_STORAGE_MODE=postgres
export CLASSIFLOW_POSTGRES_URL=postgresql://user:pass@host/db
classiflow ui serve --projects-root ./projects
```

### Databricks (Optional, Read-Only)

For enterprise deployments with centralized metadata:
```bash
export CLASSIFLOW_STORAGE_MODE=databricks
export DATABRICKS_HOST=https://your-workspace.databricks.com
export DATABRICKS_TOKEN=your-token
export CLASSIFLOW_DATABRICKS_CATALOG=ml_catalog
export CLASSIFLOW_DATABRICKS_SCHEMA=classiflow
classiflow ui serve
```

## Project Structure

```
classiflow-ui/
├── src/
│   ├── api/           # API client
│   ├── components/    # Reusable UI components
│   ├── hooks/         # React Query hooks
│   ├── pages/         # Page components
│   └── types/         # TypeScript types
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

## Tech Stack

- **Frontend**: React 18, TypeScript, Tailwind CSS, React Query, React Router
- **Backend**: FastAPI, Pydantic, SQLite
- **Build**: Vite

## License

MIT
