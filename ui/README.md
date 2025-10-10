# Wikipedia Maturity Dashboard

A professional JavaScript dashboard for visualizing Wikipedia article maturity scores and peer comparisons.

## Features

- **Radar Chart Visualization**: Interactive radar chart showing pillar scores (Structure, Sourcing, Editorial, Network)
- **Peer Comparison Table**: Sortable table comparing articles within the same category
- **Live Article Search**: Real-time article analysis with Wikipedia API integration
- **Professional Design**: Clean, minimalist interface with custom styling
- **Responsive Layout**: Works on desktop and mobile devices

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Server**:
   ```bash
   python app.py
   ```

3. **Open Dashboard**:
   Navigate to `http://localhost:5000` in your browser

## API Endpoints

- `GET /api/article/<title>` - Get maturity score for an article
- `GET /api/peers/<title>` - Get peer articles for comparison
- `GET /api/search?q=<query>` - Search for articles
- `GET /api/health` - Health check

## Architecture

- **Frontend**: Vanilla JavaScript with Chart.js for visualizations
- **Backend**: Flask API serving Wikipedia data
- **Styling**: Custom CSS with professional design system
- **Data**: Integrates with existing Wikipedia maturity scoring model

## Design System

- **Typography**: Space Grotesk font family
- **Colors**: Flat color palette with blue primary (#2563eb)
- **Layout**: Grid-based responsive design
- **Components**: Custom-styled elements with consistent spacing
