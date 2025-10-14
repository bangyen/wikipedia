# Wikipedia Maturity Dashboard

A focused, professional dashboard for analyzing Wikipedia article quality across four key pillars: structure, sourcing, editorial, and network.

## Features

- **Dual-View Interface**: Clean sidebar navigation with Search and Compare views
- **Article Search**: Real-time article analysis with live Wikipedia API integration
- **Radar Chart Visualization**: Interactive radar chart showing pillar scores
- **Peer Comparison**: Sortable table comparing articles within peer groups
- **Professional Design**: Clean, minimalist interface inspired by modern analytics platforms
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile devices

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

## Project Structure

```
ui/
├── app.py                 # Flask application with API endpoints
├── templates/             # HTML templates
│   └── dashboard.html     # Main dashboard template
├── static/                # Static assets
│   ├── css/
│   │   └── style.css      # Dashboard styles
│   └── js/
│       └── dashboard.js   # Dashboard JavaScript
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## API Endpoints

- `GET /` - Main dashboard interface
- `GET /api/article/<title>` - Get maturity score for a specific article
- `GET /api/peers/<title>` - Get peer articles for comparison
- `GET /api/search?q=<query>` - Search for articles
- `GET /api/healthz` - Health check endpoint

## Views

### 1. Search View
- Search for any Wikipedia article by title
- View detailed pillar scores in a radar chart
- See overall maturity score with color-coded indicator

### 2. Compare View
- Side-by-side comparison of peer articles
- Sortable table by any metric column
- Color-coded scores for quick identification

## Architecture

- **Frontend**: Vanilla JavaScript with Chart.js for visualizations
- **Backend**: Flask application with REST API
- **Styling**: Custom CSS design system with Wikipedia branding
- **Data**: Integrates with HeuristicBaselineModel for maturity scoring
- **Templates**: Jinja2 templates for server-side rendering

## Design System

- **Typography**: Space Grotesk font family for modern, readable text
- **Colors**: Wikipedia-inspired blue palette (#339AF0 primary)
- **Layout**: Sidebar navigation with flexible content area
- **Components**: Metric cards, chart containers, data tables
- **Responsive**: Mobile-first approach with breakpoints at 768px and 480px
