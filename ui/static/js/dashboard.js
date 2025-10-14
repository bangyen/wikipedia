const COLORS = {
    primary: '#339AF0',
    secondary: '#1C7ED6',
    tertiary: '#74C0FC',
    quaternary: '#4DABF7',
    background: '#F8F9FA',
    text: '#1D1D1F',
    success: '#51CF66',
    warning: '#FCC419',
    error: '#FF6B6B'
};

const CHART_CONFIG = {
    responsive: true,
    maintainAspectRatio: true,
    aspectRatio: 2,
    plugins: {
        legend: {
            display: true,
            position: 'bottom',
            labels: {
                font: { family: 'Space Grotesk', size: 12, weight: '500' },
                color: '#6E6E73',
                padding: 16,
                usePointStyle: true,
                pointStyle: 'rect'
            }
        }
    },
    scales: {
        x: {
            grid: { display: false, drawBorder: false },
            ticks: {
                font: { family: 'Space Grotesk', size: 11 },
                color: '#86868B'
            }
        },
        y: {
            grid: { color: '#E1E4E8', drawBorder: false },
            ticks: {
                font: { family: 'Space Grotesk', size: 11 },
                color: '#86868B'
            }
        }
    }
};

class WikipediaDashboard {
    constructor() {
        this.currentArticle = null;
        this.peerArticles = [];
        this.charts = {};
        this.sortState = { column: null, direction: 'asc' };
        
        this.initializeElements();
        this.initNavigation();
        this.bindEvents();
        this.loadSampleData();
    }

    initializeElements() {
        this.searchInput = document.getElementById('articleSearch');
        this.searchBtn = document.getElementById('searchBtn');
        this.articleResult = document.getElementById('articleResult');
        this.loading = document.getElementById('loading');
        
        // Score elements
        this.overallScore = document.getElementById('overallScore');
        this.articleTitle = document.getElementById('articleTitle');
        this.structureScore = document.getElementById('structureScore');
        this.sourcingScore = document.getElementById('sourcingScore');
        this.editorialScore = document.getElementById('editorialScore');
        this.networkScore = document.getElementById('networkScore');
        
        // Charts
        this.radarCanvas = document.getElementById('radarChart');
        this.peerTableBody = document.getElementById('peerTableBody');
    }

    initNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        const views = document.querySelectorAll('.view-container');
        
        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const viewName = item.getAttribute('data-view');
                
                navItems.forEach(n => n.classList.remove('active'));
                item.classList.add('active');
                
                views.forEach(v => v.classList.add('hidden'));
                document.getElementById(`${viewName}-view`).classList.remove('hidden');
                
                // Update page title based on view
                const titles = {
                    'search': 'Article Search',
                    'compare': 'Article Comparison'
                };
                document.querySelector('.page-title').textContent = titles[viewName] || 'Dashboard';
                
                // Load data for specific views
                if (viewName === 'compare') {
                    if (this.peerArticles.length > 0) {
                        // Render existing peer data if available
                        this.renderPeerTable();
                    } else if (this.currentArticle) {
                        // Load peer articles if we have a current article but no peers yet
                        this.loadPeerArticles(this.currentArticle.title);
                    }
                }
            });
        });
    }

    bindEvents() {
        this.searchBtn.addEventListener('click', () => this.searchArticle());
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.searchArticle();
        });

        // Refresh button
        const refreshBtn = document.getElementById('refresh-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                if (this.currentArticle) {
                    this.searchArticle();
                }
            });
        }

        // Table sorting
        document.querySelectorAll('.sortable').forEach(header => {
            header.addEventListener('click', (e) => {
                const column = e.currentTarget.dataset.column;
                this.sortTable(column);
            });
        });
    }

    async searchArticle() {
        const query = this.searchInput.value.trim();
        if (!query) return;

        this.showLoading();
        
        try {
            const articleData = await this.fetchArticleData(query);
            this.displayArticle(articleData);
        } catch (error) {
            console.error('Error fetching article:', error);
            this.showError('Failed to fetch article data. Please check the article title.');
        } finally {
            this.hideLoading();
        }
    }

    async fetchArticleData(title) {
        try {
            const response = await fetch(`/api/article/${encodeURIComponent(title)}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('API error:', error);
            throw error;
        }
    }

    displayArticle(articleData) {
        this.currentArticle = articleData;
        
        // Update score display
        this.overallScore.textContent = articleData.maturity_score.toFixed(1);
        this.articleTitle.textContent = articleData.title;
        this.structureScore.textContent = articleData.pillar_scores.structure.toFixed(1);
        this.sourcingScore.textContent = articleData.pillar_scores.sourcing.toFixed(1);
        this.editorialScore.textContent = articleData.pillar_scores.editorial.toFixed(1);
        this.networkScore.textContent = articleData.pillar_scores.network.toFixed(1);
        
        // Update score circle color
        this.updateScoreCircle(articleData.maturity_score);
        
        // Create radar chart
        this.createRadarChart(articleData.pillar_scores);
        
        // Load peer articles for comparison view
        this.loadPeerArticles(articleData.title);
        
        // Show result
        this.articleResult.style.display = 'block';
    }

    updateScoreCircle(score) {
        const scoreCircle = document.querySelector('.score-circle');
        let colorClass = 'score-low';
        
        if (score >= 70) colorClass = 'score-high';
        else if (score >= 50) colorClass = 'score-medium';
        
        scoreCircle.className = `score-circle ${colorClass}`;
    }

    createRadarChart(pillarScores) {
        if (this.charts.radar) {
            this.charts.radar.destroy();
        }

        const ctx = this.radarCanvas.getContext('2d');
        
        this.charts.radar = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Structure', 'Sourcing', 'Editorial', 'Network'],
                datasets: [{
                    label: 'Pillar Scores',
                    data: [
                        pillarScores.structure,
                        pillarScores.sourcing,
                        pillarScores.editorial,
                        pillarScores.network
                    ],
                    backgroundColor: 'rgba(51, 154, 240, 0.1)',
                    borderColor: COLORS.primary,
                    borderWidth: 2,
                    pointBackgroundColor: COLORS.primary,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20,
                            font: {
                                family: 'Space Grotesk',
                                size: 12
                            },
                            color: '#6E6E73'
                        },
                        grid: {
                            color: '#E1E4E8'
                        },
                        angleLines: {
                            color: '#E1E4E8'
                        },
                        pointLabels: {
                            font: {
                                family: 'Space Grotesk',
                                size: 14,
                                weight: '500'
                            },
                            color: '#1D1D1F'
                        }
                    }
                }
            }
        });
    }

    async loadPeerArticles(currentTitle) {
        try {
            const response = await fetch(`/api/peers/${encodeURIComponent(currentTitle)}`);
            if (response.ok) {
                const peerData = await response.json();
                this.peerArticles = [this.currentArticle, ...peerData];
                
                // Only render table if Compare view is currently active
                const compareView = document.getElementById('compare-view');
                if (compareView && !compareView.classList.contains('hidden')) {
                    this.renderPeerTable();
                }
            } else {
                throw new Error('Failed to fetch peer articles');
            }
        } catch (error) {
            console.error('Error fetching peer articles:', error);
            const compareView = document.getElementById('compare-view');
            if (compareView && !compareView.classList.contains('hidden')) {
                this.peerTableBody.innerHTML = '<tr><td colspan="6" class="loading">Failed to load peer articles</td></tr>';
            }
        }
    }

    renderPeerTable() {
        this.peerTableBody.innerHTML = '';
        
        this.peerArticles.forEach(article => {
            const row = document.createElement('tr');
            const isCurrent = article.title === this.currentArticle?.title;
            
            if (isCurrent) {
                row.style.backgroundColor = 'rgba(51, 154, 240, 0.05)';
                row.style.fontWeight = '600';
            }
            
            row.innerHTML = `
                <td>
                    <a href="https://en.wikipedia.org/wiki/${encodeURIComponent(article.title)}" 
                       target="_blank" class="article-link">
                        ${article.title}
                    </a>
                </td>
                <td class="score-cell ${this.getScoreClass(article.maturity_score)}">
                    ${article.maturity_score.toFixed(1)}
                </td>
                <td class="score-cell ${this.getScoreClass(article.pillar_scores.structure)}">
                    ${article.pillar_scores.structure.toFixed(1)}
                </td>
                <td class="score-cell ${this.getScoreClass(article.pillar_scores.sourcing)}">
                    ${article.pillar_scores.sourcing.toFixed(1)}
                </td>
                <td class="score-cell ${this.getScoreClass(article.pillar_scores.editorial)}">
                    ${article.pillar_scores.editorial.toFixed(1)}
                </td>
                <td class="score-cell ${this.getScoreClass(article.pillar_scores.network)}">
                    ${article.pillar_scores.network.toFixed(1)}
                </td>
            `;
            
            this.peerTableBody.appendChild(row);
        });
    }

    getScoreClass(score) {
        if (score >= 70) return 'score-high';
        if (score >= 50) return 'score-medium';
        return 'score-low';
    }

    sortTable(column) {
        const direction = this.sortState.column === column && this.sortState.direction === 'asc' ? 'desc' : 'asc';
        this.sortState = { column, direction };
        
        // Sort peer articles
        this.peerArticles.sort((a, b) => {
            let aVal, bVal;
            
            if (column === 'title') {
                aVal = a.title.toLowerCase();
                bVal = b.title.toLowerCase();
            } else if (column === 'overall') {
                aVal = a.maturity_score;
                bVal = b.maturity_score;
            } else {
                aVal = a.pillar_scores[column];
                bVal = b.pillar_scores[column];
            }
            
            if (direction === 'asc') {
                return aVal > bVal ? 1 : -1;
            } else {
                return aVal < bVal ? 1 : -1;
            }
        });
        
        this.renderPeerTable();
    }

    loadSampleData() {
        // Load Albert Einstein as default
        this.searchInput.value = 'Albert Einstein';
        this.searchArticle();
    }

    showLoading() {
        this.loading.style.display = 'flex';
        this.searchBtn.disabled = true;
    }

    hideLoading() {
        this.loading.style.display = 'none';
        this.searchBtn.disabled = false;
    }

    showError(message) {
        alert(message);
    }
}

// Initialize the dashboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new WikipediaDashboard();
});

