class WikipediaDashboard {
    constructor() {
        this.currentArticle = null;
        this.peerArticles = [];
        this.radarChart = null;
        this.sortState = { column: null, direction: 'asc' };
        
        this.initializeElements();
        this.bindEvents();
        this.loadSampleData();
    }

    initializeElements() {
        this.searchInput = document.getElementById('articleSearch');
        this.searchBtn = document.getElementById('searchBtn');
        this.searchResults = document.getElementById('searchResults');
        this.dashboard = document.getElementById('dashboard');
        this.loading = document.getElementById('loading');
        
        // Score elements
        this.overallScore = document.getElementById('overallScore');
        this.articleTitle = document.getElementById('articleTitle');
        this.structureScore = document.getElementById('structureScore');
        this.sourcingScore = document.getElementById('sourcingScore');
        this.editorialScore = document.getElementById('editorialScore');
        this.networkScore = document.getElementById('networkScore');
        
        // Chart and table
        this.radarCanvas = document.getElementById('radarChart');
        this.peerTableBody = document.getElementById('peerTableBody');
    }

    bindEvents() {
        this.searchBtn.addEventListener('click', () => this.searchArticle());
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.searchArticle();
        });

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
            // For demo purposes, we'll use sample data
            // In production, this would call the Python backend
            const articleData = await this.fetchArticleData(query);
            this.displayArticle(articleData);
        } catch (error) {
            console.error('Error fetching article:', error);
            this.showError('Failed to fetch article data');
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
            console.log(`Fetched real data for: ${title}`, data);
            return data;
        } catch (error) {
            console.error('API error:', error);
            this.showError(`Failed to fetch article "${title}". Please check the article title.`);
            throw error;
        }
    }

    getSampleData(title) {
        // Sample articles demonstrating full quality spectrum (0-100)
        const sampleArticles = {
            // Featured/Exceptional tier (85-90) - Featured Articles with academic sources
            'Albert Einstein': {
                title: 'Albert Einstein',
                maturity_score: 89.5,
                pillar_scores: {
                    structure: 90.0,
                    sourcing: 100.0,
                    editorial: 74.2,
                    network: 76.7
                }
            },
            'Coffee': {
                title: 'Coffee',
                maturity_score: 89.5,
                pillar_scores: {
                    structure: 90.0,
                    sourcing: 100.0,
                    editorial: 74.2,
                    network: 76.7
                }
            },
            'World War II': {
                title: 'World War II',
                maturity_score: 89.0,
                pillar_scores: {
                    structure: 90.0,
                    sourcing: 100.0,
                    editorial: 71.9,
                    network: 76.7
                }
            },
            'Zoboomafoo': {
                title: 'Zoboomafoo',
                maturity_score: 85.0,
                pillar_scores: {
                    structure: 77.8,
                    sourcing: 100.0,
                    editorial: 70.0,
                    network: 76.7
                }
            },
            // High Quality tier (80-85) - Good Articles
            'Python (programming language)': {
                title: 'Python (programming language)',
                maturity_score: 82.4,
                pillar_scores: {
                    structure: 88.8,
                    sourcing: 77.5,
                    editorial: 85.2,
                    network: 76.7
                }
            },
            'Taylor Swift': {
                title: 'Taylor Swift',
                maturity_score: 81.8,
                pillar_scores: {
                    structure: 90.0,
                    sourcing: 75.0,
                    editorial: 85.6,
                    network: 76.7
                }
            },
            // Good/Developing tier (50-74)
            'Banana slug': {
                title: 'Banana slug',
                maturity_score: 65.2,
                pillar_scores: {
                    structure: 54.0,
                    sourcing: 75.0,
                    editorial: 57.0,
                    network: 76.0
                }
            },
            // Stub tier (0-49)
            'List of colours': {
                title: 'List of colours',
                maturity_score: 9.8,
                pillar_scores: {
                    structure: 9.0,
                    sourcing: 0.0,
                    editorial: 22.0,
                    network: 28.0
                }
            }
        };
        
        return sampleArticles[title];
    }

    generateMockData(title) {
        // Generate realistic mock data on 0-100 scale
        const baseScore = Math.random() * 60 + 30; // 30-90 range
        const structure = Math.random() * 60 + 20; // 20-80
        const sourcing = Math.random() * 60 + 20; // 20-80
        const editorial = Math.random() * 60 + 20; // 20-80
        const network = Math.random() * 60 + 20; // 20-80
        
        return {
            title: title,
            maturity_score: baseScore,
            pillar_scores: {
                structure: structure,
                sourcing: sourcing,
                editorial: editorial,
                network: network
            }
        };
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
        
        // Update score circle color based on score
        this.updateScoreCircle(articleData.maturity_score);
        
        // Create radar chart
        this.createRadarChart(articleData.pillar_scores);
        
        // Load peer articles
        this.loadPeerArticles(articleData.title);
        
        // Show dashboard
        this.dashboard.style.display = 'block';
    }

    updateScoreCircle(score) {
        const scoreCircle = document.querySelector('.score-circle');
        let colorClass = 'score-low';
        
        if (score >= 70) colorClass = 'score-high';
        else if (score >= 50) colorClass = 'score-medium';
        
        scoreCircle.className = `score-circle ${colorClass}`;
    }

    createRadarChart(pillarScores) {
        if (this.radarChart) {
            this.radarChart.destroy();
        }

        const ctx = this.radarCanvas.getContext('2d');
        
        this.radarChart = new Chart(ctx, {
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
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderColor: 'rgb(37, 99, 235)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgb(37, 99, 235)',
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
                            color: '#64748b'
                        },
                        grid: {
                            color: '#e2e8f0'
                        },
                        angleLines: {
                            color: '#e2e8f0'
                        },
                        pointLabels: {
                            font: {
                                family: 'Space Grotesk',
                                size: 14,
                                weight: '500'
                            },
                            color: '#334155'
                        }
                    }
                },
                elements: {
                    line: {
                        tension: 0.1
                    }
                }
            }
        });
    }

    async loadPeerArticles(currentTitle) {
        try {
            // Try to fetch real peer articles from API
            const response = await fetch(`/api/peers/${encodeURIComponent(currentTitle)}`);
            if (response.ok) {
                const peerData = await response.json();
                this.peerArticles = [this.currentArticle, ...peerData];
            } else {
                throw new Error('Failed to fetch peer articles');
            }
        } catch (error) {
            console.error('Error fetching peer articles:', error);
            // Fallback to sample peer articles
            const peerTitles = [
                'Machine Learning',
                'Artificial Intelligence',
                'Computer Science',
                'Data Science',
                'Neural Networks',
                'Deep Learning'
            ].filter(title => title !== currentTitle).slice(0, 6);

            this.peerArticles = [this.currentArticle];
            
            for (const title of peerTitles) {
                const peerData = await this.fetchArticleData(title);
                this.peerArticles.push(peerData);
            }
        }
        
        this.renderPeerTable();
    }

    renderPeerTable() {
        this.peerTableBody.innerHTML = '';
        
        this.peerArticles.forEach(article => {
            const row = document.createElement('tr');
            const isCurrent = article.title === this.currentArticle?.title;
            
            if (isCurrent) {
                row.style.backgroundColor = 'rgba(37, 99, 235, 0.05)';
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
        
        // Update sort indicators
        document.querySelectorAll('.sortable').forEach(header => {
            header.classList.remove('sorted');
            if (header.dataset.column === column) {
                header.classList.add('sorted');
            }
        });
        
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
        // Simple error display - could be enhanced with a toast notification
        alert(message);
    }
}

// Initialize the dashboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new WikipediaDashboard();
});
