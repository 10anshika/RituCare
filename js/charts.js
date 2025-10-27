// RituCare - Charts Module
// Chart.js wrappers for cycle trends, symptoms, and risk gauges

class ChartsManager {
    constructor() {
        this.charts = {};
    }

    // Create cycle length trend chart
    createCycleTrendChart(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;

        const trendData = cycleManager.getCycleLengthTrend();

        if (trendData.length === 0) {
            canvas.parentElement.innerHTML = '<p class="text-muted">Not enough cycle data to show trends.</p>';
            return null;
        }

        const data = {
            labels: trendData.map(d => `Cycle ${d.cycle}`),
            datasets: [{
                label: 'Cycle Length (days)',
                data: trendData.map(d => d.length),
                borderColor: 'rgb(78, 205, 196)',
                backgroundColor: 'rgba(78, 205, 196, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        };

        const config = {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Cycle ${context.dataIndex + 1}: ${context.parsed.y} days`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        min: 20,
                        max: 40,
                        title: {
                            display: true,
                            text: 'Days'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Cycle Number'
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        };

        this.charts[canvasId] = new Chart(canvas, config);
        return this.charts[canvasId];
    }

    // Create symptom frequency chart
    createSymptomFrequencyChart(canvasId) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;

        const symptomData = cycleManager.getSymptomFrequency();

        if (symptomData.length === 0) {
            canvas.parentElement.innerHTML = '<p class="text-muted">No symptom data available.</p>';
            return null;
        }

        // Take top 8 symptoms
        const topSymptoms = symptomData.slice(0, 8);

        const data = {
            labels: topSymptoms.map(d => this.formatSymptomLabel(d.symptom)),
            datasets: [{
                label: 'Frequency',
                data: topSymptoms.map(d => d.count),
                backgroundColor: [
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 205, 86, 0.8)',
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(153, 102, 255, 0.8)',
                    'rgba(255, 159, 64, 0.8)',
                    'rgba(199, 199, 199, 0.8)',
                    'rgba(83, 102, 255, 0.8)'
                ],
                borderColor: [
                    'rgb(255, 99, 132)',
                    'rgb(54, 162, 235)',
                    'rgb(255, 205, 86)',
                    'rgb(75, 192, 192)',
                    'rgb(153, 102, 255)',
                    'rgb(255, 159, 64)',
                    'rgb(199, 199, 199)',
                    'rgb(83, 102, 255)'
                ],
                borderWidth: 1
            }]
        };

        const config = {
            type: 'bar',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const symptom = topSymptoms[context.dataIndex];
                                return `${symptom.count} cycles (${symptom.percentage.toFixed(1)}%)`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Number of Cycles'
                        },
                        ticks: {
                            stepSize: 1
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Symptoms'
                        }
                    }
                }
            }
        };

        this.charts[canvasId] = new Chart(canvas, config);
        return this.charts[canvasId];
    }

    // Create PCOS risk gauge
    createPCOSRiskGauge(canvasId, score, level) {
        const canvas = document.getElementById(canvasId);
        if (!canvas) return null;

        const maxScore = 10;
        const percentage = (score / maxScore) * 100;

        const data = {
            datasets: [{
                data: [percentage, 100 - percentage],
                backgroundColor: [
                    pcosManager.getRiskLevelColor(level),
                    'rgba(233, 236, 239, 0.3)'
                ],
                borderWidth: 0,
                cutout: '70%',
                circumference: 180,
                rotation: 270
            }]
        };

        const config = {
            type: 'doughnut',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                }
            },
            plugins: [{
                id: 'gaugeCenterText',
                beforeDraw: function(chart) {
                    const { ctx, chartArea: { width, height } } = chart;

                    ctx.save();
                    ctx.font = 'bold 24px Inter';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillStyle = '#333';

                    const centerX = width / 2;
                    const centerY = height / 2 + 20;

                    // Score
                    ctx.fillText(score.toString(), centerX, centerY - 10);

                    // Level
                    ctx.font = '14px Inter';
                    ctx.fillStyle = pcosManager.getRiskLevelColor(level);
                    ctx.fillText(level, centerX, centerY + 15);

                    ctx.restore();
                }
            }]
        };

        this.charts[canvasId] = new Chart(canvas, config);
        return this.charts[canvasId];
    }

    // Create mini calendar
    createMiniCalendar(containerId, year, month) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const calendarData = cycleManager.generateCalendarData(year, month);

        let html = '<div class="calendar">';
        html += '<div class="calendar-header">Sun</div>';
        html += '<div class="calendar-header">Mon</div>';
        html += '<div class="calendar-header">Tue</div>';
        html += '<div class="calendar-header">Wed</div>';
        html += '<div class="calendar-header">Thu</div>';
        html += '<div class="calendar-header">Fri</div>';
        html += '<div class="calendar-header">Sat</div>';

        calendarData.forEach(day => {
            let classes = 'calendar-day';
            if (!day.isCurrentMonth) classes += ' text-muted';
            if (day.isToday) classes += ' today';
            if (day.isPeriod) classes += ' period';
            if (day.isPredictedPeriod) classes += ' predicted';
            if (day.isFertile) classes += ' fertile';

            html += `<div class="${classes}" title="${this.getDayTooltip(day)}">${day.day}</div>`;
        });

        html += '</div>';
        container.innerHTML = html;
    }

    // Get tooltip text for calendar day
    getDayTooltip(day) {
        let tooltip = cycleManager.formatDate(day.date);

        if (day.isPeriod) {
            tooltip += ' - Period day';
            if (day.symptoms && day.symptoms.length > 0) {
                tooltip += ` (${day.symptoms.join(', ')})`;
            }
        }

        if (day.isPredictedPeriod) {
            tooltip += ' - Predicted period start';
        }

        if (day.isFertile) {
            tooltip += ' - Fertile window';
        }

        return tooltip;
    }

    // Format symptom label for charts
    formatSymptomLabel(symptom) {
        const labels = {
            cramps: 'Cramps',
            heavy_flow: 'Heavy Flow',
            mood_swings: 'Mood Swings',
            bloating: 'Bloating',
            headache: 'Headache',
            fatigue: 'Fatigue',
            breast_tenderness: 'Breast Tenderness',
            acne: 'Acne',
            cravings: 'Cravings'
        };
        return labels[symptom] || symptom.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    // Update chart data
    updateChart(chartId, newData) {
        const chart = this.charts[chartId];
        if (!chart) return;

        if (newData.labels) chart.data.labels = newData.labels;
        if (newData.datasets) chart.data.datasets = newData.datasets;

        chart.update();
    }

    // Destroy chart
    destroyChart(chartId) {
        const chart = this.charts[chartId];
        if (chart) {
            chart.destroy();
            delete this.charts[chartId];
        }
    }

    // Destroy all charts
    destroyAllCharts() {
        Object.keys(this.charts).forEach(chartId => {
            this.destroyChart(chartId);
        });
    }

    // Resize all charts
    resizeAllCharts() {
        Object.values(this.charts).forEach(chart => {
            chart.resize();
        });
    }

    // Create hydration progress indicator
    createHydrationProgress(containerId, percentage = 0) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.style.width = `${percentage}%`;
        container.setAttribute('aria-valuenow', percentage);
    }

    // Create phase progress indicator
    createPhaseProgress(containerId, currentDay, totalDays) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const percentage = totalDays > 0 ? (currentDay / totalDays) * 100 : 0;

        container.innerHTML = `
            <div class="progress" style="height: 8px;">
                <div class="progress-bar" role="progressbar"
                     style="width: ${percentage}%; background-color: ${nutritionManager.getPhaseColor(containerId.split('-')[0])}"
                     aria-valuenow="${percentage}" aria-valuemin="0" aria-valuemax="100">
                </div>
            </div>
            <small class="text-muted">Day ${currentDay} of ${totalDays}</small>
        `;
    }

    // Create symptom selector chips
    createSymptomChips(containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const symptoms = [
            { id: 'cramps', label: 'Cramps' },
            { id: 'heavy_flow', label: 'Heavy Flow' },
            { id: 'mood_swings', label: 'Mood Swings' },
            { id: 'bloating', label: 'Bloating' },
            { id: 'headache', label: 'Headache' },
            { id: 'fatigue', label: 'Fatigue' },
            { id: 'breast_tenderness', label: 'Breast Tenderness' },
            { id: 'acne', label: 'Acne' },
            { id: 'cravings', label: 'Cravings' }
        ];

        let html = '';
        symptoms.forEach(symptom => {
            html += `
                <div class="symptom-chip" data-symptom="${symptom.id}" onclick="chartsManager.toggleSymptom('${symptom.id}')">
                    ${symptom.label}
                </div>
            `;
        });

        container.innerHTML = html;
    }

    // Toggle symptom selection
    toggleSymptom(symptom) {
        const chip = document.querySelector(`[data-symptom="${symptom}"]`);
        if (chip) {
            chip.classList.toggle('active');
        }
    }

    // Create empty state illustration
    createEmptyState(containerId, type) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const emptyStates = {
            cycles: {
                icon: 'calendar',
                title: 'No Cycles Logged',
                message: 'Start tracking your periods to see insights and predictions.'
            },
            assessments: {
                icon: 'activity',
                title: 'No PCOS Assessments',
                message: 'Take the PCOS assessment to evaluate your risk level.'
            },
            nutrition: {
                icon: 'apple',
                title: 'Nutrition Data Loading',
                message: 'Your personalized nutrition recommendations will appear here.'
            }
        };

        const state = emptyStates[type] || emptyStates.cycles;

        container.innerHTML = `
            <div class="empty-state">
                <i data-feather="${state.icon}"></i>
                <h3>${state.title}</h3>
                <p>${state.message}</p>
            </div>
        `;

        // Re-initialize Feather icons
        if (window.feather) {
            window.feather.replace();
        }
    }
}

// Global charts manager instance
const chartsManager = new ChartsManager();
