// RituCare - Women's Health Companion
// Single-page application with localStorage persistence

class RituCareApp {
    constructor() {
        this.currentUser = null;
        this.featureFlags = {};
        this.init();
    }

    async init() {
        await this.loadFeatureFlags();
        await this.checkAuthStatus();
        this.initializeUI();
        this.loadAllData();
        this.setupEventListeners();
    }

    // Load feature flags
    async loadFeatureFlags() {
        const settings = storage.getData('settings') || {};
        this.featureFlags = {
            useFreeAPI: settings.use_free_api || false,
            enableCharts: true,
            enableOfflineMode: true,
            enableExport: true,
            enableNotifications: 'Notification' in window,
            useBackend: false // Disable backend for GitHub Pages
        };
    }

    // Check authentication status (for future backend integration)
    async checkAuthStatus() {
        // For GitHub Pages, we'll use localStorage only
        const userData = storage.getData('current_user');
        if (userData) {
            this.currentUser = userData;
        }
    }

    // Initialize UI components
    initializeUI() {
        // Initialize Feather icons
        if (typeof feather !== 'undefined') {
            feather.replace();
        }

        // Load profile data
        this.loadProfileData();

        // Initialize tabs
        this.initializeTabs();

        // Initialize modals
        this.initializeModals();
    }

    // Initialize Bootstrap tabs
    initializeTabs() {
        const triggerTabList = [].slice.call(document.querySelectorAll('#mainTabs button'));
        triggerTabList.forEach(function (triggerEl) {
            const tabTrigger = new bootstrap.Tab(triggerEl);
            triggerEl.addEventListener('click', function (event) {
                event.preventDefault();
                tabTrigger.show();
            });
        });
    }

    // Initialize modals
    initializeModals() {
        // Log Period Modal
        const logPeriodModal = document.getElementById('logPeriodModal');
        if (logPeriodModal) {
            this.logPeriodModal = new bootstrap.Modal(logPeriodModal);
        }
    }

    // Setup event listeners
    setupEventListeners() {
        // Tab change events
        document.querySelectorAll('#mainTabs button').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (event) => {
                const targetTab = event.target.getAttribute('data-bs-target').substring(1);
                this.onTabChange(targetTab);
            });
        });

        // Form submissions
        document.getElementById('profile-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveProfile();
        });

        document.getElementById('cycle-prefs-form')?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveCyclePrefs();
        });
    }

    // Handle tab changes
    onTabChange(tabName) {
        switch(tabName) {
            case 'dashboard':
                this.loadDashboardData();
                break;
            case 'tracker':
                this.loadTrackerData();
                break;
            case 'pcos':
                this.loadPCOSData();
                break;
            case 'nutrition':
                this.loadNutritionData();
                break;
            case 'profile':
                this.loadProfileData();
                break;
        }
    }

    // Load all data on app start
    loadAllData() {
        this.loadDashboardData();
        this.loadTrackerData();
        this.loadPCOSData();
        this.loadNutritionData();
        this.loadProfileData();
    }

    // Dashboard functions
    loadDashboardData() {
        // Load cycle info
        const cycleInfo = document.getElementById('cycle-info');
        if (cycleInfo) {
            const currentPhase = cycleManager.getCurrentPhase();
            const nextPeriod = cycleManager.getDaysUntilNextPeriod();
            const nextOvulation = cycleManager.getDaysUntilOvulation();

            let html = '';

            if (currentPhase) {
                html += `<p><strong>Current Phase:</strong> ${currentPhase.phase} (Day ${currentPhase.day})</p>`;
            }

            if (nextPeriod !== null) {
                html += `<p><strong>Next Period:</strong> ${nextPeriod === 0 ? 'Today' : `In ${nextPeriod} days`}</p>`;
            }

            if (nextOvulation !== null && nextOvulation > 0) {
                html += `<p><strong>Ovulation:</strong> In ${nextOvulation} days</p>`;
            }

            if (!html) {
                html = '<p>Log your first period to see cycle information.</p>';
            }

            cycleInfo.innerHTML = html;
        }

        // Load PCOS risk
        const pcosRisk = document.getElementById('pcos-risk');
        if (pcosRisk) {
            const latestAssessment = pcosManager.getLatestAssessment();

            if (latestAssessment) {
                pcosRisk.innerHTML = `
                    <p><strong>Risk Level:</strong> ${latestAssessment.level}</p>
                    <p>Score: ${latestAssessment.score}</p>
                    <small>Last assessed: ${cycleManager.formatDate(latestAssessment.date)}</small>
                `;
            } else {
                pcosRisk.innerHTML = '<p>No assessment yet. Take the PCOS assessment to view your risk level.</p>';
            }
        }

        // Load nutrition tip
        const nutritionTip = document.getElementById('nutrition-tip');
        if (nutritionTip) {
            const suggestion = nutritionManager.getDailySuggestion();

            if (suggestion) {
                nutritionTip.innerHTML = `<p>${suggestion.suggestions[0]}</p>`;
            } else {
                nutritionTip.innerHTML = '<p>Log cycles to get personalized nutrition tips.</p>';
            }
        }

        // Load mini calendar
        const miniCalendar = document.getElementById('mini-calendar');
        if (miniCalendar) {
            const now = new Date();
            chartsManager.createMiniCalendar('mini-calendar', now.getFullYear(), now.getMonth());
        }
    }

    // Tracker functions
    loadTrackerData() {
        // Load calendar - show month of last period or current month
        const periodCalendar = document.getElementById('period-calendar');
        if (periodCalendar) {
            const cycles = cycleManager.getCycles();
            let year, month;

            if (cycles.length > 0) {
                // Show month of the last logged period
                const lastCycle = cycles[cycles.length - 1];
                const lastPeriodDate = new Date(lastCycle.start);
                year = lastPeriodDate.getFullYear();
                month = lastPeriodDate.getMonth();
            } else {
                // No periods logged, show current month
                const now = new Date();
                year = now.getFullYear();
                month = now.getMonth();
            }

            chartsManager.createMiniCalendar('period-calendar', year, month);
        }

        // Load cycle stats
        const cycleStats = document.getElementById('cycle-stats');
        if (cycleStats) {
            const stats = cycleManager.getCycleStats();

            if (stats) {
                cycleStats.innerHTML = `
                    <p><strong>Average Cycle Length:</strong> ${stats.averageLength} days</p>
                    <p><strong>Regularity:</strong> ${stats.regularity}</p>
                    <p><strong>Cycles Tracked:</strong> ${stats.totalCycles}</p>
                `;
            } else {
                cycleStats.innerHTML = '<p>Log more cycles to see statistics.</p>';
            }
        }

        // Load recent periods
        const recentPeriods = document.getElementById('recent-periods');
        if (recentPeriods) {
            const cycles = cycleManager.getCycles().slice(-5); // Last 5 periods

            if (cycles.length === 0) {
                recentPeriods.innerHTML = '<p>No periods logged yet.</p>';
                return;
            }

            let html = '';
            cycles.forEach(cycle => {
                const endDate = cycle.end ? cycleManager.formatDate(cycle.end) : 'Ongoing';
                html += `
                    <div class="period-log">
                        <div class="dates">
                            ${cycleManager.formatDate(cycle.start)} - ${endDate}
                        </div>
                        <div class="flow ${cycle.flow}">${cycle.flow}</div>
                        <div class="symptoms">
                            ${cycle.symptoms && cycle.symptoms.length > 0 ? cycle.symptoms.join(', ') : 'No symptoms logged'}
                        </div>
                    </div>
                `;
            });

            recentPeriods.innerHTML = html;
        }

        // Initialize charts
        this.initializeCharts();
    }

    // PCOS functions
    loadPCOSData() {
        // Populate assessment form with profile data
        pcosManager.populateAssessmentForm();

        // Load assessment history
        pcosManager.displayAssessmentHistory();
    }

    // Nutrition functions
    loadNutritionData() {
        // Update current phase
        const currentPhaseEl = document.getElementById('current-phase');
        if (currentPhaseEl) {
            const phase = cycleManager.getCurrentPhase();
            if (phase) {
                currentPhaseEl.innerHTML = `
                    <div class="phase-indicator ${phase.phase}">
                        ${nutritionManager.getPhaseDisplayName(phase.phase)}
                    </div>
                    <p>Day ${phase.day} of your cycle</p>
                `;
            } else {
                currentPhaseEl.innerHTML = '<p>Log your cycle to see current phase.</p>';
            }
        }

        // Update daily suggestion
        const dailySuggestionEl = document.getElementById('daily-suggestion');
        if (dailySuggestionEl) {
            const suggestion = nutritionManager.getDailySuggestion();
            if (suggestion) {
                dailySuggestionEl.innerHTML = `
                    <ul>
                        ${suggestion.suggestions.map(item => `<li>${item}</li>`).join('')}
                    </ul>
                `;
            } else {
                dailySuggestionEl.innerHTML = '<p>Log cycles to get daily nutrition suggestions.</p>';
            }
        }

        // Update meal plan
        const mealPlanEl = document.getElementById('meal-plan');
        if (mealPlanEl) {
            const mealPlan = nutritionManager.getMealPlan();
            if (mealPlan) {
                mealPlanEl.innerHTML = `
                    <div class="meal-item">
                        <div class="meal-item-icon">🌅</div>
                        <div class="meal-item-content">
                            <h5>Breakfast</h5>
                            <p>${mealPlan.meals.breakfast}</p>
                        </div>
                    </div>
                    <div class="meal-item">
                        <div class="meal-item-icon">☀️</div>
                        <div class="meal-item-content">
                            <h5>Lunch</h5>
                            <p>${mealPlan.meals.lunch}</p>
                        </div>
                    </div>
                    <div class="meal-item">
                        <div class="meal-item-icon">🌆</div>
                        <div class="meal-item-content">
                            <h5>Dinner</h5>
                            <p>${mealPlan.meals.dinner}</p>
                        </div>
                    </div>
                    <div class="meal-item">
                        <div class="meal-item-icon">🍪</div>
                        <div class="meal-item-content">
                            <h5>Snacks</h5>
                            <p>${mealPlan.meals.snacks}</p>
                        </div>
                    </div>
                `;
            } else {
                mealPlanEl.innerHTML = '<p>Log cycles to get meal plan suggestions.</p>';
            }
        }

        // Update shopping list
        const shoppingListEl = document.getElementById('shopping-list');
        if (shoppingListEl) {
            const shoppingList = nutritionManager.getShoppingList();
            if (shoppingList.length > 0) {
                shoppingListEl.innerHTML = shoppingList.map(item => `
                    <div class="shopping-item">
                        <input type="checkbox" id="item-${item.item}" ${item.checked ? 'checked' : ''}>
                        <label for="item-${item.item}">${item.item}</label>
                    </div>
                `).join('');
            }
        }
    }

    // Profile functions
    loadProfileData() {
        const profile = storage.getData('profile') || {};

        // Populate form fields
        const nameField = document.getElementById('profile-name');
        const ageField = document.getElementById('profile-age');
        const heightField = document.getElementById('profile-height');
        const weightField = document.getElementById('profile-weight');

        if (nameField) nameField.value = profile.name || '';
        if (ageField) ageField.value = profile.age || '';
        if (heightField) heightField.value = profile.height || '';
        if (weightField) weightField.value = profile.weight || '';

        // Load cycle preferences
        const cyclePrefs = storage.getData('cycle_preferences') || {};
        const predictionWindow = document.getElementById('prediction-window');
        const cycleLength = document.getElementById('cycle-length');

        if (predictionWindow) predictionWindow.value = cyclePrefs.predictionWindow || 6;
        if (cycleLength) cycleLength.value = cyclePrefs.cycleLength || 28;
    }

    // Save profile
    saveProfile() {
        const profile = {
            name: document.getElementById('profile-name')?.value || '',
            age: parseInt(document.getElementById('profile-age')?.value) || null,
            height: parseInt(document.getElementById('profile-height')?.value) || null,
            weight: parseInt(document.getElementById('profile-weight')?.value) || null,
            updatedAt: new Date().toISOString()
        };

        storage.saveData('profile', profile);

        // Show success message
        this.showAlert('Profile saved successfully!', 'success');

        // Reload dashboard data
        this.loadDashboardData();
    }

    // Save cycle preferences
    saveCyclePrefs() {
        const cyclePrefs = {
            predictionWindow: parseInt(document.getElementById('prediction-window')?.value) || 6,
            cycleLength: parseInt(document.getElementById('cycle-length')?.value) || 28,
            updatedAt: new Date().toISOString()
        };

        storage.saveData('cycle_preferences', cyclePrefs);

        // Show success message
        this.showAlert('Cycle preferences saved successfully!', 'success');

        // Reload cycle data
        cycleManager.loadCyclePreferences();
        this.loadTrackerData();
    }

    // Initialize charts
    initializeCharts() {
        // Cycle trend chart
        const cycleTrendCanvas = document.getElementById('cycle-trend-chart');
        if (cycleTrendCanvas) {
            chartsManager.createCycleTrendChart('cycle-trend-chart');
        }

        // Symptom frequency chart
        const symptomChartCanvas = document.getElementById('symptom-frequency-chart');
        if (symptomChartCanvas) {
            chartsManager.createSymptomFrequencyChart('symptom-frequency-chart');
        }
    }

    // Utility functions
    showAlert(message, type = 'info') {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(alertDiv);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    // Global functions for HTML onclick handlers
    logPeriod() {
        this.showLogPeriodModal();
    }

    logSymptoms() {
        // Implement symptom logging
        this.showAlert('Symptom logging feature coming soon!', 'info');
    }

    recalculateCycle() {
        cycleManager.recalculateCycles();
        this.loadTrackerData();
        this.showAlert('Cycle data recalculated!', 'success');
    }

    showLogPeriodModal() {
        if (this.logPeriodModal) {
            // Populate symptom chips
            const symptomChips = document.getElementById('symptom-chips');
            if (symptomChips) {
                chartsManager.createSymptomChips('symptom-chips');
            }
            this.logPeriodModal.show();
        }
    }

    savePeriod() {
        const periodData = {
            start: document.getElementById('period-start')?.value,
            end: document.getElementById('period-end')?.value || null,
            flow: document.getElementById('flow-intensity')?.value,
            symptoms: this.getSelectedSymptoms(),
            loggedAt: new Date().toISOString()
        };

        if (!periodData.start) {
            this.showAlert('Please select a start date', 'danger');
            return;
        }

        storage.addCycle(periodData);

        // Close modal and refresh data
        if (this.logPeriodModal) {
            this.logPeriodModal.hide();
        }

        this.loadTrackerData();
        this.loadDashboardData();
        this.showAlert('Period logged successfully!', 'success');

        // Clear form
        document.getElementById('log-period-form')?.reset();
    }

    getSelectedSymptoms() {
        const selectedSymptoms = [];
        document.querySelectorAll('#symptom-chips .symptom-chip.selected').forEach(chip => {
            selectedSymptoms.push(chip.dataset.symptom);
        });
        return selectedSymptoms;
    }

    calculatePCOSRisk() {
        const formData = pcosManager.getAssessmentFormData();
        const errors = pcosManager.validateAssessmentData(formData);

        if (errors.length > 0) {
            this.showAlert('Please fix the following errors:\n' + errors.join('\n'), 'danger');
            return;
        }

        const assessment = pcosManager.saveAssessment(formData);
        pcosManager.displayAssessmentResults(assessment);

        this.loadPCOSData();
        this.loadDashboardData();
        this.showAlert('PCOS assessment completed!', 'success');
    }

    exportData() {
        const allData = {
            profile: storage.getData('profile'),
            cycle_preferences: storage.getData('cycle_preferences'),
            periods: storage.getData('periods'),
            pcos_assessment: storage.getData('pcos_assessment'),
            nutrition_log: storage.getData('nutrition_log'),
            settings: storage.getData('settings'),
            exportedAt: new Date().toISOString()
        };

        const dataStr = JSON.stringify(allData, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});

        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `ritucare-data-${new Date().toISOString().split('T')[0]}.json`;
        link.click();

        this.showAlert('Data exported successfully!', 'success');
    }

    importData() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json';

        input.onchange = (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    try {
                        const importedData = JSON.parse(e.target.result);

                        // Import each data type
                        if (importedData.profile) storage.saveData('profile', importedData.profile);
                        if (importedData.cycle_preferences) storage.saveData('cycle_preferences', importedData.cycle_preferences);
                        if (importedData.periods) storage.saveData('periods', importedData.periods);
                        if (importedData.pcos_assessment) storage.saveData('pcos_assessment', importedData.pcos_assessment);
                        if (importedData.nutrition_log) storage.saveData('nutrition_log', importedData.nutrition_log);
                        if (importedData.settings) storage.saveData('settings', importedData.settings);

                        // Reload all data
                        this.loadAllData();
                        this.showAlert('Data imported successfully!', 'success');
                    } catch (error) {
                        this.showAlert('Error importing data. Please check the file format.', 'danger');
                    }
                };
                reader.readAsText(file);
            }
        };

        input.click();
    }

    resetData() {
        if (confirm('Are you sure you want to reset all data? This action cannot be undone.')) {
            storage.clearAllData();
            this.loadAllData();
            this.showAlert('All data has been reset.', 'warning');
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.ritucareApp = new RituCareApp();
});

// Global functions for HTML onclick handlers
function logPeriod() { window.ritucareApp?.logPeriod(); }
function logSymptoms() { window.ritucareApp?.logSymptoms(); }
function recalculateCycle() { window.ritucareApp?.recalculateCycle(); }
function showLogPeriodModal() { window.ritucareApp?.showLogPeriodModal(); }
function savePeriod() { window.ritucareApp?.savePeriod(); }
function saveProfile() { window.ritucareApp?.saveProfile(); }
function saveCyclePrefs() { window.ritucareApp?.saveCyclePrefs(); }
function calculatePCOSRisk() { window.ritucareApp?.calculatePCOSRisk(); }
function exportData() { window.ritucareApp?.exportData(); }
function importData() { window.ritucareApp?.importData(); }
function resetData() { window.ritucareApp?.resetData(); }
