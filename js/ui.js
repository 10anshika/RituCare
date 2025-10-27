// RituCare - UI Module
// Handles UI components, modals, form validation, and accessibility

class UIManager {
    constructor() {
        this.toasts = [];
        this.modals = {};
        this.init();
    }

    // Initialize UI components
    init() {
        this.setupEventListeners();
        this.initializeFeatherIcons();
        this.setupFormValidation();
        this.setupKeyboardNavigation();
    }

    // Setup global event listeners
    setupEventListeners() {
        // Handle online/offline status
        window.addEventListener('online', () => this.showToast('Back online', 'success'));
        window.addEventListener('offline', () => this.showToast('You are offline', 'warning'));

        // Handle window resize for charts
        window.addEventListener('resize', () => {
            setTimeout(() => chartsManager.resizeAllCharts(), 250);
        });

        // Handle theme toggle (if implemented)
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }
    }

    // Initialize Feather icons
    initializeFeatherIcons() {
        if (window.feather) {
            window.feather.replace();
        }
    }

    // Setup form validation
    setupFormValidation() {
        // Add validation to all forms
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', (e) => {
                if (!this.validateForm(form)) {
                    e.preventDefault();
                    return false;
                }
            });
        });

        // Real-time validation
        document.querySelectorAll('input, select, textarea').forEach(field => {
            field.addEventListener('blur', () => this.validateField(field));
            field.addEventListener('input', () => this.clearFieldError(field));
        });
    }

    // Setup keyboard navigation
    setupKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            // ESC to close modals
            if (e.key === 'Escape') {
                this.closeAllModals();
            }

            // Ctrl/Cmd + / for help (if implemented)
            if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                e.preventDefault();
                this.showHelp();
            }
        });
    }

    // Validate entire form
    validateForm(form) {
        let isValid = true;
        const fields = form.querySelectorAll('input, select, textarea');

        fields.forEach(field => {
            if (!this.validateField(field)) {
                isValid = false;
            }
        });

        return isValid;
    }

    // Validate individual field
    validateField(field) {
        const value = field.value.trim();
        const fieldName = field.getAttribute('data-label') || field.name || field.id;
        let isValid = true;
        let errorMessage = '';

        // Required field validation
        if (field.hasAttribute('required') && !value) {
            isValid = false;
            errorMessage = `${fieldName} is required`;
        }

        // Type-specific validation
        if (value) {
            switch (field.type) {
                case 'email':
                    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                    if (!emailRegex.test(value)) {
                        isValid = false;
                        errorMessage = 'Please enter a valid email address';
                    }
                    break;

                case 'number':
                    const numValue = parseFloat(value);
                    const min = field.getAttribute('min');
                    const max = field.getAttribute('max');

                    if (min && numValue < parseFloat(min)) {
                        isValid = false;
                        errorMessage = `Minimum value is ${min}`;
                    }

                    if (max && numValue > parseFloat(max)) {
                        isValid = false;
                        errorMessage = `Maximum value is ${max}`;
                    }
                    break;

                case 'date':
                    const dateValue = new Date(value);
                    if (isNaN(dateValue.getTime())) {
                        isValid = false;
                        errorMessage = 'Please enter a valid date';
                    }
                    break;
            }
        }

        if (!isValid) {
            this.showFieldError(field, errorMessage);
        } else {
            this.clearFieldError(field);
        }

        return isValid;
    }

    // Show field error
    showFieldError(field, message) {
        field.classList.add('is-invalid');

        // Remove existing error message
        const existingError = field.parentElement.querySelector('.invalid-feedback');
        if (existingError) {
            existingError.remove();
        }

        // Add error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'invalid-feedback';
        errorDiv.textContent = message;
        field.parentElement.appendChild(errorDiv);
    }

    // Clear field error
    clearFieldError(field) {
        field.classList.remove('is-invalid');
        const errorMessage = field.parentElement.querySelector('.invalid-feedback');
        if (errorMessage) {
            errorMessage.remove();
        }
    }

    // Show toast notification
    showToast(message, type = 'info', duration = 3000) {
        const toastId = Date.now();
        const toast = {
            id: toastId,
            message,
            type,
            duration
        };

        this.toasts.push(toast);
        this.renderToast(toast);

        // Auto remove
        if (duration > 0) {
            setTimeout(() => this.removeToast(toastId), duration);
        }

        return toastId;
    }

    // Render toast
    renderToast(toast) {
        const toastContainer = this.getToastContainer();

        const toastEl = document.createElement('div');
        toastEl.className = `toast align-items-center text-white bg-${toast.type} border-0`;
        toastEl.setAttribute('role', 'alert');
        toastEl.setAttribute('aria-live', 'assertive');
        toastEl.setAttribute('aria-atomic', 'true');
        toastEl.id = `toast-${toast.id}`;

        toastEl.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${toast.message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" onclick="uiManager.removeToast(${toast.id})"></button>
            </div>
        `;

        toastContainer.appendChild(toastEl);

        // Show toast
        const bsToast = new bootstrap.Toast(toastEl);
        bsToast.show();

        // Remove from DOM after hide
        toastEl.addEventListener('hidden.bs.toast', () => {
            toastEl.remove();
        });
    }

    // Remove toast
    removeToast(toastId) {
        const toastEl = document.getElementById(`toast-${toastId}`);
        if (toastEl) {
            const bsToast = bootstrap.Toast.getInstance(toastEl);
            if (bsToast) {
                bsToast.hide();
            }
        }

        this.toasts = this.toasts.filter(t => t.id !== toastId);
    }

    // Get or create toast container
    getToastContainer() {
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }
        return container;
    }

    // Show modal
    showModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            const bsModal = new bootstrap.Modal(modal);
            bsModal.show();
        }
    }

    // Hide modal
    hideModal(modalId) {
        const modal = document.getElementById(modalId);
        if (modal) {
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) {
                bsModal.hide();
            }
        }
    }

    // Close all modals
    closeAllModals() {
        document.querySelectorAll('.modal').forEach(modal => {
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) {
                bsModal.hide();
            }
        });
    }

    // Show loading state
    showLoading(elementId, text = 'Loading...') {
        const element = document.getElementById(elementId);
        if (!element) return;

        element.innerHTML = `
            <div class="d-flex align-items-center justify-content-center p-4">
                <div class="spinner-border spinner-border-sm me-2" role="status">
                    <span class="sr-only">Loading...</span>
                </div>
                ${text}
            </div>
        `;
    }

    // Hide loading state
    hideLoading(elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = '';
        }
    }

    // Toggle theme (light/dark)
    toggleTheme() {
        const html = document.documentElement;
        const currentTheme = html.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        html.setAttribute('data-theme', newTheme);
        localStorage.setItem('ritucare_theme', newTheme);

        this.showToast(`Switched to ${newTheme} mode`, 'info');
    }

    // Load saved theme
    loadTheme() {
        const savedTheme = localStorage.getItem('ritucare_theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
    }

    // Show help dialog
    showHelp() {
        const helpContent = `
            <h5>Keyboard Shortcuts</h5>
            <ul>
                <li><kbd>Esc</kbd> - Close modals</li>
                <li><kbd>Ctrl</kbd> + <kbd>/</kbd> - Show this help</li>
            </ul>
            <h5>Features</h5>
            <ul>
                <li>Track your menstrual cycles</li>
                <li>Get PCOS risk assessment</li>
                <li>Receive cycle-synced nutrition tips</li>
                <li>All data stays on your device</li>
            </ul>
        `;

        this.showCustomDialog('Help', helpContent);
    }

    // Show custom dialog
    showCustomDialog(title, content, buttons = [{ text: 'OK', action: () => {} }]) {
        const modalId = 'custom-dialog';
        let modal = document.getElementById(modalId);

        if (!modal) {
            modal = document.createElement('div');
            modal.className = 'modal fade';
            modal.id = modalId;
            modal.innerHTML = `
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title"></h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body"></div>
                        <div class="modal-footer"></div>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
        }

        modal.querySelector('.modal-title').textContent = title;
        modal.querySelector('.modal-body').innerHTML = content;

        const footer = modal.querySelector('.modal-footer');
        footer.innerHTML = '';

        buttons.forEach(button => {
            const btn = document.createElement('button');
            btn.className = 'btn btn-primary';
            btn.textContent = button.text;
            btn.onclick = () => {
                button.action();
                this.hideModal(modalId);
            };
            footer.appendChild(btn);
        });

        this.showModal(modalId);
    }

    // Confirm dialog
    confirm(message, onConfirm, onCancel = () => {}) {
        this.showCustomDialog(
            'Confirm',
            message,
            [
                { text: 'Cancel', action: onCancel },
                { text: 'OK', action: onConfirm }
            ]
        );
    }

    // Update page title
    updatePageTitle(title) {
        document.title = title ? `${title} - RituCare` : 'RituCare';
    }

    // Scroll to element
    scrollToElement(elementId, offset = 0) {
        const element = document.getElementById(elementId);
        if (element) {
            const elementPosition = element.offsetTop;
            const offsetPosition = elementPosition - offset;

            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });
        }
    }

    // Update navigation active state
    updateNavigationActive(currentPage) {
        document.querySelectorAll('.navbar-nav .nav-link').forEach(link => {
            link.classList.remove('active');
        });

        const activeLink = document.querySelector(`.navbar-nav .nav-link[href*="${currentPage}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
    }

    // Handle page transitions
    transitionPage(fromPage, toPage, callback) {
        // Simple fade transition
        const content = document.querySelector('main');
        if (content) {
            content.style.opacity = '0';
            setTimeout(() => {
                callback();
                content.style.opacity = '1';
            }, 150);
        } else {
            callback();
        }
    }

    // Accessibility: Announce content changes
    announceContentChange(message) {
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'polite');
        announcement.setAttribute('aria-atomic', 'true');
        announcement.className = 'sr-only';
        announcement.textContent = message;

        document.body.appendChild(announcement);

        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }

    // Check if user prefers reduced motion
    prefersReducedMotion() {
        return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }

    // Debounce function for performance
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Format currency (for future use)
    formatCurrency(amount, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency
        }).format(amount);
    }

    // Get current page name
    getCurrentPage() {
        const path = window.location.pathname;
        const page = path.split('/').pop().replace('.html', '') || 'dashboard';
        return page;
    }

    // Initialize page-specific UI
    initializePageUI() {
        const page = this.getCurrentPage();

        switch (page) {
            case 'tracker':
                this.initializeTrackerUI();
                break;
            case 'pcos':
                this.initializePCOSUI();
                break;
            case 'nutrition':
                this.initializeNutritionUI();
                break;
            case 'profile':
                this.initializeProfileUI();
                break;
            default:
                this.initializeDashboardUI();
        }

        this.updateNavigationActive(page);
        this.updatePageTitle(page.charAt(0).toUpperCase() + page.slice(1));
    }

    // Initialize dashboard UI
    initializeDashboardUI() {
        this.updateCycleInfo();
        this.updatePCOSRisk();
        this.updateNutritionTip();
        this.updateMiniCalendar();
    }

    // Initialize tracker UI
    initializeTrackerUI() {
        chartsManager.createSymptomChips('symptom-chips');
        this.updateCycleStats();
        this.updatePeriodCalendar();
        chartsManager.createCycleTrendChart('cycle-trend-chart');
        chartsManager.createSymptomFrequencyChart('symptom-frequency-chart');
        this.updateRecentPeriods();
    }

    // Initialize PCOS UI
    initializePCOSUI() {
        pcosManager.populateAssessmentForm();
    }

    // Initialize nutrition UI
    initializeNutritionUI() {
        this.updateCurrentPhase();
        this.updateDailySuggestion();
        this.updateMealPlan();
        this.updateShoppingList();
    }

    // Initialize profile UI
    initializeProfileUI() {
        this.loadProfileData();
        this.loadSettings();
    }

    // Update methods for dashboard
    updateCycleInfo() {
        const cycleInfoEl = document.getElementById('cycle-info');
        if (!cycleInfoEl) return;

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

        cycleInfoEl.innerHTML = html;
    }

    updatePCOSRisk() {
        const riskEl = document.getElementById('pcos-risk');
        if (!riskEl) return;

        const latestAssessment = pcosManager.getLatestAssessment();

        if (latestAssessment) {
            riskEl.innerHTML = `
                <p><strong>Risk Level:</strong> ${latestAssessment.level}</p>
                <p>Score: ${latestAssessment.score}</p>
                <small>Last assessed: ${cycleManager.formatDate(latestAssessment.date)}</small>
            `;
        }
    }

    updateNutritionTip() {
        const tipEl = document.getElementById('nutrition-tip');
        if (!tipEl) return;

        const suggestion = nutritionManager.getDailySuggestion();

        if (suggestion) {
            tipEl.innerHTML = `<p>${suggestion.suggestions[0]}</p>`;
        } else {
            tipEl.innerHTML = '<p>Log cycles to get personalized nutrition tips.</p>';
        }
    }

    updateMiniCalendar() {
        const calendarEl = document.getElementById('mini-calendar');
        if (!calendarEl) return;

        const now = new Date();
        chartsManager.createMiniCalendar('mini-calendar', now.getFullYear(), now.getMonth());
    }

    // Update methods for tracker
    updateCycleStats() {
        const statsEl = document.getElementById('cycle-stats');
        if (!statsEl) return;

        const stats = cycleManager.getCycleStats();

        if (stats) {
            statsEl.innerHTML = `
                <p><strong>Average Cycle Length:</strong> ${stats.averageLength} days</p>
                <p><strong>Regularity:</strong> ${stats.regularity}</p>
                <p><strong>Cycles Tracked:</strong> ${stats.totalCycles}</p>
            `;
        } else {
            statsEl.innerHTML = '<p>Log more cycles to see statistics.</p>';
        }
    }

    updatePeriodCalendar() {
        const calendarEl = document.getElementById('period-calendar');
        if (!calendarEl) return;

        const now = new Date();
        chartsManager.createMiniCalendar('period-calendar', now.getFullYear(), now.getMonth());
    }

    updateRecentPeriods() {
        const periodsEl = document.getElementById('recent-periods');
        if (!periodsEl) return;

        const cycles = cycleManager.getCycles().slice(-5); // Last 5 periods

        if (cycles.length === 0) {
            periodsEl.innerHTML = '<p>No periods logged yet.</p>';
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

        periodsEl.innerHTML = html;
    }

    // Update methods for nutrition
    updateCurrentPhase() {
        const phaseEl = document.getElementById('current-phase');
        if (!phaseEl) return;

        const phase = cycleManager.getCurrentPhase();

        if (phase) {
            phaseEl.innerHTML = `
                <div class="phase-indicator ${phase.phase}">
                    ${nutritionManager.getPhaseDisplayName(phase.phase)}
                </div>
                <p>Day ${phase.day} of your cycle</p>
            `;
        } else {
            phaseEl.innerHTML = '<p>Log your cycle to see current phase.</p>';
        }
    }

    updateDailySuggestion() {
        const suggestionEl = document.getElementById('daily-suggestion');
        if (!suggestionEl) return;

        const suggestion = nutritionManager.getDailySuggestion();

        if (suggestion) {
            suggestionEl.innerHTML = `
                <ul>
                    ${suggestion.suggestions.map(item => `<li>${item}</li>`).join('')}
                </ul>
            `;
        } else {
            suggestionEl.innerHTML = '<p>Log cycles to get daily nutrition suggestions.</p>';
        }
    }

    updateMealPlan() {
        const mealPlanEl = document.getElementById('meal-plan');
        if (!mealPlanEl) return;

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

    updateShoppingList() {
        const shoppingEl = document.getElementById('shopping-list');
        if (!shoppingEl) return;

        const shoppingList = nutritionManager.getShoppingList();

        if (shoppingList.length > 0) {
            shoppingEl.innerHTML = shoppingList.map(item => `
                <div class="shopping-item">
                    <input type="checkbox" id="item-${item.item}" ${item.checked ? 'checked' : ''}>
                    <label for="item-${item.item}">${item.item}</label>
                </div>
            `).join('');
        }
    }

    // Load profile data
    loadProfileData() {
        const profile = storage.getData('profile');

        if (profile.name) document.getElementById('profile-name').value = profile.name;
        if (profile.age) document.getElementById('profile-age').value = profile.age;
        if (profile.height_cm) document.getElementById('profile-height').value = profile.height_cm;
        if (profile.weight_kg) document.getElementById('profile-weight').value = profile.weight_kg;
    }

    // Load settings
    loadSettings() {
        const settings = storage.getData('settings');
        const prefs = storage.getData('nutrition_prefs');

        // Cycle settings
        if (settings.prediction_window) document.getElementById('prediction-window').value = settings.prediction_window;
        if (settings.cycle_length) document.getElementById('cycle-length').value = settings.cycle_length;

        // Dietary preferences
        if (prefs.filters) {
            prefs.filters.forEach(filter => {
                const checkbox = document.getElementById(filter.replace('_', '-'));
                if (checkbox) checkbox.checked = true;
            });
        }

        if (prefs.dislikes) document.getElementById('dislikes').value = prefs.dislikes.join(', ');

        // Feature flags
        if (settings.use_free_api) document.getElementById('use-free-api').checked = true;
        this.toggleAPIKeysSection();

        // Notification preferences
        const profile = storage.getData('profile');
        if (profile.notification_prefs) {
            Object.keys(profile.notification_prefs).forEach(pref => {
                const checkbox = document.getElementById(pref.replace('_', '-'));
                if (checkbox) checkbox.checked = profile.notification_prefs[pref];
            });
        }
    }

    // Toggle API keys section
    toggleAPIKeysSection() {
        const useAPI = document.getElementById('use-free-api').checked;
        const apiSection = document.getElementById('api-keys-section');
        if (apiSection) {
            apiSection.style.display = useAPI ? 'block' : 'none';
        }
    }
}

// Global UI manager instance
const uiManager = new UIManager();
