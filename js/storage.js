// RituCare - Storage Module
// Handles localStorage persistence, data export/import

class StorageManager {
    constructor() {
        this.storageKey = 'ritucare_data';
        this.initializeData();
    }

    // Initialize default data structure
    initializeData() {
        if (!localStorage.getItem(this.storageKey)) {
            const defaultData = {
                profile: {
                    name: '',
                    age: null,
                    height_cm: null,
                    weight_kg: null,
                    dietary: {
                        vegetarian: false,
                        vegan: false,
                        dairy_free: false,
                        gluten_free: false
                    },
                    notification_prefs: {
                        period_reminders: true,
                        ovulation_reminders: true,
                        nutrition_reminders: true
                    }
                },
                cycles: [],
                assessments: [],
                nutrition_prefs: {
                    filters: [],
                    dislikes: []
                },
                settings: {
                    use_free_api: false,
                    api_keys: {
                        usda: '',
                        edamam: { id: '', key: '' }
                    },
                    prediction_window: 6,
                    cycle_length: 28
                }
            };
            localStorage.setItem(this.storageKey, JSON.stringify(defaultData));
        }
    }

    // Get all data
    getAllData() {
        try {
            const data = localStorage.getItem(this.storageKey);
            return data ? JSON.parse(data) : {};
        } catch (error) {
            console.error('Error reading data from localStorage:', error);
            return {};
        }
    }

    // Save all data
    saveAllData(data) {
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(data));
            return true;
        } catch (error) {
            console.error('Error saving data to localStorage:', error);
            return false;
        }
    }

    // Get specific data section
    getData(section) {
        const data = this.getAllData();
        return data[section] || {};
    }

    // Save specific data section
    saveData(section, sectionData) {
        const data = this.getAllData();
        data[section] = sectionData;
        return this.saveAllData(data);
    }

    // Update profile
    updateProfile(profileData) {
        const data = this.getAllData();
        data.profile = { ...data.profile, ...profileData };
        return this.saveAllData(data);
    }

    // Add cycle entry
    addCycle(cycleData) {
        const data = this.getAllData();
        data.cycles.push({
            ...cycleData,
            id: Date.now(),
            date_logged: new Date().toISOString()
        });
        // Sort cycles by start date
        data.cycles.sort((a, b) => new Date(a.start) - new Date(b.start));
        return this.saveAllData(data);
    }

    // Update cycle entry
    updateCycle(cycleId, cycleData) {
        const data = this.getAllData();
        const index = data.cycles.findIndex(c => c.id === cycleId);
        if (index !== -1) {
            data.cycles[index] = { ...data.cycles[index], ...cycleData };
            return this.saveAllData(data);
        }
        return false;
    }

    // Delete cycle entry
    deleteCycle(cycleId) {
        const data = this.getAllData();
        data.cycles = data.cycles.filter(c => c.id !== cycleId);
        return this.saveAllData(data);
    }

    // Add PCOS assessment
    addAssessment(assessmentData) {
        const data = this.getAllData();
        data.assessments.push({
            ...assessmentData,
            id: Date.now(),
            date: new Date().toISOString()
        });
        // Keep only last 10 assessments
        if (data.assessments.length > 10) {
            data.assessments = data.assessments.slice(-10);
        }
        return this.saveAllData(data);
    }

    // Update settings
    updateSettings(settingsData) {
        const data = this.getAllData();
        data.settings = { ...data.settings, ...settingsData };
        return this.saveAllData(data);
    }

    // Update nutrition preferences
    updateNutritionPrefs(prefsData) {
        const data = this.getAllData();
        data.nutrition_prefs = { ...data.nutrition_prefs, ...prefsData };
        return this.saveAllData(data);
    }

    // Export data as JSON
    exportData() {
        const data = this.getAllData();
        const dataStr = JSON.stringify(data, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });

        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `ritucare-data-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    // Import data from JSON
    importData(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const importedData = JSON.parse(e.target.result);

                    // Validate data structure
                    if (this.validateDataStructure(importedData)) {
                        // Merge or replace based on user preference
                        const mergeData = confirm('Do you want to merge with existing data? Click OK to merge, Cancel to replace.');

                        if (mergeData) {
                            const existingData = this.getAllData();
                            const mergedData = this.mergeData(existingData, importedData);
                            this.saveAllData(mergedData);
                        } else {
                            this.saveAllData(importedData);
                        }

                        resolve(true);
                    } else {
                        reject(new Error('Invalid data structure'));
                    }
                } catch (error) {
                    reject(new Error('Invalid JSON file'));
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    // Validate data structure
    validateDataStructure(data) {
        const requiredSections = ['profile', 'cycles', 'assessments', 'nutrition_prefs', 'settings'];
        return requiredSections.every(section => data.hasOwnProperty(section));
    }

    // Merge imported data with existing data
    mergeData(existing, imported) {
        return {
            profile: { ...existing.profile, ...imported.profile },
            cycles: [...existing.cycles, ...imported.cycles].sort((a, b) => new Date(a.start) - new Date(b.start)),
            assessments: [...existing.assessments, ...imported.assessments].sort((a, b) => new Date(a.date) - new Date(b.date)),
            nutrition_prefs: { ...existing.nutrition_prefs, ...imported.nutrition_prefs },
            settings: { ...existing.settings, ...imported.settings }
        };
    }

    // Reset all data
    resetData() {
        if (confirm('Are you sure you want to reset all data? This cannot be undone.')) {
            localStorage.removeItem(this.storageKey);
            this.initializeData();
            return true;
        }
        return false;
    }

    // Load sample data for demo
    loadSampleData() {
        if (confirm('This will replace your current data with sample data. Continue?')) {
            fetch('data/sample-data.json')
                .then(response => response.json())
                .then(sampleData => {
                    this.saveAllData(sampleData);
                    location.reload();
                })
                .catch(error => {
                    console.error('Error loading sample data:', error);
                    alert('Failed to load sample data');
                });
        }
    }

    // Get storage usage info
    getStorageInfo() {
        const data = localStorage.getItem(this.storageKey);
        const size = data ? new Blob([data]).size : 0;
        return {
            size: size,
            sizeFormatted: this.formatBytes(size),
            available: this.getAvailableStorage()
        };
    }

    // Format bytes for display
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Estimate available storage (rough approximation)
    getAvailableStorage() {
        try {
            const testKey = '__storage_test__';
            let testData = 'x';
            while (true) {
                localStorage.setItem(testKey, testData);
                testData += testData;
            }
        } catch (e) {
            const available = localStorage.getItem(testKey).length * 2; // Rough estimate
            localStorage.removeItem(testKey);
            return this.formatBytes(available);
        }
    }
}

// Global storage instance
const storage = new StorageManager();
