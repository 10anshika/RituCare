// RituCare - Cycle Calculation Module
// Handles period tracking, predictions, and cycle analysis

class CycleManager {
    constructor() {
        this.settings = storage.getData('settings');
    }

    // Calculate BMI
    calculateBMI(heightCm, weightKg) {
        if (!heightCm || !weightKg) return null;
        const heightM = heightCm / 100;
        return (weightKg / (heightM * heightM)).toFixed(1);
    }

    // Get all cycles sorted by start date
    getCycles() {
        return storage.getData('cycles').sort((a, b) => new Date(a.start) - new Date(b.start));
    }

    // Get recent cycles for averaging
    getRecentCycles(count = this.settings.prediction_window || 6) {
        const cycles = this.getCycles();
        return cycles.slice(-count);
    }

    // Calculate average cycle length
    getAverageCycleLength() {
        const cycles = this.getRecentCycles();
        if (cycles.length < 2) return this.settings.cycle_length || 28;

        const lengths = [];
        for (let i = 0; i < cycles.length - 1; i++) {
            const current = cycles[i];
            const next = cycles[i + 1];
            const length = Math.round((new Date(next.start) - new Date(current.start)) / (1000 * 60 * 60 * 24));
            lengths.push(length);
        }

        return lengths.length > 0 ? Math.round(lengths.reduce((a, b) => a + b) / lengths.length) : this.settings.cycle_length || 28;
    }

    // Calculate average period length
    getAveragePeriodLength() {
        const cycles = this.getRecentCycles();
        const periods = cycles.filter(c => c.end).map(c => {
            return Math.round((new Date(c.end) - new Date(c.start)) / (1000 * 60 * 60 * 24)) + 1;
        });

        return periods.length > 0 ? Math.round(periods.reduce((a, b) => a + b) / periods.length) : 5;
    }

    // Get last period start date
    getLastPeriodStart() {
        const cycles = this.getCycles();
        return cycles.length > 0 ? new Date(cycles[cycles.length - 1].start) : null;
    }

    // Predict next period
    predictNextPeriod() {
        const lastStart = this.getLastPeriodStart();
        if (!lastStart) return null;

        const avgLength = this.getAverageCycleLength();
        const nextPeriod = new Date(lastStart);
        nextPeriod.setDate(nextPeriod.getDate() + avgLength);

        return nextPeriod;
    }

    // Predict ovulation date
    predictOvulation() {
        const nextPeriod = this.predictNextPeriod();
        if (!nextPeriod) return null;

        // Ovulation typically occurs 14 days before next period
        const ovulation = new Date(nextPeriod);
        ovulation.setDate(ovulation.getDate() - 14);

        return ovulation;
    }

    // Get fertile window (around ovulation)
    getFertileWindow() {
        const ovulation = this.predictOvulation();
        if (!ovulation) return null;

        const start = new Date(ovulation);
        start.setDate(start.getDate() - 2);

        const end = new Date(ovulation);
        end.setDate(end.getDate() + 2);

        return { start, end };
    }

    // Get current cycle day
    getCurrentCycleDay() {
        const lastStart = this.getLastPeriodStart();
        if (!lastStart) return null;

        const today = new Date();
        const diffTime = today - lastStart;
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

        return diffDays;
    }

    // Get current cycle phase
    getCurrentPhase() {
        const cycleDay = this.getCurrentCycleDay();
        if (!cycleDay) return null;

        const avgCycleLength = this.getAverageCycleLength();

        // Adjust phase lengths based on actual cycle
        const menstrualLength = 5;
        const follicularLength = Math.max(1, Math.round((avgCycleLength - 14) * 0.4)); // ~40% of cycle minus luteal
        const ovulatoryDay = avgCycleLength - 14;
        const lutealLength = 14;

        if (cycleDay <= menstrualLength) {
            return { phase: 'menstrual', day: cycleDay, totalDays: menstrualLength };
        } else if (cycleDay <= menstrualLength + follicularLength) {
            return { phase: 'follicular', day: cycleDay - menstrualLength, totalDays: follicularLength };
        } else if (cycleDay === ovulatoryDay) {
            return { phase: 'ovulatory', day: 1, totalDays: 1 };
        } else if (cycleDay <= avgCycleLength) {
            return { phase: 'luteal', day: cycleDay - ovulatoryDay, totalDays: lutealLength };
        } else {
            // Post-cycle, predict next
            return { phase: 'post-cycle', day: cycleDay - avgCycleLength, totalDays: null };
        }
    }

    // Get cycle statistics
    getCycleStats() {
        const cycles = this.getRecentCycles();
        if (cycles.length === 0) return null;

        const lengths = [];
        for (let i = 0; i < cycles.length - 1; i++) {
            const length = Math.round((new Date(cycles[i + 1].start) - new Date(cycles[i].start)) / (1000 * 60 * 60 * 24));
            lengths.push(length);
        }

        const avgLength = lengths.reduce((a, b) => a + b, 0) / lengths.length;
        const minLength = Math.min(...lengths);
        const maxLength = Math.max(...lengths);
        const variability = maxLength - minLength;

        return {
            averageLength: Math.round(avgLength),
            minLength,
            maxLength,
            variability,
            totalCycles: cycles.length,
            regularity: variability <= 7 ? 'regular' : 'irregular'
        };
    }

    // Check for cycle irregularities
    getCycleIrregularities() {
        const stats = this.getCycleStats();
        if (!stats) return [];

        const irregularities = [];

        if (stats.variability > 7) {
            irregularities.push('irregular');
        }

        if (stats.averageLength < 21) {
            irregularities.push('short');
        }

        if (stats.averageLength > 35) {
            irregularities.push('long');
        }

        if (this.checkForMissedPeriods()) {
            irregularities.push('missed');
        }

        return irregularities;
    }

    // Check for missed periods (gap > 60 days)
    checkForMissedPeriods() {
        const cycles = this.getCycles();
        for (let i = 0; i < cycles.length - 1; i++) {
            const gap = (new Date(cycles[i + 1].start) - new Date(cycles[i].end || cycles[i].start)) / (1000 * 60 * 60 * 24);
            if (gap > 60) return true;
        }
        return false;
    }

    // Get symptom frequency data
    getSymptomFrequency() {
        const cycles = this.getRecentCycles();
        const symptomCounts = {};

        cycles.forEach(cycle => {
            if (cycle.symptoms) {
                cycle.symptoms.forEach(symptom => {
                    symptomCounts[symptom] = (symptomCounts[symptom] || 0) + 1;
                });
            }
        });

        return Object.entries(symptomCounts)
            .map(([symptom, count]) => ({ symptom, count, percentage: (count / cycles.length) * 100 }))
            .sort((a, b) => b.count - a.count);
    }

    // Get cycle length trend data for charts
    getCycleLengthTrend() {
        const cycles = this.getCycles();
        const trendData = [];

        for (let i = 0; i < cycles.length - 1; i++) {
            const length = Math.round((new Date(cycles[i + 1].start) - new Date(cycles[i].start)) / (1000 * 60 * 60 * 24));
            trendData.push({
                cycle: i + 1,
                length: length,
                date: cycles[i].start
            });
        }

        return trendData;
    }

    // Generate calendar data for a month
    generateCalendarData(year, month) {
        const cycles = this.getCycles();
        const nextPeriod = this.predictNextPeriod();
        const fertileWindow = this.getFertileWindow();

        const firstDay = new Date(year, month, 1);
        const lastDay = new Date(year, month + 1, 0);
        const startDate = new Date(firstDay);
        startDate.setDate(startDate.getDate() - firstDay.getDay());

        const calendarDays = [];

        for (let d = new Date(startDate); d <= lastDay; d.setDate(d.getDate() + 1)) {
            const dayData = {
                date: new Date(d),
                day: d.getDate(),
                isCurrentMonth: d.getMonth() === month,
                isToday: this.isToday(d),
                isPeriod: false,
                isPredictedPeriod: false,
                isFertile: false,
                symptoms: []
            };

            // Check if day is in any logged period
            cycles.forEach(cycle => {
                const periodStart = new Date(cycle.start);
                const periodEnd = cycle.end ? new Date(cycle.end) : new Date(periodStart);

                if (d >= periodStart && d <= periodEnd) {
                    dayData.isPeriod = true;
                    dayData.symptoms = cycle.symptoms || [];
                }
            });

            // Check predicted period
            if (nextPeriod && this.isSameDay(d, nextPeriod)) {
                dayData.isPredictedPeriod = true;
            }

            // Check fertile window
            if (fertileWindow && d >= fertileWindow.start && d <= fertileWindow.end) {
                dayData.isFertile = true;
            }

            calendarDays.push(dayData);
        }

        return calendarDays;
    }

    // Helper: Check if date is today
    isToday(date) {
        const today = new Date();
        return date.getDate() === today.getDate() &&
               date.getMonth() === today.getMonth() &&
               date.getFullYear() === today.getFullYear();
    }

    // Helper: Check if two dates are the same day
    isSameDay(date1, date2) {
        return date1.getDate() === date2.getDate() &&
               date1.getMonth() === date2.getMonth() &&
               date1.getFullYear() === date2.getFullYear();
    }

    // Format date for display
    formatDate(date, options = {}) {
        if (!date) return '';

        const defaultOptions = {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        };

        return new Intl.DateTimeFormat('en-US', { ...defaultOptions, ...options }).format(new Date(date));
    }

    // Get days until next period
    getDaysUntilNextPeriod() {
        const nextPeriod = this.predictNextPeriod();
        if (!nextPeriod) return null;

        const today = new Date();
        const diffTime = nextPeriod - today;
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

        return diffDays > 0 ? diffDays : 0;
    }

    // Get days until ovulation
    getDaysUntilOvulation() {
        const ovulation = this.predictOvulation();
        if (!ovulation) return null;

        const today = new Date();
        const diffTime = ovulation - today;
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

        return diffDays > 0 ? diffDays : 0;
    }
}

// Global cycle manager instance
const cycleManager = new CycleManager();
