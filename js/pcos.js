// RituCare - PCOS Assessment Module
// Handles PCOS risk calculation and assessment logic

class PCOSManager {
    constructor() {
        this.symptoms = [
            'irregular_periods',
            'hirsutism',
            'acne',
            'weight_gain',
            'hair_loss',
            'fertility_difficulty',
            'mood_changes',
            'skin_darkening'
        ];
    }

    // Calculate PCOS risk score
    calculateRiskScore(assessmentData) {
        let score = 0;

        // BMI factor
        const bmi = cycleManager.calculateBMI(assessmentData.height, assessmentData.weight);
        if (bmi >= 25) score += 2;

        // Cycle regularity factor
        if (assessmentData.irregular_cycles) score += 3;

        // Symptom factors (0.5 points each)
        const symptomCount = this.symptoms.filter(symptom => assessmentData[symptom]).length;
        score += symptomCount * 0.5;

        return {
            score: Math.round(score * 10) / 10, // Round to 1 decimal
            bmi: bmi,
            level: this.getRiskLevel(score),
            factors: this.getContributingFactors(assessmentData, bmi)
        };
    }

    // Get risk level based on score
    getRiskLevel(score) {
        if (score < 3) return 'Low';
        if (score < 5.5) return 'Medium';
        return 'High';
    }

    // Get contributing factors
    getContributingFactors(assessmentData, bmi) {
        const factors = [];

        if (bmi >= 25) {
            factors.push('BMI ≥ 25');
        }

        if (assessmentData.irregular_cycles) {
            factors.push('Irregular cycles');
        }

        this.symptoms.forEach(symptom => {
            if (assessmentData[symptom]) {
                factors.push(this.formatSymptomName(symptom));
            }
        });

        return factors;
    }

    // Format symptom name for display
    formatSymptomName(symptom) {
        const names = {
            irregular_periods: 'Irregular periods',
            hirsutism: 'Hirsutism',
            acne: 'Acne',
            weight_gain: 'Weight gain',
            hair_loss: 'Hair loss',
            fertility_difficulty: 'Fertility difficulty',
            mood_changes: 'Mood changes',
            skin_darkening: 'Skin darkening'
        };
        return names[symptom] || symptom;
    }

    // Get recommendations based on risk level
    getRecommendations(riskLevel, factors) {
        const recommendations = {
            Low: [
                'Maintain healthy lifestyle',
                'Regular exercise and balanced diet',
                'Monitor cycle regularity',
                'Consider annual check-ups'
            ],
            Medium: [
                'Consult healthcare provider for evaluation',
                'Consider lifestyle modifications',
                'Track symptoms and cycles regularly',
                'Discuss hormonal testing if concerned'
            ],
            High: [
                'Consult endocrinologist or gynecologist',
                'Consider comprehensive PCOS evaluation',
                'Discuss treatment options (medication, lifestyle)',
                'Regular monitoring of metabolic health',
                'Consider fertility consultation if planning pregnancy'
            ]
        };

        return recommendations[riskLevel] || [];
    }

    // Get lifestyle tips based on risk factors
    getLifestyleTips(factors) {
        const tips = [];

        if (factors.includes('BMI ≥ 25')) {
            tips.push('Focus on gradual weight management through diet and exercise');
            tips.push('Aim for 150 minutes of moderate exercise per week');
        }

        if (factors.includes('Irregular cycles')) {
            tips.push('Track your cycles consistently to identify patterns');
            tips.push('Consider cycle-syncing nutrition and lifestyle');
        }

        if (factors.some(f => ['Hirsutism', 'Acne', 'Hair loss'].includes(f))) {
            tips.push('Consult dermatologist for hormonal acne or hair concerns');
        }

        if (factors.includes('Fertility difficulty')) {
            tips.push('Track ovulation and consider fertility awareness methods');
        }

        if (factors.includes('Mood changes')) {
            tips.push('Practice stress management techniques');
            tips.push('Consider adequate sleep and nutrition for mood stability');
        }

        return tips;
    }

    // Save assessment
    saveAssessment(assessmentData) {
        const result = this.calculateRiskScore(assessmentData);
        const assessment = {
            ...assessmentData,
            ...result,
            recommendations: this.getRecommendations(result.level, result.factors),
            lifestyleTips: this.getLifestyleTips(result.factors)
        };

        return storage.addAssessment(assessment);
    }

    // Get assessment history
    getAssessmentHistory() {
        return storage.getData('assessments').sort((a, b) => new Date(b.date) - new Date(a.date));
    }

    // Get latest assessment
    getLatestAssessment() {
        const history = this.getAssessmentHistory();
        return history.length > 0 ? history[0] : null;
    }

    // Get risk level color
    getRiskLevelColor(level) {
        const colors = {
            Low: '#00b894',
            Medium: '#fdcb6e',
            High: '#e17055'
        };
        return colors[level] || '#6c757d';
    }

    // Get risk level description
    getRiskLevelDescription(level) {
        const descriptions = {
            Low: 'Your symptoms suggest a low likelihood of PCOS. Continue healthy lifestyle habits.',
            Medium: 'Your symptoms are concerning for PCOS. Consider professional evaluation.',
            High: 'Your symptoms strongly suggest PCOS. Please consult a healthcare provider.'
        };
        return descriptions[level] || '';
    }

    // Validate assessment data
    validateAssessmentData(data) {
        const errors = [];

        if (!data.age || data.age < 13 || data.age > 100) {
            errors.push('Please enter a valid age (13-100)');
        }

        if (!data.height || data.height < 100 || data.height > 250) {
            errors.push('Please enter a valid height (100-250 cm)');
        }

        if (!data.weight || data.weight < 30 || data.weight > 300) {
            errors.push('Please enter a valid weight (30-300 kg)');
        }

        return errors;
    }

    // Get assessment form data from DOM
    getAssessmentFormData() {
        return {
            age: parseInt(document.getElementById('age')?.value) || null,
            height: parseFloat(document.getElementById('height')?.value) || null,
            weight: parseFloat(document.getElementById('weight')?.value) || null,
            irregular_cycles: document.getElementById('irregular')?.checked || false,
            irregular_periods: document.getElementById('irregular_periods')?.checked || false,
            hirsutism: document.getElementById('hirsutism')?.checked || false,
            acne: document.getElementById('acne')?.checked || false,
            weight_gain: document.getElementById('weight_gain')?.checked || false,
            hair_loss: document.getElementById('hair_loss')?.checked || false,
            fertility_difficulty: document.getElementById('fertility_difficulty')?.checked || false,
            mood_changes: document.getElementById('mood_changes')?.checked || false,
            skin_darkening: document.getElementById('skin_darkening')?.checked || false
        };
    }

    // Populate assessment form
    populateAssessmentForm() {
        const profile = storage.getData('profile');

        if (profile.age) document.getElementById('age').value = profile.age;
        if (profile.height_cm) document.getElementById('height').value = profile.height_cm;
        if (profile.weight_kg) document.getElementById('weight').value = profile.weight_kg;

        // Calculate and display BMI
        this.updateBMIDisplay();
    }

    // Update BMI display
    updateBMIDisplay() {
        const height = parseFloat(document.getElementById('height')?.value);
        const weight = parseFloat(document.getElementById('weight')?.value);
        const bmiDisplay = document.getElementById('bmi-display');

        if (bmiDisplay && height && weight) {
            const bmi = cycleManager.calculateBMI(height, weight);
            bmiDisplay.value = bmi ? `${bmi} (${this.getBMICategory(bmi)})` : '';
        }
    }

    // Get BMI category
    getBMICategory(bmi) {
        if (bmi < 18.5) return 'Underweight';
        if (bmi < 25) return 'Normal';
        if (bmi < 30) return 'Overweight';
        return 'Obese';
    }

    // Display assessment results
    displayAssessmentResults(assessment) {
        // Update risk level
        const riskLevelEl = document.getElementById('risk-level');
        if (riskLevelEl) {
            riskLevelEl.innerHTML = `
                <div class="alert alert-${assessment.level.toLowerCase() === 'low' ? 'success' : assessment.level.toLowerCase() === 'medium' ? 'warning' : 'danger'}">
                    <h5>Risk Level: ${assessment.level}</h5>
                    <p>${this.getRiskLevelDescription(assessment.level)}</p>
                    <p><strong>Score: ${assessment.score}</strong></p>
                </div>
            `;
        }

        // Update risk gauge
        this.updateRiskGauge(assessment.score, assessment.level);

        // Update recommendations
        const recommendationsEl = document.getElementById('pcos-recommendations');
        if (recommendationsEl) {
            const recommendationsHTML = assessment.recommendations.map(rec => `<li>${rec}</li>`).join('');
            const tipsHTML = assessment.lifestyleTips.map(tip => `<li>${tip}</li>`).join('');

            recommendationsEl.innerHTML = `
                <h6>Recommendations:</h6>
                <ul>${recommendationsHTML}</ul>
                ${tipsHTML.length > 0 ? `<h6>Lifestyle Tips:</h6><ul>${tipsHTML}</ul>` : ''}
                <p class="text-muted mt-2">This assessment is for informational purposes only. Please consult a healthcare professional for diagnosis and treatment.</p>
            `;
        }

        // Update assessment history
        this.displayAssessmentHistory();
    }

    // Update risk gauge chart
    updateRiskGauge(score, level) {
        const ctx = document.getElementById('pcos-risk-gauge');
        if (!ctx) return;

        const data = {
            datasets: [{
                data: [score, 10 - score],
                backgroundColor: [
                    this.getRiskLevelColor(level),
                    '#e9ecef'
                ],
                borderWidth: 0
            }]
        };

        const config = {
            type: 'doughnut',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
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
                id: 'centerText',
                beforeDraw: function(chart) {
                    const width = chart.width;
                    const height = chart.height;
                    const ctx = chart.ctx;

                    ctx.restore();
                    ctx.font = 'bold 20px Inter';
                    ctx.textBaseline = 'middle';
                    ctx.fillStyle = '#333';

                    const text = score.toString();
                    const textX = Math.round((width - ctx.measureText(text).width) / 2);
                    const textY = height / 2;

                    ctx.fillText(text, textX, textY);
                    ctx.save();
                }
            }]
        };

        new Chart(ctx, config);
    }

    // Display assessment history
    displayAssessmentHistory() {
        const historyEl = document.getElementById('assessment-history');
        if (!historyEl) return;

        const history = this.getAssessmentHistory();

        if (history.length === 0) {
            historyEl.innerHTML = '<p>No assessments completed yet.</p>';
            return;
        }

        const historyHTML = history.map(assessment => `
            <div class="assessment-item">
                <div class="date">${cycleManager.formatDate(assessment.date)}</div>
                <div class="score">${assessment.score}</div>
                <div class="level ${assessment.level.toLowerCase()}">${assessment.level}</div>
            </div>
        `).join('');

        historyEl.innerHTML = historyHTML;
    }
}

// Global PCOS manager instance
const pcosManager = new PCOSManager();
