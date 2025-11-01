// RituCare - Nutrition Module
// Handles cycle-synced nutrition recommendations

class NutritionManager {
    constructor() {
        this.nutritionData = {};
        this.loadNutritionData();
    }

    // Load nutrition database
    async loadNutritionData() {
        try {
            const response = await fetch('data/nutrition-db.json');
            this.nutritionData = await response.json();
        } catch (error) {
            console.error('Error loading nutrition data:', error);
            // Fallback to hardcoded data
            this.nutritionData = this.getFallbackNutritionData();
        }
    }

    // Fallback nutrition data
    getFallbackNutritionData() {
        return {
            menstrual: {
                recommended: [
                    'Iron-rich foods (spinach, red meat, lentils)',
                    'Vitamin C foods (citrus, bell peppers)',
                    'Anti-inflammatory foods (ginger, turmeric)',
                    'Complex carbohydrates (whole grains, sweet potatoes)',
                    'Magnesium-rich foods (nuts, seeds, dark chocolate)'
                ],
                avoid: [
                    'Caffeine (coffee, tea, chocolate)',
                    'Alcohol',
                    'Processed foods',
                    'Excessive salt',
                    'Dairy (if sensitive)'
                ],
                tips: [
                    'Stay hydrated with herbal teas',
                    'Focus on warming, nourishing foods',
                    'Include protein with every meal',
                    'Consider gentle exercise like walking'
                ]
            },
            follicular: {
                recommended: [
                    'High-quality proteins (eggs, fish, lean meats)',
                    'Healthy fats (avocado, nuts, olive oil)',
                    'Colorful vegetables and fruits',
                    'Whole grains and legumes',
                    'Fermented foods (yogurt, sauerkraut)'
                ],
                avoid: [
                    'Excessive sugar',
                    'Refined carbohydrates',
                    'Trans fats',
                    'Artificial sweeteners'
                ],
                tips: [
                    'Build energy with nutrient-dense foods',
                    'Include healthy fats for hormone production',
                    'Try new recipes and flavors',
                    'Focus on variety in your diet'
                ]
            },
            ovulatory: {
                recommended: [
                    'Antioxidant-rich foods (berries, dark leafy greens)',
                    'Healthy fats (seeds, nuts, fatty fish)',
                    'Whole grains and legumes',
                    'Colorful vegetables',
                    'Herbs and spices (turmeric, cinnamon)'
                ],
                avoid: [
                    'Alcohol',
                    'Excessive caffeine',
                    'Processed foods',
                    'Artificial additives'
                ],
                tips: [
                    'Focus on fertility-supporting nutrients',
                    'Include zinc-rich foods (pumpkin seeds, oysters)',
                    'Stay hydrated',
                    'Enjoy light, fresh meals'
                ]
            },
            luteal: {
                recommended: [
                    'Complex carbohydrates (sweet potatoes, whole grains)',
                    'Magnesium-rich foods (dark chocolate, bananas, spinach)',
                    'Calcium-rich foods (kale, almonds, dairy)',
                    'B vitamins (leafy greens, eggs, legumes)',
                    'Healthy fats (avocado, nuts, seeds)'
                ],
                avoid: [
                    'Sugar and refined carbs',
                    'Caffeine',
                    'Alcohol',
                    'Excessive salt'
                ],
                tips: [
                    'Choose complex carbs to stabilize blood sugar',
                    'Include magnesium-rich foods for PMS relief',
                    'Practice mindful eating',
                    'Consider herbal teas for relaxation'
                ]
            }
        };
    }

    // Get current phase nutrition
    getCurrentPhaseNutrition() {
        const phase = cycleManager.getCurrentPhase();
        if (!phase) return null;

        return {
            phase: phase.phase,
            day: phase.day,
            data: this.nutritionData[phase.phase] || this.nutritionData.follicular
        };
    }

    // Get nutrition for specific phase
    getPhaseNutrition(phase) {
        return this.nutritionData[phase] || this.nutritionData.follicular;
    }

    // Get daily suggestion
    getDailySuggestion() {
        const phaseNutrition = this.getCurrentPhaseNutrition();
        if (!phaseNutrition) return null;

        const { data } = phaseNutrition;
        const suggestions = [];

        // Pick 2-3 recommended foods
        if (data.recommended && data.recommended.length > 0) {
            const recommended = this.shuffleArray(data.recommended).slice(0, 3);
            suggestions.push(...recommended);
        }

        // Pick 1 tip
        if (data.tips && data.tips.length > 0) {
            const tip = this.shuffleArray(data.tips)[0];
            suggestions.push(`Tip: ${tip}`);
        }

        return {
            phase: phaseNutrition.phase,
            suggestions: suggestions
        };
    }

    // Get meal plan for current phase
    getMealPlan() {
        const phaseNutrition = this.getCurrentPhaseNutrition();
        if (!phaseNutrition) return null;

        const { data } = phaseNutrition;
        const meals = {
            breakfast: this.generateMeal('breakfast', data.recommended),
            lunch: this.generateMeal('lunch', data.recommended),
            dinner: this.generateMeal('dinner', data.recommended),
            snacks: this.generateMeal('snacks', data.recommended)
        };

        return {
            phase: phaseNutrition.phase,
            meals: meals
        };
    }

    // Generate sample meal
    generateMeal(mealType, recommendations) {
        if (!recommendations || recommendations.length === 0) {
            return `Sample ${mealType} based on current phase`;
        }

        // Simple meal generation based on recommendations
        const mealTemplates = {
            breakfast: [
                'Oatmeal with {food1} and {food2}',
                'Smoothie with {food1}, {food2}, and yogurt',
                'Eggs with {food1} and whole grain toast',
                '{food1} parfait with nuts and seeds'
            ],
            lunch: [
                'Salad with {food1}, {food2}, and grilled protein',
                'Whole grain wrap with {food1} and vegetables',
                'Stir-fry with {food1}, {food2}, and brown rice',
                'Soup with {food1} and whole grain bread'
            ],
            dinner: [
                'Grilled protein with {food1} and quinoa',
                'Baked dish with {food1}, {food2}, and sweet potato',
                'Stir-fried vegetables with {food1} and tofu',
                'Whole grain pasta with {food1} and herbs'
            ],
            snacks: [
                '{food1} with a handful of nuts',
                'Fresh fruit with {food2}',
                'Vegetable sticks with hummus',
                'Greek yogurt with berries'
            ]
        };

        const templates = mealTemplates[mealType] || ['{food1} and {food2}'];
        const template = this.shuffleArray(templates)[0];

        // Extract food items from recommendations
        const foods = recommendations.map(rec => {
            // Extract main food items from recommendation strings
            const matches = rec.match(/([A-Za-z\s]+)(?:\s*\([^)]*\))?/);
            return matches ? matches[1].trim() : rec;
        });

        const food1 = foods[0] || 'seasonal vegetables';
        const food2 = foods[1] || 'lean protein';

        return template.replace('{food1}', food1).replace('{food2}', food2);
    }

    // Get shopping list suggestions
    getShoppingList() {
        const phaseNutrition = this.getCurrentPhaseNutrition();
        if (!phaseNutrition) return [];

        const { data } = phaseNutrition;
        const items = [];

        // Add recommended foods to shopping list
        if (data.recommended) {
            data.recommended.forEach(rec => {
                // Extract specific food items
                const foodMatches = rec.match(/\(([^)]+)\)/);
                if (foodMatches) {
                    const foods = foodMatches[1].split(', ');
                    items.push(...foods);
                }
            });
        }

        // Remove duplicates and format
        return [...new Set(items)].map(item => ({
            item: item.trim(),
            checked: false
        }));
    }

    // Filter nutrition based on dietary preferences
    filterByDietaryPrefs(nutritionData) {
        const prefs = storage.getData('nutrition_prefs');
        const profile = storage.getData('profile');

        let filtered = { ...nutritionData };

        // Apply dietary filters
        if (profile.dietary) {
            if (profile.dietary.vegetarian) {
                filtered = this.applyVegetarianFilter(filtered);
            }
            if (profile.dietary.vegan) {
                filtered = this.applyVeganFilter(filtered);
            }
            if (profile.dietary.dairy_free) {
                filtered = this.applyDairyFreeFilter(filtered);
            }
            if (profile.dietary.gluten_free) {
                filtered = this.applyGlutenFreeFilter(filtered);
            }
        }

        // Apply dislikes
        if (prefs.dislikes && prefs.dislikes.length > 0) {
            filtered = this.applyDislikesFilter(filtered, prefs.dislikes);
        }

        return filtered;
    }

    // Apply vegetarian filter
    applyVegetarianFilter(data) {
        const filtered = { ...data };
        Object.keys(filtered).forEach(phase => {
            if (filtered[phase].recommended) {
                filtered[phase].recommended = filtered[phase].recommended.filter(item =>
                    !item.toLowerCase().includes('meat') &&
                    !item.toLowerCase().includes('fish') &&
                    !item.toLowerCase().includes('chicken') &&
                    !item.toLowerCase().includes('beef') &&
                    !item.toLowerCase().includes('pork')
                );
            }
        });
        return filtered;
    }

    // Apply vegan filter
    applyVeganFilter(data) {
        let filtered = this.applyVegetarianFilter(data);

        Object.keys(filtered).forEach(phase => {
            if (filtered[phase].recommended) {
                filtered[phase].recommended = filtered[phase].recommended.filter(item =>
                    !item.toLowerCase().includes('dairy') &&
                    !item.toLowerCase().includes('yogurt') &&
                    !item.toLowerCase().includes('cheese') &&
                    !item.toLowerCase().includes('eggs') &&
                    !item.toLowerCase().includes('honey')
                );
            }
        });

        return filtered;
    }

    // Apply dairy-free filter
    applyDairyFreeFilter(data) {
        const filtered = { ...data };
        Object.keys(filtered).forEach(phase => {
            if (filtered[phase].recommended) {
                filtered[phase].recommended = filtered[phase].recommended.filter(item =>
                    !item.toLowerCase().includes('dairy') &&
                    !item.toLowerCase().includes('yogurt') &&
                    !item.toLowerCase().includes('cheese') &&
                    !item.toLowerCase().includes('milk')
                );
            }
        });
        return filtered;
    }

    // Apply gluten-free filter
    applyGlutenFreeFilter(data) {
        const filtered = { ...data };
        Object.keys(filtered).forEach(phase => {
            if (filtered[phase].recommended) {
                filtered[phase].recommended = filtered[phase].recommended.filter(item =>
                    !item.toLowerCase().includes('wheat') &&
                    !item.toLowerCase().includes('bread') &&
                    !item.toLowerCase().includes('pasta') &&
                    !item.toLowerCase().includes('flour')
                );
            }
        });
        return filtered;
    }

    // Apply dislikes filter
    applyDislikesFilter(data, dislikes) {
        const filtered = { ...data };
        const dislikeSet = new Set(dislikes.map(d => d.toLowerCase()));

        Object.keys(filtered).forEach(phase => {
            if (filtered[phase].recommended) {
                filtered[phase].recommended = filtered[phase].recommended.filter(item =>
                    !dislikeSet.has(item.toLowerCase())
                );
            }
        });

        return filtered;
    }

    // Utility: Shuffle array
    shuffleArray(array) {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }

    // Get phase color for UI
    getPhaseColor(phase) {
        const colors = {
            menstrual: '#ffeaa7',
            follicular: '#a29bfe',
            ovulatory: '#fd79a8',
            luteal: '#6c5ce7'
        };
        return colors[phase] || '#636e72';
    }

    // Get phase display name
    getPhaseDisplayName(phase) {
        const names = {
            menstrual: 'Menstrual Phase',
            follicular: 'Follicular Phase',
            ovulatory: 'Ovulatory Phase',
            luteal: 'Luteal Phase'
        };
        return names[phase] || phase;
    }

    // Save nutrition preferences
    saveNutritionPrefs(prefs) {
        return storage.updateNutritionPrefs(prefs);
    }

    // Get nutrition preferences
    getNutritionPrefs() {
        return storage.getData('nutrition_prefs');
    }
}

// Global nutrition manager instance
const nutritionManager = new NutritionManager();
