import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Apple, Leaf, Utensils, TrendingUp } from "lucide-react";
import { toast } from "sonner";

export default function Nutrition() {
  const cyclePhase = "Follicular Phase";

  const nutritionTips = [
    {
      phase: "Menstrual Phase",
      icon: Apple,
      color: "text-primary",
      bg: "bg-primary/10",
      foods: ["Iron-rich foods (spinach, lentils)", "Omega-3 (salmon, walnuts)", "Dark chocolate", "Ginger tea"],
      avoid: ["Caffeine", "Processed foods", "High salt foods"],
    },
    {
      phase: "Follicular Phase",
      icon: Leaf,
      color: "text-secondary",
      bg: "bg-secondary/10",
      foods: ["Lean proteins (chicken, tofu)", "Fresh vegetables", "Fermented foods", "Whole grains"],
      avoid: ["Refined sugar", "Alcohol", "Heavy meals"],
    },
    {
      phase: "Ovulation Phase",
      icon: TrendingUp,
      color: "text-accent",
      bg: "bg-accent/10",
      foods: ["Antioxidant-rich fruits", "Quinoa", "Raw vegetables", "Nuts and seeds"],
      avoid: ["Inflammatory foods", "Excess dairy", "Fried foods"],
    },
    {
      phase: "Luteal Phase",
      icon: Utensils,
      color: "text-primary",
      bg: "bg-primary/10",
      foods: ["Complex carbs", "Leafy greens", "Calcium-rich foods", "Magnesium sources"],
      avoid: ["High sodium", "Caffeine", "Refined carbs"],
    },
  ];

  const mealPlan = {
    breakfast: [
      { name: "Oatmeal with berries and almonds", calories: 350 },
      { name: "Greek yogurt with honey and granola", calories: 280 },
      { name: "Avocado toast with eggs", calories: 400 },
    ],
    lunch: [
      { name: "Quinoa bowl with chickpeas and vegetables", calories: 450 },
      { name: "Grilled chicken salad", calories: 380 },
      { name: "Lentil soup with whole grain bread", calories: 420 },
    ],
    dinner: [
      { name: "Baked salmon with sweet potato", calories: 520 },
      { name: "Stir-fried tofu with brown rice", calories: 480 },
      { name: "Turkey meatballs with zucchini noodles", calories: 460 },
    ],
  };

  const logMeal = (meal: string) => {
    toast.success("Meal logged!", {
      description: meal,
    });
  };

  return (
    <div className="min-h-screen bg-[var(--gradient-subtle)]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-gradient-to-br from-primary/20 to-primary/10 rounded-xl">
              <Apple className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-foreground">Nutrition</h1>
              <p className="text-muted-foreground">Cycle-based nutrition guidance</p>
            </div>
          </div>
          <Badge className="bg-secondary/20 text-secondary hover:bg-secondary/30">
            Current: {cyclePhase}
          </Badge>
        </div>

        {/* Cycle-Based Nutrition */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {nutritionTips.map((tip, idx) => (
            <Card key={idx} className="p-6 hover:shadow-[var(--shadow-card)] transition-[var(--transition-smooth)]">
              <div className="flex items-center gap-3 mb-4">
                <div className={`p-2 rounded-lg ${tip.bg}`}>
                  <tip.icon className={`w-5 h-5 ${tip.color}`} />
                </div>
                <h3 className="text-lg font-semibold text-foreground">{tip.phase}</h3>
              </div>

              <div className="space-y-4">
                <div>
                  <h4 className="text-sm font-medium text-foreground mb-2">Recommended Foods</h4>
                  <div className="flex flex-wrap gap-2">
                    {tip.foods.map((food, i) => (
                      <Badge key={i} variant="outline" className="bg-secondary/5">
                        {food}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-foreground mb-2">Foods to Limit</h4>
                  <div className="flex flex-wrap gap-2">
                    {tip.avoid.map((food, i) => (
                      <Badge key={i} variant="outline" className="bg-destructive/5 text-destructive">
                        {food}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </div>

        {/* Daily Meal Plan */}
        <Card className="p-6">
          <h2 className="text-xl font-semibold text-foreground mb-6">Today's Meal Suggestions</h2>

          <div className="space-y-6">
            {Object.entries(mealPlan).map(([mealType, meals]) => (
              <div key={mealType}>
                <h3 className="text-lg font-medium text-foreground mb-3 capitalize">{mealType}</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {meals.map((meal, idx) => (
                    <Card
                      key={idx}
                      className="p-4 bg-gradient-to-br from-muted/50 to-muted/30 hover:shadow-[var(--shadow-card)] transition-[var(--transition-smooth)]"
                    >
                      <div className="space-y-3">
                        <div>
                          <p className="font-medium text-foreground">{meal.name}</p>
                          <p className="text-sm text-muted-foreground">{meal.calories} calories</p>
                        </div>
                        <Button
                          onClick={() => logMeal(meal.name)}
                          variant="outline"
                          size="sm"
                          className="w-full hover:border-primary hover:bg-primary/5"
                        >
                          Log Meal
                        </Button>
                      </div>
                    </Card>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Nutrition Tips */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="p-6 bg-gradient-to-br from-primary/10 to-primary/5">
            <h3 className="font-semibold text-foreground mb-2">Hydration</h3>
            <p className="text-sm text-muted-foreground">
              Aim for 8-10 glasses of water daily. Increase during menstruation to reduce bloating.
            </p>
          </Card>

          <Card className="p-6 bg-gradient-to-br from-accent/10 to-accent/5">
            <h3 className="font-semibold text-foreground mb-2">Iron Intake</h3>
            <p className="text-sm text-muted-foreground">
              Include iron-rich foods during and after periods to replenish iron stores.
            </p>
          </Card>

          <Card className="p-6 bg-gradient-to-br from-secondary/10 to-secondary/5">
            <h3 className="font-semibold text-foreground mb-2">Balance</h3>
            <p className="text-sm text-muted-foreground">
              Focus on balanced meals with proteins, healthy fats, and complex carbs.
            </p>
          </Card>
        </div>
      </div>
    </div>
  );
}
