import { Card } from "@/components/ui/card";
import { Lightbulb, Apple, Activity } from "lucide-react";

export const QuickTips = () => {
  const tips = [
    {
      icon: Apple,
      title: "Nutrition Tip",
      description: "Iron-rich foods like spinach and lentils help during menstruation",
      gradient: "from-primary/10 to-primary/5",
      iconColor: "text-primary",
    },
    {
      icon: Activity,
      title: "Exercise",
      description: "Light yoga can help reduce cramps and improve mood",
      gradient: "from-accent/10 to-accent/5",
      iconColor: "text-accent",
    },
    {
      icon: Lightbulb,
      title: "Wellness",
      description: "Track your symptoms to identify patterns in your cycle",
      gradient: "from-secondary/10 to-secondary/5",
      iconColor: "text-secondary",
    },
  ];

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-foreground">Health Tips</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {tips.map((tip, index) => (
          <Card
            key={index}
            className={`p-4 bg-gradient-to-br ${tip.gradient} border-primary/10 hover:shadow-[var(--shadow-card)] transition-[var(--transition-smooth)]`}
          >
            <div className="flex items-start gap-3">
              <div className={`p-2 bg-white/50 rounded-lg ${tip.iconColor}`}>
                <tip.icon className="w-5 h-5" />
              </div>
              <div>
                <h4 className="font-semibold text-foreground mb-1">{tip.title}</h4>
                <p className="text-sm text-muted-foreground">{tip.description}</p>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
};
