import { Card } from "@/components/ui/card";
import { Droplets, Heart, Calendar } from "lucide-react";

export const CycleOverview = () => {
  const currentDay = 12;
  const cycleLength = 28;
  const nextPeriod = 16;

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <Card className="p-8 bg-gradient-to-br from-primary/10 to-primary/5 border-primary/20 hover:shadow-[var(--shadow-card)] transition-[var(--transition-smooth)]">
        <div className="flex items-center gap-4 mb-4">
          <div className="p-3 bg-primary/20 rounded-full">
            <Calendar className="w-6 h-6 text-primary" />
          </div>
          <h3 className="font-semibold text-foreground">Current Day</h3>
        </div>
        <p className="text-4xl font-bold text-primary mb-2">Day {currentDay}</p>
        <p className="text-base text-muted-foreground">of {cycleLength}-day cycle</p>
      </Card>

      <Card className="p-8 bg-gradient-to-br from-accent/10 to-accent/5 border-accent/20 hover:shadow-[var(--shadow-card)] transition-[var(--transition-smooth)]">
        <div className="flex items-center gap-4 mb-4">
          <div className="p-3 bg-accent/20 rounded-full">
            <Heart className="w-6 h-6 text-accent" />
          </div>
          <h3 className="font-semibold text-foreground">Next Period</h3>
        </div>
        <p className="text-4xl font-bold text-accent mb-2">{nextPeriod} days</p>
        <p className="text-base text-muted-foreground">Expected on Dec 28</p>
      </Card>

      <Card className="p-8 bg-gradient-to-br from-secondary/10 to-secondary/5 border-secondary/20 hover:shadow-[var(--shadow-card)] transition-[var(--transition-smooth)]">
        <div className="flex items-center gap-4 mb-4">
          <div className="p-3 bg-secondary/20 rounded-full">
            <Droplets className="w-6 h-6 text-secondary" />
          </div>
          <h3 className="font-semibold text-foreground">Water Today</h3>
        </div>
        <p className="text-4xl font-bold text-secondary mb-2">6/8</p>
        <p className="text-base text-muted-foreground">glasses logged</p>
      </Card>
    </div>
  );
};
