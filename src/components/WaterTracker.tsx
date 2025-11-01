import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Droplets, Plus, Minus } from "lucide-react";
import { toast } from "sonner";

export const WaterTracker = () => {
  const [waterCount, setWaterCount] = useState(6);
  const goal = 10;

  const addWater = () => {
    if (waterCount < 12) {
      setWaterCount(waterCount + 1);
      toast.success("Water logged!", {
        description: `${waterCount + 1} of ${goal} glasses today`,
      });
    }
  };

  const removeWater = () => {
    if (waterCount > 0) {
      setWaterCount(waterCount - 1);
    }
  };

  const percentage = (waterCount / goal) * 100;

  return (
    <Card className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-full bg-gradient-to-br from-secondary/20 to-accent/20">
            <Droplets className="w-6 h-6 text-secondary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-foreground">Water Intake</h3>
            <p className="text-sm text-muted-foreground">Stay hydrated throughout your cycle</p>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="flex items-center justify-center gap-4">
          <Button
            onClick={removeWater}
            variant="outline"
            size="icon"
            className="rounded-full hover:border-secondary hover:bg-secondary/10"
          >
            <Minus className="w-4 h-4" />
          </Button>

          <div className="text-center">
            <p className="text-4xl font-bold text-secondary">{waterCount}</p>
            <p className="text-sm text-muted-foreground">of {goal} glasses</p>
          </div>

          <Button
            onClick={addWater}
            size="icon"
            className="rounded-full bg-gradient-to-br from-secondary to-accent hover:opacity-90"
          >
            <Plus className="w-4 h-4" />
          </Button>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm text-muted-foreground">
            <span>Progress</span>
            <span>{Math.round(percentage)}%</span>
          </div>
          <div className="h-3 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-secondary to-accent transition-all duration-500 rounded-full"
              style={{ width: `${Math.min(percentage, 100)}%` }}
            />
          </div>
        </div>

        <div className="grid grid-cols-4 gap-2 mt-4">
          {Array.from({ length: goal }).map((_, i) => (
            <div
              key={i}
              className={`h-12 rounded-lg flex items-center justify-center transition-[var(--transition-smooth)] ${
                i < waterCount
                  ? "bg-gradient-to-br from-secondary to-accent text-white"
                  : "bg-muted"
              }`}
            >
              <Droplets className={`w-4 h-4 ${i < waterCount ? "opacity-100" : "opacity-30"}`} />
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};
