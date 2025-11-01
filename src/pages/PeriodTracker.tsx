import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Calendar } from "@/components/ui/calendar";
import { Badge } from "@/components/ui/badge";
import { Calendar as CalendarIcon, Plus, TrendingUp } from "lucide-react";
import { toast } from "sonner";

export default function PeriodTracker() {
  const [selectedDate, setSelectedDate] = useState<Date | undefined>(new Date());
  const [periodDates, setPeriodDates] = useState<Date[]>([
    new Date(2024, 11, 1),
    new Date(2024, 11, 2),
    new Date(2024, 11, 3),
    new Date(2024, 11, 4),
    new Date(2024, 11, 5),
  ]);

  const [symptoms, setSymptoms] = useState({
    cramps: false,
    bloating: false,
    headache: false,
    mood: false,
    fatigue: false,
  });

  const toggleSymptom = (symptom: keyof typeof symptoms) => {
    setSymptoms((prev) => ({ ...prev, [symptom]: !prev[symptom] }));
    toast.success(`Symptom ${symptoms[symptom] ? "removed" : "logged"}`);
  };

  const logPeriod = () => {
    if (selectedDate) {
      toast.success("Period logged successfully", {
        description: `Date: ${selectedDate.toLocaleDateString()}`,
      });
    }
  };

  return (
    <div className="min-h-screen bg-[var(--gradient-subtle)]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <div className="p-3 bg-gradient-to-br from-primary/20 to-primary/10 rounded-xl">
            <CalendarIcon className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-foreground">Period Tracker</h1>
            <p className="text-muted-foreground">Log and track your menstrual cycle</p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Calendar Section */}
          <Card className="p-6">
            <h2 className="text-lg font-semibold text-foreground mb-4">Cycle Calendar</h2>
            <Calendar
              mode="single"
              selected={selectedDate}
              onSelect={setSelectedDate}
              className="rounded-lg border-0"
              modifiers={{
                period: periodDates,
              }}
              modifiersStyles={{
                period: {
                  backgroundColor: "hsl(var(--primary))",
                  color: "white",
                  fontWeight: "bold",
                },
              }}
            />
            <div className="mt-4 flex gap-4 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-primary" />
                <span className="text-muted-foreground">Period Days</span>
              </div>
            </div>
            <Button onClick={logPeriod} className="w-full mt-4 bg-gradient-to-r from-primary to-primary/80">
              <Plus className="w-4 h-4 mr-2" />
              Log Period for Selected Date
            </Button>
          </Card>

          {/* Symptoms Tracker */}
          <div className="space-y-6">
            <Card className="p-6">
              <h2 className="text-lg font-semibold text-foreground mb-4">Track Symptoms</h2>
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(symptoms).map(([key, value]) => (
                  <Button
                    key={key}
                    onClick={() => toggleSymptom(key as keyof typeof symptoms)}
                    variant={value ? "default" : "outline"}
                    className={
                      value
                        ? "bg-gradient-to-br from-primary to-primary/80"
                        : "hover:border-primary"
                    }
                  >
                    {key.charAt(0).toUpperCase() + key.slice(1)}
                  </Button>
                ))}
              </div>
            </Card>

            {/* Cycle Stats */}
            <Card className="p-6">
              <div className="flex items-center gap-3 mb-4">
                <TrendingUp className="w-5 h-5 text-primary" />
                <h2 className="text-lg font-semibold text-foreground">Cycle Statistics</h2>
              </div>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Average Cycle Length</span>
                  <Badge className="bg-primary/20 text-primary hover:bg-primary/30">28 days</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Average Period Length</span>
                  <Badge className="bg-primary/20 text-primary hover:bg-primary/30">5 days</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Last Period</span>
                  <Badge className="bg-primary/20 text-primary hover:bg-primary/30">Dec 1, 2024</Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Next Expected</span>
                  <Badge className="bg-accent/20 text-accent hover:bg-accent/30">Dec 28, 2024</Badge>
                </div>
              </div>
            </Card>
          </div>
        </div>

        {/* Recent History */}
        <Card className="p-6">
          <h2 className="text-lg font-semibold text-foreground mb-4">Recent Cycles</h2>
          <div className="space-y-3">
            {[
              { start: "Dec 1, 2024", end: "Dec 5, 2024", length: 5, symptoms: 3 },
              { start: "Nov 3, 2024", end: "Nov 7, 2024", length: 5, symptoms: 2 },
              { start: "Oct 6, 2024", end: "Oct 10, 2024", length: 5, symptoms: 4 },
            ].map((cycle, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-4 rounded-lg bg-gradient-to-br from-muted/50 to-muted/30 hover:shadow-[var(--shadow-card)] transition-[var(--transition-smooth)]"
              >
                <div className="flex items-center gap-4">
                  <div className="p-2 bg-primary/20 rounded-lg">
                    <CalendarIcon className="w-5 h-5 text-primary" />
                  </div>
                  <div>
                    <p className="font-medium text-foreground">
                      {cycle.start} - {cycle.end}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {cycle.length} days • {cycle.symptoms} symptoms logged
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
