import { Card } from "@/components/ui/card";
import { Calendar } from "@/components/ui/calendar";
import { useState } from "react";

export const CycleCalendar = () => {
  const [date, setDate] = useState<Date | undefined>(new Date());

  // Sample period dates (in a real app, this would come from user data)
  const periodDates = [
    new Date(2024, 11, 1),
    new Date(2024, 11, 2),
    new Date(2024, 11, 3),
    new Date(2024, 11, 4),
    new Date(2024, 11, 5),
  ];

  const fertileDates = [
    new Date(2024, 11, 13),
    new Date(2024, 11, 14),
    new Date(2024, 11, 15),
    new Date(2024, 11, 16),
    new Date(2024, 11, 17),
  ];

  return (
    <Card className="p-4">
      <div className="mb-4">
        <h3 className="text-xl font-semibold text-foreground mb-3">Cycle Calendar</h3>
        <div className="flex gap-6 text-base">
          <div className="flex items-center gap-3">
            <div className="w-5 h-5 rounded-full bg-primary" />
            <span className="text-muted-foreground">Period</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-5 h-5 rounded-full bg-accent" />
            <span className="text-muted-foreground">Fertile Window</span>
          </div>
        </div>
      </div>

      <Calendar
        mode="single"
        selected={date}
        onSelect={setDate}
        className="rounded-lg border-0 w-full max-w-md mx-auto"
        modifiers={{
          period: periodDates,
          fertile: fertileDates,
        }}
        modifiersStyles={{
          period: {
            backgroundColor: "hsl(var(--primary))",
            color: "white",
            fontWeight: "bold",
          },
          fertile: {
            backgroundColor: "hsl(var(--accent))",
            color: "white",
          },
        }}
      />
    </Card>
  );
};
