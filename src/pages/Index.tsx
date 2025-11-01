import { CycleOverview } from "@/components/CycleOverview";
import { WaterTracker } from "@/components/WaterTracker";
import { CycleCalendar } from "@/components/CycleCalendar";
import { QuickTips } from "@/components/QuickTips";

const Index = () => {
  return (
    <div className="min-h-screen bg-[var(--gradient-subtle)]">
      {/* Main Content */}
      <main className="w-full px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Overview Cards */}
        <section>
          <CycleOverview />
        </section>

        {/* Calendar and Water Tracker */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <CycleCalendar />
          <WaterTracker />
        </section>

        {/* Health Tips */}
        <section>
          <QuickTips />
        </section>
      </main>
    </div>
  );
};

export default Index;
